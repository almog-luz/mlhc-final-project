import os
import json
import time
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import optuna  # type: ignore
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier  # type: ignore
try:
    from xgboost.callback import EarlyStopping  # type: ignore
except Exception:  # pragma: no cover
    EarlyStopping = None  # type: ignore
import joblib

# Lightweight fast tuner that reuses a precomputed feature matrix + preprocessor output.
# It performs CV in-memory and reports mean ROC AUC for readmission_label only.


def parse_args():
    p = argparse.ArgumentParser(description="Fast Optuna XGBoost tuner for readmission.")
    p.add_argument("--labels", required=True, help="Path to labels.csv containing readmission_label column (and subject_id index alignment with features).")
    p.add_argument("--features", required=True, help="Path to features_full.parquet (precomputed features).")
    p.add_argument("--preprocessor", required=True, help="Path to preprocessor.joblib (fit on same features matrix).")
    p.add_argument("--output", required=True, help="Directory to store tuning study artifacts.")
    p.add_argument("--trials", type=int, default=50, help="Number of Optuna trials (default 50).")
    p.add_argument("--splits", required=False, help="Optional splits.json to restrict tuning to train subjects only.")
    p.add_argument("--cv-folds", type=int, default=3, help="Number of CV folds (default 3).")
    p.add_argument("--early-stopping-rounds", type=int, default=30, help="Early stopping rounds per fold (default 30).")
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument("--max-depth", nargs=2, type=int, default=[3,7], help="Min and max for max_depth search integer range.")
    p.add_argument("--n-estimators", nargs=2, type=int, default=[300,1200], help="Min/max boosting rounds search range.")
    p.add_argument("--learning-rate", nargs=2, type=float, default=[0.01,0.1], help="Min/max learning rate log range.")
    p.add_argument("--subsample", nargs=2, type=float, default=[0.6,1.0], help="Min/max subsample range.")
    p.add_argument("--colsample-bytree", nargs=2, type=float, default=[0.5,1.0], help="Min/max colsample_bytree range.")
    p.add_argument("--reg-lambda", nargs=2, type=float, default=[0.2,3.0], help="Min/max L2 regularization (log range).")
    p.add_argument("--reg-alpha", nargs=2, type=float, default=[0.0,1.0], help="Min/max L1 regularization (linear range).")
    p.add_argument("--scale-pos-weight", nargs=2, type=float, default=[0.5,4.0], help="Min/max multiplier around empirical n_neg/n_pos ratio.")
    p.add_argument("--no-pruning", action="store_true", help="Disable Optuna median pruning.")
    p.add_argument("--verbosity", type=int, default=0, help="XGBoost verbosity (0=silent).")
    return p.parse_args()


def load_data(labels_path: str, features_path: str, preprocessor_path: str, splits_path: str | None = None):
    labels_df = pd.read_csv(labels_path)
    if 'readmission_label' not in labels_df.columns:
        raise SystemExit("labels file must contain readmission_label column")
    feat = pd.read_parquet(features_path)
    if 'subject_id' in feat.columns and feat.index.name != 'subject_id':
        try:
            feat = feat.set_index('subject_id')
        except Exception:
            pass
    # Align order
    feat = feat.reindex(labels_df['subject_id']).fillna(0.0)
    if splits_path:
        try:
            import json as _json
            with open(splits_path, 'r', encoding='utf-8') as f:
                splits = _json.load(f)
            train_ids = set(splits.get('train', []))
            mask = labels_df['subject_id'].isin(train_ids)
            labels_df = labels_df[mask].reset_index(drop=True)
            feat = feat.loc[labels_df['subject_id']]
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to apply splits restriction: {e}")
    pre = joblib.load(preprocessor_path)
    X_t = pre.transform(feat)
    y = labels_df['readmission_label'].astype(int).to_numpy()
    return X_t, y, list(feat.columns)


def build_param_space(trial: optuna.Trial, args, base_spw: float):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', args.n_estimators[0], args.n_estimators[1], step=50),
        'learning_rate': trial.suggest_float('learning_rate', args.learning_rate[0], args.learning_rate[1], log=True),
        'max_depth': trial.suggest_int('max_depth', args.max_depth[0], args.max_depth[1]),
        'subsample': trial.suggest_float('subsample', args.subsample[0], args.subsample[1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', args.colsample_bytree[0], args.colsample_bytree[1]),
        'reg_lambda': trial.suggest_float('reg_lambda', args.reg_lambda[0], args.reg_lambda[1], log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', args.reg_alpha[0], args.reg_alpha[1]),
        'scale_pos_weight': base_spw * trial.suggest_float('spw_mult', args.scale_pos_weight[0], args.scale_pos_weight[1]),
    }
    return params


def objective_factory(X_t, y, args):
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    base_spw = (n_neg / max(1, n_pos))

    def objective(trial: optuna.Trial) -> float:
        params = build_param_space(trial, args, base_spw)
        fold_aucs: List[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(y), y)):
            X_tr = X_t[tr_idx]
            y_tr = y[tr_idx]
            X_va = X_t[va_idx]
            y_va = y[va_idx]
            model = XGBClassifier(
                **{
                    'n_estimators': params['n_estimators'],
                    'learning_rate': params['learning_rate'],
                    'max_depth': params['max_depth'],
                    'subsample': params['subsample'],
                    'colsample_bytree': params['colsample_bytree'],
                    'reg_lambda': params['reg_lambda'],
                    'reg_alpha': params['reg_alpha'],
                    'scale_pos_weight': params['scale_pos_weight'],
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'tree_method': 'hist',
                    'n_jobs': 4,
                    'verbosity': args.verbosity,
                }
            )
            # Early stopping: use built-in evaluation set
            # Attempt early stopping; if unsupported fall back to plain fit.
            fitted = False
            for attempt in ("early_stopping_rounds", "plain"):
                if fitted:
                    break
                try:
                    if attempt == "early_stopping_rounds":
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_va, y_va)],
                            verbose=False,
                            early_stopping_rounds=args.early_stopping_rounds,
                        )
                    else:
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_va, y_va)],
                            verbose=False,
                        )
                    fitted = True
                except TypeError:
                    continue
            proba = model.predict_proba(X_va)[:,1]
            auc = float(roc_auc_score(y_va, proba))
            fold_aucs.append(auc)
            # Report intermediate; allow pruning
            trial.report(auc, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_auc = float(np.mean(fold_aucs))
        trial.set_user_attr('fold_aucs', fold_aucs)
        return mean_auc

    return objective


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    X_t, y, feature_cols = load_data(args.labels, args.features, args.preprocessor, args.splits)

    pruner = optuna.pruners.NopPruner() if args.no_pruning else optuna.pruners.MedianPruner(n_startup_trials=max(5, args.cv_folds))
    study = optuna.create_study(direction='maximize', pruner=pruner)
    objective = objective_factory(X_t, y, args)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Persist results
    best = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
    }
    with open(os.path.join(args.output, 'best.json'), 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2)

    rows = []
    for t in study.trials:
        rows.append({
            'number': t.number,
            'value': t.value,
            'state': str(t.state),
            **{f'param_{k}': v for k,v in t.params.items()},
            'fold_aucs': getattr(t, 'user_attrs', {}).get('fold_aucs'),
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.output, 'trials.csv'), index=False)
    print(f"Best AUC: {best['best_value']:.6f}")
    print("Best Params:")
    for k,v in best['best_params'].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
