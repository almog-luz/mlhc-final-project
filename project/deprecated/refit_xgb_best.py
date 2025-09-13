import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier  # type: ignore
from .metrics_utils import compute_binary_metrics, metrics_to_dict
from sklearn.calibration import CalibratedClassifierCV

# Refit XGBoost on full dataset using best params saved by fast tuner (best.json)
# Expects: labels.csv with readmission_label, features_full.parquet, preprocessor.joblib, best.json

def parse_args():
    p = argparse.ArgumentParser(description="Refit best XGBoost params on full dataset and compute metrics.")
    p.add_argument("--labels", required=True, help="Path to labels.csv containing readmission_label column")
    p.add_argument("--features", required=True, help="Path to features_full.parquet")
    p.add_argument("--preprocessor", required=True, help="Path to preprocessor.joblib")
    p.add_argument("--best-json", required=True, help="Path to best.json produced by xgb_optuna_fast")
    p.add_argument("--output", required=True, help="Directory to write refit model + metrics")
    p.add_argument("--splits", required=True, help="Path to persistent splits.json with train/val/test subject_ids")
    p.add_argument("--eval-set", choices=['val','train'], default='val', help="Which split to evaluate on (default val). Test is forbidden until explicitly allowed.")
    p.add_argument("--no-threshold", action="store_true", help="Skip threshold selection (report only scores)")
    p.add_argument("--calibrate", action="store_true", help="Apply probability calibration on training split")
    p.add_argument("--calib-method", choices=['sigmoid','isotonic'], default='sigmoid', help="Calibration method (default sigmoid)")
    p.add_argument("--calib-cv", type=int, default=5, help="Inner CV folds for calibration (default 5)")
    p.add_argument("--save-proba", action="store_true", help="If set, save evaluation probabilities to proba.csv in output dir")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    labels = pd.read_csv(args.labels)
    if 'readmission_label' not in labels.columns:
        raise SystemExit("labels file must contain readmission_label")

    feat = pd.read_parquet(args.features)
    if 'subject_id' in feat.columns and feat.index.name != 'subject_id':
        try:
            feat = feat.set_index('subject_id')
        except Exception:
            pass
    feat = feat.reindex(labels['subject_id']).fillna(0.0)

    # Load splits
    with open(args.splits, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    train_ids = set(splits['train'])
    val_ids = set(splits['val'])
    if 'test' not in splits:
        raise SystemExit('splits file missing test set (expected key test)')
    test_ids = set(splits['test'])

    # Build boolean masks
    sid_series = labels['subject_id']
    train_mask = sid_series.isin(train_ids)
    val_mask = sid_series.isin(val_ids)
    test_mask = sid_series.isin(test_ids)
    # Safety: ensure disjoint
    if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
        raise SystemExit('Overlapping splits detected; aborting.')

    pre = joblib.load(args.preprocessor)
    X_all = pre.transform(feat)
    y_all = labels['readmission_label'].astype(int).to_numpy()

    import numpy as _np  # local import to avoid top-level clutter
    train_bool = _np.asarray(train_mask.values, dtype=bool)
    train_idx = _np.where(train_bool)[0]
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    eval_indices = train_idx  # default
    if args.eval_set == 'val':
        val_bool = _np.asarray(val_mask.values, dtype=bool)
        val_idx = _np.where(val_bool)[0]
        X_eval = X_all[val_idx]
        y_eval = y_all[val_idx]
        eval_name = 'val'
        eval_indices = val_idx
    else:
        X_eval = X_train
        y_eval = y_train
        eval_name = 'train'

    best = json.load(open(args.best_json, 'r', encoding='utf-8'))
    params = best.get('best_params', {})
    # Translate spw_mult into scale_pos_weight if present
    if 'spw_mult' in params:
        # base ratio = (n_neg / n_pos) estimated in tuner; here recompute
        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())
        base_ratio = n_neg / max(1, n_pos)
        params['scale_pos_weight'] = base_ratio * params.pop('spw_mult')

    model = XGBClassifier(
        n_estimators=params.get('n_estimators', 600),
        learning_rate=params.get('learning_rate', 0.03),
        max_depth=params.get('max_depth', 4),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        reg_lambda=params.get('reg_lambda', 1.0),
        reg_alpha=params.get('reg_alpha', 0.0),
        scale_pos_weight=params.get('scale_pos_weight', 1.0),
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        n_jobs=4,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    calibrated = False
    if args.calibrate:
        # Wrap in CalibratedClassifierCV using only training data; evaluate on separate val
        # Refit underlying base model inside calibrator; we pass prefit model to avoid refit? Keep consistent: refit for safety.
        base = model
        try:
            try:
                cal = CalibratedClassifierCV(estimator=base, method=args.calib_method, cv=args.calib_cv)
            except TypeError:
                cal = CalibratedClassifierCV(base_estimator=base, method=args.calib_method, cv=args.calib_cv)
            cal.fit(X_train, y_train)
            model = cal  # replace reference with calibrated model
            calibrated = True
        except Exception as e:  # pragma: no cover
            print(f"Warning: calibration failed: {e}")

    proba = model.predict_proba(X_eval)[:,1]
    metrics = compute_binary_metrics(
        y_true=y_eval,
        y_proba=proba,
        threshold_objective=None if args.no_threshold else 'f1',
        n_calib_bins=10,
        threshold_min_recall=None,
    )
    metrics_dict = { 'readmission': metrics_to_dict(metrics), 'meta': {'evaluation_split': eval_name, 'train_size': int(train_mask.sum()), 'eval_size': int((val_mask if eval_name=='val' else train_mask).sum())} }

    with open(os.path.join(args.output, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)

    joblib.dump(model, os.path.join(args.output, 'model_readmission.joblib'))
    if args.save_proba:
        import pandas as _pd
        select_subjects = labels.iloc[eval_indices]['subject_id'].to_list()
        proba_df = _pd.DataFrame({'subject_id': select_subjects,'readmission_label': y_eval,'proba': proba})
        proba_df.to_csv(os.path.join(args.output, 'proba.csv'), index=False)
    # Save feature columns for traceability
    with open(os.path.join(args.output, 'feature_columns.json'), 'w', encoding='utf-8') as f:
        json.dump({'feature_columns': list(feat.columns)}, f, indent=2)
    # Save run metadata
    with open(os.path.join(args.output, 'run_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'source_best_json': os.path.abspath(args.best_json)}, f, indent=2)

    print(f"Refit complete. Evaluated on {eval_name} split. Metrics written to {os.path.join(args.output, 'metrics.json')}")
    if calibrated:
        print("Calibration applied:", args.calib_method, f"(cv={args.calib_cv})")
    print("ROC AUC:", metrics.roc_auc, "PR AUC:", metrics.pr_auc)

if __name__ == '__main__':
    main()
