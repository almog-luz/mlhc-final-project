import os
import argparse
import json
import math
import shutil
import time
from typing import Optional, Dict, Any

import optuna  # type: ignore
# Safe pandas import (type checker friendly)
try:  # pragma: no cover
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from .train import train_from_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuning for XGBoost readmission model using reused features.")
    p.add_argument("--input", required=True, help="Labels CSV with subject_id and readmission_label column.")
    p.add_argument("--features", required=True, help="Path to features_full.parquet (reused).")
    p.add_argument("--preprocessor", required=False, default=None, help="Optional preprocessor.joblib path (reuse transform only).")
    p.add_argument("--output-dir", required=True, help="Directory to store study artifacts.")
    p.add_argument("--trials", type=int, default=10, help="Number of Optuna trials (default 10).")
    p.add_argument("--project-id", required=False, default="ml-for-healthcare-2025", help="GCP project ID (only used if extraction required; reused features bypass queries).")
    p.add_argument("--seed", type=int, default=42, help="Base random seed.")
    p.add_argument("--no-calibration", action="store_true", help="Disable calibration for faster AUC optimization.")
    p.add_argument("--calib-cv", type=int, default=3, help="Calibration CV folds if calibration enabled.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Persist a manifest for reproducibility
    manifest = {
        'input': args.input,
        'features': args.features,
        'preprocessor': args.preprocessor,
        'trials': args.trials,
        'seed': args.seed,
        'calibration': not args.no_calibration,
        'calib_cv': args.calib_cv,
    }
    with open(os.path.join(args.output_dir, 'tune_manifest.json'), 'w', encoding='utf-8') as mf:
        json.dump(manifest, mf, indent=2)

    # Upfront validation: ensure readmission label present (directly or via rename map in train.py)
    if pd is None:
        raise SystemExit("pandas is required for tuning validation but is not installed.")
    try:
        head = pd.read_csv(args.input, nrows=5)  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to read labels CSV {args.input}: {e}")
    # Accept if any column that train.py will normalize to readmission_label exists
    acceptable_cols = {c.lower() for c in head.columns}
    has_readm = ("readmission_label" in acceptable_cols) or ("readmission" in acceptable_cols)
    if not has_readm:
        raise SystemExit(
            f"Labels file {args.input} missing readmission label column. Present columns: {list(head.columns)}. "
            "Expected at least 'subject_id' and either 'readmission' or 'readmission_label'."
        )

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        }
        trial_id = f"trial_{trial.number:03d}"
        trial_dir = os.path.join(args.output_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        t0 = time.time()
        metrics = train_from_labels(
            project_id=args.project_id,
            input_csv=args.input,
            output_dir=trial_dir,
            test_size=0.3,
            random_state=args.seed + trial.number,
            generate_artifacts=False,
            use_cache_first=False,
            show_progress=False,
            profile=False,
            model_type='xgb',
            calib_cv=args.calib_cv,
            skip_importance=True,
            cv_folds=0,
            finalize_mode=False,
            readmission_model_override='same',
            readmission_no_calibration=args.no_calibration,
            only_readmission=True,
            calib_progress=False,
            readmission_min_recall=None,
            readmission_pos_weight=1.0,
            write_provenance=False,
            allow_history=False,
            phenotype_clusters=0,
            stacking_meta=False,
            feature_select_target=None,
            feature_select_C=0.5,
            reuse_features=args.features,
            reuse_preprocessor=args.preprocessor,
            xgb_params=params,
        )
        dur = time.time() - t0
        # Extract AUC; if missing, prune trial so Optuna doesn't treat as failure with NaN
        if 'readmission' not in metrics or 'roc_auc' not in metrics['readmission']:
            raise optuna.TrialPruned("readmission metrics missing (check label file and training output)")
        auc = float(metrics['readmission']['roc_auc'])
        # Save quick summary for this trial
        summary = {
            'trial': trial.number,
            'params': params,
            'roc_auc': auc,
            'duration_sec': dur,
        }
        with open(os.path.join(trial_dir, 'summary.json'), 'w', encoding='utf-8') as sf:
            json.dump(summary, sf, indent=2)
        # Report intermediate value
        trial.report(auc, step=0)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # Persist study results
    best = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
    }
    with open(os.path.join(args.output_dir, 'best_trial.json'), 'w', encoding='utf-8') as bf:
        json.dump(best, bf, indent=2)
    # All trials CSV
    rows = []
    for t in study.trials:
        r = {'number': t.number, 'value': t.value, 'state': str(t.state)}
        r.update({f"param_{k}": v for k, v in t.params.items()})
        rows.append(r)
    if pd is not None:  # type: ignore
        pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'trials.csv'), index=False)  # type: ignore[attr-defined]
    print("Best AUC:", best['best_value'])
    print("Best Params:", best['best_params'])


if __name__ == "__main__":
    main()
