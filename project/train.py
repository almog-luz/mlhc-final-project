# Training CLI to fit calibrated models from a labeled cohort
from __future__ import annotations
import os
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Support both package and script contexts
try:
    from .extract import (
        get_first_admissions,
        get_demographics,
        get_vitals_48h,
        get_labs_48h,
        get_prescriptions_48h,
        get_procedures_48h,
    )
    from .features import build_features
    from .metrics_utils import compute_binary_metrics, metrics_to_dict
except ImportError:  # pragma: no cover - script mode
    from extract import (
        get_first_admissions,
        get_demographics,
        get_vitals_48h,
        get_labs_48h,
        get_prescriptions_48h,
        get_procedures_48h,
    )
    from features import build_features
    from metrics_utils import compute_binary_metrics, metrics_to_dict


def parse_args():
    p = argparse.ArgumentParser(description="Train calibrated models for available targets using a labeled cohort.")
    p.add_argument("--project-id", required=True, help="GCP Project ID for BigQuery")
    p.add_argument("--input", required=True, help="CSV with subject_id and label columns (mortality_label, prolonged_los_label, readmission_label as available)")
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), 'models'), help="Directory to save artifacts")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for splits")
    return p.parse_args()


def _load_and_build_features(client: bigquery.Client, subject_ids: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    first_adm = get_first_admissions(client, subject_ids)
    if first_adm.empty:
        return first_adm, pd.DataFrame()
    first_adm['admittime'] = pd.to_datetime(first_adm['admittime'])
    first_adm['dischtime'] = pd.to_datetime(first_adm['dischtime'])
    first_adm['los_hours'] = (first_adm['dischtime'] - first_adm['admittime']).dt.total_seconds() / 3600.0
    first_adm = first_adm[first_adm['los_hours'] >= 54].copy()
    hadm_ids = first_adm['hadm_id'].dropna().astype(int).tolist()

    demo = get_demographics(client, first_adm['subject_id'].dropna().astype(int).tolist())
    vitals = get_vitals_48h(client, hadm_ids)
    labs = get_labs_48h(client, hadm_ids)
    rx = get_prescriptions_48h(client, hadm_ids)
    proc = get_procedures_48h(client, hadm_ids)

    X = build_features(first_adm, demo, vitals, labs, rx, proc)
    return first_adm, X


def _build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    # All features are numeric; impute missing with median, then scale for linear model stability
    num_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    pre = ColumnTransformer(transformers=[("num", num_transform, feature_cols)], remainder="drop")
    return pre


def _fit_model(X_t, y: np.ndarray) -> CalibratedClassifierCV:
    # Base estimator: logistic regression with class_weight balanced
    base = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    # Choose calibration method
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    method = "isotonic" if positives >= 50 and negatives >= 50 else "sigmoid"
    clf = CalibratedClassifierCV(base_estimator=base, method=method, cv=5)
    clf.fit(X_t, y)
    return clf


def train_from_labels(project_id: str, input_csv: str, output_dir: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict]:
    os.makedirs(output_dir, exist_ok=True)

    labels_df = pd.read_csv(input_csv)
    if 'subject_id' not in labels_df.columns:
        raise ValueError("Input CSV must contain subject_id")

    # Normalize label column names to expected
    rename_map = {
        "prolonged_los": "prolonged_los_label",
        "prolonged_LOS": "prolonged_los_label",
        "prolonged_LOS_label": "prolonged_los_label",
        "mortality": "mortality_label",
        "readmission": "readmission_label",
    }
    cols_lower = {c: c for c in labels_df.columns}
    for c in list(labels_df.columns):
        lc = c.lower()
        if lc in rename_map:
            cols_lower[c] = rename_map[lc]
        elif lc.endswith("_label"):
            cols_lower[c] = lc
    labels_df = labels_df.rename(columns=cols_lower)

    subject_ids = labels_df['subject_id'].dropna().astype(int).tolist()

    client = bigquery.Client(project=project_id)
    first_adm, X = _load_and_build_features(client, subject_ids)
    if X.empty:
        raise RuntimeError("No features built; check cohort and extraction.")

    # Align to labels_df subjects
    X = X.reindex(index=labels_df['subject_id']).fillna(0.0)

    feature_cols = list(X.columns)
    pre = _build_preprocessor(feature_cols)
    X_t = pre.fit_transform(X)

    # Persist preprocessor and feature columns now
    joblib.dump(pre, os.path.join(output_dir, 'preprocessor.joblib'))
    with open(os.path.join(output_dir, 'feature_columns.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f)

    # Train models for available targets
    targets = [
        ("mortality_label", "model_mortality.joblib", "mortality"),
        ("prolonged_los_label", "model_prolonged_los.joblib", "prolonged_los"),
        ("readmission_label", "model_readmission.joblib", "readmission"),
    ]

    metrics_report: Dict[str, Dict] = {}

    for y_col, fname, name in targets:
        if y_col not in labels_df.columns:
            continue
        y_all = labels_df[y_col].astype(int).values
        # Stratified split if both classes present
        strat = y_all if len(np.unique(y_all)) > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(X_t, y_all, test_size=test_size, random_state=random_state, stratify=strat)
        model = _fit_model(X_tr, y_tr)
        joblib.dump(model, os.path.join(output_dir, fname))

        # Evaluate
        proba_te = model.predict_proba(X_te)[:, 1]
        m = compute_binary_metrics(y_te, proba_te, threshold_objective="f1")
        metrics_report[name] = metrics_to_dict(m)

    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=2)

    return metrics_report


def main():
    args = parse_args()
    report = train_from_labels(args.project_id, args.input, args.output_dir, test_size=args.test_size, random_state=args.random_state)
    print(f"Saved artifacts to {args.output_dir}")
    if report:
        print("\nTest metrics summary:")
        for k, v in report.items():
            print(f"- {k}: n={v.get('n')}, prev={v.get('prevalence'):.3f}, AUC={v.get('roc_auc'):.3f}, PR-AUC={v.get('pr_auc'):.3f}, Brier={v.get('brier'):.3f}, ECE={v.get('ece'):.3f}")


if __name__ == "__main__":
    main()
