# Training CLI to fit calibrated models from a labeled cohort
from __future__ import annotations
import os
import json
import argparse
from typing import Dict, List, Tuple, Optional
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

# Optional progress bar with safe widget fallback (avoid IProgress warning)
try:  # pragma: no cover
    from IPython import get_ipython  # type: ignore
except Exception:  # pragma: no cover
    get_ipython = lambda: None  # type: ignore

def _resolve_tqdm():  # pragma: no cover
    try:
        from tqdm.auto import tqdm as _auto  # type: ignore
        # Attempt to detect ipywidgets availability
        try:
            import ipywidgets  # noqa: F401
            return _auto
        except Exception:
            from tqdm import tqdm as _plain  # type: ignore
            return _plain
    except Exception:
        try:
            from tqdm import tqdm as _plain  # type: ignore
            return _plain
        except Exception:
            return None

tqdm = _resolve_tqdm()
import joblib

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
from .evaluation import (
    compute_curve_data,
    save_curve_data,
    maybe_plot_curves,
    permutation_importance_report,
    save_feature_importance,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train calibrated models for available targets using a labeled cohort.")
    p.add_argument("--project-id", required=True, help="GCP Project ID for BigQuery")
    p.add_argument("--input", required=True, help="CSV with subject_id and label columns (mortality_label, prolonged_los_label, readmission_label as available)")
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), 'models'), help="Directory to save artifacts")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for splits")
    p.add_argument("--use-cache-first", action="store_true", help="If set, attempt to load cached parquet extracts in data/extracted_cache before querying BigQuery.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--profile", action="store_true", help="Collect and print per-stage timing information.")
    return p.parse_args()


def _load_cached_extracts(cache_dir: str) -> Dict[str, pd.DataFrame]:
    """Load previously cached extraction parquet files if present.

    Returns mapping name->DataFrame. Missing files yield empty DataFrames.
    """
    out: Dict[str, pd.DataFrame] = {}
    names = [
        ("first_admissions", "first_admissions.parquet"),
        ("demographics", "demographics.parquet"),
        ("vitals_48h", "vitals_48h.parquet"),
        ("labs_48h", "labs_48h.parquet"),
        ("prescriptions_48h", "prescriptions_48h.parquet"),
        ("procedures_48h", "procedures_48h.parquet"),
    ]
    for key, fname in names:
        fp = os.path.join(cache_dir, fname)
        if os.path.exists(fp):
            try:
                out[key] = pd.read_parquet(fp)
            except Exception as e:  # pragma: no cover
                print(f"Warning: failed to read cache file {fp}: {e}")
                out[key] = pd.DataFrame()
        else:
            out[key] = pd.DataFrame()
    return out


def _load_and_build_features(client: Optional[bigquery.Client], subject_ids: List[int], use_cache_first: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if client is None:
        # Without a client we cannot query; return empty frames to trigger cache fallback logic upstream.
        return pd.DataFrame(), pd.DataFrame()
    first_adm = get_first_admissions(client, subject_ids)
    if first_adm.empty:
        return first_adm, pd.DataFrame()
    first_adm['admittime'] = pd.to_datetime(first_adm['admittime'])
    first_adm['dischtime'] = pd.to_datetime(first_adm['dischtime'])
    first_adm['los_hours'] = (first_adm['dischtime'] - first_adm['admittime']).dt.total_seconds() / 3600.0
    first_adm = first_adm[first_adm['los_hours'] >= 54].copy()
    hadm_ids = first_adm['hadm_id'].dropna().astype(int).tolist()

    demo = vitals = labs = rx = proc = pd.DataFrame()

    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'extracted_cache')
    cache_used = False
    if use_cache_first:
        cached = _load_cached_extracts(cache_dir)
        # Basic sanity: require at least demographics + one time-series modality
        if not cached['demographics'].empty and (not cached['vitals_48h'].empty or not cached['labs_48h'].empty):
            demo = cached['demographics']
            vitals = cached['vitals_48h']
            labs = cached['labs_48h']
            rx = cached['prescriptions_48h']
            proc = cached['procedures_48h']
            cache_used = True
            print("Loaded features from cache directory (skipped BigQuery):", cache_dir)

    if not cache_used and client is not None:
        demo = get_demographics(client, first_adm['subject_id'].dropna().astype(int).tolist())
        vitals = get_vitals_48h(client, hadm_ids)
        labs = get_labs_48h(client, hadm_ids)
        rx = get_prescriptions_48h(client, hadm_ids)
        proc = get_procedures_48h(client, hadm_ids)
        # Opportunistically cache extracts for reproducibility
        try:
            os.makedirs(cache_dir, exist_ok=True)
            if not first_adm.empty:
                first_adm.to_parquet(os.path.join(cache_dir, 'first_admissions.parquet'), index=False)
            if not demo.empty:
                demo.to_parquet(os.path.join(cache_dir, 'demographics.parquet'), index=False)
            if not vitals.empty:
                vitals.to_parquet(os.path.join(cache_dir, 'vitals_48h.parquet'), index=False)
            if not labs.empty:
                labs.to_parquet(os.path.join(cache_dir, 'labs_48h.parquet'), index=False)
            if not rx.empty:
                rx.to_parquet(os.path.join(cache_dir, 'prescriptions_48h.parquet'), index=False)
            if not proc.empty:
                proc.to_parquet(os.path.join(cache_dir, 'procedures_48h.parquet'), index=False)
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to write cache extracts: {e}")

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
    """Fit a calibrated logistic regression model.

    Newer scikit-learn versions deprecate the 'base_estimator' kwarg in favor of 'estimator'.
    We attempt to pass 'estimator' first and fall back for older versions for compatibility.
    """
    base = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    method = "isotonic" if positives >= 50 and negatives >= 50 else "sigmoid"
    # Always prefer the modern 'estimator' kwarg. If this environment has only the old signature
    # we raise a clear error telling the user to upgrade scikit-learn rather than silently using
    # the deprecated path that is now invalid in newer versions.
    try:
        clf = CalibratedClassifierCV(estimator=base, method=method, cv=5)
        print(f"CalibratedClassifierCV created with method={method} (using 'estimator' kwarg).")
    except TypeError as e:
        raise RuntimeError(
            "CalibratedClassifierCV signature mismatch. This code expects a scikit-learn version that "
            "supports the 'estimator' kwarg. Please upgrade scikit-learn (>=1.5). Original error: " + str(e)
        )
    clf.fit(X_t, y)
    return clf


def train_from_labels(project_id: str, input_csv: str, output_dir: str, test_size: float = 0.2, random_state: int = 42,
                      generate_artifacts: bool = True, use_cache_first: bool = False,
                      show_progress: bool = True, profile: bool = False) -> Dict[str, Dict]:
    """End-to-end training entry point.

    Returns mapping target_name -> metrics dictionary (serializable JSON structure).
    Raises RuntimeError if feature construction produced an empty matrix.
    """
    os.makedirs(output_dir, exist_ok=True)

    if show_progress:
        print("[1/5] Loading labels CSV ...")
    import time
    _t0_global = time.perf_counter()
    timings: Dict[str, float] = {}

    def _mark(stage: str, t_start: float):
        if profile:
            timings[stage] = timings.get(stage, 0.0) + (time.perf_counter() - t_start)

    t_stage = time.perf_counter()
    labels_df = pd.read_csv(input_csv)
    _mark("load_labels", t_stage)
    # Persist a copy of the labels used for this run (one canonical location) in the parent of output_dir
    try:
        import shutil
        run_root = os.path.abspath(os.path.join(output_dir, os.pardir))
        os.makedirs(run_root, exist_ok=True)
        labels_copy_path = os.path.join(run_root, 'labels.csv')
        shutil.copy2(input_csv, labels_copy_path)
        if show_progress:
            print(f"Copied labels to {labels_copy_path}")
    except Exception as e:  # pragma: no cover
        print(f"Warning: failed to copy labels file for run reproducibility: {e}")
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

    # BigQuery client initialization may fail if credentials are missing; provide helpful message
    client = None
    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        if not use_cache_first:
            print("Warning: Failed to initialize BigQuery client (" + str(e) + "). Attempting cache-only mode.")
            use_cache_first = True
    if show_progress:
        print("[2/5] Building / loading feature matrices ...")
    t_stage = time.perf_counter()
    first_adm, X = _load_and_build_features(client, subject_ids, use_cache_first=use_cache_first)
    _mark("feature_extraction_total", t_stage)
    # If initial attempt failed (empty) and cache flag is set, attempt pure cache load bypassing BigQuery.
    if X.empty and use_cache_first:
        cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'extracted_cache')
        cached = _load_cached_extracts(cache_dir)
        first_adm = cached.get('first_admissions', pd.DataFrame())
        # Minimal LOS filter if admissions cached
        if not first_adm.empty and 'admittime' in first_adm and 'dischtime' in first_adm:
            first_adm['admittime'] = pd.to_datetime(first_adm['admittime'])
            first_adm['dischtime'] = pd.to_datetime(first_adm['dischtime'])
            first_adm['los_hours'] = (first_adm['dischtime'] - first_adm['admittime']).dt.total_seconds() / 3600.0
            first_adm = first_adm[first_adm['los_hours'] >= 54].copy()
        X = build_features(
            first_adm,
            cached.get('demographics', pd.DataFrame()),
            cached.get('vitals_48h', pd.DataFrame()),
            cached.get('labs_48h', pd.DataFrame()),
            cached.get('prescriptions_48h', pd.DataFrame()),
            cached.get('procedures_48h', pd.DataFrame()),
        )
    if X.empty:
        raise RuntimeError("No features built; check cohort and extraction.")

    # Align to labels_df subjects
    X = X.reindex(index=labels_df['subject_id']).fillna(0.0)

    feature_cols = list(X.columns)
    if show_progress:
        print(f"[3/5] Fitting preprocessing pipeline on {X.shape[0]} patients x {X.shape[1]} features ...")
    pre = _build_preprocessor(feature_cols)
    t_stage = time.perf_counter()
    X_t = pre.fit_transform(X)
    _mark("preprocessor_fit_transform", t_stage)

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

    iterable = targets
    if show_progress and tqdm is not None:
        iterable = tqdm(targets, desc="[4/5] Training targets", unit="model")  # type: ignore
    for y_col, fname, target_name in iterable:  # type: ignore
        if y_col not in labels_df.columns:
            continue
        if show_progress and tqdm is None:
            print(f"  - Training {target_name} ...")
        y_series = labels_df[y_col].astype(int)
        y_all = y_series.to_numpy(copy=True)
        unique_classes = np.unique(y_all)
        strat = y_all if unique_classes.size > 1 else None
        t_split = time.perf_counter()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_t, y_all, test_size=test_size, random_state=random_state, stratify=strat
        )
        _mark(f"split_{target_name}", t_split)
        t_model = time.perf_counter()
        model = _fit_model(X_tr, y_tr)
        _mark(f"model_fit_{target_name}", t_model)
        joblib.dump(model, os.path.join(output_dir, fname))

        proba_te = model.predict_proba(X_te)[:, 1]
        t_metrics = time.perf_counter()
        m = compute_binary_metrics(y_te, proba_te, threshold_objective="f1")
        _mark(f"metrics_{target_name}", t_metrics)
        metrics_report[target_name] = metrics_to_dict(m)

        if generate_artifacts:
            if show_progress and tqdm is None:
                print(f"    > Computing curve + importance artifacts for {target_name} ...")
            t_curves = time.perf_counter()
            curves = compute_curve_data(y_te, proba_te)
            save_curve_data(curves, os.path.join(output_dir, f"{target_name}_curves.json"))
            _mark(f"curves_{target_name}", t_curves)
            t_plot = time.perf_counter()
            maybe_plot_curves(curves, target_name, output_dir)
            _mark(f"plot_{target_name}", t_plot)
            try:
                t_imp = time.perf_counter()
                imp_df = permutation_importance_report(model, X_te, y_te, feature_cols)
                save_feature_importance(imp_df, os.path.join(output_dir, f"feature_importance_{target_name}.csv"))
                _mark(f"importance_{target_name}", t_imp)
            except Exception:  # pragma: no cover
                pass

    # Save metrics
    if show_progress:
        print("[5/5] Writing metrics.json")
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=2)

    _mark("total", _t0_global)

    if profile:
        # Aggregate & print a concise sorted report
        print("\n=== STAGE TIMINGS (seconds) ===")
        for name, dur in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:30s} {dur:8.3f}")
        # Summaries: total model fit, total importance
        def _sum(prefix: str) -> float:
            return sum(v for k, v in timings.items() if k.startswith(prefix))
        print("\nTotals:")
        for pref in ["model_fit_", "importance_", "curves_", "plot_"]:
            print(f"{pref:<12} {_sum(pref):8.3f}")

    return metrics_report


def main():
    args = parse_args()
    report = train_from_labels(args.project_id, args.input, args.output_dir, test_size=args.test_size, random_state=args.random_state,
                               use_cache_first=args.use_cache_first, show_progress=not args.no_progress, profile=args.profile)
    print(f"Saved artifacts to {args.output_dir}")
    if report:
        print("\nTest metrics summary:")
        for k, v in report.items():
            print(f"- {k}: n={v.get('n')}, prev={v.get('prevalence'):.3f}, AUC={v.get('roc_auc'):.3f}, PR-AUC={v.get('pr_auc'):.3f}, Brier={v.get('brier'):.3f}, ECE={v.get('ece'):.3f}")


if __name__ == "__main__":
    main()
