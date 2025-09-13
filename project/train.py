# Training CLI to fit calibrated models from a labeled cohort
from __future__ import annotations
import os
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any, Iterable, Sequence
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore
try:  # sklearn >=1.3 exposes _safe_indexing internally; fallback to simple slicing if absent
    from sklearn.utils import _safe_indexing as safe_indexing  # type: ignore
except Exception:  # pragma: no cover
    def safe_indexing(X, indices):  # type: ignore
        return X[indices]
from sklearn.calibration import CalibratedClassifierCV

# Optional progress bar with safe widget fallback (avoid IProgress warning)
try:  # pragma: no cover
    from IPython import get_ipython  # type: ignore
except Exception:  # pragma: no cover
    get_ipython = lambda: None  # type: ignore

def _resolve_tqdm():  # pragma: no cover
    try:
        from tqdm.auto import tqdm as _auto  # type: ignore
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
    get_all_admissions,
    get_demographics,
    get_vitals_48h,
    get_labs_48h,
    get_prescriptions_48h,
    get_procedures_48h,
    get_all_admission_diagnoses,
)
from .features import build_features, build_feature_provenance, FEATURE_SCHEMA_VERSION
from .comorbidity_features import build_comorbidity_frame
from .utilization_features import build_utilization_features
from .variability_features import build_variability_features
from .medication_features import build_medication_features
from .phenotype_clustering import build_phenotype_clusters
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
    p.add_argument("--project-id", required=False, help="GCP Project ID for BigQuery (or set MLHC_PROJECT_ID env var)")
    p.add_argument("--input", required=True, help="CSV with subject_id and label columns (mortality_label, prolonged_los_label, readmission_label as available)")
    p.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), 'models'), help="Directory to save artifacts")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default 0.2)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed for splits")
    p.add_argument("--use-cache-first", action="store_true", help="If set, attempt to load cached parquet extracts in data/extracted_cache before querying BigQuery.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--profile", action="store_true", help="Collect and print per-stage timing information.")
    p.add_argument("--model", choices=["logreg","hgb","xgb"], default="logreg", help="Base model type (logreg, hist gradient boosting, or xgboost + calibration).")
    p.add_argument("--calib-cv", type=int, default=5, help="CV folds for calibration wrapper (logreg only).")
    p.add_argument("--skip-importance", action="store_true", help="Skip permutation importance to save time.")
    p.add_argument("--cv-folds", type=int, default=0, help="If >0, perform K-fold CV for metrics (no held-out split) before final fit; 0 keeps single holdout split.")
    p.add_argument("--finalize-mode", action="store_true", help="Enable final thorough training: full feature set, calibrate ALL targets (isotonic when possible) with provided calib-cv (min 5), ALWAYS run permutation importance, ignore --skip-importance if passed.")
    # Readmission-focused experimental knobs
    p.add_argument("--readmission-model", choices=["same","logreg","hgb","xgb"], default="same", help="Override base model just for readmission (same=use --model choice).")
    p.add_argument("--readmission-no-calibration", action="store_true", help="Disable probability calibration specifically for readmission (use raw estimator outputs).")
    p.add_argument("--only-readmission", action="store_true", help="Train only the readmission target (skip mortality and prolonged LOS) for faster iteration.")
    p.add_argument("--calib-progress", action="store_true", help="Print per-target calibration progress timings.")
    # Readmission tuning extras
    p.add_argument("--readmission-min-recall", type=float, default=None, help="If set, choose threshold maximizing F1 subject to recall >= this value (readmission only).")
    p.add_argument("--readmission-pos-weight", type=float, default=1.0, help="Multiplier applied to positive class weight for logistic regression when training readmission.")
    p.add_argument("--logreg-C", type=float, default=1.0, help="Inverse regularization strength C for logistic regression (ignored for hgb).")
    p.add_argument("--no-provenance", action="store_true", help="Skip writing feature_provenance.json (all features currently baseline).")
    p.add_argument("--allow-history", action="store_true", help="Enable use of pre-index admission history (comorbidity/utilization). OFF by default for strict first-visit compliance.")
    p.add_argument("--phenotype-clusters", type=int, default=0, help="If >0, derive KMeans phenotype clusters (n clusters) from current feature matrix and append one-hot features.")
    p.add_argument("--stack-readmission", action="store_true", help="If set, build out-of-fold mortality & prolonged_los prediction features for readmission model (meta-features).")
    p.add_argument("--feature-select-l1", type=str, default="", help="If set to a target name (mortality|prolonged_los|readmission), run an L1 logistic pre-selection to keep non-zero coefficient features for that target before training final models.")
    p.add_argument("--feature-select-C", type=float, default=0.5, help="Inverse regularization strength C for L1 feature selection logistic (smaller=stronger sparsity).")
    # XGBoost specific hyperparameters
    p.add_argument("--xgb-n-estimators", type=int, default=600, help="XGBoost number of boosting rounds.")
    p.add_argument("--xgb-learning-rate", type=float, default=0.03, help="XGBoost learning rate (eta).")
    p.add_argument("--xgb-max-depth", type=int, default=4, help="XGBoost max tree depth.")
    p.add_argument("--xgb-subsample", type=float, default=0.8, help="XGBoost subsample ratio per tree.")
    p.add_argument("--xgb-colsample-bytree", type=float, default=0.8, help="XGBoost feature subsample per tree.")
    p.add_argument("--xgb-reg-lambda", type=float, default=1.0, help="XGBoost L2 regularization term.")
    p.add_argument("--xgb-reg-alpha", type=float, default=0.0, help="XGBoost L1 regularization term.")
    p.add_argument("--reuse-features", type=str, default=None, help="Path to precomputed feature matrix parquet to reuse (skip extraction & engineering)")
    p.add_argument("--reuse-preprocessor", type=str, default=None, help="Path to existing preprocessor.joblib to reuse (skip fitting the preprocessing pipeline)")
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


def _load_and_build_features(client: Optional[bigquery.Client], subject_ids: List[int], use_cache_first: bool = False, allow_history: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    # Always admissible (within first admission 48h) variability + medication features
    try:
        var_feats = build_variability_features(vitals, labs)
        if not var_feats.empty:
            X = X.merge(var_feats, left_index=True, right_index=True, how='left')
        med_feats = build_medication_features(rx)
        if not med_feats.empty:
            X = X.merge(med_feats, left_index=True, right_index=True, how='left')
    except Exception as e:  # pragma: no cover
        print(f"Warning: variability/medication feature build failed: {e}")
    # History-derived features only if explicitly allowed
    if allow_history:
        try:
            if client is not None and not first_adm.empty:
                all_dx = get_all_admission_diagnoses(client, subject_ids)
                cmb = build_comorbidity_frame(all_dx, first_adm)
                if not cmb.empty:
                    X = X.merge(cmb, left_index=True, right_index=True, how='left')
                all_adm_hist = get_all_admissions(client, subject_ids)
                utl = build_utilization_features(all_adm_hist, first_adm)
                if not utl.empty:
                    X = X.merge(utl, left_index=True, right_index=True, how='left')
        except Exception as e:  # pragma: no cover
            print(f"Warning: history feature build failed: {e}")
    # Backward compatibility / leakage mitigation: ensure los_hours not retained if present
    if 'los_hours' in X.columns:
        X = X.drop(columns=['los_hours'])
    return first_adm, X


def _build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    # All features are numeric; impute missing with median, then scale for linear model stability
    num_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    pre = ColumnTransformer(transformers=[("num", num_transform, feature_cols)], remainder="drop")
    return pre


def _fit_model(
    X_t,
    y: np.ndarray,
    model_type: str = "logreg",
    calib_cv: int = 5,
    calibrate: bool = True,
    method_override: Optional[str] = None,
    progress: bool = False,
    target_name: str = "",
    pos_weight_multiplier: float = 1.0,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Fit a (optionally calibrated) model.

    Parameters
    ----------
    X_t : array-like or sparse matrix
        Transformed feature matrix.
    y : np.ndarray
        Binary target (0/1) values.
    model_type : str
        'logreg' (LogisticRegression) or 'hgb' (HistGradientBoostingClassifier) or 'xgb' (XGBoost gradient boosting trees).
    calib_cv : int
        Number of folds for CalibratedClassifierCV when calibration enabled.
    calibrate : bool
        If False, return a base estimator fitted once (faster for iteration).
    method_override : Optional[str]
        Force calibration method ('isotonic' or 'sigmoid'). If None, heuristic based on class counts.
    """
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    if model_type == "logreg":
        # Balanced base weights then scale positive class if requested
        if pos_weight_multiplier != 1.0:
            # Derive manual weights: w_pos = (n_neg/n_pos) * multiplier, w_neg = 1.0
            n_pos = max(1, positives)
            n_neg = max(1, negatives)
            w_pos = (n_neg / n_pos) * pos_weight_multiplier
            class_weight = {0: 1.0, 1: w_pos}
        else:
            class_weight = "balanced"
        C = globals().get("_GLOBAL_LOGREG_C", 1.0)
        base = LogisticRegression(max_iter=1000, class_weight=class_weight, solver="liblinear", C=C)
    elif model_type == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost not installed; please add to requirements and pip install.")
        # Compute scale_pos_weight ~ n_neg/n_pos * multiplier (similar intent to logistic)
        n_pos = max(1, positives)
        n_neg = max(1, negatives)
        scale_pos = (n_neg / n_pos) * pos_weight_multiplier
        # Use explicit params dict supplied by caller (no globals). Provide defaults.
        p = xgb_params or {}
        base = XGBClassifier(
            n_estimators=int(p.get("n_estimators", 600)),
            learning_rate=float(p.get("learning_rate", 0.03)),
            max_depth=int(p.get("max_depth", 4)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.8)),
            reg_lambda=float(p.get("reg_lambda", 1.0)),
            reg_alpha=float(p.get("reg_alpha", 0.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos,
            n_jobs=4,
            verbosity=0,
        )
    else:
        base = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.05, max_iter=200, class_weight="balanced")

    if not calibrate:
        if progress:
            print(f"[CALIB] Skipping calibration for {target_name or model_type}; fitting base {model_type} ...", flush=True)
        base.fit(X_t, y)
        return base

    method = method_override or ("isotonic" if positives >= 50 and negatives >= 50 else "sigmoid")
    try:
        if progress:
            import time as _t
            start_t = _t.time()
            print(f"[CALIB] Starting calibration for {target_name or model_type}: method={method} cv={calib_cv}", flush=True)
        clf = CalibratedClassifierCV(estimator=base, method=method, cv=calib_cv)  # type: ignore[arg-type]
    except TypeError as e:  # pragma: no cover
        raise RuntimeError(
            "CalibratedClassifierCV signature mismatch; requires scikit-learn supporting 'estimator'. Upgrade sklearn. " + str(e)
        )
    clf.fit(X_t, y)
    if progress:
        dur = _t.time() - start_t  # type: ignore[name-defined]
        print(f"[CALIB] Done {target_name or model_type} in {dur:.2f}s", flush=True)
    return clf


def train_from_labels(project_id: str, input_csv: str, output_dir: str, test_size: float = 0.2, random_state: int = 42,
                      generate_artifacts: bool = True, use_cache_first: bool = False,
                      show_progress: bool = True, profile: bool = False,
                      model_type: str = "logreg", calib_cv: int = 5, skip_importance: bool = False, cv_folds: int = 0,
                      finalize_mode: bool = False,
                      readmission_model_override: str = "same", readmission_no_calibration: bool = False,
                      only_readmission: bool = False,
                      calib_progress: bool = False,
                      readmission_min_recall: Optional[float] = None,
                      readmission_pos_weight: float = 1.0,
                      write_provenance: bool = True,
                      allow_history: bool = False,
                      phenotype_clusters: int = 0,
                      stacking_meta: bool = False,
                      feature_select_target: str | None = None,
                      feature_select_C: float = 0.5,
                      reuse_features: Optional[str] = None,
                      reuse_preprocessor: Optional[str] = None,
                      xgb_params: Optional[Dict[str, Any]] = None) -> Dict[str, Dict]:
    """End-to-end training entry point.

    Returns mapping target_name -> metrics dictionary (serializable JSON structure).
    Raises RuntimeError if feature construction produced an empty matrix.
    """
    os.makedirs(output_dir, exist_ok=True)

    # No fast mode: always consider full feature set (except finalize adjustments below)

    if finalize_mode:
        # Enforce thorough settings
        if calib_cv < 5:
            calib_cv = 5
        skip_importance = False  # always compute
        if show_progress:
            print("[FINALIZE] Finalize mode active: full calibration for ALL targets, calib_cv>=5, permutation importance enabled.")

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
    # Persist a copy of the labels used for this run (one canonical location) with Windows-safe fallback
    def _copy_labels(src: str, dst_dir: str):
        import shutil, time, datetime
        os.makedirs(dst_dir, exist_ok=True)
        base_target = os.path.join(dst_dir, 'labels.csv')
        # If already exists with identical size + mtime, skip
        try:
            if os.path.exists(base_target):
                if os.path.getsize(base_target) == os.path.getsize(src):
                    return base_target
        except Exception:
            pass
        # Try direct copy with a couple retries for transient WinError 32
        for attempt in range(3):
            try:
                shutil.copy2(src, base_target)
                return base_target
            except Exception as ex:  # pragma: no cover
                msg = str(ex)
                if 'WinError 32' in msg and attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                # Fallback: manual read/write unique timestamped filename
                try:
                    with open(src, 'rb') as rf:
                        data = rf.read()
                    ts_name = f"labels_copy_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.csv"
                    alt_path = os.path.join(dst_dir, ts_name)
                    with open(alt_path, 'wb') as wf:
                        wf.write(data)
                    return alt_path
                except Exception:
                    raise
        return base_target
    run_root = os.path.abspath(os.path.join(output_dir, os.pardir))
    try:
        copied_path = _copy_labels(input_csv, run_root)
        if show_progress:
            print(f"Stored labels snapshot at {copied_path}")
    except Exception as e:  # pragma: no cover
        print(f"Warning: could not persist labels snapshot: {e}")
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
    if reuse_features:
        if show_progress:
            print(f"[2/5] Reusing precomputed feature matrix from {reuse_features} ...")
        t_stage = time.perf_counter()
        X = pd.read_parquet(reuse_features)
        if 'subject_id' in X.columns and (X.index.name != 'subject_id'):
            try:
                X = X.set_index('subject_id')
            except Exception:
                pass
        first_adm = pd.DataFrame()
        _mark("feature_load_reuse", t_stage)
    else:
        if show_progress:
            print("[2/5] Building / loading feature matrices ...")
        t_stage = time.perf_counter()
        first_adm, X = _load_and_build_features(client, subject_ids, use_cache_first=use_cache_first, allow_history=allow_history)
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
        # Even if cache had historical derived columns from prior runs, we'll drop them later if disallowed
        if 'los_hours' in X.columns:
            X = X.drop(columns=['los_hours'])
    if X.empty:
        raise RuntimeError("No features built; check cohort and extraction.")

    # Strict first-visit compliance: drop any inadvertent history-derived columns if not allowed
    if not allow_history:
        hist_prefixes = ('cmb_', 'cmbgrp_', 'utl_')
        drop_cols = [c for c in X.columns if c.startswith(hist_prefixes)]
        if drop_cols:
            X = X.drop(columns=drop_cols)

    # Align to labels_df subjects
    X = X.reindex(index=labels_df['subject_id']).fillna(0.0)

    # finalize_mode keeps full feature set (no action needed). Fast-mode removed.

    # Fail-safe: detect Block1 schema (missingness features). If missing many '__measured' columns when reuse requested, warn user.
    if reuse_features:
        meas_cols = [c for c in X.columns if c.endswith('__measured')]
        if len(meas_cols) < 5:  # heuristic threshold; Block1 should create several
            print("[SCHEMA WARNING] Reused feature matrix appears to lack Block1 missingness/count features. "
                  "Consider rebuilding features (omit --reuse-features) after updating code. Current schema version expected:",
                  FEATURE_SCHEMA_VERSION)

    # Persist full feature matrix for future reuse if it came from fresh build (not reuse_features)
    if reuse_features is None:
        try:
            persist_path = os.path.join(output_dir, 'features_full.parquet')
            if not os.path.exists(persist_path):
                X.to_parquet(persist_path)
                if show_progress:
                    print(f"Persisted feature matrix to {persist_path}")
        except Exception as e:  # pragma: no cover
            if show_progress:
                print(f"Warning: failed to persist feature matrix: {e}")

    feature_cols = list(X.columns)
    # Optional phenotype clustering (unsupervised) before preprocessing
    # Optional phenotype clustering (unsupervised) before preprocessing
    if phenotype_clusters and phenotype_clusters > 0:
        try:
            ph = build_phenotype_clusters(X, n_clusters=phenotype_clusters, random_state=random_state)
            # Avoid collision; if clusters already built skip
            new_cols = [c for c in ph.columns if c not in X.columns]
            if new_cols:
                X = X.merge(ph[new_cols], left_index=True, right_index=True, how='left')
        except Exception as e:  # pragma: no cover
            if show_progress:
                print(f"Warning: phenotype clustering failed: {e}")
    feature_cols = list(X.columns)

    # L1 Feature Selection (pre-processing): run a sparse logistic regression on chosen target to filter features.
    if feature_select_target and feature_select_target in {'mortality','prolonged_los','readmission'}:
        target_col = f"{feature_select_target}_label"
        if target_col in labels_df.columns:
            y_sel = labels_df[target_col].astype(int).reindex(X.index).fillna(0).to_numpy()
            # Build a simple impute+scale pipeline then pull sparse coef mask
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.linear_model import LogisticRegression
            sel_pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=max(1e-4, feature_select_C), class_weight="balanced", max_iter=500)),
            ])
            try:
                sel_pipe.fit(X, y_sel)
                clf = sel_pipe.named_steps['clf']
                coefs = clf.coef_.ravel()
                mask = coefs != 0
                kept = [col for col, keep in zip(X.columns, mask) if keep]
                if len(kept) > 0 and len(kept) < len(X.columns):
                    X = X[kept]
                    if show_progress:
                        print(f"[FS] L1 kept {len(kept)}/{len(mask)} features for target {feature_select_target} (C={feature_select_C}).")
                else:
                    if show_progress:
                        print(f"[FS] L1 selection kept all features ({len(X.columns)}); skipping reduction.")
            except Exception as e:
                if show_progress:
                    print(f"Warning: L1 feature selection failed: {e}")
        else:
            if show_progress:
                print(f"[FS] Target column {target_col} not in labels; skipping L1 feature selection.")
    feature_cols = list(X.columns)

    # Stacking meta-features: build OOF predictions for mortality & prolonged LOS (internal 5-fold) then append
    if stacking_meta and only_readmission and 'readmission_label' in labels_df.columns:
        base_targets = []
        for col in ['mortality_label', 'prolonged_los_label']:
            if col in labels_df.columns and labels_df[col].notna().any():
                base_targets.append(col)
        if base_targets:
            from sklearn.model_selection import StratifiedKFold
            from sklearn.linear_model import LogisticRegression
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            for tgt in base_targets:
                y_full = labels_df[tgt].astype(int).to_numpy()
                oof = np.zeros_like(y_full, dtype=float)
                # Reuse numeric matrix before existing preprocessor to avoid leakage of meta features into themselves
                # We'll fit a simple pipeline: median impute + standardize + logistic
                from sklearn.impute import SimpleImputer
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline
                for tr_idx, te_idx in skf.split(X.values, y_full):
                    X_tr_fold = X.iloc[tr_idx]
                    X_te_fold = X.iloc[te_idx]
                    pipe = Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False)),
                        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")),
                    ])
                    pipe.fit(X_tr_fold, y_full[tr_idx])
                    oof[te_idx] = pipe.predict_proba(X_te_fold)[:, 1]
                X[f'stack_{tgt.replace("_label","")}_proba'] = oof
            feature_cols = list(X.columns)
    if 'reuse_preprocessor' in locals() and reuse_preprocessor:
        if show_progress:
            print(f"[3/5] Loading existing preprocessor from {reuse_preprocessor} and transforming {X.shape[0]}x{X.shape[1]} ...")
        t_stage = time.perf_counter()
        pre = joblib.load(reuse_preprocessor)
        X_t = pre.transform(X)
        _mark("preprocessor_transform_only", t_stage)
        # Still write feature_columns.json so downstream consumers have consistent set
        with open(os.path.join(output_dir, 'feature_columns.json'), 'w', encoding='utf-8') as f:
            json.dump(feature_cols, f)
    else:
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
    # Write baseline provenance mapping (all features baseline group) unless disabled
    if write_provenance:
        try:
            prov = build_feature_provenance(X, default_group="baseline")
            for k in list(prov.keys()):
                if k.startswith('var_'):
                    prov[k]['group'] = 'variability'
                if k.startswith('med_'):
                    prov[k]['group'] = 'medication'
                if k.startswith('pheno_cluster'):
                    prov[k]['group'] = 'phenotype'
                if k.startswith('stack_'):
                    prov[k]['group'] = 'stacking'
                if allow_history and (k.startswith('cmb_') or k.startswith('cmbgrp_')):
                    prov[k]['group'] = 'comorbidity'
                if allow_history and k.startswith('utl_'):
                    prov[k]['group'] = 'utilization'
            with open(os.path.join(output_dir, 'feature_provenance.json'), 'w', encoding='utf-8') as pf:
                json.dump(prov, pf, indent=2)
        except Exception as e:  # pragma: no cover
            if show_progress:
                print(f"Warning: failed to write feature_provenance.json: {e}")

    # Train models for available targets
    targets = [
        ("mortality_label", "model_mortality.joblib", "mortality"),
        ("prolonged_los_label", "model_prolonged_los.joblib", "prolonged_los"),
        ("readmission_label", "model_readmission.joblib", "readmission"),
    ]
    if only_readmission:
        targets = [t for t in targets if t[2] == "readmission"]

    metrics_report: Dict[str, Dict] = {}

    iterable = targets
    if show_progress and tqdm is not None:
        iterable = tqdm(targets, desc="[4/5] Training targets", unit="model")  # type: ignore
    # In fast mode, calibrate only readmission to save time unless user supplied cv_folds >1
    # In finalize_mode we calibrate ALL targets fully; otherwise still calibrate all (fast mode removed)
    # Determine user overrides for readmission behavior via environment variables set by CLI main (persisted below).
    # This avoids threading many parameters; main() will set sentinel env vars before calling train_from_labels.
    READM_OVERRIDE_MODEL = readmission_model_override if readmission_model_override in {"logreg","hgb"} else ""
    READM_NO_CALIB = readmission_no_calibration

    for y_col, fname, target_name in iterable:  # type: ignore
        if y_col not in labels_df.columns:
            continue
        if show_progress and tqdm is None:
            print(f"  - Training {target_name} ...")
        y_series = labels_df[y_col].astype(int)
        y_all = y_series.to_numpy(copy=True)
        unique_classes = np.unique(y_all)
        strat = y_all if unique_classes.size > 1 else None
        # Determine per-target overrides
        this_model_type = model_type
        this_calibrate = True
        if target_name == "readmission" and READM_OVERRIDE_MODEL in {"logreg", "hgb"}:
            this_model_type = READM_OVERRIDE_MODEL
        if target_name == "readmission" and READM_NO_CALIB:
            this_calibrate = False
        y_eval = None  # type: ignore
        proba_all = None  # type: ignore
        if cv_folds and cv_folds > 1:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state) if strat is not None else None
            fold_metrics = []
            for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_t, y_all) if skf else [(np.arange(len(y_all)), np.arange(len(y_all)))]):
                t_model = time.perf_counter()
                X_tr_fold = safe_indexing(X_t, tr_idx)
                model_cv = _fit_model(
                    X_tr_fold,
                    y_all[tr_idx],
                    model_type=this_model_type,
                    calib_cv=calib_cv,
                    calibrate=this_calibrate,
                    method_override=None,
                    progress=calib_progress,
                    target_name=target_name,
                    pos_weight_multiplier=(args.readmission_pos_weight if target_name == "readmission" else 1.0) if False else (1.0),
                    xgb_params=xgb_params if this_model_type == "xgb" else None,
                )
                _mark(f"model_fit_{target_name}_cv{fold_idx}", t_model)
                X_te_fold = safe_indexing(X_t, te_idx)
                proba_te = model_cv.predict_proba(X_te_fold)[:, 1]
                t_metrics = time.perf_counter()
                m = compute_binary_metrics(y_all[te_idx], proba_te, threshold_objective="f1")
                _mark(f"metrics_{target_name}_cv{fold_idx}", t_metrics)
                fold_metrics.append(metrics_to_dict(m))
            agg = {}
            for fm in fold_metrics:
                for k, v in fm.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        agg.setdefault(k, []).append(v)
            cv_summary = {k: float(np.nanmean(vs)) for k, vs in agg.items()}
            metrics_report[target_name] = {**cv_summary, 'cv_folds': cv_folds}
            # Final model trained on all data for artifacts
            model = _fit_model(
                X_t,
                y_all,
                model_type=this_model_type,
                calib_cv=calib_cv,
                calibrate=this_calibrate,
                method_override=None,
                progress=calib_progress,
                target_name=target_name,
                pos_weight_multiplier=(readmission_pos_weight if target_name == "readmission" else 1.0),
                xgb_params=xgb_params if this_model_type == "xgb" else None,
            )
            joblib.dump(model, os.path.join(output_dir, fname))
            proba_all = model.predict_proba(X_t)[:, 1]
            y_eval = y_all
        else:
            t_split = time.perf_counter()
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_t, y_all, test_size=test_size, random_state=random_state, stratify=strat
            )
            _mark(f"split_{target_name}", t_split)
            t_model = time.perf_counter()
            model = _fit_model(
                X_tr,
                y_tr,
                model_type=this_model_type,
                calib_cv=calib_cv,
                calibrate=this_calibrate,
                method_override=None,
                progress=calib_progress,
                target_name=target_name,
                pos_weight_multiplier=(readmission_pos_weight if target_name == "readmission" else 1.0),
                xgb_params=xgb_params if this_model_type == "xgb" else None,
            )
            _mark(f"model_fit_{target_name}", t_model)
            joblib.dump(model, os.path.join(output_dir, fname))
            proba_te = model.predict_proba(X_te)[:, 1]
            t_metrics = time.perf_counter()
            threshold_min_recall = readmission_min_recall if target_name == "readmission" else None
            m = compute_binary_metrics(y_te, proba_te, threshold_objective="f1", threshold_min_recall=threshold_min_recall)
            _mark(f"metrics_{target_name}", t_metrics)
            metrics_report[target_name] = metrics_to_dict(m)
            proba_all = proba_te
            y_eval = y_te

        if generate_artifacts and y_eval is not None and proba_all is not None:
            if show_progress and tqdm is None:
                print(f"    > Computing curve + importance artifacts for {target_name} ...")
            t_curves = time.perf_counter()
            curves = compute_curve_data(y_eval, proba_all)
            save_curve_data(curves, os.path.join(output_dir, f"{target_name}_curves.json"))
            _mark(f"curves_{target_name}", t_curves)
            t_plot = time.perf_counter()
            maybe_plot_curves(curves, target_name, output_dir)
            _mark(f"plot_{target_name}", t_plot)
            if not skip_importance and not (cv_folds and cv_folds > 1) and 'X_te' in locals():
                try:  # pragma: no cover
                    t_imp = time.perf_counter()
                    imp_df = permutation_importance_report(model, X_te, y_te, feature_cols)  # type: ignore[name-defined]
                    save_feature_importance(imp_df, os.path.join(output_dir, f"feature_importance_{target_name}.csv"))
                    _mark(f"importance_{target_name}", t_imp)
                except Exception:
                    pass

    # Save metrics
    if show_progress:
        print("[5/5] Writing metrics.json")
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=2)

    _mark("total", _t0_global)

    # Minimal run metadata (separate from metrics for clarity)
    try:  # pragma: no cover
        import datetime, subprocess
        start_iso = datetime.datetime.fromtimestamp(_t0_global, datetime.timezone.utc).isoformat()
        end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        duration = None
        if 'total' in timings:
            duration = timings['total']
        else:
            # fallback approximate
            duration = (datetime.datetime.now().timestamp() - _t0_global)
        git_commit = None
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
        effective_config = {
            'model_type': model_type,
            'calib_cv': calib_cv,
            'skip_importance_effective': skip_importance,
            'cv_folds': cv_folds,
            'finalize_mode': finalize_mode,
            'feature_count': len(feature_cols),
            'test_size': test_size,
            'random_state': random_state,
            'allow_history': allow_history,
        }
        metadata = {
            'start_time_utc': start_iso,
            'end_time_utc': end_iso,
            'duration_seconds': duration,
            'git_commit': git_commit,
            'config': effective_config,
            'timings': timings if profile else None,
        }
        with open(os.path.join(output_dir, 'run_metadata.json'), 'w', encoding='utf-8') as mf:
            json.dump(metadata, mf, indent=2)
    except Exception:
        pass

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
    # Project ID resolution: CLI arg > env var > default used in notebooks
    project_id = args.project_id or os.environ.get("MLHC_PROJECT_ID") or os.environ.get("GCP_PROJECT_ID") or "ml-for-healthcare-2025"
    # Extract readmission-specific tuning params
    readmission_min_recall = args.readmission_min_recall
    readmission_pos_weight = max(0.01, args.readmission_pos_weight)
    # Stash global logistic C so _fit_model can access without threading param explosion
    global _GLOBAL_LOGREG_C
    _GLOBAL_LOGREG_C = max(1e-4, float(args.logreg_C))
    # Collect XGBoost hyperparameters into a dict passed explicitly (no globals)
    xgb_params = {
        'n_estimators': int(args.xgb_n_estimators),
        'learning_rate': float(args.xgb_learning_rate),
        'max_depth': int(args.xgb_max_depth),
        'subsample': float(args.xgb_subsample),
        'colsample_bytree': float(args.xgb_colsample_bytree),
        'reg_lambda': float(args.xgb_reg_lambda),
        'reg_alpha': float(args.xgb_reg_alpha),
    }
    report = train_from_labels(
        project_id,
        args.input,
        args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        use_cache_first=args.use_cache_first,
        show_progress=not args.no_progress,
        profile=args.profile,
        model_type=args.model,
        calib_cv=args.calib_cv,
        skip_importance=args.skip_importance,
        cv_folds=args.cv_folds,
        finalize_mode=args.finalize_mode,
        readmission_model_override=args.readmission_model,
        readmission_no_calibration=args.readmission_no_calibration,
        only_readmission=args.only_readmission,
        calib_progress=args.calib_progress,
        readmission_min_recall=readmission_min_recall,
        readmission_pos_weight=readmission_pos_weight,
        write_provenance=not args.no_provenance,
        allow_history=args.allow_history,
        phenotype_clusters=args.phenotype_clusters,
    stacking_meta=args.stack_readmission,
        feature_select_target=(args.feature_select_l1 if args.feature_select_l1 else None),
        feature_select_C=args.feature_select_C,
        reuse_features=args.reuse_features,
        reuse_preprocessor=args.reuse_preprocessor,
        xgb_params=xgb_params if args.model == 'xgb' or args.readmission_model == 'xgb' else None,
    )
    print(f"Saved artifacts to {args.output_dir}")
    if report:
        print("\nTest metrics summary:")
        for k, v in report.items():
            def _fmt(name):
                val = v.get(name)
                if val is None:
                    return "NA"
                try:
                    return f"{float(val):.3f}"
                except Exception:
                    return str(val)
            print(f"- {k}: n={v.get('n')}, prev={_fmt('prevalence')}, AUC={_fmt('roc_auc')}, PR-AUC={_fmt('pr_auc')}, Brier={_fmt('brier')}, ECE={_fmt('ece')}")


if __name__ == "__main__":
    main()
