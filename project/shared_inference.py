import os
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable, Union


# Default guard thresholds (env override capable)
_MIN_VARIABLE_FRAC = float(os.getenv("MLHC_MIN_SCORING_VARIABLE_FRAC", "0.10"))
_RAISE_ON_LOW_VARIANCE = os.getenv("MLHC_SCORING_STRICT", "0").lower() in {"1", "true", "yes"}


def _assert_aligned_matrix_health(df: pd.DataFrame,
                                  min_variable_frac: float = _MIN_VARIABLE_FRAC,
                                  strict: Optional[bool] = None) -> float:
    """Validate that the aligned feature matrix retains sufficient variability.

    Parameters
    ----------
    df : DataFrame
        Aligned feature matrix (post reindex) used for scoring.
    min_variable_frac : float
        Minimum acceptable fraction of columns with >1 unique value.
    strict : bool, optional
        If True raise on failure; if False just warn. Defaults to env flag.

    Returns
    -------
    float
        Observed variable fraction.
    """
    if strict is None:
        strict = _RAISE_ON_LOW_VARIANCE
    if df is None or df.empty:
        msg = "Aligned matrix empty or None at scoring time."
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)
        return 0.0
    # Compute population variance; avoid memory blow-up by using nunique
    nunq = df.nunique(dropna=False)
    variable_frac = (nunq > 1).mean() if len(nunq) else 0.0
    if variable_frac < min_variable_frac:
        msg = (f"Aligned feature matrix variable fraction {variable_frac:.2%} < threshold "
               f"{min_variable_frac:.2%}. Possible schema mismatch / zero fill collapse.")
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)
    return float(variable_frac)


def _resolve_models_dir() -> Path:
    """Resolve canonical artifacts directory (legacy models/ removed).

    Returns
    -------
    Path
        Path object pointing to project/artifacts. Presence of feature_columns.json is validated.
    """
    base_dir = Path(__file__).parent
    artifacts_dir = base_dir / 'artifacts'
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found at expected path: {artifacts_dir}")
    if not (artifacts_dir / 'feature_columns.json').exists():
        raise FileNotFoundError(
            f"feature_columns.json missing in artifacts directory: {artifacts_dir}."
        )
    return artifacts_dir


def _load_artifacts(models_dir: Optional[Union[str, Path]] = None) -> Tuple[Dict[str, Any], Any, Any, Any, Optional[Any]]:
    """Load feature columns, base models, optional calibrators, and optional preprocessor.

    Returns
    -------
    (meta, model_mortality, model_prolonged_los, model_readmission, preprocessor)
    meta: dict with keys:
        feature_cols: list[str]
        models_dir: str
        calibrators: dict[str, Callable] (keys: mortality, prolonged_los, readmission) if present
    """
    md_path = Path(models_dir) if models_dir is not None else _resolve_models_dir()
    md = str(md_path)
    feat_cols_path = os.path.join(md, 'feature_columns.json')
    required = [
        feat_cols_path,
        os.path.join(md, 'model_mortality.joblib'),
        os.path.join(md, 'model_prolonged_los.joblib'),
        os.path.join(md, 'model_readmission.joblib'),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError('Missing required model artifacts: ' + ', '.join(missing))
    with open(feat_cols_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
    model_mort = joblib.load(os.path.join(md, 'model_mortality.joblib'))
    model_los = joblib.load(os.path.join(md, 'model_prolonged_los.joblib'))
    model_readm = joblib.load(os.path.join(md, 'model_readmission.joblib'))
    preprocessor = None
    preproc_path = os.path.join(md, 'preprocessor.joblib')
    if os.path.exists(preproc_path):
        try:
            preprocessor = joblib.load(preproc_path)
        except Exception as e:  # pragma: no cover
            print('WARNING: failed to load preprocessor.joblib ->', e)
    # Optional isotonic calibrators
    calibrators: Dict[str, Any] = {}
    iso_map = {
        'mortality': 'isotonic_mortality.joblib',
        'prolonged_los': 'isotonic_prolonged_los.joblib',
        'readmission': 'isotonic_readmission.joblib',
    }
    for key, fname in iso_map.items():
        fpath = os.path.join(md, fname)
        if os.path.exists(fpath):
            try:
                calibrators[key] = joblib.load(fpath)
            except Exception as e:  # pragma: no cover
                print(f"WARNING: failed to load calibrator {fname} -> {e}")
    return ({'feature_cols': feature_cols, 'models_dir': md, 'calibrators': calibrators}, model_mort, model_los, model_readm, preprocessor)


def score_features(features: pd.DataFrame,
                   models_dir: Optional[str] = None,
                   enforce_variance_guard: bool = True,
                   min_variable_frac: Optional[float] = None,
                   strict: Optional[bool] = None) -> pd.DataFrame:
    """Score a feature matrix using saved models.

    Parameters
    ----------
    features : DataFrame
        Feature matrix containing at least all training-time feature columns.
        Index must be subject_id (int-like) to propagate into output.
    models_dir : str, optional
        Explicit path to artifacts/models directory. If omitted, auto-resolve.

    Returns
    -------
    DataFrame
        Columns: mortality_proba, prolonged_LOS_proba, readmission_proba indexed by subject_id.
    """
    meta, model_mort, model_los, model_readm, preprocessor = _load_artifacts(models_dir)
    feature_cols = meta['feature_cols']
    missing_cols = [c for c in feature_cols if c not in features.columns]
    if missing_cols:
        # Vectorized addition to avoid fragmentation warnings from repeated assignment.
        print(
            f"WARNING: {len(missing_cols)} missing feature columns; filling with zeros (added in batch). "
            f"First few: {missing_cols[:10]}"
        )
        add_df = pd.DataFrame({c: 0.0 for c in missing_cols}, index=features.index)
        # Concatenate then reassign to original variable (preserve original reference semantics for caller)
        features = pd.concat([features, add_df], axis=1)
    X = features.reindex(columns=feature_cols, fill_value=0)
    if enforce_variance_guard:
        _assert_aligned_matrix_health(
            X,
            min_variable_frac=(min_variable_frac if min_variable_frac is not None else _MIN_VARIABLE_FRAC),
            strict=strict,
        )
    if preprocessor is not None:
        X_t = preprocessor.transform(X)
    else:
        if X.isna().any().any():
            X = X.fillna(0.0)
        X_t = X.values
    mort_raw = model_mort.predict_proba(X_t)[:, 1]
    los_raw = model_los.predict_proba(X_t)[:, 1]
    readm_raw = model_readm.predict_proba(X_t)[:, 1]

    calibrators = meta.get('calibrators', {})
    def _maybe_calibrate(arr, key):
        cal = calibrators.get(key)
        if cal is None:
            return arr
        # Try common APIs
        try:
            if hasattr(cal, 'predict'):
                return cal.predict(arr)
        except Exception as e:  # pragma: no cover
            print(f"WARNING: calibrator 'predict' failed for {key} -> {e}; falling back to raw probabilities")
            return arr
        try:
            if hasattr(cal, 'transform'):
                return cal.transform(arr)
        except Exception as e:  # pragma: no cover
            print(f"WARNING: calibrator 'transform' failed for {key} -> {e}; falling back to raw probabilities")
            return arr
        print(f"WARNING: calibrator for {key} has no usable interface; returning raw probabilities")
        return arr

    mort = _maybe_calibrate(mort_raw, 'mortality')
    los = _maybe_calibrate(los_raw, 'prolonged_los')
    readm = _maybe_calibrate(readm_raw, 'readmission')
    out = pd.DataFrame({
        'mortality_proba': mort,
        'prolonged_LOS_proba': los,
        'readmission_proba': readm,
    }, index=X.index)
    return out


# Convenience end-to-end helper (optional reuse in notebooks)
def score_raw_features_df(features: pd.DataFrame, models_dir: Optional[str] = None) -> pd.DataFrame:
    """Alias for score_features for external clarity."""
    return score_features(features, models_dir=models_dir)



def get_model_and_calibrator(target: str, models_dir: Optional[Union[str, Path]] = None) -> Tuple[Any, Optional[Any]]:
    """Return the trained model and optional isotonic calibrator for a given target.

    Parameters
    ----------
    target : str
        One of 'mortality', 'prolonged_los', 'readmission'. (Case-insensitive; underscores permitted.)
    models_dir : str, optional
        Explicit artifacts directory; auto-resolved if omitted.

    Returns
    -------
    (model, calibrator_or_None)
    """
    t = target.lower()
    norm_map = {
        'mortality': 'mortality',
        'prolonged_los': 'prolonged_los',
        'prolonged-los': 'prolonged_los',
        'readmission': 'readmission',
    }
    if t not in norm_map:
        raise ValueError(f"Unsupported target '{target}'. Expected one of {list(norm_map.keys())}.")
    t_norm = norm_map[t]
    md_path = Path(models_dir) if models_dir is not None else _resolve_models_dir()
    model_path = md_path / f"model_{t_norm}.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    iso_path = md_path / f"isotonic_{t_norm}.joblib"
    calibrator = None
    if iso_path.exists():
        try:
            calibrator = joblib.load(iso_path)
        except Exception as e:  # pragma: no cover
            print(f"WARNING: failed to load calibrator for {t_norm} -> {e}")
    return model, calibrator


def apply_calibration(raw_probs, calibrator):
    """Apply calibration using a calibrator that may expose either transform() or predict()."""
    if calibrator is None:
        return raw_probs
    if hasattr(calibrator, 'transform'):
        try:
            return calibrator.transform(raw_probs)
        except Exception:
            pass
    if hasattr(calibrator, 'predict'):
        try:
            return calibrator.predict(raw_probs)
        except Exception:
            pass
    print('WARNING: calibrator has neither usable transform nor predict; returning raw probabilities')
    return raw_probs
