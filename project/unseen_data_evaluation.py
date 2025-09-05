from typing import List
import os
import json
import pandas as pd
from google.cloud import bigquery as bq
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


def run_pipeline_on_unseen_data(subject_ids: List[int], client: bq.Client) -> pd.DataFrame:
    """Run inference pipeline on unseen patients.

    Steps:
      1. Pull first admissions and filter by required LOS window (>=54h).
      2. Extract 0-48h modalities (vitals, labs, prescriptions, procedures, demographics).
      3. Build feature matrix aligned to training feature set.
      4. Load fitted preprocessor + calibrated models and produce probabilities.

    Returns
    -------
    DataFrame
        Columns: subject_id, mortality_proba, prolonged_LOS_proba, readmission_proba
        Includes NA rows for subjects excluded by LOS filtering to keep alignment.
    """
    # 1) First admission per subject and LOS >= 54h
    first_adm = get_first_admissions(client, subject_ids)
    if first_adm.empty:
        return pd.DataFrame({
            'subject_id': subject_ids,
            'mortality_proba': [float('nan')]*len(subject_ids),
            'prolonged_LOS_proba': [float('nan')]*len(subject_ids),
            'readmission_proba': [float('nan')]*len(subject_ids),
        })
    first_adm['admittime'] = pd.to_datetime(first_adm['admittime'])
    first_adm['dischtime'] = pd.to_datetime(first_adm['dischtime'])
    first_adm['los_hours'] = (first_adm['dischtime'] - first_adm['admittime']).dt.total_seconds() / 3600.0
    first_adm = first_adm[first_adm['los_hours'] >= 54].copy()
    hadm_ids = first_adm['hadm_id'].dropna().astype(int).tolist()

    # 2) Extract modalities within 0-48h
    demo = get_demographics(client, first_adm['subject_id'].dropna().astype(int).tolist())
    vitals = get_vitals_48h(client, hadm_ids)
    labs = get_labs_48h(client, hadm_ids)
    rx = get_prescriptions_48h(client, hadm_ids)
    proc = get_procedures_48h(client, hadm_ids)

    # 3) Build features
    features = build_features(first_adm, demo, vitals, labs, rx, proc)

    # 4) Load artifacts strictly from MLHC_MODELS_DIR (no discovery fallback)
    models_dir = os.environ.get('MLHC_MODELS_DIR')
    core_files = [
        'preprocessor.joblib',
        'feature_columns.json',
        'model_mortality.joblib',
        'model_prolonged_los.joblib',
        'model_readmission.joblib'
    ]
    if not models_dir or not all(os.path.exists(os.path.join(models_dir, f)) for f in core_files):
        raise FileNotFoundError(
            "Model artifacts missing or MLHC_MODELS_DIR unset. Set MLHC_MODELS_DIR to a run's models directory containing: "
            + ", ".join(core_files)
        )
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    feat_cols_path = os.path.join(models_dir, 'feature_columns.json')
    m_mort_path = os.path.join(models_dir, 'model_mortality.joblib')
    m_los_path = os.path.join(models_dir, 'model_prolonged_los.joblib')
    m_readm_path = os.path.join(models_dir, 'model_readmission.joblib')

    preprocessor = joblib.load(preprocessor_path)
    with open(feat_cols_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
    model_mort = joblib.load(m_mort_path)
    model_los = joblib.load(m_los_path)
    model_readm = joblib.load(m_readm_path)

    # 5) Align feature columns (vectorized to avoid fragmentation)
    missing_cols = [c for c in feature_cols if c not in features.columns]
    if missing_cols:
        zeros_df = pd.DataFrame(0, index=features.index, columns=missing_cols)
        features = pd.concat([features, zeros_df], axis=1)
    # Reorder exactly as training feature column order
    features = features.reindex(columns=feature_cols, fill_value=0)
    if not feature_cols:
        return pd.DataFrame({
            'subject_id': subject_ids,
            'mortality_proba': [float('nan')]*len(subject_ids),
            'prolonged_LOS_proba': [float('nan')]*len(subject_ids),
            'readmission_proba': [float('nan')]*len(subject_ids),
        })
    X = features[feature_cols].copy()

    # 6) Transform and predict
    X_t = preprocessor.transform(X)
    mort_proba = model_mort.predict_proba(X_t)[:, 1]
    los_proba = model_los.predict_proba(X_t)[:, 1]
    readm_proba = model_readm.predict_proba(X_t)[:, 1]

    # 7) Build output
    out = pd.DataFrame({
        'subject_id': X.index.astype(int),
        'mortality_proba': mort_proba,
        'prolonged_LOS_proba': los_proba,
        'readmission_proba': readm_proba,
    })
    missing = set(subject_ids) - set(out['subject_id'].tolist())
    if missing:
        out = pd.concat([
            out,
            pd.DataFrame({
                'subject_id': list(missing),
                'mortality_proba': [float('nan')]*len(missing),
                'prolonged_LOS_proba': [float('nan')]*len(missing),
                'readmission_proba': [float('nan')]*len(missing),
            })
        ], axis=0, ignore_index=True)
    return out.sort_values('subject_id').reset_index(drop=True)
