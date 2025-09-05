from typing import List
import os
import json
import pandas as pd
from google.cloud import bigquery as bq
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


def run_pipeline_on_unseen_data(subject_ids: List[int], client: bq.Client) -> pd.DataFrame:
    """
    Run the pipeline on unseen subject_ids: extract 0-48h features from BigQuery,
    apply saved preprocessor and models, and return calibrated probabilities.

    Returns a DataFrame with columns: subject_id, mortality_proba, prolonged_LOS_proba, readmission_proba
    """
    # 1) First admission per subject and LOS >= 54h
    first_adm = get_first_admissions(client, subject_ids)
    if first_adm.empty:
        # Return empty with required columns
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

    # 4) Load artifacts
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    feat_cols_path = os.path.join(models_dir, 'feature_columns.json')
    m_mort_path = os.path.join(models_dir, 'model_mortality.joblib')
    m_los_path = os.path.join(models_dir, 'model_prolonged_los.joblib')
    m_readm_path = os.path.join(models_dir, 'model_readmission.joblib')

    if not (os.path.exists(preprocessor_path) and os.path.exists(feat_cols_path) and os.path.exists(m_mort_path) and os.path.exists(m_los_path) and os.path.exists(m_readm_path)):
        raise FileNotFoundError("Model artifacts not found in project/models. Train and save models first.")

    preprocessor = joblib.load(preprocessor_path)
    with open(feat_cols_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
    model_mort = joblib.load(m_mort_path)
    model_los = joblib.load(m_los_path)
    model_readm = joblib.load(m_readm_path)

    # 5) Align feature columns
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0
    X = features[feature_cols].copy()

    # 6) Transform and predict
    X_t = preprocessor.transform(X)
    mort_proba = model_mort.predict_proba(X_t)[:, 1]
    los_proba = model_los.predict_proba(X_t)[:, 1]
    readm_proba = model_readm.predict_proba(X_t)[:, 1]

    # 7) Build output aligned to requested subject_ids (include NaN for those filtered out)
    out = pd.DataFrame({
        'subject_id': X.index.astype(int),
        'mortality_proba': mort_proba,
        'prolonged_LOS_proba': los_proba,
        'readmission_proba': readm_proba,
    })

    # Insert missing subjects with NaN to satisfy API contract
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
