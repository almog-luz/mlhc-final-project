from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .features import build_features, build_feature_provenance
from .labels import build_labels_duckdb
from .shared_inference import score_features


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def extract_first_admissions_duckdb(con, subject_ids: Iterable[int]) -> pd.DataFrame:
    ids_csv = ",".join(map(str, subject_ids))
    if not ids_csv:
        return pd.DataFrame(columns=[
            "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
            "admission_type", "admission_location", "discharge_location",
            "diagnosis", "insurance", "language", "marital_status", "ethnicity",
            "los_hours"
        ])
    sql = f"""
        SELECT subject_id, hadm_id, admittime, dischtime, deathtime, admission_type,
               admission_location, discharge_location, diagnosis, insurance, language,
               marital_status, ethnicity
        FROM admissions
        WHERE subject_id IN ({ids_csv})
        ORDER BY subject_id, admittime
    """
    df = con.execute(sql).fetchdf()
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    df = (df.sort_values(["subject_id", "admittime"])  # first admission only
            .groupby("subject_id", as_index=False)
            .first())
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["los_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds()/3600.0
    df = df[df["los_hours"] >= 54].copy()
    return df


def extract_labels_and_filter(con, first_adm: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if first_adm.empty:
        return pd.DataFrame(columns=["subject_id"]), first_adm
    subject_ids = first_adm["subject_id"].astype(int).tolist()
    labels_df = build_labels_duckdb(con, subject_ids)
    keep = set(labels_df["subject_id"].tolist())
    first_adm_f = first_adm[first_adm["subject_id"].isin(keep)].copy()
    return labels_df, first_adm_f

def extract_modalities_duckdb(con, first_adm: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if first_adm.empty:
        return {"demo": pd.DataFrame(), "vitals": pd.DataFrame(), "labs": pd.DataFrame(),
                "rx": pd.DataFrame(), "proc": pd.DataFrame()}
    hadm_ids = first_adm["hadm_id"].dropna().astype(int).tolist()
    hadm_csv = ",".join(map(str, hadm_ids)) if hadm_ids else "-1"

    demo = first_adm[["subject_id", "hadm_id", "admission_type", "admission_location",
                      "discharge_location", "diagnosis", "insurance", "language",
                      "marital_status", "ethnicity"]].copy()

    def _safe_query(sql: str) -> pd.DataFrame:
        try:
            return con.execute(sql).fetchdf()
        except Exception:
            return pd.DataFrame()

    vitals = _safe_query(f"SELECT * FROM vitals WHERE hadm_id IN ({hadm_csv})")
    labs = _safe_query(f"SELECT * FROM labs WHERE hadm_id IN ({hadm_csv})")
    rx = _safe_query(f"SELECT * FROM prescriptions WHERE hadm_id IN ({hadm_csv})")
    proc = _safe_query(f"SELECT * FROM procedures WHERE hadm_id IN ({hadm_csv})")

    return {"demo": demo, "vitals": vitals, "labs": labs, "rx": rx, "proc": proc}


def build_feature_matrix(first_adm: pd.DataFrame, modalities: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if first_adm.empty:
        return pd.DataFrame()
    feats = build_features(
        first_adm,
        modalities.get("demo", pd.DataFrame()),
        modalities.get("vitals", pd.DataFrame()),
        modalities.get("labs", pd.DataFrame()),
        modalities.get("rx", pd.DataFrame()),
        modalities.get("proc", pd.DataFrame()),
    )
    leakage_cols = [c for c in feats.columns if "los_hours" in c.lower()]
    if leakage_cols:
        feats = feats.drop(columns=leakage_cols)
    return feats


def persist_feature_artifacts(features: pd.DataFrame, artifacts_dir: Path) -> None:
    if features.empty:
        return
    _ensure_dir(artifacts_dir)
    # to_parquet returns None; use pandas IO
    features.to_parquet(artifacts_dir / "features_full.parquet")
    prov = build_feature_provenance(features)
    (artifacts_dir / "feature_provenance.json").write_text(json.dumps(prov, indent=2))
    (artifacts_dir / "feature_columns.json").write_text(json.dumps(list(features.columns), indent=2))

def run_training_side_pipeline(con, cohort_subject_ids: Iterable[int], artifacts_dir: Path, persist: bool = True) -> pd.DataFrame:
    """End-to-end feature build (training side) matching notebook extraction logic.

    The extraction notebook should call THIS function instead of re-implementing
    the steps. Any future changes in extraction should land here first, then the
    notebook refreshed to prevent divergence.
    """
    first_adm = extract_first_admissions_duckdb(con, cohort_subject_ids)
    _labels_df, first_adm = extract_labels_and_filter(con, first_adm)
    modalities = extract_modalities_duckdb(con, first_adm)
    features = build_feature_matrix(first_adm, modalities)
    if persist:
        persist_feature_artifacts(features, artifacts_dir)
    return features


def run_inference_pipeline(
    con,
    subject_ids: Iterable[int],
    artifacts_dir: Path,
    models_dir: Path,
) -> pd.DataFrame:
    first_adm = extract_first_admissions_duckdb(con, subject_ids)
    _labels, first_adm = extract_labels_and_filter(con, first_adm)
    modalities = extract_modalities_duckdb(con, first_adm)
    features = build_feature_matrix(first_adm, modalities)
    if features.empty:
        return pd.DataFrame(columns=["subject_id"])  # maintain schema
    scored = score_features(features, models_dir=str(models_dir))
    # Ensure subject_id as column for convenience
    if 'subject_id' not in scored.columns:
        scored = scored.reset_index().rename(columns={'index': 'subject_id'})
    return scored


__all__ = [
    "extract_first_admissions_duckdb",
    "extract_labels_and_filter",
    "extract_modalities_duckdb",
    "build_feature_matrix",
    "persist_feature_artifacts",
    "run_training_side_pipeline",
    "run_inference_pipeline",
]
