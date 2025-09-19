from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .features import build_features, build_feature_provenance
from .labels import build_labels_duckdb
from .shared_inference import score_features
from .extract import (
    get_first_admissions_duckdb,
    get_demographics_duckdb,
    get_vitals_48h_duckdb,
    get_labs_48h_duckdb,
    get_prescriptions_48h_duckdb,
    get_procedures_48h_duckdb,
)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def extract_first_admissions_duckdb(con, subject_ids: Iterable[int]) -> pd.DataFrame:
    """Return first admissions with computed LOS; keep stays >=54h."""
    subject_ids_list = list(subject_ids)
    df = get_first_admissions_duckdb(con, subject_ids_list)
    if df.empty:
        return df
    # Compute los_hours (not persisted in helper) and filter
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['los_hours'] = (df['dischtime'] - df['admittime']).dt.total_seconds()/3600.0
    return df[df['los_hours'] >= 54].copy()


def extract_labels_and_filter(con, first_adm: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if first_adm.empty:
        return pd.DataFrame(columns=["subject_id"]), first_adm
    subject_ids = first_adm["subject_id"].astype(int).tolist()
    labels_df = build_labels_duckdb(con, subject_ids)
    keep = set(labels_df["subject_id"].tolist())
    first_adm_f = first_adm[first_adm["subject_id"].isin(keep)].copy()
    return labels_df, first_adm_f

def extract_modalities_duckdb(con, first_adm: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Delegate modality extraction to extract.py DuckDB helpers for parity."""
    if first_adm.empty:
        return {"demo": pd.DataFrame(), "vitals": pd.DataFrame(), "labs": pd.DataFrame(), "rx": pd.DataFrame(), "proc": pd.DataFrame()}
    subject_ids = first_adm['subject_id'].astype(int).tolist()
    hadm_ids = first_adm['hadm_id'].dropna().astype(int).tolist()
    demo = get_demographics_duckdb(con, subject_ids)
    vitals = get_vitals_48h_duckdb(con, hadm_ids)
    labs = get_labs_48h_duckdb(con, hadm_ids)
    rx = get_prescriptions_48h_duckdb(con, hadm_ids)
    proc = get_procedures_48h_duckdb(con, hadm_ids)
    return {"demo": demo, "vitals": vitals, "labs": labs, "rx": rx, "proc": proc}


def build_feature_matrix(first_adm: pd.DataFrame, modalities: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construct and prune feature matrix with variance guard."""
    if first_adm.empty:
        return pd.DataFrame()
    feats_raw = build_features(
        first_adm,
        modalities.get("demo", pd.DataFrame()),
        modalities.get("vitals", pd.DataFrame()),
        modalities.get("labs", pd.DataFrame()),
        modalities.get("rx", pd.DataFrame()),
        modalities.get("proc", pd.DataFrame()),
    )
    leakage_cols = [c for c in feats_raw.columns if "los_hours" in c.lower()]
    if leakage_cols:
        feats_raw = feats_raw.drop(columns=leakage_cols)
    # Apply pruning (honors env overrides)
    feats_pruned = prune_features(feats_raw)
    if feats_pruned.empty:
        raise RuntimeError("Pruning resulted in empty feature matrix; investigate extraction / thresholds.")
    # Fail-fast variance guard
    nunq = feats_pruned.nunique(dropna=True)
    variable = (nunq > 1).sum()
    variable_frac = variable / len(nunq)
    if variable_frac < 0.2:
        raise RuntimeError(
            f"High constant-rate after pruning: only {variable} / {len(nunq)} (={variable_frac:.3%}) columns have variance >1. "
            "Adjust pruning or investigate extraction."
        )
    return feats_pruned


def prune_features(features: pd.DataFrame, min_support: int | None = None, target_max: int | None = None) -> pd.DataFrame:
    """Variance-first pruning with optional composite ranking.

    Steps:
      1. Read env overrides (MLHC_MIN_SUPPORT, MLHC_TARGET_MAX)
      2. Compute support & variance (mean-imputed for calculation only)
      3. Drop zero-variance columns
      4. Apply support threshold (default 10)
      5. If still above cap, rank by 0.7*variance + 0.3*support and keep top target_max
      6. Print concise diagnostics
    """
    if features.empty:
        return features

    import os
    if min_support is None:
        try:
            min_support = int(os.getenv("MLHC_MIN_SUPPORT", "10"))
        except ValueError:
            min_support = 10
    if target_max is None:
        try:
            target_max = int(os.getenv("MLHC_TARGET_MAX", "1500"))
        except ValueError:
            target_max = 1500
    if min_support < 1:
        min_support = 1
    if target_max is not None and target_max < 1:
        target_max = None

    orig_cols = list(features.columns)
    support = features.notna().sum(axis=0)
    var_series = features.apply(lambda s: s.fillna(s.mean()).var(ddof=0))
    zero_var_cols = var_series[var_series == 0].index.tolist()
    kept = [c for c in features.columns if c not in zero_var_cols]
    work = features[kept]
    support = support[kept]
    var_series = var_series[kept]
    keep_support = support[support >= min_support].index.tolist()
    work = work[keep_support]
    support = support[keep_support]
    var_series = var_series[keep_support]
    if target_max and work.shape[1] > target_max:
        var_sub = var_series[work.columns].astype(float)
        sup_sub = support[work.columns].astype(float)
        var_norm = (var_sub - var_sub.mean()) / (var_sub.std() + 1e-12)
        sup_norm = (sup_sub - sup_sub.mean()) / (sup_sub.std() + 1e-12)
        score = 0.7 * var_norm + 0.3 * sup_norm
        rank_df = (
            pd.DataFrame({
                'col': work.columns,
                'variance': var_sub.values,
                'support': sup_sub.values,
                'score': score.values,
            })
            .sort_values('score', ascending=False)
        )
        keep_n = int(target_max)
        keep_cols = rank_df.iloc[:keep_n]['col'].tolist()
        work = work[keep_cols]
    removed = set(orig_cols) - set(work.columns)
    const_dropped = len(zero_var_cols)
    low_support_removed = [c for c in removed if c in support.index and support[c] < min_support]
    print(
        f"Prune summary: original={len(orig_cols)} kept={work.shape[1]} removed={len(removed)} | "
        f"zero_var_dropped={const_dropped} low_support_removed={len(low_support_removed)}"
    )
    try:
        if work.shape[1] > 0:
            subset_var = var_series[work.columns]
            pairs = list(zip(subset_var.index.tolist(), subset_var.values.tolist()))
            pairs.sort(key=lambda x: x[1], reverse=True)
            sample_preview = [p[0] for p in pairs[:5]]
            print("  Top variance retained (sample):", sample_preview)
    except Exception:  # pragma: no cover
        pass
    return work


def persist_feature_artifacts(features: pd.DataFrame, artifacts_dir: Path, raw_features: pd.DataFrame | None = None, persist_raw: bool | None = None) -> None:
    if features.empty:
        return
    # Fail-fast: reject persistence if excessive constants (safeguard against silent degradation)
    nunq = features.nunique(dropna=True)
    variable = (nunq > 1).sum()
    if variable == 0:
        raise RuntimeError("Refusing to persist: all features constant.")
    variable_frac = variable / len(nunq)
    if variable_frac < 0.5:
        raise RuntimeError(
            f"Refusing to persist: only {variable} / {len(nunq)} (={variable_frac:.2%}) columns show variance >1."
        )
    _ensure_dir(artifacts_dir)
    # Optional detailed health snapshot (env: MLHC_PERSIST_VARIANCE_SNAPSHOT=1) to aid post-mortem of flatness events
    try:
        import os as _os, hashlib, math
        if _os.getenv("MLHC_PERSIST_VARIANCE_SNAPSHOT", "0").lower() in {"1","true","yes"}:
            num = features.select_dtypes(include=["number"])  # ignore object columns
            health = {}
            if not num.empty:
                desc_min = num.min()
                desc_max = num.max()
                desc_std = num.std(ddof=0)
                nunique = num.nunique(dropna=True)
                miss = num.isna().mean()
                for c in num.columns:
                    v = num[c]
                    # sample hash of first 10 non-null values for cheap drift signal
                    sample_vals = tuple(v.dropna().head(10).tolist())
                    sample_hash = hashlib.sha1(str(sample_vals).encode()).hexdigest() if sample_vals else None
                    health[c] = {
                        "min": None if pd.isna(desc_min[c]) else float(desc_min[c]),
                        "max": None if pd.isna(desc_max[c]) else float(desc_max[c]),
                        "std": None if pd.isna(desc_std[c]) else float(desc_std[c]),
                        "n_unique": int(nunique[c]),
                        "missing_frac": float(miss[c]),
                        "all_constant": bool(nunique[c] <= 1),
                        "sample10_hash": sample_hash,
                    }
            snapshot_path = artifacts_dir / "features_health.json"
            snapshot_path.write_text(json.dumps({
                "column_count": int(features.shape[1]),
                "row_count": int(features.shape[0]),
                "variable_columns": int(variable),
                "variable_fraction": float(variable_frac),
                "numeric_profile": health,
            }, indent=2))
            print("Persisted feature health snapshot ->", snapshot_path.name)
    except Exception as e:  # pragma: no cover
        print("WARNING: failed to write feature health snapshot ->", e)
    # to_parquet returns None; use pandas IO
    features.to_parquet(artifacts_dir / "features_full.parquet")
    # Optional raw (pre-prune) persistence for variance audits
    if persist_raw is None:
        import os as _os
        persist_raw = _os.getenv("MLHC_PERSIST_RAW_FEATURES", "0").lower() in {"1","true","yes"}
    if persist_raw and raw_features is not None and not raw_features.empty:
        try:
            raw_features.to_parquet(artifacts_dir / "features_raw.parquet")
            print("Persisted raw (pre-prune) feature matrix -> features_raw.parquet")
        except Exception as e:  # pragma: no cover
            print("WARNING: failed to persist raw features ->", e)
    prov = build_feature_provenance(features)
    (artifacts_dir / "feature_provenance.json").write_text(json.dumps(prov, indent=2))
    (artifacts_dir / "feature_columns.json").write_text(json.dumps(list(features.columns), indent=2))

def run_training_side_pipeline(con, cohort_subject_ids: Iterable[int], debug: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build training-side features and labels; returns DataFrames + optional debug info."""
    cohort_subject_ids_list = list(cohort_subject_ids)
    first_adm = extract_first_admissions_duckdb(con, cohort_subject_ids_list)
    labels_df, first_adm_filtered = extract_labels_and_filter(con, first_adm)
    modalities = extract_modalities_duckdb(con, first_adm_filtered)

    if debug:
        print("first_adm_filtered shape:", first_adm_filtered.shape)
        try:
            print("Distinct admittime:", first_adm_filtered['admittime'].nunique())
        except Exception:
            pass
        print("Demographics columns:", modalities['demo'].columns.tolist(), "rows:", len(modalities['demo']))
        print("Vitals rows / subjects:", len(modalities['vitals']), modalities['vitals']['subject_id'].nunique() if not modalities['vitals'].empty else 0)
        print("Labs rows / subjects:", len(modalities['labs']), modalities['labs']['subject_id'].nunique() if not modalities['labs'].empty else 0)
        print("Prescriptions rows / subjects:", len(modalities['rx']), modalities['rx']['subject_id'].nunique() if not modalities['rx'].empty else 0)
        print("Procedures rows / subjects:", len(modalities['proc']), modalities['proc']['subject_id'].nunique() if not modalities['proc'].empty else 0)

    features = build_feature_matrix(first_adm_filtered, modalities)
    # Alignment assertion: ensure subjects match labels 1-1 and ordering preserved after a stable sort
    label_subjects = labels_df['subject_id'].astype(int).tolist()
    feat_subjects = features.index.astype(int).tolist()
    if set(label_subjects) != set(feat_subjects):
        missing_in_feats = set(label_subjects) - set(feat_subjects)
        missing_in_labels = set(feat_subjects) - set(label_subjects)
        raise RuntimeError(
            f"Subject mismatch between labels and features. Missing_in_features={len(missing_in_feats)} Missing_in_labels={len(missing_in_labels)}"
        )
    # Order check (after sorting both) to detect unintended shuffles
    if sorted(label_subjects) != sorted(feat_subjects):  # logically redundant but explicit
        raise RuntimeError("Sorted subject ID sets differ between labels and features.")

    raw = build_features(first_adm_filtered, modalities['demo'], modalities['vitals'],
                        modalities['labs'], modalities['rx'], modalities['proc'])
    if debug:
        support = raw.notna().sum()
        print("Raw feature shape:", raw.shape)
        print("Support quantiles:", support.describe())
        print("Columns with support>=1:", (support>=1).sum(), ">=5:", (support>=5).sum(), ">=10:", (support>=10).sum())
        const_raw = raw.nunique(dropna=True)
        print("Non-constant columns pre-prune:", (const_raw>1).sum(), "Constant:", (const_raw<=1).sum())
        print("Sample non-constant names:", [c for c in raw.columns[const_raw>1][:10]])
        from project.pipeline_core import prune_features as _pf_dbg
        debug_pruned = _pf_dbg(raw.copy(), min_support=1, target_max=None)
        print("Debug pruned (min_support=1, no cap) shape:", debug_pruned.shape)
        print("Non-constant after relaxed pruning:", (debug_pruned.nunique(dropna=True)>1).sum())

    # (No persistence: caller responsible for saving features/labels if desired.)
    if debug:
        debug_info = {
            "initial_subject_count": len(set(cohort_subject_ids_list)),
            "first_admissions_count": 0 if first_adm is None else len(first_adm),  # already LOS filtered
            "post_label_filter_subjects": 0 if first_adm_filtered is None else len(first_adm_filtered),
            "labels_count": 0 if labels_df is None else len(labels_df),
            "modalities_rows": {k: (0 if v is None else len(v)) for k, v in modalities.items()},
            "feature_shape": features.shape,
            "feature_column_count": 0 if features is None else features.shape[1],
        }
    else:
        debug_info = {}

    debug_info["labels"] = labels_df.copy() if labels_df is not None else pd.DataFrame()
    return features, labels_df, debug_info


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
