from typing import List, Any
import pandas as pd
from pathlib import Path
from .pipeline_core import run_inference_pipeline


def run_pipeline_on_unseen_data(subject_ids: List[int], client: Any) -> pd.DataFrame:
    """Unified unseen inference entrypoint (delegates to pipeline_core).

    Public signature preserved for external callers. Internally we reuse the
    shared DuckDB extraction + feature build logic to guarantee parity with
    training-side feature generation.
    """
    if client is None:
        raise ValueError("client (duckdb.Connection) is required")
    if not subject_ids:
        raise ValueError("No subject_ids provided")
    # Derive canonical artifact/model directory (co-located in project/artifacts)
    artifacts_dir = Path(__file__).parent / 'artifacts'
    models_dir = artifacts_dir  # models + feature_columns.json live together
    scored = run_inference_pipeline(
        client,
        subject_ids=subject_ids,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
    )
    # Ensure column order & presence
    expected_cols = [
        'subject_id', 'mortality_proba', 'prolonged_LOS_proba', 'readmission_proba'
    ]
    for col in expected_cols:
        if col not in scored.columns:
            scored[col] = pd.NA
    return scored[expected_cols].sort_values('subject_id').reset_index(drop=True)
