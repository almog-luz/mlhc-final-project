import json
import argparse
from pathlib import Path
import pandas as pd
import sys
from pathlib import Path as _P

# Ensure package imports work when executed as a script from repo root
ROOT = _P(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.extract import get_first_admissions, get_all_admissions, get_demographics  # type: ignore
from project.labels import build_labels  # reuse LOS>=54h filtering


def audit_gap_bigquery(subject_ids, client, output_json: str):
    # Reconstruct first admissions (filtered already by build_labels) to align with modeling cohort
    labels_df = build_labels(client, subject_ids)
    if labels_df.empty:
        result = {
            "total_subjects": 0,
            "checked_admissions": 0,
            "violations": {},
            "status": "no data"
        }
        Path(output_json).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    first_adm = get_first_admissions(client, labels_df.subject_id.tolist())
    all_adm = get_all_admissions(client, labels_df.subject_id.tolist())
    demo = get_demographics(client, labels_df.subject_id.tolist())

    # Parse times
    for df, cols in [
        (first_adm, ["admittime", "dischtime", "deathtime"]),
        (all_adm, ["admittime", "dischtime", "deathtime"]),
        (demo, ["dod"])]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])

    # Merge death/discharge events for first admission only
    fa = first_adm.sort_values(["subject_id", "admittime"]).groupby("subject_id", as_index=False).first()
    fa = fa[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"]]

    # Identify death time (in-hospital or date of death) relative to admission
    demo = demo[["subject_id", "dod"]].drop_duplicates("subject_id")
    fa = fa.merge(demo, on="subject_id", how="left")

    fa["hours_to_discharge"] = (fa["dischtime"] - fa["admittime"]).dt.total_seconds() / 3600.0
    fa["hours_to_in_hosp_death"] = (fa["deathtime"] - fa["admittime"]).dt.total_seconds() / 3600.0
    fa["hours_to_dod"] = (fa["dod"] - fa["admittime"]).dt.total_seconds() / 3600.0

    # Readmission: get next admission start after first discharge
    all_adm_sorted = all_adm.sort_values(["subject_id", "admittime"]).copy()
    next_adm = all_adm_sorted.groupby("subject_id").nth(1).reset_index()  # second admission if exists
    if not next_adm.empty:
        # Align dischtime for subjects with a second admission
        disch_map = fa.set_index("subject_id")["dischtime"]
        aligned_discharge = disch_map.reindex(next_adm["subject_id"]).reset_index(drop=True)
        adm_times = next_adm["admittime"].reset_index(drop=True)
        deltas = adm_times - aligned_discharge
        next_adm["hours_to_next_admission"] = deltas.dt.total_seconds() / 3600.0
    else:
        next_adm["hours_to_next_admission"] = []

    # Evaluate violations (<=54h)
    def count_condition(series):
        mask = (series.notna()) & (series <= 54)
        return int(mask.sum())

    violations = {
        "discharge_leq_54h": count_condition(fa["hours_to_discharge"]),
        "in_hosp_death_leq_54h": count_condition(fa["hours_to_in_hosp_death"]),
        "dod_leq_54h": count_condition(fa["hours_to_dod"]),
        "readmission_leq_54h": count_condition(next_adm.get("hours_to_next_admission", pd.Series(dtype=float)))
    }

    # Totals
    result = {
        "total_subjects": int(fa.shape[0]),
        "checked_admissions": int(fa.shape[0]),
        "violations": violations,
        "any_violation": any(v > 0 for v in violations.values()),
        "violation_subject_ids_sample": {
            k: fa.loc[(fa[col] <= 54) & fa[col].notna(), "subject_id"].head(5).tolist() if k != "readmission_leq_54h" else []
            for k, col in [
                ("discharge_leq_54h", "hours_to_discharge"),
                ("in_hosp_death_leq_54h", "hours_to_in_hosp_death"),
                ("dod_leq_54h", "hours_to_dod"),
            ]
        }
    }

    Path(output_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Audit 54h gap for death/discharge/readmission")
    parser.add_argument("--output", default="gap_audit.json")
    parser.add_argument("--subject_csv", default="./data/initial_cohort.csv", help="CSV with subject_id column")
    args = parser.parse_args()

    # For local DuckDB only context (no BigQuery client), we can't run this fully.
    try:
        from google.cloud import bigquery as bq  # type: ignore
        client = bq.Client()
    except Exception:
        print("BigQuery client unavailable - gap audit requires BigQuery for admission timestamps.")
        return

    subj_df = pd.read_csv(args.subject_csv)
    if "subject_id" not in subj_df.columns:
        raise ValueError("subject_csv must contain subject_id column")

    subject_ids = subj_df["subject_id"].astype(int).tolist()
    audit_gap_bigquery(subject_ids, client, args.output)


if __name__ == "__main__":
    main()
