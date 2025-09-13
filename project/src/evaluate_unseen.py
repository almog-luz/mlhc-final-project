import argparse
import os
import sys
import json
import pandas as pd
from google.cloud import bigquery

from .unseen_data_evaluation import run_pipeline_on_unseen_data
from .metrics_utils import compute_binary_metrics, metrics_to_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run unseen evaluation on subject IDs.")
    p.add_argument("--project-id", required=True, help="GCP Project ID for BigQuery")
    p.add_argument("--input", required=True, help="CSV with a subject_id column")
    p.add_argument("--output", required=True, help="Output CSV path for predictions")
    # Optional evaluation against provided labels
    p.add_argument("--labels", help="Optional CSV with ground-truth labels (columns: subject_id and any of mortality_label, prolonged_los_label, readmission_label)")
    p.add_argument("--metrics-output", help="Optional path for metrics JSON; defaults to <output> with _metrics.json suffix")
    p.add_argument("--threshold-objective", default="f1", help="Objective to choose threshold when reporting point metrics (default: f1)")
    return p.parse_args()


def _validate_labels_df(df: pd.DataFrame) -> pd.DataFrame:
    if "subject_id" not in df.columns:
        raise ValueError("Labels CSV must contain a 'subject_id' column")
    # normalize expected label column names
    rename_map = {
        "prolonged_los": "prolonged_los_label",
        "prolonged_LOS": "prolonged_los_label",
        "prolonged_LOS_label": "prolonged_los_label",
        "mortality": "mortality_label",
        "readmission": "readmission_label",
    }
    cols_lower = {c: c for c in df.columns}
    # try simple lower-case normalization for robustness
    for c in list(df.columns):
        lc = c.lower()
        if lc in rename_map:
            cols_lower[c] = rename_map[lc]
        elif lc.endswith("_label"):
            cols_lower[c] = lc
    df = df.rename(columns=cols_lower)
    return df


def _compute_and_save_metrics(preds: pd.DataFrame, labels_df: pd.DataFrame, metrics_path: str, threshold_objective: str = "f1") -> dict:
    # Merge on subject_id
    merged = preds.merge(labels_df, on="subject_id", how="inner")
    results = {}
    mapping = [
        ("mortality_label", "mortality_proba", "mortality"),
        ("prolonged_los_label", "prolonged_LOS_proba", "prolonged_los"),
        ("readmission_label", "readmission_proba", "readmission"),
    ]
    for y_col, p_col, name in mapping:
        if y_col in merged.columns and p_col in merged.columns:
            y = merged[y_col].dropna().astype(int)
            p = merged.loc[y.index, p_col].astype(float)
            if y.shape[0] == 0:
                continue
            # Ensure plain numpy arrays (avoid potential pandas ExtensionArray typing noise)
            m = compute_binary_metrics(y.to_numpy(), p.to_numpy(), threshold_objective=threshold_objective)
            results[name] = metrics_to_dict(m)
    # Write JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    if 'subject_id' not in df.columns:
        print("Input CSV must contain a 'subject_id' column", file=sys.stderr)
        sys.exit(1)

    subject_ids = df['subject_id'].dropna().astype(int).tolist()

    client = bigquery.Client(project=args.project_id)

    preds = run_pipeline_on_unseen_data(subject_ids, client)
    preds.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")

    # Optional: compute metrics if labels CSV provided
    if args.labels:
        if not os.path.exists(args.labels):
            print(f"Labels file not found: {args.labels}", file=sys.stderr)
            sys.exit(1)
        labels_df = pd.read_csv(args.labels)
        try:
            labels_df = _validate_labels_df(labels_df)
        except Exception as e:
            print(f"Invalid labels CSV: {e}", file=sys.stderr)
            sys.exit(1)
        metrics_path = args.metrics_output or os.path.splitext(args.output)[0] + "_metrics.json"
        results = _compute_and_save_metrics(preds, labels_df, metrics_path, threshold_objective=args.threshold_objective)
        # Print concise summary
        if results:
            print("\nEvaluation summary:")
            for name, vals in results.items():
                print(f"- {name}: n={vals.get('n')}, prev={vals.get('prevalence'):.3f}, AUC={vals.get('roc_auc'):.3f}, PR-AUC={vals.get('pr_auc'):.3f}, Brier={vals.get('brier'):.3f}, ECE={vals.get('ece'):.3f}")
                th = vals.get('threshold')
                if th is not None:
                    print(f"  @thr≈{th:.3f}: P={vals.get('precision'):.3f}, R={vals.get('recall'):.3f}, F1={vals.get('f1'):.3f}, Acc={vals.get('accuracy'):.3f}, Spec={vals.get('specificity'):.3f}")
            print(f"Metrics JSON written to {metrics_path}")
        else:
            print("No overlapping labels found to compute metrics.")


if __name__ == "__main__":
    main()
