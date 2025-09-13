"""Hyperparameter sweep utility for readmission model.

Runs multiple configurations of logistic regression (and optional HGB) for the readmission task
by invoking train.train_from_labels programmatically (faster than spawning new processes) and
collects key metrics into a CSV summary.

Example:
python -m project.hparam_tune \
  --labels project/runs/20250905_143527/labels.csv \
  --output-root project/runs/20250905_143527/tuning_readmission \
  --project-id ml-for-healthcare-2025 \
  --pos-weights 1 1.5 2 3 \
  --logreg-C 0.1 0.5 1 2 5 \
  --min-recall 0.15 0.20
"""
from __future__ import annotations
import os
import argparse
import itertools
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd

from .train import train_from_labels  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter sweep for readmission model.")
    p.add_argument("--labels", required=True, help="Path to labels.csv containing readmission_label column")
    p.add_argument("--output-root", required=True, help="Directory root for sweep runs (one subdir per config)")
    p.add_argument("--project-id", required=False, help="GCP Project ID (or env MLHC_PROJECT_ID)")
    p.add_argument("--pos-weights", nargs="*", type=float, default=[1.0, 1.5, 2.0, 3.0], help="Positive class weight multipliers to test")
    p.add_argument("--logreg-C", nargs="*", type=float, default=[1.0], help="Logistic regression C values to test")
    p.add_argument("--min-recall", nargs="*", type=float, default=[], help="Optional recall floors for threshold selection (readmission only)")
    p.add_argument("--model-types", nargs="*", default=["logreg"], choices=["logreg","hgb","xgb"], help="Model types to include in sweep (now supports xgb)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    p.add_argument("--use-cache-first", action="store_true", help="Use cached feature extracts if available")
    p.add_argument("--limit", type=int, default=0, help="Optional early stop after N configs (debug)")
    # Optuna
    p.add_argument("--optuna", action="store_true", help="Use Optuna instead of grid search")
    p.add_argument("--optuna-trials", type=int, default=30, help="Number of Optuna trials")
    p.add_argument("--optuna-metric", choices=["roc_auc","pr_auc","f1"], default="roc_auc", help="Metric to maximize in Optuna")
    p.add_argument("--optuna-storage", default=None, help="Optional Optuna storage URL (e.g., sqlite:///study.db)")
    p.add_argument("--optuna-study", default="readmission_tune", help="Optuna study name")
    p.add_argument("--enable-fs", action="store_true", help="Allow feature selection search (L1 for readmission)")
    p.add_argument("--fs-target", default="readmission", choices=["readmission","mortality","prolonged_los"], help="Target for L1 feature selection")
    p.add_argument("--phenotype-options", nargs="*", type=int, default=[0,4,6], help="Phenotype cluster options to consider")
    p.add_argument("--stacking-options", nargs="*", type=int, default=[0,1], help="Whether to include stacking meta-features (0/1)")
    return p.parse_args()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def run_one(config: Dict[str, Any]) -> Dict[str, Any]:
    # Prepare output directory name
    tag_parts = [config["model_type"], f"pw{config['pos_weight']}", f"C{config['C']}"]
    if config.get("feature_select_C"):
        tag_parts.append(f"fs{config['feature_select_C']}")
    if config.get("phenotype_clusters") is not None:
        tag_parts.append(f"ph{config['phenotype_clusters']}")
    if config.get("stack_readmission"):
        tag_parts.append("stack")
    if config.get("min_recall") is not None:
        tag_parts.append(f"r{config['min_recall']}")
    run_dir = os.path.join(config["output_root"], "_".join(tag_parts))
    ensure_dir(run_dir)
    # Stash global C value for logistic inside train module
    if config["model_type"] == "logreg":
        import project.train as train_mod  # local import to access global
        train_mod._GLOBAL_LOGREG_C = config['C']  # type: ignore
    metrics = train_from_labels(
        project_id=config["project_id"],
        input_csv=config["labels_path"],
        output_dir=run_dir,
        test_size=config["test_size"],
        random_state=config["seed"],
        use_cache_first=config["use_cache_first"],
        show_progress=False,
        profile=False,
        model_type=config["model_type"],
        calib_cv=5,
        skip_importance=True,
        cv_folds=0,
        finalize_mode=False,
        readmission_model_override="same",
        readmission_no_calibration=False,
        only_readmission=True,
        calib_progress=False,
        readmission_min_recall=config.get("min_recall"),
        readmission_pos_weight=config['pos_weight'],
        phenotype_clusters=config.get("phenotype_clusters", 0),
        stacking_meta=bool(config.get("stack_readmission", 0)),
        feature_select_target=("readmission" if config.get("feature_select_C") else None),
        feature_select_C=config.get("feature_select_C", 0.5),
    )
    m = metrics.get("readmission", {})
    summary = {
        "model_type": config["model_type"],
        "pos_weight": config['pos_weight'],
        "C": config['C'],
        "min_recall": config.get("min_recall"),
        "feature_select_C": config.get("feature_select_C"),
        "phenotype_clusters": config.get("phenotype_clusters"),
        "stack_readmission": config.get("stack_readmission"),
        "roc_auc": m.get("roc_auc"),
        "pr_auc": m.get("pr_auc"),
        "f1": m.get("f1"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "brier": m.get("brier"),
        "ece": m.get("ece"),
        "threshold": m.get("threshold"),
        "run_dir": run_dir,
    }
    # Persist individual summary JSON
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    args = parse_args()
    project_id = args.project_id or os.environ.get("MLHC_PROJECT_ID") or os.environ.get("GCP_PROJECT_ID") or "ml-for-healthcare-2025"
    ensure_dir(args.output_root)
    # Build recall floor list including 'no constraint' (None)
    min_recall_values_raw = [None] + list(args.min_recall)
    min_recall_values: List[Optional[float]] = [mr for mr in min_recall_values_raw]
    if args.optuna:
        import optuna
        def objective(trial: 'optuna.trial.Trial') -> float:  # type: ignore
            model_type = trial.suggest_categorical("model_type", args.model_types)
            # Log scale for C, linear for pos weight
            C = trial.suggest_float("C", 0.05, 5.0, log=True)
            pos_weight = trial.suggest_float("pos_weight", 0.5, 4.0)
            min_recall = trial.suggest_categorical("min_recall", min_recall_values)
            phenotype_clusters = trial.suggest_categorical("phenotype_clusters", args.phenotype_options)
            stack_flag = trial.suggest_categorical("stack_readmission", args.stacking_options)
            if args.enable_fs:
                fs_C = trial.suggest_float("feature_select_C", 0.05, 1.5, log=True)
            else:
                fs_C = None
            cfg = {
                "model_type": model_type,
                "pos_weight": pos_weight,
                "C": C,
                "min_recall": min_recall,
                "phenotype_clusters": phenotype_clusters,
                "stack_readmission": int(stack_flag),
                "feature_select_C": fs_C,
                "project_id": project_id,
                "labels_path": args.labels,
                "output_root": os.path.join(args.output_root, f"trial_{trial.number}"),
                "seed": args.seed,
                "test_size": args.test_size,
                "use_cache_first": args.use_cache_first,
            }
            res = run_one(cfg)
            metric = res.get(args.optuna_metric, None)
            if metric is None:
                raise RuntimeError("Metric missing")
            # Record summary as user attrs
            trial.set_user_attr("summary", res)
            return float(metric)
        study = optuna.create_study(
            study_name=args.optuna_study,
            storage=args.optuna_storage,
            load_if_exists=True,
            direction="maximize",
        )
        t0 = time.time()
        study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=True)
        dt = time.time() - t0
        print(f"Optuna completed {len(study.trials)} trials in {dt:.1f}s")
        best = study.best_trial
        best_summary = best.user_attrs.get("summary", {})
        with open(os.path.join(args.output_root, "optuna_best.json"), "w", encoding="utf-8") as f:
            json.dump({"value": best.value, "params": best.params, "summary": best_summary}, f, indent=2)
        # Trials dataframe
        try:
            df_trials = study.trials_dataframe()
            df_trials.to_csv(os.path.join(args.output_root, "optuna_trials.csv"), index=False)
        except Exception:
            pass
        print("Best params:", best.params)
        print("Best metric (", args.optuna_metric, ")=", best.value)
    else:
        configs = []
        for model_type in args.model_types:
            for pw, C, mr in itertools.product(args.pos_weights, args.logreg_C, min_recall_values):
                configs.append({
                    "model_type": model_type,
                    "pos_weight": float(pw),
                    "C": float(C),
                    "min_recall": mr,
                    "project_id": project_id,
                    "labels_path": args.labels,
                    "output_root": args.output_root,
                    "seed": args.seed,
                    "test_size": args.test_size,
                    "use_cache_first": args.use_cache_first,
                })
        if args.limit and args.limit > 0:
            configs = configs[:args.limit]
        results = []
        t0 = time.time()
        for i, cfg in enumerate(configs, 1):
            print(f"[{i}/{len(configs)}] {cfg['model_type']} pw={cfg['pos_weight']} C={cfg['C']} min_recall={cfg['min_recall']}")
            try:
                res = run_one(cfg)
                results.append(res)
            except Exception as e:  # pragma: no cover
                print("  ERROR:", e)
        dt = time.time() - t0
        if results:
            df = pd.DataFrame(results)
            df_sorted = df.sort_values(["roc_auc", "pr_auc"], ascending=[False, False])
            csv_path = os.path.join(args.output_root, "sweep_summary.csv")
            df_sorted.to_csv(csv_path, index=False)
            print(f"Saved summary to {csv_path} (top 5 shown):")
            print(df_sorted.head(5).to_string(index=False))
        print(f"Completed {len(results)} configs in {dt:.1f}s")


if __name__ == "__main__":
    main()
