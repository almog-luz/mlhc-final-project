"""Model evaluation utilities for training pipeline.

Generates:
 - ROC & PR curve data (JSON)
 - Calibration curve data (JSON)
 - Permutation feature importance (CSV)
 - Optional PNG plots (if matplotlib is installed)

These utilities are intentionally lightweight so that notebooks can
call into them without duplicating logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, List
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance

from .metrics_utils import calibration_bins


@dataclass
class CurveData:
    fpr: List[float]
    tpr: List[float]
    roc_thresholds: List[float]
    precision: List[float]
    recall: List[float]
    pr_thresholds: List[float]
    calib: pd.DataFrame

    def to_json_dict(self) -> Dict:
        return {
            "roc_fpr": list(map(float, self.fpr)),
            "roc_tpr": list(map(float, self.tpr)),
            "roc_thresholds": list(map(float, self.roc_thresholds)),
            "pr_precision": list(map(float, self.precision)),
            "pr_recall": list(map(float, self.recall)),
            "pr_thresholds": list(map(float, self.pr_thresholds)),
            "calibration_mean_proba": self.calib.get("mean_proba", pd.Series([], dtype=float)).astype(float).tolist(),
            "calibration_empirical_rate": self.calib.get("empirical_rate", pd.Series([], dtype=float)).astype(float).tolist(),
            "calibration_count": self.calib.get("count", pd.Series([], dtype=float)).astype(float).tolist(),
        }


def compute_curve_data(y_true: np.ndarray, y_proba: np.ndarray, n_calib_bins: int = 10) -> CurveData:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if y_true.size == 0:
        return CurveData([], [], [], [], [], [], pd.DataFrame())
    fpr, tpr, roc_th = roc_curve(y_true, y_proba)
    pr_prec, pr_rec, pr_th = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns last threshold omitted; ensure lengths consistent by padding
    if pr_th.size + 1 == pr_prec.size:
        pr_th = np.concatenate([pr_th, [1.0]])
    calib = calibration_bins(y_true, y_proba, n_bins=n_calib_bins)
    return CurveData(
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        roc_thresholds=roc_th.tolist(),
        precision=pr_prec.tolist(),
        recall=pr_rec.tolist(),
        pr_thresholds=pr_th.tolist(),
        calib=calib,
    )


def save_curve_data(curves: CurveData, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(curves.to_json_dict(), f, indent=2)


def maybe_plot_curves(curves: CurveData, target_name: str, out_dir: str) -> None:
    try:  # optional dependency
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover
        return
    os.makedirs(out_dir, exist_ok=True)
    # ROC
    plt.figure(figsize=(4, 4))
    plt.plot(curves.fpr, curves.tpr, label=f"ROC {target_name}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC - {target_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{target_name}_roc.png"), dpi=150)
    plt.close()
    # PR
    plt.figure(figsize=(4, 4))
    plt.plot(curves.recall, curves.precision, label=f"PR {target_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {target_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{target_name}_pr.png"), dpi=150)
    plt.close()
    # Calibration
    if not curves.calib.empty:
        plt.figure(figsize=(4, 4))
        plt.plot(curves.calib["mean_proba"], curves.calib["empirical_rate"], marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Rate")
        plt.title(f"Calibration - {target_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{target_name}_calibration.png"), dpi=150)
        plt.close()


def permutation_importance_report(model, X_matrix, y_true: np.ndarray, feature_names: Sequence[str], n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Compute permutation importance for a fitted (calibrated) model.

    model: object with predict_proba.
    X_matrix: ndarray or sparse matrix (already transformed) – we cannot revert to raw features for linear coefficients due to calibration wrapper.
    feature_names: names aligned to the columns used to build X_matrix.
    Returns DataFrame sorted by mean importance descending.
    """
    try:
        r = permutation_importance(model, X_matrix, y_true, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc")
    except Exception:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    importances = getattr(r, "importances", None)
    importances_mean = getattr(r, "importances_mean", None)
    importances_std = getattr(r, "importances_std", None)
    if importances is None or importances_mean is None or importances_std is None:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    if len(feature_names) != importances.shape[0]:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    imp = pd.DataFrame({
        "feature": list(feature_names),
        "importance_mean": importances_mean,
        "importance_std": importances_std,
    }).sort_values("importance_mean", ascending=False)
    return imp


def save_feature_importance(df: pd.DataFrame, out_path: str) -> None:
    if df is None or df.empty:
        return
    df.to_csv(out_path, index=False)
