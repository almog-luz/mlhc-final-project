# Helper utilities for evaluating binary classifiers
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


@dataclass
class BinaryMetrics:
    n: int
    positives: int
    prevalence: float
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float
    threshold: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    specificity: Optional[float]
    accuracy: Optional[float]
    cm: Optional[Dict[str, int]]
    calibration: Optional[pd.DataFrame]


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray, kind: str = "roc") -> float:
    try:
        if kind == "roc":
            return float(roc_auc_score(y_true, y_proba))
        return float(average_precision_score(y_true, y_proba))
    except Exception:
        return float("nan")


def calibration_bins(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    if len(y_true) == 0:
        return pd.DataFrame(columns=["bin", "count", "mean_proba", "empirical_rate"])
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    # Avoid edge effects: clip proba into [0, 1]
    df["p"] = df["p"].clip(0.0, 1.0)
    # Use quantile bins for better balance if enough unique probs
    try:
        if df["p"].nunique() >= n_bins:
            df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
        else:
            df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    except Exception:
        df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    calib = df.groupby("bin").agg(
        count=("y", "size"),
        mean_proba=("p", "mean"),
        empirical_rate=("y", "mean"),
    ).reset_index()
    return calib


def expected_calibration_error(calibration_df: pd.DataFrame) -> float:
    if calibration_df is None or calibration_df.empty:
        return float("nan")
    n = calibration_df["count"].sum()
    if n == 0:
        return float("nan")
    diffs = (calibration_df["mean_proba"] - calibration_df["empirical_rate"]).abs()
    weights = calibration_df["count"] / n
    return float((diffs * weights).sum())


def best_threshold(y_true: np.ndarray, y_proba: np.ndarray, objective: str = "f1") -> Tuple[Optional[float], Dict[str, float]]:
    if len(y_true) == 0:
        return None, {}
    # Use unique probability values as candidate thresholds
    thresholds = np.unique(y_proba)
    if thresholds.size == 0:
        return None, {}
    best_t = None
    best_val = -np.inf
    best_stats: Dict[str, float] = {}
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        try:
            if objective == "f1":
                val = f1_score(y_true, y_pred, zero_division=0)
            else:
                # default to f1
                val = f1_score(y_true, y_pred, zero_division=0)
        except Exception:
            val = -np.inf
        if val > best_val:
            best_val = val
            best_t = float(t)
            # compute stats at this threshold
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = (tp + tn) / max(1, (tp + tn + fp + fn))
            specificity = tn / max(1, (tn + fp))
            best_stats = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(acc),
                "specificity": float(specificity),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
    return best_t, best_stats


def compute_binary_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold_objective: Optional[str] = None,
    n_calib_bins: int = 10,
) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    n = int(y_true.shape[0])
    positives = int(y_true.sum())
    prevalence = float(positives / n) if n > 0 else float("nan")

    roc = _safe_auc(y_true, y_proba, kind="roc")
    pr = _safe_auc(y_true, y_proba, kind="pr")
    try:
        brier = float(brier_score_loss(y_true, y_proba))
    except Exception:
        brier = float("nan")

    calib = calibration_bins(y_true, y_proba, n_bins=n_calib_bins)
    ece = expected_calibration_error(calib)

    t = None
    stats: Dict[str, float] = {}
    if threshold_objective:
        t, stats = best_threshold(y_true, y_proba, objective=threshold_objective)

    cm: Optional[Dict[str, int]] = None
    if stats:
        cm = {k: int(v) for k, v in stats.items() if k in {"tp", "tn", "fp", "fn"}}

    return BinaryMetrics(
        n=n,
        positives=positives,
        prevalence=prevalence,
        roc_auc=roc,
        pr_auc=pr,
        brier=brier,
        ece=ece,
        threshold=t,
        precision=stats.get("precision"),
        recall=stats.get("recall"),
        f1=stats.get("f1"),
        specificity=stats.get("specificity"),
        accuracy=stats.get("accuracy"),
        cm=cm,
        calibration=calib,
    )


def metrics_to_dict(m: BinaryMetrics) -> Dict:
    out = {
        "n": m.n,
        "positives": m.positives,
        "prevalence": m.prevalence,
        "roc_auc": m.roc_auc,
        "pr_auc": m.pr_auc,
        "brier": m.brier,
        "ece": m.ece,
        "threshold": m.threshold,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "specificity": m.specificity,
        "accuracy": m.accuracy,
        "cm": m.cm,
    }
    if m.calibration is not None:
        out["calibration_bins"] = m.calibration.to_dict(orient="list")
    return out
