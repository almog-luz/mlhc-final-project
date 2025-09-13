from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from .features import build_features, FEATURE_SCHEMA_VERSION
from .variability_features import build_variability_features
from .medication_features import build_medication_features
from .metrics_utils import compute_binary_metrics, metrics_to_dict
from .extract import (
    get_first_admissions,
    get_demographics,
    get_vitals_48h,
    get_labs_48h,
    get_prescriptions_48h,
    get_procedures_48h,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore
from sklearn.calibration import CalibratedClassifierCV


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    num_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    return ColumnTransformer(transformers=[("num", num_transform, feature_cols)], remainder="drop")


def fit_model(X, y, model_type: str = "xgb", calibrate: bool = True, calib_cv: int = 5, xgb_params: Optional[Dict[str, Any]] = None):
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    if model_type == "logreg":
        base = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    elif model_type == "hgb":
        base = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced")
    else:
        if XGBClassifier is None:
            raise RuntimeError("xgboost not installed")
        x = xgb_params or {}
        base = XGBClassifier(
            n_estimators=int(x.get("n_estimators", 600)),
            learning_rate=float(x.get("learning_rate", 0.03)),
            max_depth=int(x.get("max_depth", 4)),
            subsample=float(x.get("subsample", 0.8)),
            colsample_bytree=float(x.get("colsample_bytree", 0.8)),
            reg_lambda=float(x.get("reg_lambda", 1.0)),
            reg_alpha=float(x.get("reg_alpha", 0.0)),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            verbosity=0,
            scale_pos_weight=(negatives / max(positives,1)),
        )
    if not calibrate:
        base.fit(X, y)
        return base
    method = "isotonic" if positives >= 50 and negatives >= 50 else "sigmoid"
    clf = CalibratedClassifierCV(estimator=base, method=method, cv=calib_cv)  # type: ignore[arg-type]
    clf.fit(X, y)
    return clf


def train_readmission(feature_matrix: pd.DataFrame, labels: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                      model_type: str = "xgb", calibrate: bool = True, xgb_params: Optional[Dict[str, Any]] = None,
                      threshold_objective: str = "f1") -> Dict[str, Any]:
    y = labels['readmission_label'].astype(int).reindex(feature_matrix.index).fillna(0).to_numpy()
    X = feature_matrix.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    model = fit_model(Xtr, ytr, model_type=model_type, calibrate=calibrate, xgb_params=xgb_params)
    proba = model.predict_proba(Xte)[:,1]
    metrics = compute_binary_metrics(yte, proba, threshold_objective=threshold_objective)
    return {"model": model, "metrics": metrics_to_dict(metrics), "proba": proba, "y_true": yte}


def save_artifacts(out_dir: str, model, preprocessor, feature_cols: List[str], metrics: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, 'model_readmission.joblib'))
    joblib.dump(preprocessor, os.path.join(out_dir, 'preprocessor.joblib'))
    with open(os.path.join(out_dir, 'feature_columns.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f)
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'readmission': metrics}, f, indent=2)
    with open(os.path.join(out_dir, 'feature_schema_version.txt'), 'w', encoding='utf-8') as f:
        f.write(FEATURE_SCHEMA_VERSION + "\n")

