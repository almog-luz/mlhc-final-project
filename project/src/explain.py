from __future__ import annotations
import os, json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import joblib

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore

def compute_shap(model, preprocessor, X_sample: pd.DataFrame, feature_cols: List[str], model_type: str = 'xgb', background_size: int = 50):
    X_t = preprocessor.transform(X_sample[feature_cols])
    if model_type == 'xgb' and shap is not None and hasattr(model, 'get_booster'):
        try:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X_t)
            return vals
        except Exception:
            pass
    if shap is None:
        raise RuntimeError('shap not installed')
    import numpy as np
    ns = min(background_size, X_t.shape[0])
    rng = np.random.default_rng(42)
    bg = X_t[rng.choice(X_t.shape[0], size=ns, replace=False)]
    def _predict(p):
        return model.predict_proba(p)[:,1]
    expl = shap.KernelExplainer(_predict, bg)
    vals = expl.shap_values(X_t, nsamples=min(500, 50 * X_t.shape[1]))
    return vals

def summarize_shap(shap_vals, feature_cols: List[str]) -> pd.DataFrame:
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    abs_mean = np.abs(shap_vals).mean(axis=0)
    return pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': abs_mean}).sort_values('mean_abs_shap', ascending=False)

def group_shap(summary_df: pd.DataFrame, provenance_path: str | None) -> pd.DataFrame:
    prov = {}
    if provenance_path and os.path.exists(provenance_path):
        with open(provenance_path, 'r', encoding='utf-8') as f:
            prov = json.load(f)
    def infer_group(f: str) -> str:
        if f.startswith('var_'): return 'variability'
        if f.startswith('med_'): return 'medication'
        if f.startswith('stack_'): return 'stacking'
        if f.startswith('pheno_cluster'): return 'phenotype'
        if f.endswith('__measured') or f.endswith('__count'): return 'missingness'
        if f.startswith('cmb_') or f.startswith('cmbgrp_'): return 'comorbidity'
        if f.startswith('utl_'): return 'utilization'
        return 'baseline'
    groups = []
    for feat in summary_df['feature']:
        if feat in prov:
            groups.append(prov[feat].get('group', infer_group(feat)))
        else:
            groups.append(infer_group(feat))
    summary_df = summary_df.copy()
    summary_df['group'] = groups
    agg = summary_df.groupby('group', as_index=False).agg(feature_count=('feature','count'), total_mean_abs_shap=('mean_abs_shap','sum'))
    total = agg['total_mean_abs_shap'].sum()
    agg['pct_of_total'] = agg['total_mean_abs_shap'] / total if total else 0.0
    agg['mean_mean_abs_shap'] = agg['total_mean_abs_shap'] / agg['feature_count']
    return agg.sort_values('total_mean_abs_shap', ascending=False)
