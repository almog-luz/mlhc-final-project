from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Sequence, Optional
import pandas.api.types as ptypes
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_phenotype_clusters(features: pd.DataFrame, n_clusters: int = 5, random_state: int = 42,
                             max_features: int = 200) -> pd.DataFrame:
    """Derive coarse patient phenotype clusters from existing numeric feature matrix.

    Strategy:
      * Select up to `max_features` low-missingness, non-constant numeric columns.
      * Standardize.
      * Run KMeans to obtain cluster labels.
      * Return one-hot encoded cluster membership plus raw label.
    This is purposely lightweight (no PCA) to stay reproducible and fast.
    """
    if features is None or features.empty:
        return pd.DataFrame(columns=['subject_id']).set_index('subject_id')
    X = features.copy()
    # Drop obviously categorical one-hot columns already (binary 0/1) only if there are far too many; keep a sample.
    # Identify numeric columns
    num_cols = [c for c in X.columns if ptypes.is_numeric_dtype(X[c])]
    if not num_cols:
        return pd.DataFrame(columns=['subject_id']).set_index('subject_id')
    # Missingness filter & variance filter
    miss_frac = X[num_cols].isna().mean()
    good = miss_frac[miss_frac < 0.3].index.tolist()
    if not good:
        good = num_cols
    # Variance > 0
    var = X[good].var()
    good = [c for c in good if var[c] > 0]
    if not good:
        return pd.DataFrame(columns=['subject_id']).set_index('subject_id')
    # Cap dimensionality for clustering stability
    good = good[:max_features]
    sub = X[good].fillna(X[good].median())
    scaler = StandardScaler(with_mean=True, with_std=True)
    sub_s = scaler.fit_transform(sub)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(sub_s)
    out = pd.DataFrame({'subject_id': X.index, 'pheno_cluster': labels}).set_index('subject_id')
    # One-hot
    for k in range(n_clusters):
        out[f'pheno_cluster_{k}'] = (out['pheno_cluster'] == k).astype(int)
    return out
