from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Set, List

# Abnormal thresholds (rough heuristic ranges) for selected vitals/labs.
ABNORMAL_SPEC = {
    'heart rate': {'low': 50, 'high': 110},
    'respiratory rate': {'low': 10, 'high': 24},
    'temperature': {'low': 35.5, 'high': 38.5},
    'systolic': {'low': 90, 'high': 160},
    'diastolic': {'low': 50, 'high': 100},
    'mean arterial': {'low': 60, 'high': 110},
    'spo2': {'low': 92, 'high': 100},
    'wbc': {'low': 4, 'high': 12},
    'hemoglobin': {'low': 10, 'high': 17},
    'platelet': {'low': 150, 'high': 450},
    'sodium': {'low': 130, 'high': 150},
    'potassium': {'low': 3.2, 'high': 5.2},
    'chloride': {'low': 95, 'high': 110},
    'bicarbonate': {'low': 18, 'high': 32},
    'bun': {'low': 5, 'high': 30},
    'creatinine': {'low': 0.4, 'high': 2.0},
    'glucose': {'low': 60, 'high': 200},
    'lactate': {'low': 0.5, 'high': 3.0},
}

# Label whitelist mirrors core feature builder.
VALID_MEASUREMENT_LABELS: Set[str] = set(ABNORMAL_SPEC.keys())
_MAX_VAR_LABELS = 60  # stricter than base aggregation – variability should stay compact

_REQUIRED_MIN_POINTS = 3

def _select_labels(d: pd.DataFrame) -> List[str]:
    counts = d['item_label'].value_counts()
    selected = [lab for lab in VALID_MEASUREMENT_LABELS if lab in counts.index]
    if len(selected) < 5:  # backfill with most frequent non‑whitelist up to cap
        for lab in counts.index:
            if lab not in selected:
                selected.append(lab)
            if len(selected) >= _MAX_VAR_LABELS:
                break
    else:  # optionally extend with frequent non‑whitelist labels (still capped)
        for lab in counts.index:
            if lab in selected:
                continue
            selected.append(lab)
            if len(selected) >= _MAX_VAR_LABELS:
                break
    return selected[:_MAX_VAR_LABELS]


def _compute_group_metrics(sub: pd.DataFrame, norm_label: str):
    # Expect sub has columns hours, valuenum
    if sub.shape[0] < _REQUIRED_MIN_POINTS:
        return None
    x = sub['hours'].to_numpy()
    y = sub['valuenum'].to_numpy()
    x_bar = x.mean()
    y_bar = y.mean()
    denom = ((x - x_bar) ** 2).sum()
    slope = float(((x - x_bar) * (y - y_bar)).sum() / denom) if denom > 0 else 0.0
    std = float(y.std(ddof=0))
    cv = float(std / y_bar) if y_bar not in (0, np.nan) else 0.0
    spec = ABNORMAL_SPEC.get(norm_label)
    if spec:
        abn_low = int((y < spec['low']).sum())
        abn_high = int((y > spec['high']).sum())
    else:
        abn_low = 0
        abn_high = 0
    return slope, std, cv, abn_low, abn_high


def _variability_from(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols_needed = {'subject_id', 'charttime', 'item_label', 'valuenum'}
    missing = cols_needed - set(df.columns)
    if missing:
        return pd.DataFrame()
    d = df[list(cols_needed)].dropna(subset=['valuenum']).copy()
    if d.empty:
        return pd.DataFrame()
    d['charttime'] = pd.to_datetime(d['charttime'])
    d['item_label'] = d['item_label'].astype(str).str.strip().str.lower()
    labels = _select_labels(d)
    d = d[d['item_label'].isin(labels)]
    if d.empty:
        return pd.DataFrame()
    # Relative hour within each (subject,label)
    d['hours'] = (d['charttime'] - d.groupby(['subject_id', 'item_label'])['charttime'].transform('min')).dt.total_seconds() / 3600.0
    # Iterate groups (bounded by subjects * labels; labels capped) – memory safe
    rows = []
    for (sid, lab), sub in d.groupby(['subject_id', 'item_label']):
        res = _compute_group_metrics(sub, lab)
        if res is None:
            continue
        rows.append((sid, lab, *res))
    if not rows:
        return pd.DataFrame()
    tidy = pd.DataFrame(rows, columns=['subject_id', 'item_label', 'slope', 'std', 'cv', 'abn_low_ct', 'abn_high_ct'])
    # Wide pivot with MultiIndex then flatten
    tidy_wide = tidy.set_index(['subject_id', 'item_label'])[['slope', 'std', 'cv', 'abn_low_ct', 'abn_high_ct']].unstack('item_label')
    # If only a single label survives, unstack can yield a Series per metric; ensure DataFrame
    if not isinstance(tidy_wide, pd.DataFrame):  # pragma: no cover - defensive
        tidy_wide = tidy_wide.to_frame()
    # Flatten columns: (metric, label) -> var_<label>__<metric>
    new_cols: List[str] = []
    for metric, label in tidy_wide.columns:
        new_cols.append(f"var_{label}__{metric}")
    tidy_wide.columns = new_cols
    tidy_wide.index.name = 'subject_id'
    return tidy_wide


def build_variability_features(vitals: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    # Build separately then outer-join on subject_id index.
    vit = _variability_from(vitals)
    lab = _variability_from(labs)
    if vit.empty and lab.empty:
        return pd.DataFrame(columns=['subject_id']).set_index('subject_id')
    if vit.empty:
        return lab
    if lab.empty:
        return vit
    return vit.join(lab, how='outer')
