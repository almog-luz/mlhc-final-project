# Shared feature engineering utilities
from __future__ import annotations
from typing import List, Optional
import pandas as pd


def aggregate_events(df: pd.DataFrame, value_col: str, time_col: str, label_col: str) -> pd.DataFrame:
    """Aggregate time-series events into subject-level wide features.

    For each item label compute mean/min/max/last of the numeric value. Column naming pattern:
    <original_label>__{mean|min|max|last}.
    Returns DataFrame with one row per subject_id.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['subject_id'])
    d = df.dropna(subset=[value_col]).copy()
    if d.empty:
        return pd.DataFrame(columns=['subject_id'])
    d[time_col] = pd.to_datetime(d[time_col])
    last_vals = d.sort_values(["subject_id", label_col, time_col]).groupby(['subject_id', label_col]).tail(1)
    agg = d.groupby(['subject_id', label_col])[value_col].agg(['mean','min','max']).reset_index()
    last = last_vals[['subject_id', label_col, value_col]].rename(columns={value_col:'last'})
    wide = agg.merge(last, on=['subject_id', label_col], how='left')
    mats = []
    for stat in ['mean','min','max','last']:
        pivot = wide.pivot_table(index='subject_id', columns=label_col, values=stat)
        pivot.columns = [f"{str(c)}__{stat}" for c in pivot.columns]
        mats.append(pivot)
    out = pd.concat(mats, axis=1).reset_index()
    return out


def build_features(first_adm: pd.DataFrame,
                   demo: pd.DataFrame,
                   vitals: Optional[pd.DataFrame],
                   labs: Optional[pd.DataFrame],
                   rx: Optional[pd.DataFrame],
                   proc: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Create unified feature matrix.

    Parameters are individual modality data frames; missing/empty frames are handled gracefully.
    Returns a numeric feature matrix indexed by subject_id (index name subject_id).
    """
    # Demographics: age (cap 90), gender one-hot, ethnicity buckets
    feat_demo = pd.DataFrame()
    if first_adm is not None and not first_adm.empty and demo is not None and not demo.empty:
        tmp = first_adm[['subject_id','admittime','ethnicity']].merge(demo[['subject_id','gender','dob']], on='subject_id', how='left')
        tmp['age'] = (pd.to_datetime(tmp['admittime']) - pd.to_datetime(tmp['dob'])).dt.days / 365.25
        tmp['age'] = tmp['age'].clip(lower=0)
        tmp.loc[tmp['age'] >= 89, 'age'] = 90
        tmp['gender_M'] = (tmp['gender'].astype(str).str.upper() == 'M').astype(int)
        tmp['gender_F'] = (tmp['gender'].astype(str).str.upper() == 'F').astype(int)
        def _eth_bucket(x: str) -> str:
            s = str(x).lower()
            if 'white' in s:
                return 'WHITE'
            if 'black' in s:
                return 'BLACK'
            if 'asian' in s:
                return 'ASIAN'
            if 'hisp' in s or 'latino' in s or 'latina' in s:
                return 'HISPANIC'
            return 'OTHER'
        tmp['eth_bucket'] = tmp['ethnicity'].apply(_eth_bucket)
        eth_dummies = pd.get_dummies(tmp['eth_bucket'], prefix='eth', dtype=int)
        feat_demo = pd.concat([tmp[['subject_id','age','gender_M','gender_F']], eth_dummies], axis=1)

    feat_vitals = aggregate_events(vitals, value_col='valuenum', time_col='charttime', label_col='item_label') if vitals is not None else pd.DataFrame()
    feat_labs = aggregate_events(labs, value_col='valuenum', time_col='charttime', label_col='item_label') if labs is not None else pd.DataFrame()

    # Prescriptions: flags and simple counts
    feat_rx = pd.DataFrame()
    if rx is not None and not rx.empty:
        r = rx.copy()
        r['drug_low'] = r['drug'].astype(str).str.lower()
        def any_pattern(series: pd.Series, patterns: List[str]) -> pd.Series:
            return series.str.contains('|'.join(patterns), na=False)
        abx_patterns = ['cillin','cef','ceph','penem','floxacin','vancomycin','metronidazole','piperacillin','tazobactam']
        insulin_patterns = ['insulin']
        diuretic_patterns = ['furosemide','lasix','bumetanide','torsemide','hydrochlorothiazide','hctz','spironolactone']
        steroid_patterns = ['predni','methylpred','hydrocortisone','dexamethasone']
        grp = r.groupby('subject_id')
        feat_rx = pd.DataFrame({
            'subject_id': grp.size().index,
            'rx_total_mentions': grp.size().to_list(),
            'rx_unique_drugs': grp['drug_low'].nunique().to_list(),
        })
        rx_flags = r[['subject_id','drug_low']].copy()
        flags = rx_flags.groupby('subject_id').agg({
            'drug_low': lambda s: pd.Series({
                'rx_any_abx': any_pattern(s, abx_patterns).any(),
                'rx_any_insulin': any_pattern(s, insulin_patterns).any(),
                'rx_any_diuretic': any_pattern(s, diuretic_patterns).any(),
                'rx_any_steroid': any_pattern(s, steroid_patterns).any(),
            })
        })
        flags = pd.DataFrame(flags['drug_low'].to_list(), index=flags.index).reset_index().rename(columns={'index':'subject_id'})
        for c in flags.columns:
            if c != 'subject_id':
                flags[c] = flags[c].astype(int)
        feat_rx = feat_rx.merge(flags, on='subject_id', how='left')

    # Procedures: heuristic flags
    feat_proc = pd.DataFrame()
    if proc is not None and not proc.empty:
        p = proc.copy()
        label_col = 'item_label' if 'item_label' in p.columns else 'ordercategorydescription'
        p['lbl_low'] = p[label_col].astype(str).str.lower()
        def flag_any(df: pd.DataFrame, patterns: List[str]) -> pd.Series:
            pattern_re = '|'.join(patterns)
            # Avoid deprecated groupby.apply on grouping columns: aggregate boolean then cast
            hits = df[['subject_id','lbl_low']].assign(_hit=df['lbl_low'].str.contains(pattern_re, na=False))
            agg = hits.groupby('subject_id', observed=False)['_hit'].any()
            return agg.astype(int)
        vent = flag_any(p, ['vent','intubat','endotracheal','peep','tidal volume'])
        rrt = flag_any(p, ['dialysis','crrt','hemodialysis'])
        cl = flag_any(p, ['central line','cvc','subclavian','internal jugular','femoral line'])
        feat_proc = pd.DataFrame({
            'subject_id': vent.index,
            'proc_vent_any': vent.values,
            'proc_rrt_any': rrt.values,
            'proc_central_line_any': cl.values,
        })

    # Merge all
    features = feat_demo.copy() if not feat_demo.empty else pd.DataFrame({'subject_id': []})
    for df in [feat_vitals, feat_labs, feat_rx, feat_proc]:
        if df is not None and not df.empty:
            features = features.merge(df, on='subject_id', how='outer')
    features = features.drop_duplicates('subject_id').set_index('subject_id')
    # Ensure string column names
    if not features.empty:
        features.columns = features.columns.map(str)
    return features
