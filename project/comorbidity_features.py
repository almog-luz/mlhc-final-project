from __future__ import annotations
import pandas as pd
from typing import Dict, List

# Simplified ICD9 prefix mappings for Charlson & Elixhauser style groupings.
# This is a pragmatic subset (not exhaustive) chosen for incremental lift.
CHARLSON_MAP = {
    'MI': ['410', '412'],
    'CHF': ['428'],
    'PVD': ['440', '441', '4439', '7854', 'V434'],
    'CVD': ['430', '431', '432', '433', '434', '435', '437', '438'],
    'Dementia': ['290'],
    'COPD': ['490', '491', '492', '493', '494', '495', '496'],
    'Rheumatic': ['710', '714', '725'],
    'PUD': ['531', '532', '533', '534'],
    'MildLiver': ['570', '5712', '5714', '5715', '5716', '5718', '5719', '5733', '5734', '5738', '5739', 'V427'],
    'Diabetes': ['2500', '2501', '2502', '2503', '2508', '2509'],
    'DiabComp': ['2504', '2505', '2506', '2507'],
    'Paralysis': ['342', '343', '344'],
    'Renal': ['582', '583', '585', '586', '588'],
    'Cancer': ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '170', '171', '172', '174', '175', '176', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '200', '201', '202', '203', '204', '205', '206', '207', '208'],
    'Metastatic': ['196', '197', '198', '199'],
    'SevereLiver': ['5722', '5723', '5724', '5728'],
    'HIV': ['042', '043', '044'],
}

# Helper broader groupings for potential aggregation (cardio/metabolic etc.)
BROAD_GROUPS = {
    'cardio': ['MI', 'CHF', 'PVD', 'CVD'],
    'metabolic': ['Diabetes', 'DiabComp'],
    'hepatic': ['MildLiver', 'SevereLiver'],
    'renal': ['Renal'],
    'oncologic': ['Cancer', 'Metastatic'],
    'pulmonary': ['COPD'],
    'neuro': ['Dementia', 'Paralysis'],
    'immune': ['HIV', 'Rheumatic'],
}

def _code_matches_any(icd: str, prefixes: List[str]) -> bool:
    for p in prefixes:
        if icd.startswith(p):
            return True
    return False

def build_comorbidity_frame(all_dx: pd.DataFrame, first_adm: pd.DataFrame) -> pd.DataFrame:
    """Construct comorbidity indicator features using ONLY prior admissions codes.

    Parameters
    ----------
    all_dx : DataFrame with columns [subject_id, hadm_id, admittime, dischtime, icd9_code]
    first_adm : First admission frame (used to get index admittime)

    Returns
    -------
    DataFrame indexed by subject_id with binary indicators and group counts.
    """
    if all_dx is None or all_dx.empty or first_adm is None or first_adm.empty:
        return pd.DataFrame(columns=['subject_id']).set_index('subject_id')

    idx_times = first_adm.set_index('subject_id')['admittime']
    # Filter to codes occurring strictly before the index admission start
    merged = all_dx.copy()
    merged['admittime'] = pd.to_datetime(merged['admittime'])
    merged = merged.merge(idx_times.rename('index_admit'), left_on='subject_id', right_index=True, how='inner')
    prior = merged[merged['admittime'] < merged['index_admit']].copy()
    if prior.empty:
        return pd.DataFrame(columns=['subject_id']).set_index('subject_id')

    prior['icd9_code'] = prior['icd9_code'].astype(str).str.replace('[^0-9A-Za-z]', '', regex=True)

    records: List[Dict[str, int]] = []
    for subject_id, sub_df in prior.groupby('subject_id'):
        codes = sub_df['icd9_code'].dropna().unique().tolist()
        feats: Dict[str, int] = {'subject_id': int(sub_df['subject_id'].iloc[0])}
        for name, pref_list in CHARLSON_MAP.items():
            feats[f"cmb_{name}"] = int(any(_code_matches_any(c, pref_list) for c in codes))
        for bname, comp_list in BROAD_GROUPS.items():
            feats[f"cmbgrp_{bname}"] = int(any(feats.get(f"cmb_{c}", 0) for c in comp_list))
        feats['cmb_total_conditions'] = int(sum(feats[k] for k in feats if k.startswith('cmb_') and not k.startswith('cmbgrp_')))
        records.append(feats)
    if not records:
        return pd.DataFrame(columns=['subject_id']).set_index('subject_id')
    out = pd.DataFrame(records).set_index('subject_id')
    return out
