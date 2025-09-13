from __future__ import annotations
import pandas as pd
from typing import Dict, List

CLASS_PATTERNS: Dict[str, List[str]] = {
    'abx_beta_lactam': ['cillin', 'cef', 'ceph', 'penem', 'piperacillin', 'tazobactam'],
    'abx_glycopeptide': ['vancomycin'],
    'abx_metronidazole': ['metronidazole'],
    'abx_fluoroquinolone': ['floxacin'],
    'insulin': ['insulin'],
    'diuretic': ['furosemide','lasix','bumetanide','torsemide','hydrochlorothiazide','hctz','spironolactone'],
    'steroid': ['predni','methylpred','hydrocortisone','dexamethasone'],
    'anticoagulant': ['heparin','warfarin','enoxaparin'],
    'opioid': ['morphine','fentanyl','hydromorphone','oxycodone'],
}


def build_medication_features(rx: pd.DataFrame) -> pd.DataFrame:
    if rx is None or rx.empty:
        return pd.DataFrame(columns=['subject_id']).set_index('subject_id')
    r = rx.copy()
    r['drug_low'] = r['drug'].astype(str).str.lower()
    grp = r.groupby('subject_id')
    base = pd.DataFrame({'subject_id': grp.size().index})
    base['med_total_mentions'] = grp.size().values
    base['med_unique_drugs'] = grp['drug_low'].nunique().values
    # Class flags
    for cls, pats in CLASS_PATTERNS.items():
        pattern = '|'.join(pats)
        flag = grp['drug_low'].apply(lambda s: int(s.str.contains(pattern, regex=True, na=False).any()))
        base[f'med_cls_{cls}_any'] = flag.values
    # Counts for antibiotic subclasses
    abx_cols = [c for c in base.columns if c.startswith('med_cls_abx_')]
    if abx_cols:
        base['med_abx_class_count'] = base[abx_cols].sum(axis=1)
    return base.set_index('subject_id')
