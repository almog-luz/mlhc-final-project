# Label construction utilities extracted from exploration notebook
from __future__ import annotations
from typing import List
import pandas as pd

try:  # support package and script usage
    from .extract import (
        get_first_admissions,
        get_demographics,
        get_all_admissions,
    )
except ImportError:  # pragma: no cover
    from extract import (
        get_first_admissions,
        get_demographics,
        get_all_admissions,
    )


def build_labels(client, subject_ids: List[int]) -> pd.DataFrame:
    """Build mortality, prolonged LOS (>7d), and 30-day readmission labels.

    Steps:
      1. Get first admission per subject, compute LOS, filter LOS >=54h (model requirement).
      2. Mortality: in-hospital death (deathtime not null) OR death date (dod) within 30 days post discharge.
      3. Prolonged LOS: > 7 days.
      4. Readmission: any subsequent admission within 30 days of first discharge.
    Returns a DataFrame[subject_id, hadm_id, mortality_label, prolonged_los_label, readmission_label].
    """
    if not subject_ids:
        return pd.DataFrame(columns=["subject_id","hadm_id","mortality_label","prolonged_los_label","readmission_label"])  # pragma: no cover

    first_adm = get_first_admissions(client, subject_ids)
    if first_adm.empty:
        return pd.DataFrame(columns=["subject_id","hadm_id","mortality_label","prolonged_los_label","readmission_label"])  # pragma: no cover

    first_adm['admittime'] = pd.to_datetime(first_adm['admittime'])
    first_adm['dischtime'] = pd.to_datetime(first_adm['dischtime'])
    first_adm['los_hours'] = (first_adm['dischtime'] - first_adm['admittime']).dt.total_seconds() / 3600.0
    first_adm = first_adm[first_adm['los_hours'] >= 54].copy()
    if first_adm.empty:
        return pd.DataFrame(columns=["subject_id","hadm_id","mortality_label","prolonged_los_label","readmission_label"])  # pragma: no cover

    # All admissions for readmission + in-hospital death check
    all_adm = get_all_admissions(client, first_adm['subject_id'].tolist())
    all_adm['admittime'] = pd.to_datetime(all_adm['admittime'])
    all_adm['dischtime'] = pd.to_datetime(all_adm['dischtime'])
    all_adm['deathtime'] = pd.to_datetime(all_adm['deathtime'])

    # Demographics for date of death (dod)
    demo = get_demographics(client, first_adm['subject_id'].tolist())
    if 'dod' not in demo.columns:
        demo['dod'] = pd.NaT
    demo['dod'] = pd.to_datetime(demo['dod'])

    # Mortality
    mort = first_adm[['subject_id','hadm_id','dischtime']].merge(
        all_adm[['hadm_id','deathtime']], on='hadm_id', how='left'
    ).merge(demo[['subject_id','dod']], on='subject_id', how='left')
    death_in_hosp = mort['deathtime'].notna()
    death_within_30 = (mort['dod'].notna()) & (mort['dod'] <= (mort['dischtime'] + pd.Timedelta(days=30))) & (mort['dod'] >= mort['dischtime'])
    mort['mortality_label'] = (death_in_hosp | death_within_30).astype(int)

    # Prolonged LOS > 7 days
    pl = first_adm[['subject_id','hadm_id','los_hours']].copy()
    pl['prolonged_los_label'] = (pl['los_hours'] > 7*24).astype(int)

    # Readmission within 30 days
    fa = first_adm[['subject_id','hadm_id','dischtime']].rename(columns={'hadm_id':'first_hadm_id','dischtime':'first_discharge'})
    nxt = all_adm.sort_values(['subject_id','admittime']).merge(fa[['subject_id','first_hadm_id','first_discharge']], on='subject_id', how='left')
    after_first = nxt[nxt['admittime'] > nxt['first_discharge']].copy()
    within_30 = after_first.groupby('subject_id').apply(lambda g: (g['admittime'] <= g['first_discharge'].iloc[0] + pd.Timedelta(days=30)).any())
    readmit = within_30.rename('readmission_label').reset_index()

    labels = first_adm[['subject_id','hadm_id']].merge(mort[['subject_id','hadm_id','mortality_label']], on=['subject_id','hadm_id'], how='left')
    labels = labels.merge(pl[['subject_id','hadm_id','prolonged_los_label']], on=['subject_id','hadm_id'], how='left')
    labels = labels.merge(readmit, on='subject_id', how='left')
    labels['readmission_label'] = labels['readmission_label'].fillna(False).astype(int)
    labels[['mortality_label','prolonged_los_label']] = labels[['mortality_label','prolonged_los_label']].fillna(0).astype(int)
    return labels[['subject_id','hadm_id','mortality_label','prolonged_los_label','readmission_label']]


def build_and_save_labels(client, subject_ids: List[int], output_csv: str) -> pd.DataFrame:
    df = build_labels(client, subject_ids)
    df.to_csv(output_csv, index=False)
    return df

