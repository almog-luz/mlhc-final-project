# Label construction utilities extracted from exploration notebook
from __future__ import annotations
from typing import List
import pandas as pd

from .extract import (
    get_first_admissions,
    get_demographics,
    get_all_admissions,
    get_first_admissions_duckdb,
    get_all_admissions_duckdb,
    get_demographics_duckdb,
)
from google.cloud import bigquery as bq


def build_labels(client: bq.Client, subject_ids: List[int]) -> pd.DataFrame:
    """Build mortality, prolonged LOS (>7d), and 30-day readmission labels.

    Steps:
      1. Get first admission per subject, compute LOS, filter LOS >=54h (model requirement).
      2. Mortality: in-hospital death (deathtime not null) OR death date (dod) within 30 days post discharge.
      3. Prolonged LOS: > 7 days.
      4. Readmission: any subsequent admission within 30 days of first discharge.
      5. Post-filter: REMOVE any subject whose discharge, in-hospital death, date-of-death (dod), or next admission (readmission) occurs within 54h of first admission (enforcing a 6h prediction gap after a 48h observation window).
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
    fa = first_adm[['subject_id','hadm_id','admittime','dischtime']].rename(columns={'hadm_id':'first_hadm_id','dischtime':'first_discharge'})
    nxt = all_adm.sort_values(['subject_id','admittime']).merge(fa[['subject_id','first_hadm_id','first_discharge']], on='subject_id', how='left')
    after_first = nxt[nxt['admittime'] > nxt['first_discharge']].copy()
    # Vectorized check: admission within 30 days of first discharge
    after_first['within_30'] = after_first['admittime'] <= (after_first['first_discharge'] + pd.Timedelta(days=30))
    within_30 = after_first.groupby('subject_id', observed=False)['within_30'].any()
    readmit = within_30.rename('readmission_label').reset_index()

    # ------------------------------------------------------------------
    # Early event filtering (strict >54h horizon enforcement)
    # Compute event times relative to first admission start and drop any
    # subject with an event (discharge, death, dod, next admission) <=54h.
    # ------------------------------------------------------------------
    fa_times = first_adm[['subject_id','admittime','dischtime','deathtime']].copy()
    # Explicit datetime coercion (defensive) before timedeltas
    for _col in ['admittime','dischtime','deathtime']:
        if _col in fa_times.columns:
            fa_times[_col] = pd.to_datetime(fa_times[_col], errors='coerce')
    fa_times['hours_to_discharge'] = (fa_times['dischtime'] - fa_times['admittime']).dt.total_seconds() / 3600.0
    fa_times['hours_to_in_hosp_death'] = (fa_times['deathtime'] - fa_times['admittime']).dt.total_seconds() / 3600.0
    # Merge dod for timing
    dod_map = demo[['subject_id','dod']].drop_duplicates('subject_id')
    fa_times = fa_times.merge(dod_map, on='subject_id', how='left')
    if 'dod' in fa_times.columns:
        fa_times['dod'] = pd.to_datetime(fa_times['dod'], errors='coerce')
    fa_times['hours_to_dod'] = (fa_times['dod'] - fa_times['admittime']).dt.total_seconds() / 3600.0
    # Next admission (readmission) time relative to first admittime
    next_adm = after_first.sort_values(['subject_id','admittime']).groupby('subject_id', as_index=False).first()
    if not next_adm.empty:
        # Merge to align columns as Series and compute timedelta safely
        tmp = next_adm[['subject_id','admittime']].merge(
            fa_times[['subject_id','admittime']].rename(columns={'admittime':'first_admit_start'}),
            on='subject_id', how='left')
        tmp['admittime'] = pd.to_datetime(tmp['admittime'])
        tmp['first_admit_start'] = pd.to_datetime(tmp['first_admit_start'])
        delta_series = (tmp['admittime'] - tmp['first_admit_start']).dt.total_seconds()/3600.0
        next_adm_times = pd.DataFrame({'subject_id': tmp['subject_id'], 'hours_to_next_admission': delta_series})
    else:
        next_adm_times = pd.DataFrame(columns=['subject_id','hours_to_next_admission'])
    fa_times = fa_times.merge(next_adm_times, on='subject_id', how='left')

    def _leq_54(series: pd.Series) -> pd.Series:
        return (series.notna()) & (series <= 54)

    early_mask = (
        _leq_54(fa_times['hours_to_discharge']) |
        _leq_54(fa_times['hours_to_in_hosp_death']) |
        _leq_54(fa_times['hours_to_dod']) |
        _leq_54(fa_times.get('hours_to_next_admission', pd.Series([], dtype=float)))
    )
    early_subjects = set(fa_times.loc[early_mask, 'subject_id'].tolist())

    if early_subjects:
        # Drop these from first_adm before label merges so they are excluded globally.
        first_adm = first_adm[~first_adm['subject_id'].isin(early_subjects)].copy()
        # If all removed, return empty
        if first_adm.empty:
            return pd.DataFrame(columns=["subject_id","hadm_id","mortality_label","prolonged_los_label","readmission_label"])  # pragma: no cover

        # Recompute components limited to remaining subjects
        mort = mort[~mort['subject_id'].isin(early_subjects)].copy()
        pl = pl[~pl['subject_id'].isin(early_subjects)].copy()
        readmit = readmit[~readmit['subject_id'].isin(early_subjects)].copy()

    labels = first_adm[['subject_id','hadm_id']].merge(mort[['subject_id','hadm_id','mortality_label']], on=['subject_id','hadm_id'], how='left')
    labels = labels.merge(pl[['subject_id','hadm_id','prolonged_los_label']], on=['subject_id','hadm_id'], how='left')
    labels = labels.merge(readmit, on='subject_id', how='left')
    # Use nullable boolean dtype before fillna to avoid FutureWarning about implicit downcasting
    labels['readmission_label'] = (
        labels['readmission_label']
        .astype('boolean')  # pandas nullable boolean
        .fillna(False)
        .astype(int)
    )
    labels[['mortality_label','prolonged_los_label']] = labels[['mortality_label','prolonged_los_label']].fillna(0).astype(int)
    return labels[['subject_id','hadm_id','mortality_label','prolonged_los_label','readmission_label']]


def build_and_save_labels(client: bq.Client, subject_ids: List[int], output_csv: str) -> pd.DataFrame:
    """Convenience wrapper that builds labels and writes them to CSV.

    Returns the labels DataFrame for further use in a pipeline.
    """
    df = build_labels(client, subject_ids)
    df.to_csv(output_csv, index=False)
    return df


# ----------------------------------------------------------------------------
# DuckDB variant (mirrors BigQuery logic)
# ----------------------------------------------------------------------------
def build_labels_duckdb(con, subject_ids: List[int]) -> pd.DataFrame:
    """DuckDB version of label builder with early-event (>54h horizon) enforcement.

    Applies same logic as build_labels (BigQuery) using DuckDB extraction helpers.
    Drops subjects whose first admission LOS <54h or any outcome/readmission event occurs ≤54h.
    """
    if not subject_ids:
        return pd.DataFrame(columns=["subject_id","hadm_id","mortality_label","prolonged_los_label","readmission_label"])  # pragma: no cover

    first_adm = get_first_admissions_duckdb(con, subject_ids)
    # Normalize column names to lowercase defensively (DuckDB may preserve case)
    if not first_adm.empty:
        first_adm.columns = [c.lower() for c in first_adm.columns]
    # Fallback alias mapping if subject id appears with alternative casing
    if 'subject_id' not in first_adm.columns:
        for alt in ['subjectid', 'SUBJECT_ID', 'Subject_ID']:
            if alt in first_adm.columns:
                first_adm = first_adm.rename(columns={alt: 'subject_id'})
                break
    if 'subject_id' not in first_adm.columns:
        print('DEBUG first_adm columns:', list(first_adm.columns))
        raise KeyError("Expected 'subject_id' in first_adm result from get_first_admissions_duckdb after normalization")
    if first_adm.empty:
        return pd.DataFrame(columns=["subject_id","hadm_id","mortality_label","prolonged_los_label","readmission_label"])  # pragma: no cover

    first_adm['admittime'] = pd.to_datetime(first_adm['admittime'])
    first_adm['dischtime'] = pd.to_datetime(first_adm['dischtime'])
    first_adm['los_hours'] = (first_adm['dischtime'] - first_adm['admittime']).dt.total_seconds() / 3600.0
    first_adm = first_adm[first_adm['los_hours'] >= 54].copy()
    if first_adm.empty:
        return pd.DataFrame(columns=["subject_id","hadm_id","mortality_label","prolonged_los_label","readmission_label"])  # pragma: no cover

    all_adm = get_all_admissions_duckdb(con, first_adm['subject_id'].tolist())
    if not all_adm.empty:
        all_adm.columns = [c.lower() for c in all_adm.columns]
        if 'subject_id' not in all_adm.columns:
            for alt in ['subjectid', 'SUBJECT_ID', 'Subject_ID']:
                if alt in all_adm.columns:
                    all_adm = all_adm.rename(columns={alt: 'subject_id'})
                    break
    if not all_adm.empty and 'subject_id' not in all_adm.columns:
        print('DEBUG all_adm columns:', list(all_adm.columns))
        raise KeyError("Expected 'subject_id' in all_adm result from get_all_admissions_duckdb after normalization")
    if all_adm.empty:
        return pd.DataFrame(columns=["subject_id","hadm_id","mortality_label","prolonged_los_label","readmission_label"])  # pragma: no cover
    all_adm['admittime'] = pd.to_datetime(all_adm['admittime'])
    all_adm['dischtime'] = pd.to_datetime(all_adm['dischtime'])
    all_adm['deathtime'] = pd.to_datetime(all_adm['deathtime'])

    demo = get_demographics_duckdb(con, first_adm['subject_id'].tolist())
    if not demo.empty:
        demo.columns = [c.lower() for c in demo.columns]
        if 'subject_id' not in demo.columns:
            for alt in ['subjectid', 'SUBJECT_ID', 'Subject_ID']:
                if alt in demo.columns:
                    demo = demo.rename(columns={alt: 'subject_id'})
                    break
    if not demo.empty and 'subject_id' not in demo.columns:
        print('DEBUG demo columns:', list(demo.columns))
        # Allow proceeding; demographics may be optional but warn
        print('WARNING: subject_id missing in demographics after normalization.')
    if 'dod' not in demo.columns:
        demo['dod'] = pd.NaT
    demo['dod'] = pd.to_datetime(demo['dod'])

    mort = first_adm[['subject_id','hadm_id','dischtime']].merge(
        all_adm[['hadm_id','deathtime']], on='hadm_id', how='left'
    ).merge(demo[['subject_id','dod']], on='subject_id', how='left')
    death_in_hosp = mort['deathtime'].notna()
    death_within_30 = (mort['dod'].notna()) & (mort['dod'] <= (mort['dischtime'] + pd.Timedelta(days=30))) & (mort['dod'] >= mort['dischtime'])
    mort['mortality_label'] = (death_in_hosp | death_within_30).astype(int)

    pl = first_adm[['subject_id','hadm_id','los_hours']].copy()
    pl['prolonged_los_label'] = (pl['los_hours'] > 7*24).astype(int)

    fa = first_adm[['subject_id','hadm_id','admittime','dischtime']].rename(columns={'hadm_id':'first_hadm_id','dischtime':'first_discharge'})
    nxt = all_adm.sort_values(['subject_id','admittime']).merge(fa[['subject_id','first_hadm_id','first_discharge']], on='subject_id', how='left')
    after_first = nxt[nxt['admittime'] > nxt['first_discharge']].copy()
    after_first['within_30'] = after_first['admittime'] <= (after_first['first_discharge'] + pd.Timedelta(days=30))
    within_30 = after_first.groupby('subject_id', observed=False)['within_30'].any()
    readmit = within_30.rename('readmission_label').reset_index()

    # Early event filtering
    fa_times = first_adm[['subject_id','admittime','dischtime','deathtime']].copy()
    for _col in ['admittime','dischtime','deathtime']:
        if _col in fa_times.columns:
            fa_times[_col] = pd.to_datetime(fa_times[_col], errors='coerce')
    fa_times['hours_to_discharge'] = (fa_times['dischtime'] - fa_times['admittime']).dt.total_seconds() / 3600.0
    fa_times['hours_to_in_hosp_death'] = (fa_times['deathtime'] - fa_times['admittime']).dt.total_seconds() / 3600.0
    dod_map = demo[['subject_id','dod']].drop_duplicates('subject_id')
    fa_times = fa_times.merge(dod_map, on='subject_id', how='left')
    if 'dod' in fa_times.columns:
        fa_times['dod'] = pd.to_datetime(fa_times['dod'], errors='coerce')
    fa_times['hours_to_dod'] = (fa_times['dod'] - fa_times['admittime']).dt.total_seconds() / 3600.0
    next_adm = after_first.sort_values(['subject_id','admittime']).groupby('subject_id', as_index=False).first()
    if not next_adm.empty:
        tmp = next_adm[['subject_id','admittime']].merge(
            fa_times[['subject_id','admittime']].rename(columns={'admittime':'first_admit_start'}),
            on='subject_id', how='left')
        tmp['admittime'] = pd.to_datetime(tmp['admittime'])
        tmp['first_admit_start'] = pd.to_datetime(tmp['first_admit_start'])
        delta_series = (tmp['admittime'] - tmp['first_admit_start']).dt.total_seconds()/3600.0
        next_adm_times = pd.DataFrame({'subject_id': tmp['subject_id'], 'hours_to_next_admission': delta_series})
    else:
        next_adm_times = pd.DataFrame(columns=['subject_id','hours_to_next_admission'])
    fa_times = fa_times.merge(next_adm_times, on='subject_id', how='left')

    def _leq_54(series: pd.Series) -> pd.Series:
        return (series.notna()) & (series <= 54)

    early_mask = (
        _leq_54(fa_times['hours_to_discharge']) |
        _leq_54(fa_times['hours_to_in_hosp_death']) |
        _leq_54(fa_times['hours_to_dod']) |
        _leq_54(fa_times.get('hours_to_next_admission', pd.Series([], dtype=float)))
    )
    early_subjects = set(fa_times.loc[early_mask, 'subject_id'].tolist())
    if early_subjects:
        first_adm = first_adm[~first_adm['subject_id'].isin(early_subjects)].copy()
        if first_adm.empty:
            return pd.DataFrame(columns=["subject_id","hadm_id","mortality_label","prolonged_los_label","readmission_label"])  # pragma: no cover
        mort = mort[~mort['subject_id'].isin(early_subjects)].copy()
        pl = pl[~pl['subject_id'].isin(early_subjects)].copy()
        readmit = readmit[~readmit['subject_id'].isin(early_subjects)].copy()

    labels = first_adm[['subject_id','hadm_id']].merge(mort[['subject_id','hadm_id','mortality_label']], on=['subject_id','hadm_id'], how='left')
    labels = labels.merge(pl[['subject_id','hadm_id','prolonged_los_label']], on=['subject_id','hadm_id'], how='left')
    labels = labels.merge(readmit, on='subject_id', how='left')
    labels['readmission_label'] = (
        labels['readmission_label']
        .astype('boolean')
        .fillna(False)
        .astype(int)
    )
    labels[['mortality_label','prolonged_los_label']] = labels[['mortality_label','prolonged_los_label']].fillna(0).astype(int)
    return labels[['subject_id','hadm_id','mortality_label','prolonged_los_label','readmission_label']]


def build_and_save_labels_duckdb(con, subject_ids: List[int], output_csv: str) -> pd.DataFrame:
    df = build_labels_duckdb(con, subject_ids)
    df.to_csv(output_csv, index=False)
    return df

