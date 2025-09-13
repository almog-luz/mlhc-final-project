from __future__ import annotations
import pandas as pd

WINDOWS_DAYS = [30, 90, 365]


def build_utilization_features(all_admissions: pd.DataFrame, first_adm: pd.DataFrame) -> pd.DataFrame:
    """Generate prior utilization features (counts & recency).

    - Counts of prior admissions in 30/90/365 day windows.
    - Days since last discharge ( -1 if none ).
    - Mean prior LOS hours.
    - Approx prior admission rate (admissions in last 365d / 365).
    """
    if all_admissions is None or all_admissions.empty or first_adm is None or first_adm.empty:
        return pd.DataFrame(columns=['subject_id']).set_index('subject_id')

    fa = first_adm[['subject_id', 'admittime']].copy()
    fa['admittime'] = pd.to_datetime(fa['admittime'])
    adm = all_admissions.copy()
    adm['admittime'] = pd.to_datetime(adm['admittime'])
    adm['dischtime'] = pd.to_datetime(adm['dischtime'])

    # Join index admission start
    merged = adm.merge(fa.rename(columns={'admittime': 'index_admit'}), on='subject_id', how='inner')
    prior = merged[merged['admittime'] < merged['index_admit']].copy()
    if prior.empty:
        base = fa[['subject_id']].copy()
        for w in WINDOWS_DAYS:
            base[f'utl_prior_adm_{w}d'] = 0
        base['utl_days_since_last_discharge'] = -1
        base['utl_mean_prior_los_hours'] = 0.0
        base['utl_prior_adm_rate_365d'] = 0.0
        return base.set_index('subject_id')

    prior['los_hours'] = (prior['dischtime'] - prior['admittime']).dt.total_seconds() / 3600.0

    records = []
    idx_series = fa.set_index('subject_id')['admittime']
    for sid, idx_time in idx_series.items():
        sub = prior[prior['subject_id'] == sid]
        rec = {'subject_id': int(sid)}
        if sub.empty:
            for w in WINDOWS_DAYS:
                rec[f'utl_prior_adm_{w}d'] = 0
            rec['utl_days_since_last_discharge'] = -1
            rec['utl_mean_prior_los_hours'] = 0.0
            rec['utl_prior_adm_rate_365d'] = 0.0
            records.append(rec)
            continue
        for w in WINDOWS_DAYS:
            win_start = idx_time - pd.Timedelta(days=w)
            rec[f'utl_prior_adm_{w}d'] = int((sub['admittime'] >= win_start).sum())
        last_discharge = sub['dischtime'].max()
        rec['utl_days_since_last_discharge'] = (idx_time - last_discharge).days if pd.notnull(last_discharge) else -1
        rec['utl_mean_prior_los_hours'] = float(sub['los_hours'].mean()) if not sub.empty else 0.0
        rec['utl_prior_adm_rate_365d'] = float(rec.get('utl_prior_adm_365d', 0)) / 365.0
        records.append(rec)

    return pd.DataFrame(records).set_index('subject_id')
