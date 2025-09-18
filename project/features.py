# Shared feature engineering utilities
from __future__ import annotations
from typing import List, Optional, Set
import pandas as pd

# Feature schema version (increment when adding/removing engineered blocks)
FEATURE_SCHEMA_VERSION = "2025-Block1-missingness-v1"

# Curated whitelist of common vital / lab labels to avoid exploding the
# feature space when raw item_label values contain highly granular or
# free‑text variants. These should be lower‑cased.
VALID_MEASUREMENT_LABELS: Set[str] = {
    'heart rate', 'respiratory rate', 'temperature', 'systolic', 'diastolic',
    'mean arterial', 'spo2', 'wbc', 'hemoglobin', 'platelet', 'sodium',
    'potassium', 'chloride', 'bicarbonate', 'bun', 'creatinine', 'glucose',
    'lactate'
}

# Hard limits to cap long‑tail label proliferation.
_MAX_LABELS_PER_MODALITY = 120  # safety upper bound
_MIN_LABEL_COUNT = 10           # drop ultra‑rare labels


def aggregate_events(df: pd.DataFrame, value_col: str, time_col: str, label_col: str) -> pd.DataFrame:
    """Aggregate time-series events into subject-level wide features.

    Extended: mean / std / min / max / range / last value per (subject_id, label).
    Column naming pattern: <original_label>__{mean|std|min|max|range|last}
    Returns DataFrame with one row per subject_id.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['subject_id'])
    if value_col not in df.columns or time_col not in df.columns or 'subject_id' not in df.columns:
        return pd.DataFrame(columns=['subject_id'])
    d = df.copy()
    # Coerce measurements to numeric; invalid strings -> NaN then dropped
    d[value_col] = pd.to_numeric(d[value_col], errors='coerce')
    d = d.dropna(subset=[value_col])
    if d.empty:
        return pd.DataFrame(columns=['subject_id'])
    # Safe datetime conversion; invalid -> NaT and subsequently dropped
    d[time_col] = pd.to_datetime(d[time_col], errors='coerce')
    d = d.dropna(subset=[time_col])
    if d.empty:
        return pd.DataFrame(columns=['subject_id'])

    # Normalize label text early to consolidate variants (case/whitespace).
    if label_col in d.columns:
        d[label_col] = d[label_col].astype(str).str.strip().str.lower()

        # Strategy: prefer curated whitelist; if insufficient coverage, fall back to
        # top-N most frequent labels after removing very rare ones. This guards
        # against accidental explosion (millions of sparse columns) from overly
        # specific labels.
        counts = d[label_col].value_counts()
        # Keep whitelist labels that appear in data
        selected = [lab for lab in VALID_MEASUREMENT_LABELS if lab in counts.index]
        # If whitelist gives too few (<5), augment with most frequent remaining labels
        if len(selected) < 5:
            frequent = counts.index.tolist()
            for lab in frequent:
                if lab not in selected:
                    selected.append(lab)
                if len(selected) >= _MAX_LABELS_PER_MODALITY:
                    break
        else:
            # Optionally extend with additional frequent labels (not in whitelist) up to cap
            for lab in counts.index:
                if lab in selected:
                    continue
                if counts[lab] < _MIN_LABEL_COUNT:
                    break  # remaining will be rarer
                selected.append(lab)
                if len(selected) >= _MAX_LABELS_PER_MODALITY:
                    break
        d = d[d[label_col].isin(selected)]
        if d.empty:
            return pd.DataFrame(columns=['subject_id'])
    # Compute base aggregations including std (population) then derive range
    agg = d.groupby(['subject_id', label_col])[value_col].agg(['mean', 'std', 'min', 'max']).reset_index()
    agg['range'] = agg['max'] - agg['min']
    last_vals = d.sort_values(["subject_id", label_col, time_col]).groupby(['subject_id', label_col]).tail(1)
    last = last_vals[['subject_id', label_col, value_col]].rename(columns={value_col: 'last'})
    wide = agg.merge(last, on=['subject_id', label_col], how='left')
    stats = ['mean', 'std', 'min', 'max', 'range', 'last']
    mats = []
    for stat in stats:
        pivot = wide.pivot_table(index='subject_id', columns=label_col, values=stat)
        pivot.columns = [f"{str(c)}__{stat}" for c in pivot.columns]
        mats.append(pivot)
    # Block1 additions: measurement counts & presence indicators (missingness)
    try:
        counts = d.groupby(['subject_id', label_col])[value_col].size().reset_index(name='__tmp_count')
        pivot_count = counts.pivot_table(index='subject_id', columns=label_col, values='__tmp_count')
        count_cols = []
        measured_cols = []
        if not pivot_count.empty:
            # Counts
            cc = pivot_count.copy()
            cc.columns = [f"{str(c)}__count" for c in cc.columns]
            mats.append(cc)
            count_cols = list(cc.columns)
            # Measured indicators (1 if any measurement existed)
            mc = pivot_count.copy()
            mc[mc.notna()] = 1  # any presence -> 1
            mc = mc.fillna(0)
            mc.columns = [f"{str(c)}__measured" for c in mc.columns]
            mats.append(mc)
            measured_cols = list(mc.columns)
        # (Optional) Could log counts here; kept silent to avoid verbose output within library function.
    except Exception:  # pragma: no cover
        pass
    out = pd.concat(mats, axis=1).reset_index()
    # Safety: enforce single row per subject_id
    if out['subject_id'].duplicated().any():
        out = out.groupby('subject_id', as_index=False).first()
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
        # Normalize column casing to lowercase for robust selection
        try:
            first_adm.columns = [c.lower() for c in first_adm.columns]
        except Exception:
            pass
        # Debug: print available columns for troubleshooting
        try:
            print('DEBUG(build_features): first_adm columns ->', list(first_adm.columns))
        except Exception:
            pass
        adm_cols_needed = ['subject_id','admittime']
        has_eth = 'ethnicity' in first_adm.columns
        # Only select ethnicity if present to avoid KeyError
        use_cols = [c for c in adm_cols_needed if c in first_adm.columns]
        if has_eth:
            use_cols.append('ethnicity')
        # Demographics may sometimes arrive missing gender or dob (partial patients table). Handle gracefully.
        try:
            print('DEBUG(build_features): demo columns ->', list(demo.columns))
        except Exception:
            pass
        # Normalize demo column names
        try:
            demo.columns = [c.lower() for c in demo.columns]
        except Exception:
            pass
        demo_cols_available = [c for c in ['subject_id','gender','dob'] if c in demo.columns]
        if 'subject_id' not in demo_cols_available:
            tmp = first_adm[use_cols].copy()
        else:
            tmp = first_adm[use_cols].merge(demo[demo_cols_available], on='subject_id', how='left')
        if 'ethnicity' not in tmp.columns:
            tmp['ethnicity'] = 'UNKNOWN'
        # Age computation only if dob present
        # Age computation with overflow and plausibility safeguards
        if 'dob' in tmp.columns:
            try:
                admit_dt = pd.to_datetime(tmp['admittime'], errors='coerce')
                dob_dt = pd.to_datetime(tmp['dob'], errors='coerce')
                # Raw age in years
                age_years = (admit_dt - dob_dt).dt.total_seconds() / (365.25 * 24 * 3600)
                # Invalidate implausible or negative ages
                invalid = (age_years.isna()) | (age_years < 0) | (age_years > 120)
                age_years[invalid] = 65  # fallback placeholder
                tmp['age'] = age_years.clip(lower=0)
                tmp.loc[tmp['age'] >= 89, 'age'] = 90
            except Exception as age_err:  # pragma: no cover
                print('DEBUG(build_features): age computation failed ->', age_err)
                tmp['age'] = 65
        else:
            tmp['age'] = 65  # no DOB available
        if 'gender' in tmp.columns:
            tmp['gender_M'] = (tmp['gender'].astype(str).str.upper() == 'M').astype(int)
            tmp['gender_F'] = (tmp['gender'].astype(str).str.upper() == 'F').astype(int)
        else:
            tmp['gender_M'] = 0
            tmp['gender_F'] = 0
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

    # Normalize modality column names to lowercase to align with aggregate_events expectations
    def _norm(df):
        if df is None or df.empty:
            return df
        try:
            df.columns = [c.lower() for c in df.columns]
        except Exception:
            pass
        return df
    vitals = _norm(vitals)
    labs = _norm(labs)
    rx = _norm(rx)
    proc = _norm(proc)
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

    # Admission temporal signals & LOS (helps readmission / LOS discrimination)
    adm_feat = pd.DataFrame()
    if first_adm is not None and not first_adm.empty:
        tmp = first_adm[['subject_id','admittime']].drop_duplicates('subject_id').copy()
        tmp['admittime'] = pd.to_datetime(tmp['admittime'])
        tmp['admit_hour'] = tmp['admittime'].dt.hour.astype(int)
        tmp['admit_weekend'] = (tmp['admittime'].dt.weekday >= 5).astype(int)
        tmp['admit_month'] = tmp['admittime'].dt.month.astype(int)
        if 'los_hours' in first_adm.columns:
            los_map = first_adm[['subject_id','los_hours']].drop_duplicates('subject_id')
            tmp = tmp.merge(los_map, on='subject_id', how='left')
        adm_feat = tmp.drop(columns=['admittime'])

    # Merge all
    features = feat_demo.copy() if not feat_demo.empty else pd.DataFrame({'subject_id': []})
    for df in [feat_vitals, feat_labs, feat_rx, feat_proc]:
        if df is not None and not df.empty:
            features = features.merge(df, on='subject_id', how='outer')
    if not adm_feat.empty:
        features = features.merge(adm_feat, on='subject_id', how='outer')
    features = features.drop_duplicates('subject_id').set_index('subject_id')
    # Ensure string column names
    if not features.empty:
        features.columns = features.columns.map(str)
    # --- Leakage Mitigation ---
    # 'los_hours' encodes the FULL realized length of stay for the index admission.
    # This is future information at prediction time when predicting prolonged LOS (>7d) and
    # yields near-perfect discrimination (confirmed by leakage audit). It must be excluded
    # from modeling features to prevent target leakage (and indirect leakage into other tasks).
    leaky_cols = [c for c in ['los_hours'] if c in features.columns]
    if leaky_cols:
        features = features.drop(columns=leaky_cols)
        # (Optional) Could log or return removed columns; kept silent for now.
    return features


def build_feature_provenance(features: pd.DataFrame, default_group: str = "baseline") -> dict:
    """Construct a simple provenance mapping for an existing feature matrix.

    Initially all features are assigned to a single baseline group. As new feature
    groups (comorbidity, utilization, variability, etc.) are added, this function
    can be extended or replaced with richer metadata (e.g., source tables, window,
    aggregation). Keeping it separate avoids changing the signature of build_features.
    Returns a dict: feature_name -> { 'group': <group_name> }
    """
    prov = {}
    if features is None or features.empty:
        return prov
    for c in features.columns:
        g = default_group
        if str(c).endswith('__measured') or str(c).endswith('__count'):
            g = 'missingness'
        prov[str(c)] = {"group": g}
    prov['__feature_schema_version__'] = {"group": FEATURE_SCHEMA_VERSION}
    return prov
