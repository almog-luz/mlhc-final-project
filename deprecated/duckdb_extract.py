"""DuckDB extraction utilities using the ORIGINAL query structures from
`extract.py` (BigQuery version) with only minimal syntactic adjustments
required for DuckDB (UNNEST -> IN list; TIMESTAMP_DIFF -> date_diff).

IMPORTANT: Per user request, no additional filtering logic, heuristics, or
label curation beyond what exists in the original BigQuery helpers.

Mirrors functions:
    - get_first_admissions  (first admission per subject)
    - get_demographics
    - get_vitals_48h        (restricted label regex set)
    - get_labs_48h          (restricted label regex set)
    - get_prescriptions_48h
    - get_procedures_48h

High level helper `extract_early_window` reuses these to produce the same
modalities dictionary consumed by downstream feature engineering.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd

TIME_WINDOW_HOURS = 48
MIN_REQUIRED_LOS_HOURS = 54

def _safe(con, sql: str) -> pd.DataFrame:
    try:
        return con.execute(sql).fetchdf()
    except Exception as e:  # pragma: no cover
        print('QUERY FAIL:', e, '\nSQL snippet:', sql[:120])
        return pd.DataFrame()

def _csv_int_list(values: List[int]) -> str:
        return ','.join(str(int(v)) for v in values) if values else '-1'


def get_first_admissions_duckdb(con, subject_ids: List[int]) -> pd.DataFrame:
        if not subject_ids:
                return pd.DataFrame()
        sid_csv = _csv_int_list(subject_ids)
        sql = f"""
        SELECT subject_id, hadm_id, admittime, dischtime, deathtime, admission_type,
                     admission_location, discharge_location, diagnosis, insurance, language,
                     marital_status, ethnicity
        FROM admissions
        WHERE subject_id IN ({sid_csv})
        ORDER BY subject_id, admittime
        """
        df = _safe(con, sql)
        if df.empty:
                return df
        return (df.sort_values(['subject_id','admittime'])
                            .groupby('subject_id', as_index=False)
                            .first())


def get_demographics_duckdb(con, subject_ids: List[int]) -> pd.DataFrame:
        if not subject_ids:
                return pd.DataFrame()
        sid_csv = _csv_int_list(subject_ids)
        sql = f"""
        SELECT subject_id, gender, dob, dod, expire_flag
        FROM patients
        WHERE subject_id IN ({sid_csv})
        """
        return _safe(con, sql)


def get_vitals_48h_duckdb(con, hadm_ids: List[int]) -> pd.DataFrame:
        if not hadm_ids:
                return pd.DataFrame()
        hadm_csv = _csv_int_list(hadm_ids)
        # Original BigQuery query structure (label regex restrictions preserved)
        sql = f"""
        WITH first_adm AS (
            SELECT hadm_id, admittime
            FROM admissions
            WHERE hadm_id IN ({hadm_csv})
        )
        SELECT ce.subject_id, ce.hadm_id, ce.icustay_id, ce.charttime,
                     di.label AS item_label, ce.valuenum, ce.valueuom
        FROM chartevents ce
        JOIN first_adm fa USING (hadm_id)
        JOIN d_items di ON di.itemid = ce.itemid
        WHERE ce.hadm_id IN ({hadm_csv})
            AND ce.valuenum IS NOT NULL
            AND date_diff('hour', fa.admittime, ce.charttime) BETWEEN 0 AND {TIME_WINDOW_HOURS}
            AND (
                lower(di.label) REGEXP '(heart rate|hr)' OR
                lower(di.label) REGEXP '(respiratory rate|rr)' OR
                lower(di.label) REGEXP 'temperature' OR
                lower(di.label) REGEXP '((non)?invasive )?systolic|sysbp|sbp' OR
                lower(di.label) REGEXP '((non)?invasive )?diastolic|diasbp|dbp' OR
                lower(di.label) REGEXP 'mean arterial|map' OR
                lower(di.label) REGEXP 'spo2|o2 saturation|oxygen saturation'
            )
        ORDER BY subject_id, charttime
        """
        return _safe(con, sql)


def get_labs_48h_duckdb(con, hadm_ids: List[int]) -> pd.DataFrame:
        if not hadm_ids:
                return pd.DataFrame()
        hadm_csv = _csv_int_list(hadm_ids)
        sql = f"""
        WITH first_adm AS (
            SELECT hadm_id, admittime
            FROM admissions
            WHERE hadm_id IN ({hadm_csv})
        )
        SELECT le.subject_id, le.hadm_id, le.charttime,
                     dl.label AS item_label, le.valuenum, le.value AS value_text,
                     le.valueuom, le.flag
        FROM labevents le
        JOIN first_adm fa USING (hadm_id)
        JOIN d_labitems dl ON dl.itemid = le.itemid
        WHERE le.hadm_id IN ({hadm_csv})
            AND le.charttime IS NOT NULL
            AND date_diff('hour', fa.admittime, le.charttime) BETWEEN 0 AND {TIME_WINDOW_HOURS}
            AND (
                lower(dl.label) REGEXP 'wbc|white blood' OR
                lower(dl.label) REGEXP 'hemoglobin|hgb' OR
                lower(dl.label) REGEXP 'hematocrit|hct' OR
                lower(dl.label) REGEXP 'platelet' OR
                lower(dl.label) REGEXP 'sodium|na\\b' OR
                lower(dl.label) REGEXP 'potassium|k\\b' OR
                lower(dl.label) REGEXP 'chloride|cl\\b' OR
                lower(dl.label) REGEXP 'bicarbonate|hco3' OR
                lower(dl.label) REGEXP 'bun|urea' OR
                lower(dl.label) REGEXP 'creatinine' OR
                lower(dl.label) REGEXP 'glucose' OR
                lower(dl.label) REGEXP 'lactate'
            )
        ORDER BY subject_id, charttime
        """
        return _safe(con, sql)


def get_prescriptions_48h_duckdb(con, hadm_ids: List[int]) -> pd.DataFrame:
        if not hadm_ids:
                return pd.DataFrame()
        hadm_csv = _csv_int_list(hadm_ids)
        sql = f"""
        WITH first_adm AS (
            SELECT hadm_id, admittime
            FROM admissions
            WHERE hadm_id IN ({hadm_csv})
        )
        SELECT pr.subject_id, pr.hadm_id, pr.startdate, pr.enddate,
                     pr.drug, pr.drug_type, pr.formulary_drug_cd, pr.route
        FROM prescriptions pr
        JOIN first_adm fa USING (hadm_id)
        WHERE pr.hadm_id IN ({hadm_csv})
            AND pr.startdate IS NOT NULL
            AND date_diff('hour', fa.admittime, pr.startdate) BETWEEN 0 AND {TIME_WINDOW_HOURS}
        ORDER BY subject_id, startdate
        """
        return _safe(con, sql)


def get_procedures_48h_duckdb(con, hadm_ids: List[int]) -> pd.DataFrame:
        if not hadm_ids:
                return pd.DataFrame()
        hadm_csv = _csv_int_list(hadm_ids)
        sql = f"""
        WITH first_adm AS (
            SELECT hadm_id, admittime
            FROM admissions
            WHERE hadm_id IN ({hadm_csv})
        )
        SELECT pe.subject_id, pe.hadm_id, pe.icustay_id,
                     pe.starttime, pe.endtime,
                     pe.itemid, di.label AS item_label,
                     pe.ordercategoryname, pe.ordercategorydescription, pe.location
        FROM procedureevents_mv pe
        JOIN first_adm fa USING (hadm_id)
        LEFT JOIN d_items di ON di.itemid = pe.itemid
        WHERE pe.hadm_id IN ({hadm_csv})
            AND pe.starttime IS NOT NULL
            AND date_diff('hour', fa.admittime, pe.starttime) BETWEEN 0 AND {TIME_WINDOW_HOURS}
        ORDER BY subject_id, starttime
        """
        return _safe(con, sql)


def extract_early_window(subject_ids: List[int], con) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if not subject_ids:
                return pd.DataFrame(), {k: pd.DataFrame() for k in ['demo','vitals','labs','rx','proc']}

        first_adm = get_first_admissions_duckdb(con, subject_ids)
        if first_adm.empty:
                print('WARN: no admissions returned for provided subject_ids.')
                return first_adm, {k: pd.DataFrame() for k in ['demo','vitals','labs','rx','proc']}

        # Compute LOS hours & filter (same post-processing as previous implementation)
        first_adm['admittime'] = pd.to_datetime(first_adm['admittime'])
        first_adm['dischtime'] = pd.to_datetime(first_adm['dischtime'])
        first_adm['los_hours'] = (first_adm['dischtime'] - first_adm['admittime']).dt.total_seconds()/3600.0
        pre = len(first_adm)
        first_adm = first_adm[first_adm['los_hours'] >= MIN_REQUIRED_LOS_HOURS].copy()
        print(f'Admissions retained (LOS>={MIN_REQUIRED_LOS_HOURS}h): {len(first_adm)} / {pre}')
        if first_adm.empty:
                return first_adm, {k: pd.DataFrame() for k in ['demo','vitals','labs','rx','proc']}

        hadm_ids = first_adm['hadm_id'].dropna().astype(int).tolist()
        demo = get_demographics_duckdb(con, subject_ids)
        vitals = get_vitals_48h_duckdb(con, hadm_ids)
        labs = get_labs_48h_duckdb(con, hadm_ids)
        rx = get_prescriptions_48h_duckdb(con, hadm_ids)
        proc = get_procedures_48h_duckdb(con, hadm_ids)
        sizes = {k: len(v) for k,v in {'demo':demo,'vitals':vitals,'labs':labs,'rx':rx,'proc':proc}.items()}
        print('Modalities (rows):', sizes)
        return first_adm, {'demo': demo, 'vitals': vitals, 'labs': labs, 'rx': rx, 'proc': proc}
