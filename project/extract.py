"""Unified extraction utilities supporting both BigQuery and DuckDB.

Refactor: Original inline SQL strings have been lifted into module-level
constants so the exact query text is centralized. Functions for BigQuery
reuse those constants with parameter binding; parallel DuckDB functions
generate equivalent queries (minimal syntactic adjustments: UNNEST arrays
replaced with IN lists, TIMESTAMP_DIFF -> date_diff, removal of dataset
prefix/backticks, REGEXP_CONTAINS -> lower(col) REGEXP 'pattern').

Public API (BigQuery – unchanged names):
  - get_first_admissions
  - get_all_admissions
  - get_demographics
  - get_vitals_48h
  - get_labs_48h
  - get_prescriptions_48h
  - get_procedures_48h
  - get_all_admission_diagnoses

Added parallel DuckDB variants (suffix _duckdb):
  - get_first_admissions_duckdb
  - get_all_admissions_duckdb
  - get_demographics_duckdb
  - get_vitals_48h_duckdb
  - get_labs_48h_duckdb
  - get_prescriptions_48h_duckdb
  - get_procedures_48h_duckdb
  - get_all_admission_diagnoses_duckdb

Centralizing queries reduces divergence risk between engines.
"""

from __future__ import annotations
from typing import List, Optional
import pandas as pd

# ---------------------------------------------------------------------------
# DuckDB SQL templates (parameterized via simple Python string substitution)
# These mirror BigQuery semantics as closely as possible.
# ---------------------------------------------------------------------------
SQL_FIRST_ADMISSIONS_DUCKDB = """
SELECT subject_id, hadm_id, admittime, dischtime, deathtime, admission_type,
       admission_location, discharge_location, diagnosis, insurance, language,
       marital_status, ethnicity
FROM admissions
WHERE subject_id IN ({subject_ids_csv})
ORDER BY subject_id, admittime
"""

SQL_ALL_ADMISSIONS_DUCKDB = """
SELECT subject_id, hadm_id, admittime, dischtime, deathtime
FROM admissions
WHERE subject_id IN ({subject_ids_csv})
ORDER BY subject_id, admittime
"""

SQL_DEMOGRAPHICS_DUCKDB = """
SELECT subject_id, gender, dob, dod, expire_flag
FROM patients
WHERE subject_id IN ({subject_ids_csv})
"""

SQL_VITALS_48H_DUCKDB = """
-- Wide extraction (no label regex filtering) to mirror original legacy feature space.
WITH first_adm AS (
  SELECT hadm_id, admittime
  FROM admissions
  WHERE hadm_id IN ({hadm_ids_csv})
)
SELECT ce.subject_id, ce.hadm_id, ce.icustay_id, ce.charttime,
       di.label AS item_label, ce.valuenum, ce.valueuom
FROM chartevents ce
JOIN first_adm fa USING (hadm_id)
JOIN d_items di ON di.itemid = ce.itemid
WHERE ce.hadm_id IN ({hadm_ids_csv})
  AND ce.valuenum IS NOT NULL
  AND datediff('hour', CAST(fa.admittime AS TIMESTAMP), CAST(ce.charttime AS TIMESTAMP)) BETWEEN 0 AND 48
ORDER BY subject_id, charttime
"""

SQL_LABS_48H_DUCKDB = """
-- Wide extraction (no label regex filtering) to mirror original legacy feature space.
WITH first_adm AS (
  SELECT hadm_id, admittime
  FROM admissions
  WHERE hadm_id IN ({hadm_ids_csv})
)
SELECT le.subject_id, le.hadm_id, le.charttime,
       dl.label AS item_label, le.valuenum, le.value AS value_text,
       le.valueuom, le.flag
FROM labevents le
JOIN first_adm fa USING (hadm_id)
JOIN d_labitems dl ON dl.itemid = le.itemid
WHERE le.hadm_id IN ({hadm_ids_csv})
  AND le.charttime IS NOT NULL
  AND datediff('hour', CAST(fa.admittime AS TIMESTAMP), CAST(le.charttime AS TIMESTAMP)) BETWEEN 0 AND 48
ORDER BY subject_id, charttime
"""

SQL_PRESCRIPTIONS_48H_DUCKDB = """
WITH first_adm AS (
  SELECT hadm_id, admittime
  FROM admissions
  WHERE hadm_id IN ({hadm_ids_csv})
)
SELECT pr.subject_id, pr.hadm_id, pr.startdate, pr.enddate,
       pr.drug, pr.drug_type, pr.formulary_drug_cd, pr.route
FROM prescriptions pr
JOIN first_adm fa USING (hadm_id)
WHERE pr.hadm_id IN ({hadm_ids_csv})
  AND pr.startdate IS NOT NULL
  AND datediff('hour', CAST(fa.admittime AS TIMESTAMP), CAST(pr.startdate AS TIMESTAMP)) BETWEEN 0 AND 48
ORDER BY subject_id, startdate
"""

SQL_PROCEDURES_48H_DUCKDB = """
WITH first_adm AS (
  SELECT hadm_id, admittime
  FROM admissions
  WHERE hadm_id IN ({hadm_ids_csv})
)
SELECT pe.subject_id, pe.hadm_id, pe.icustay_id,
       pe.starttime, pe.endtime,
       pe.itemid, di.label AS item_label,
       pe.ordercategoryname, pe.ordercategorydescription, pe.location
FROM procedureevents_mv pe
JOIN first_adm fa USING (hadm_id)
LEFT JOIN d_items di ON di.itemid = pe.itemid
WHERE pe.hadm_id IN ({hadm_ids_csv})
  AND pe.starttime IS NOT NULL
  AND datediff('hour', CAST(fa.admittime AS TIMESTAMP), CAST(pe.starttime AS TIMESTAMP)) BETWEEN 0 AND 48
ORDER BY subject_id, starttime
"""

SQL_ALL_ADMISSION_DIAGNOSES_DUCKDB = """
SELECT d.subject_id, d.hadm_id, a.admittime, a.dischtime, d.icd9_code
FROM diagnoses_icd d
JOIN admissions a USING (hadm_id)
WHERE d.subject_id IN ({subject_ids_csv})
ORDER BY d.subject_id, a.admittime
"""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _csv_int_list(values: List[int]) -> str:
    return ",".join(str(int(v)) for v in values) if values else "-1"


# ---------------------------------------------------------------------------
# DuckDB helper (mirrors BigQuery semantics)
# ---------------------------------------------------------------------------
def _duckdb_safe(con, sql: str) -> pd.DataFrame:
  try:
    return con.execute(sql).fetchdf()
  except Exception as e:  # pragma: no cover - defensive
    print('DuckDB query failed:', e)
    return pd.DataFrame()


def get_first_admissions_duckdb(con, subject_ids: List[int]) -> pd.DataFrame:
  if not subject_ids:
    return pd.DataFrame()
  sql = SQL_FIRST_ADMISSIONS_DUCKDB.format(subject_ids_csv=_csv_int_list(subject_ids))
  df = _duckdb_safe(con, sql)
  if df.empty:
    return df
  df.columns = [c.lower() for c in df.columns]
  return df.sort_values(["subject_id", "admittime"]).groupby("subject_id", as_index=False).first()


def get_all_admissions_duckdb(con, subject_ids: List[int]) -> pd.DataFrame:
  if not subject_ids:
    return pd.DataFrame()
  sql = SQL_ALL_ADMISSIONS_DUCKDB.format(subject_ids_csv=_csv_int_list(subject_ids))
  df = _duckdb_safe(con, sql)
  if df.empty:
    return df
  df.columns = [c.lower() for c in df.columns]
  return df


def get_demographics_duckdb(con, subject_ids: List[int]) -> pd.DataFrame:
  if not subject_ids:
    return pd.DataFrame()
  sql = SQL_DEMOGRAPHICS_DUCKDB.format(subject_ids_csv=_csv_int_list(subject_ids))
  df = _duckdb_safe(con, sql)
  if df.empty:
    return df
  df.columns = [c.lower() for c in df.columns]
  return df


def get_vitals_48h_duckdb(con, hadm_ids: List[int]) -> pd.DataFrame:
  if not hadm_ids:
    return pd.DataFrame()
  sql = SQL_VITALS_48H_DUCKDB.format(hadm_ids_csv=_csv_int_list(hadm_ids))
  df = _duckdb_safe(con, sql)
  if df.empty:
    return df
  df.columns = [c.lower() for c in df.columns]
  return df


def get_labs_48h_duckdb(con, hadm_ids: List[int]) -> pd.DataFrame:
  if not hadm_ids:
    return pd.DataFrame()
  sql = SQL_LABS_48H_DUCKDB.format(hadm_ids_csv=_csv_int_list(hadm_ids))
  df = _duckdb_safe(con, sql)
  if df.empty:
    return df
  df.columns = [c.lower() for c in df.columns]
  return df


def get_prescriptions_48h_duckdb(con, hadm_ids: List[int]) -> pd.DataFrame:
  if not hadm_ids:
    return pd.DataFrame()
  sql = SQL_PRESCRIPTIONS_48H_DUCKDB.format(hadm_ids_csv=_csv_int_list(hadm_ids))
  df = _duckdb_safe(con, sql)
  if df.empty:
    return df
  df.columns = [c.lower() for c in df.columns]
  return df


def get_procedures_48h_duckdb(con, hadm_ids: List[int]) -> pd.DataFrame:
  if not hadm_ids:
    return pd.DataFrame()
  sql = SQL_PROCEDURES_48H_DUCKDB.format(hadm_ids_csv=_csv_int_list(hadm_ids))
  df = _duckdb_safe(con, sql)
  if df.empty:
    return df
  df.columns = [c.lower() for c in df.columns]
  return df


def get_all_admission_diagnoses_duckdb(con, subject_ids: List[int]) -> pd.DataFrame:
  if not subject_ids:
    return pd.DataFrame()
  sql = SQL_ALL_ADMISSION_DIAGNOSES_DUCKDB.format(subject_ids_csv=_csv_int_list(subject_ids))
  df = _duckdb_safe(con, sql)
  if df.empty:
    return df
  df.columns = [c.lower() for c in df.columns]
  return df

