# BigQuery extraction utilities shared across training and inference
from __future__ import annotations
from typing import List, Optional
import pandas as pd
from google.cloud import bigquery as bq


def safe_query(client: bq.Client, sql: str, job_config: Optional[bq.QueryJobConfig] = None) -> pd.DataFrame:
  """Execute a BigQuery SQL string and return a DataFrame.

  Returns an empty DataFrame on failure (and prints a warning) so callers can
  choose to short‑circuit gracefully without additional try/except noise.
  """
  try:
    return client.query(sql, job_config=job_config).to_dataframe()  # type: ignore[no-untyped-call]
  except Exception as e:  # pragma: no cover - defensive
    print(f"BigQuery query failed: {e}")
    return pd.DataFrame()


def get_first_admissions(client: bq.Client, subject_ids: List[int]) -> pd.DataFrame:
    sql = """
    SELECT subject_id, hadm_id, admittime, dischtime, deathtime, admission_type,
           admission_location, discharge_location, diagnosis, insurance, language,
           marital_status, ethnicity
    FROM `physionet-data.mimiciii_clinical.admissions`
    WHERE subject_id IN UNNEST(@subject_ids)
    ORDER BY subject_id, admittime
    """
    cfg = bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("subject_ids", "INT64", subject_ids)])
    df = safe_query(client, sql, cfg)
    if df.empty:
        return df
    first = (
        df.sort_values(["subject_id", "admittime"]).groupby("subject_id", as_index=False).first()
    )
    return first


def get_all_admissions(client: bq.Client, subject_ids: List[int]) -> pd.DataFrame:
    sql = """
    SELECT subject_id, hadm_id, admittime, dischtime, deathtime
    FROM `physionet-data.mimiciii_clinical.admissions`
    WHERE subject_id IN UNNEST(@subject_ids)
    ORDER BY subject_id, admittime
    """
    cfg = bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("subject_ids", "INT64", subject_ids)])
    return safe_query(client, sql, cfg)


def get_demographics(client: bq.Client, subject_ids: List[int]) -> pd.DataFrame:
    sql = """
    SELECT subject_id, gender, dob, dod, expire_flag
    FROM `physionet-data.mimiciii_clinical.patients`
    WHERE subject_id IN UNNEST(@subject_ids)
    """
    cfg = bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("subject_ids", "INT64", subject_ids)])
    return safe_query(client, sql, cfg)


def get_vitals_48h(client: bq.Client, hadm_ids: List[int]) -> pd.DataFrame:
    if not hadm_ids:
        return pd.DataFrame()
    sql = """
    WITH first_adm AS (
      SELECT hadm_id, admittime
      FROM `physionet-data.mimiciii_clinical.admissions`
      WHERE hadm_id IN UNNEST(@hadm_ids)
    )
    SELECT ce.subject_id, ce.hadm_id, ce.icustay_id, ce.charttime,
           di.label AS item_label, ce.valuenum, ce.valueuom
    FROM `physionet-data.mimiciii_clinical.chartevents` ce
    JOIN first_adm fa USING (hadm_id)
    JOIN `physionet-data.mimiciii_clinical.d_items` di ON di.itemid = ce.itemid
    WHERE ce.hadm_id IN UNNEST(@hadm_ids)
      AND ce.valuenum IS NOT NULL
      AND TIMESTAMP_DIFF(ce.charttime, fa.admittime, HOUR) BETWEEN 0 AND 48
      AND (
        REGEXP_CONTAINS(LOWER(di.label), r"heart rate|hr") OR
        REGEXP_CONTAINS(LOWER(di.label), r"respiratory rate|rr") OR
        REGEXP_CONTAINS(LOWER(di.label), r"temperature") OR
        REGEXP_CONTAINS(LOWER(di.label), r"(non?invasive )?systolic|sysbp|sbp") OR
        REGEXP_CONTAINS(LOWER(di.label), r"(non?invasive )?diastolic|diasbp|dbp") OR
        REGEXP_CONTAINS(LOWER(di.label), r"mean arterial|map") OR
        REGEXP_CONTAINS(LOWER(di.label), r"spo2|o2 saturation|oxygen saturation")
      )
    ORDER BY subject_id, charttime
    """
    cfg = bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)])
    return safe_query(client, sql, cfg)


def get_labs_48h(client: bq.Client, hadm_ids: List[int]) -> pd.DataFrame:
    if not hadm_ids:
        return pd.DataFrame()
    sql = """
    WITH first_adm AS (
      SELECT hadm_id, admittime
      FROM `physionet-data.mimiciii_clinical.admissions`
      WHERE hadm_id IN UNNEST(@hadm_ids)
    )
    SELECT le.subject_id, le.hadm_id, le.charttime,
           dl.label AS item_label, le.valuenum, le.value AS value_text,
           le.valueuom, le.flag
    FROM `physionet-data.mimiciii_clinical.labevents` le
    JOIN first_adm fa USING (hadm_id)
    JOIN `physionet-data.mimiciii_clinical.d_labitems` dl ON dl.itemid = le.itemid
    WHERE le.hadm_id IN UNNEST(@hadm_ids)
      AND le.charttime IS NOT NULL
      AND TIMESTAMP_DIFF(le.charttime, fa.admittime, HOUR) BETWEEN 0 AND 48
      AND (
        REGEXP_CONTAINS(LOWER(dl.label), r"wbc|white blood") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"hemoglobin|hgb") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"hematocrit|hct") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"platelet") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"sodium|na\\b") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"potassium|k\\b") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"chloride|cl\\b") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"bicarbonate|hco3") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"bun|urea") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"creatinine") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"glucose") OR
        REGEXP_CONTAINS(LOWER(dl.label), r"lactate")
      )
    ORDER BY subject_id, charttime
    """
    cfg = bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)])
    return safe_query(client, sql, cfg)


def get_prescriptions_48h(client: bq.Client, hadm_ids: List[int]) -> pd.DataFrame:
    if not hadm_ids:
        return pd.DataFrame()
    sql = """
    WITH first_adm AS (
      SELECT hadm_id, admittime
      FROM `physionet-data.mimiciii_clinical.admissions`
      WHERE hadm_id IN UNNEST(@hadm_ids)
    )
    SELECT pr.subject_id, pr.hadm_id, pr.startdate, pr.enddate,
           pr.drug, pr.drug_type, pr.formulary_drug_cd, pr.route
    FROM `physionet-data.mimiciii_clinical.prescriptions` pr
    JOIN first_adm fa USING (hadm_id)
    WHERE pr.hadm_id IN UNNEST(@hadm_ids)
      AND pr.startdate IS NOT NULL
      AND TIMESTAMP_DIFF(pr.startdate, fa.admittime, HOUR) BETWEEN 0 AND 48
    ORDER BY subject_id, startdate
    """
    cfg = bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)])
    return safe_query(client, sql, cfg)


def get_procedures_48h(client: bq.Client, hadm_ids: List[int]) -> pd.DataFrame:
    if not hadm_ids:
        return pd.DataFrame()
    sql = """
    WITH first_adm AS (
      SELECT hadm_id, admittime
      FROM `physionet-data.mimiciii_clinical.admissions`
      WHERE hadm_id IN UNNEST(@hadm_ids)
    )
    SELECT pe.subject_id, pe.hadm_id, pe.icustay_id,
           pe.starttime, pe.endtime,
           pe.itemid, di.label AS item_label,
           pe.ordercategoryname, pe.ordercategorydescription, pe.location
    FROM `physionet-data.mimiciii_clinical.procedureevents_mv` pe
    JOIN first_adm fa USING (hadm_id)
    LEFT JOIN `physionet-data.mimiciii_clinical.d_items` di ON di.itemid = pe.itemid
    WHERE pe.hadm_id IN UNNEST(@hadm_ids)
      AND pe.starttime IS NOT NULL
      AND TIMESTAMP_DIFF(pe.starttime, fa.admittime, HOUR) BETWEEN 0 AND 48
    ORDER BY subject_id, starttime
    """
    cfg = bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)])
    return safe_query(client, sql, cfg)

