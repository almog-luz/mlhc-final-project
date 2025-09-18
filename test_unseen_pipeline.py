import os
import pandas as pd
import duckdb
from project import run_pipeline_on_unseen_data
from datetime import datetime, timedelta

# Configure artifacts directory (adjust if different on your system)
ART_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
os.environ['MLHC_MODELS_DIR'] = ART_DIR

# Minimal synthetic tables
now = datetime(2020,1,1,8,0,0)
admissions = pd.DataFrame({
  'subject_id':[101,102,103],
  'hadm_id':[1101,1102,1103],
  'admittime':[now, now, now],
  'dischtime':[now+timedelta(hours=60), now+timedelta(hours=70), now+timedelta(hours=80)],
  'deathtime':[None,None,None],
  'admission_type':['EMERGENCY']*3,
  'admission_location':['EMERGENCY ROOM']*3,
  'discharge_location':['HOME']*3,
  'diagnosis':['TEST']*3,
  'insurance':['Medicare']*3,
  'language':['EN']*3,
  'marital_status':['S']*3,
  'ethnicity':['WHITE']*3
})
patients = pd.DataFrame({
  'subject_id':[101,102,103],
  'gender':['M','F','M'],
  'dob':[datetime(1950,1,1)]*3,
  'dod':[None,None,None],
  'expire_flag':[0,0,0]
})
# Empty event tables (models still produce output using demographic + admission features)
empty_cols = {
 'chartevents':['subject_id','hadm_id','charttime','itemid','valuenum','valueuom'],
 'labevents':['subject_id','hadm_id','charttime','itemid','valuenum','value','valueuom','flag'],
 'prescriptions':['subject_id','hadm_id','startdate','enddate','drug','drug_type','formulary_drug_cd','route'],
 'procedureevents_mv':['subject_id','hadm_id','starttime','endtime','itemid','ordercategoryname','ordercategorydescription','location']
}
con = duckdb.connect(':memory:')
con.register('admissions', admissions)
con.register('patients', patients)
for tbl, cols in empty_cols.items():
    con.register(tbl, pd.DataFrame(columns=cols))

# Run pipeline
subject_ids = [101,102,103]
result = run_pipeline_on_unseen_data(subject_ids, con)
print(result)
assert set(result.columns) == {'subject_id','mortality_proba','prolonged_LOS_proba','readmission_proba'}
assert len(result) == 3
print('Synthetic unseen pipeline test passed.')
