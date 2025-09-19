## ICU Prediction Models (Mortality, Prolonged LOS, Readmission)

This repository contains a modular pipeline to extract early (first 48h) EHR data, engineer features, train calibrated classification models, and audit model fairness/performance.

### Quick Start
1. Environment
   - `project/environment.yml`
2. Prepare data: `notebooks/mlhc_data_extraction_duckdb.ipynb`, run all cells
3. Train (notebook): open `notebooks/mlhc_training_pipeline.ipynb`, run all cells.
4. Fairness & bias: open `notebooks/mlhc_fairness_and_bias_analysis.ipynb` (after training artifacts exist) and run to recompute + visualize disparities.
5. Inference (programmatic example):
```python
from duckdb import connect
from project.unseen_data_evaluation import run_pipeline_on_unseen_data
con = connect('mlhc.db')
scores = run_pipeline_on_unseen_data([10001,10006], con)
print(scores.head())
```

### Notebooks
- `mlhc_data_extraction_duckdb.ipynb` – DuckDB extraction logic and schema sanity checks.
- `mlhc_training_pipeline.ipynb` – End‑to‑end training: extraction (DuckDB), feature build, split, model fit, calibration, metrics, artifact persistence (+ split subject ID manifests).
- `mlhc_fairness_and_bias_analysis.ipynb` – Test‑split fairness recomputation (AUC / PR AUC / F1 deltas), demographic distributions, disparity plots, prevalence vs AUC scatter, figure export.
- `mlhc_local_db_tester_notebook.ipynb` – Lightweight debugging of local DuckDB queries & joins.

### Core Python Modules (`project/`)
- `extract.py` – Source‑of‑truth extraction helpers (DuckDB) for demographics, labs (48h), vitals, prescriptions, procedures, first admissions.
- `labels.py` – Label construction (mortality, prolonged LOS, readmission) aligned to first admissions.
- `features.py` – Feature engineering (aggregation, pruning with variance/support heuristics, provenance capture).
- `pipelines.py` – Orchestrated training & inference pipeline steps (extraction → modalities → features → variance safeguards).
- `inference.py` – Scoring utilities: artifact resolution, feature alignment, optional isotonic calibration usage, variance guards.
- `unseen_data_evaluation.py` – Single entry function `run_pipeline_on_unseen_data(subject_ids, con)` returning probability columns for the three targets.
- `train.py` – CLI training script: builds features, splits, trains base + calibrated models, writes metrics & thresholds.
  

### Artifacts & Data Layout
- `project/artifacts/` – Canonical promoted model set: `model_*.joblib`, `isotonic_*.joblib`, `preprocessor.joblib`, `feature_columns.json`, `threshold_<target>.txt`, `metrics_<target>.json`, `metrics_all.json`, `fairness_group_metrics_test.csv`, split manifest (`data_split_manifest.json`), and subject ID text files (`train_subject_ids.txt`, etc.).
- `project/data/` – Input cohort & caches: `initial_cohort.csv`, (canonical) `labels.csv`, example `test_example.csv`, and `extracted_cache/*.parquet` generated during feature build.
  

### Fairness Outputs
- Group metrics CSV columns: `target,group_category,group_value,n,prevalence,auc,pr_auc,f1,delta_auc,delta_f1,...` restricted to the persisted test split.
- Plots emitted under `project/artifacts/fairness_plots/` (demographic distributions, disparity panels, prevalence vs AUC).
- Split reproducibility: `data_split_manifest.json` + per‑split subject ID text files enable deterministic test reuse.
