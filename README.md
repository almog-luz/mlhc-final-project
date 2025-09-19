## ICU Prediction Models (Mortality, Prolonged LOS, Readmission)

This repository contains a modular pipeline to extract early (first 48h) EHR data, engineer features, train calibrated classification models, and audit model fairness/performance.

### Quick Start
1. Environment
   - `pip install -r requirements.txt`  (or create from `project/environment.yml`)
2. Prepare a labels CSV with columns: `subject_id,mortality_label,prolonged_los_label,readmission_label`.
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
- `mlhc_training_pipeline.ipynb` – End‑to‑end training: extraction (DuckDB), feature build, split, model fit, calibration, metrics, artifact persistence (+ split subject ID manifests).
- `mlhc_fairness_and_bias_analysis.ipynb` – Test‑split fairness recomputation (AUC / PR AUC / F1 deltas), demographic distributions, disparity plots, prevalence vs AUC scatter, figure export.
- `mlhc_data_extraction_duckdb.ipynb` – Focused exploration of DuckDB extraction logic and schema sanity checks.
- `mlhc_local_db_tester_notebook.ipynb` – Lightweight debugging of local DuckDB queries & joins.

### Core Python Modules (`project/`)
- `extract.py` – Source‑of‑truth extraction helpers (DuckDB) for demographics, labs (48h), vitals, prescriptions, procedures, first admissions.
- `labels.py` – Label construction (mortality, prolonged LOS, readmission) aligned to first admissions.
- `features.py` – Feature engineering (aggregation, pruning with variance/support heuristics, provenance capture).
- `pipelines.py` – Orchestrated training & inference pipeline steps (extraction → modalities → features → variance safeguards).
- `inference.py` – Scoring utilities: artifact resolution, feature alignment, optional isotonic calibration usage, variance guards.
- `unseen_data_evaluation.py` – Single entry function `run_pipeline_on_unseen_data(subject_ids, con)` returning probability columns for the three targets.
- `metrics_utils.py` – Performance metric helpers (ROC, PR, Brier, threshold selection, calibration helpers).
- `train.py` – CLI training script: builds features, splits, trains base + calibrated models, writes metrics & thresholds.
- `evaluate_unseen.py` – CLI inference & optional evaluation given a CSV of `subject_id` (and labels if available).
  

### Artifacts & Data Layout
- `project/artifacts/` – Canonical promoted model set: `model_*.joblib`, `isotonic_*.joblib`, `preprocessor.joblib`, `feature_columns.json`, `threshold_<target>.txt`, `metrics_<target>.json`, `metrics_all.json`, `fairness_group_metrics_test.csv`, split manifest (`data_split_manifest.json`), and subject ID text files (`train_subject_ids.txt`, etc.).
- `project/data/` – Input cohort & caches: `initial_cohort.csv`, (canonical) `labels.csv`, example `test_example.csv`, and `extracted_cache/*.parquet` generated during feature build.
  

### Fairness Outputs
- Group metrics CSV columns: `target,group_category,group_value,n,prevalence,auc,pr_auc,f1,delta_auc,delta_f1,...` restricted to the persisted test split.
- Plots emitted under `project/artifacts/fairness_plots/` (demographic distributions, disparity panels, prevalence vs AUC).
- Split reproducibility: `data_split_manifest.json` + per‑split subject ID text files enable deterministic test reuse.

### Thresholding & Calibration
- Thresholds selected (validation F1) and persisted per target (`threshold_<target>.txt`).
- Calibration: isotonic models saved if sufficient positives, falling back to raw model outputs if not available.

### Key Design Choices
- Early 48h window with LOS ≥54h to ensure complete capture of first two days signals.
- Variance/support pruning with environment overrides (`MLHC_MIN_SUPPORT`, `MLHC_TARGET_MAX`).
- Strict alignment & variance guards before persisting or scoring to catch silent schema drift.
- Fairness evaluation restricted to true test partition (never training spillover) for unbiased disparity estimates.

### Command Line Examples
Train (artifacts into a timestamped run directory):
```bash
python project/train.py --input project/data/labels.csv --output-dir project/runs/20250919_120000/artifacts
```
Inference + metrics (if labels known):
```bash
python project/evaluate_unseen.py --input project/data/test_example.csv --labels project/data/labels.csv --output predictions.csv --metrics-output metrics.json
```

### Maintenance Notes
- Promote / legacy run management scripts and exploratory legacy notebooks were removed to keep the repository lean.
- Add new engineered features via `features.py`; update provenance mapping to keep downstream reproducibility.
- Regenerate fairness metrics after retraining or recalibration to monitor drift in disparities.

---
Concise module + artifact descriptions above are intended to make navigation and evaluation straightforward without reading every source file first.
