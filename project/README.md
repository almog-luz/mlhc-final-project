# MLHC Final Project - Option 1 (ICU Prediction Models)

Contents
- mlhc_project_exploration.ipynb: lean EDA/orchestration; core logic lives in Python modules.
- mlhc_project_train.ipynb: lean training notebook that calls project.train.train_from_labels and prints metrics.
- mlhc_project_inference.ipynb: lean inference notebook that loads models and runs predictions; optional metrics.
- extract.py: BigQuery extraction utilities shared across training and inference.
- features.py: feature engineering utilities (0–48h aggregation, demographics, flags).
- unseen_data_evaluation.py: programmatic API run_pipeline_on_unseen_data(subject_ids, client) that extracts 0–48h features and returns probabilities for three targets.
- train.py: CLI to train calibrated models and save artifacts + metrics.
- evaluate_unseen.py: CLI to produce predictions and (optionally) compute evaluation metrics when labels are provided.
- metrics_utils.py: metrics and calibration utilities.
- data/: initial_cohort.csv, test_example.csv, extracted_cache/ for parquet caches.
- runs/<timestamp>/models/: created by training (preprocessor + 3 models + feature_columns.json + metrics.json) plus evaluation subdirectories.

Quick start
1) Environment
   - pip install -r ../requirements.txt
   - or conda env create -f environment.yml && conda activate mlhc_project

2) Notebooks (lean)
   - Training: open mlhc_project_train.ipynb, set GCP_PROJECT_ID and LABELS_CSV (subject_id + *_label columns), run all.
   - Inference: open mlhc_project_inference.ipynb, set GCP_PROJECT_ID and SUBJECTS_CSV (subject_id), run all; set LABELS_CSV to compute metrics.

3) Train via CLI (alternative)
   - python project/train.py --project-id YOUR_GCP_PROJECT --input path/to/labels.csv --output-dir runs/$(date +%Y%m%d_%H%M%S)/models  (on Windows create the directory first or substitute a timestamp manually)

4) Unseen evaluation (script)
   - Predictions only:
     - python project/evaluate_unseen.py --project-id YOUR_GCP_PROJECT --input project/data/test_example.csv --output predictions.csv
   - Predictions + metrics:
     - python project/evaluate_unseen.py --project-id YOUR_GCP_PROJECT --input project/data/test_example.csv --output predictions.csv --labels path/to/labels.csv --metrics-output metrics.json

5) Programmatic API
```python
from google.cloud import bigquery
from project.unseen_data_evaluation import run_pipeline_on_unseen_data
client = bigquery.Client(project="YOUR_GCP_PROJECT")
subject_ids = [10001, 10006]
probs_df = run_pipeline_on_unseen_data(subject_ids, client)
print(probs_df)
```

Notes
- Uses only Labevents for labs, per instructions.
- First 48h window with LOS >=54h, 6h gap respected by design.
- Models are calibrated (isotonic if enough positives, else Platt/sigmoid).

## Readmission Hyperparameter Tuning

Use `project/hparam_tune.py` to sweep logistic regression (and optional HGB) hyperparameters for the readmission task without manual intervention.

Example:

```bash
python -m project.hparam_tune \
   --labels project/runs/20250905_143527/labels.csv \
   --output-root project/runs/20250905_143527/tuning_readmission \
   --project-id ml-for-healthcare-2025 \
   --pos-weights 1 1.5 2 3 \
   --logreg-C 0.1 0.5 1 2 5 \
   --min-recall 0.15 0.20 \
   --use-cache-first
```

The script produces one subdirectory per configuration plus a consolidated CSV `sweep_summary.csv` (sorted by ROC AUC then PR AUC) in the output root.
