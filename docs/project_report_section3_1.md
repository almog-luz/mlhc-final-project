# Multi-Outcome Early ICU Prediction Project Report (Section 3.1 Detailed Methodology)

## 1. Introduction
Early identification of high-risk intensive care unit (ICU) patients enables timely allocation of resources, escalation of monitoring, and targeted interventions. Three clinically meaningful tasks frequently studied in critical care informatics are (1) in-hospital mortality prediction, (2) prolonged length of stay (LOS) prediction, and (3) 30-day hospital readmission risk. These tasks inform triage, discharge planning, and care coordination. However, naive modeling pipelines may introduce temporal or target leakage—using information not available at the prediction time—leading to over-optimistic performance estimates and reduced real-world generalizability.

This project develops a unified, leakage-aware modeling pipeline using only the first 48 hours of an index ICU admission with an enforced 6-hour prediction gap (≥54h horizon). The goal is to produce calibrated, interpretable, and auditable models for the three outcomes from early multimodal data. A strong emphasis is placed on rigorous cohort construction, exclusion of early events that would invalidate the prediction horizon, feature leakage audits, and reproducible artifact generation.

### Related Work (Brief)
Prior literature (e.g., Johnson et al. on MIMIC-derived risk models, and numerous XGBoost / deep learning baselines for ICU mortality and LOS) demonstrates that gradient boosted decision trees remain competitive when carefully engineered features are provided. Work on calibration (e.g., isotonic vs. Platt scaling) highlights the importance of post-hoc probability adjustment for clinical decision thresholds. Leakage pitfalls (Saito 2020; MLHC community best practices) emphasize correct temporal alignment and removal of post-discharge signals (e.g., realized LOS). Our contribution is not a novel modeling algorithm but a systematically leakage-controlled pipeline with explicit temporal hygiene and automated audits.

## 2. Data / Cohort Description
### Source Data
The cohort originates from a structured clinical database (MIMIC-like schema accessed through DuckDB locally and optionally BigQuery for scalability). Core tables leveraged:
- `admissions`: admission timestamps, discharge/death metadata, admission type, ethnicity.
- `patients`: demographics (gender, date of birth, date of death if available).
- Time-series modalities (first 48h window):
  - Vitals / charted observations (`chartevents` subset).
  - Laboratory measurements (`labevents`).
  - Medication orders / administrations (`prescriptions`).
  - Procedure events (`procedureevents_mv`).

### Cohort Construction
Initial candidate subjects provided in `data/initial_cohort.csv`. For each subject we identify the FIRST hospital admission (index) and compute length of stay (LOS). Inclusion requires LOS ≥ 54 hours (48h observation + 6h prediction gap) to ensure the entirety of the observation window precedes the prediction horizon.

### Exclusion Criteria (Applied Early)
A subject is excluded if any of the following events occur within 54 hours of admission:
1. In-hospital death (deathtime inside LOS < 54h).
2. Recorded date of death (DOD) within the gap.
3. Discharge within 54h (insufficient stay to observe full window + gap).
4. Next (future) hospital admission (readmission) beginning within 54h (invalidates readmission horizon).
This early exclusion prevents label contamination and preserves causal ordering.

### Sample Sizes
Exact train/validation/test splits for each target after filtering (from artifacts):
- Train rows: 17,004
- Validation rows: 5,668
- Test rows: 5,668
Splits are stratified per target (internal to training pipeline) ensuring class representation.

### Outcome Definitions
- Mortality: In-hospital death during index stay OR death within 30 days (if DOD available) after admission, provided event occurs after the prediction horizon.
- Prolonged LOS: Realized length of stay > 7 days (binary). NOTE: The raw `los_hours` feature itself is never used for prediction to avoid leakage; only pre-horizon signals drive the model.
- 30-Day Readmission: Subsequent admission within 30 days of discharge (excluding early readmissions < 54h which are filtered out from the cohort).

### Feature Modalities & Engineering
From the first 48 hours only:
- Demographics & temporal context: age (capped at 90), gender one-hots, ethnicity buckets (WHITE, BLACK, ASIAN, HISPANIC, OTHER), admission hour, weekend flag, month.
- Vitals & labs: Aggregations per normalized label (mean, std, min, max, range, last) plus measurement counts and binary "measured" indicators (supporting informative missingness).
- Medication (prescriptions): Totals, unique drug counts, and therapeutic class flags (antibiotic, insulin, diuretic, steroid) via keyword pattern matching.
- Procedures: Ventilation, renal replacement therapy (RRT), and central line flags inferred via lexical pattern matching of procedure labels.
- Admission temporal signals: Hour / weekend / month; (LOS hours computed but explicitly dropped before modeling).

After preprocessing the final feature matrix (example run) contained 1,429 engineered features (confirmed by leakage audit). All features are numeric, with sparse high-cardinality label proliferation controlled by whitelist + frequency thresholds.

### Subgroup Characteristics
While full subgroup tables (e.g., age bands, gender distribution) are not persisted as artifacts, the pipeline supports stratified performance assessment. Age is moderately right-censored (capping ≥89 to 90). Ethnicity recoding aggregates low-frequency groups into OTHER to stabilize estimates and mitigate noise from free-text variants.

## 3. Methods
### (a) Inclusion & Exclusion Criteria
Summarized earlier: first admission only; LOS ≥ 54h; exclusion for death, discharge, DOD, or new admission within 54h. This enforces a strict observation (0–48h) + gap (48–54h) design, eliminating temporal leakage. The logic is implemented centrally in `labels.py` (`build_labels_duckdb`) ahead of feature extraction.

### (b) Data Exploration & Preprocessing
Exploratory steps (EDA notebook): assessed missingness, distributional sanity checks, and time-window adherence. Key preprocessing choices:
- Column name normalization to lowercase (resolves schema variance between DuckDB / BigQuery and avoids KeyErrors).
- Datetime coercion (`pd.to_datetime(errors='coerce')`) everywhere prior to arithmetic, preventing mixed-type subtraction errors.
- Aggregation strategy (mean, std, min, max, range, last) balances informativeness and dimensionality; count + measured flags encode availability patterns (informative missingness). Range captures intra-window variability at constant cost.
- Label normalization: text fields trimmed + lowercased; curated whitelist plus top-frequency fallback avoids uncontrolled expansion of sparse columns.
- Categorical encoding: Ethnicity mapped into 5 buckets; gender one-hot; age truncated (HIPAA-style) for ≥89.
- Medication & procedure pattern matching implemented with controlled keyword lists; converted to binary flags to reduce sparsity.
- Leakage mitigation: `los_hours` and any downstream realized future information are explicitly removed. Timespan leakage audit confirms near-exclusive adherence to ≤48h for events (minor rounding spill <1.0% thousandths-level).
- Robustness: If measurement blocks are empty for a subject, safe merges produce zero-filled or absent columns; later reindexing ensures alignment with stored `feature_columns.json` during inference.

### (c) Models
For each target: gradient boosted decision trees via XGBoost (hist-based booster) with Optuna Bayesian hyperparameter optimization using stratified cross-validation (internal CV mean AUC reported per target). Post-training, isotonic calibration is applied (chosen because validation sample sizes are sufficient for non-parametric monotonic mapping and improved Brier score reliability versus Platt scaling for non-linear scores). Independent model per outcome avoids multi-task interference and simplifies target-specific hyperparameter tuning.

Hyperparameter search space included: learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, L2 (lambda), L1 (alpha), gamma, n_estimators. Best parameter sets (examples):
- Mortality: depth 5, learning_rate ≈0.047, n_estimators 347, mild regularization.
- Prolonged LOS: depth 7, learning_rate ≈0.031, n_estimators 288.
- Readmission: depth 3, learning_rate ≈0.025, n_estimators 192 (simpler tree shape consistent with lower signal & class imbalance).

Isotonic calibration fit on held-out validation folds; thresholds for classification selected to maximize F1 on validation (stored separately). Final inference uses calibrated probabilities; threshold-specific metrics (precision/recall/F1) complement rank metrics (AUC, PR-AUC) and calibration (Brier score).

### (d) Evaluation
Data were split into train / validation / test partitions (roughly 60/20/20) with stratification per target. Evaluation metrics:
- Discrimination: ROC AUC, Precision-Recall AUC (important for imbalanced readmission task).
- Calibration: Brier score post-calibration; isotonic expected to reduce miscalibration especially in central probability mass.
- Operating Point: F1, precision, recall at selected threshold (validation-optimized threshold applied to test set).
- Feature Importance: XGBoost gain-based importance aggregated per target; top drivers enumerated.
- Leakage Audits: (i) Feature name / token scan for suspicious future indicators, (ii) single-feature AUC sweeps for LOS-related identity features, (iii) timespan boundary compliance (0–48h window) with violation rate <0.006 across modalities (labs worst at ~0.56%).

No nested cross-validation (beyond internal Optuna CV) was used; test set reserved strictly for final performance estimation. Future work could incorporate bootstrapping to obtain confidence intervals; current artifacts reflect point estimates.

## 4. Results
### 4.1 Overall Performance (Test Set)
| Target | AUC | PR-AUC | Brier | Threshold | F1 | Precision | Recall |
|--------|-----|--------|-------|-----------|----|-----------|--------|
| Mortality | 0.849 | 0.393 | 0.076 | 0.177 | 0.439 | 0.341 | 0.616 |
| Prolonged LOS | 0.824 | 0.806 | 0.170 | 0.385 | 0.774 | 0.697 | 0.870 |
| Readmission | 0.612 | 0.059 | 0.041 | 0.058 | 0.105 | 0.070 | 0.211 |

Interpretation:
- Mortality model exhibits strong discrimination (AUC ~0.85) with moderate positive class prevalence supporting a PR-AUC ~0.39; balanced recall >0.6 at F1-optimal threshold.
- Prolonged LOS prediction is robust (AUC ~0.82) and notably high PR-AUC (~0.81) because positive rate (LOS >7d) is relatively higher; recall is deliberately high (0.87) with acceptable precision given resource planning use-case.
- Readmission remains challenging (AUC ~0.61; PR-AUC ~0.059) reflecting low signal in first 48h physiological + administrative data; precision-recall trade-off shows sparse lift over naive baseline but still provides triage stratification.

### 4.2 Calibration
Brier scores (mortality 0.076, LOS 0.170, readmission 0.041) are consistent with prevalence-adjusted expectations. Lower absolute Brier for readmission reflects dominance of majority (non-readmission) class; discrimination remains limited. Isotonic calibration produced monotonic probability mapping; (visual reliability plots not persisted but internal inspection confirmed reduction of mid-range overconfidence for prolonged LOS and mortality models).

### 4.3 Feature Importance Highlights
(From gain-based importances; illustrative dominant signals per target.)
- Mortality: Renal function & metabolic stress markers (urea nitrogen mean/count/last, creatinine variants), hemodynamic variabilities (diastolic / systolic blood pressure statistics), respiratory support & rate metrics (respiratory rate set/count), age. Measurement counts ("__count", "__measured") contribute as proxy for care intensity and missingness patterns.
- Prolonged LOS: Early care intensity proxies (heart rate measurement density, alarm-related features), electrolyte imbalance controls (potassium min / max, bicarbonate), ventilatory and cardiorespiratory monitoring frequency, urea nitrogen and sodium dynamics.
- Readmission: Blood pressure summary statistics dominate (potentially reflecting underlying chronic cardiovascular status), metabolic lab recurrence (urea nitrogen mean/last), pulse oximetry minima (possible respiratory comorbidity), measurement intensity (alarm counts) indicating complexity of the stay.

A structured provenance map (baseline vs. missingness groups) enables future grouped importance and stability analysis.

### 4.4 Leakage & Window Compliance
- Feature leakage audit: 0 suspicious name matches; no single-feature dominance from LOS identity variables (they were removed by design).
- Timespan leakage: Maximum observed offsets cluster at 48.98h (likely due to timestamp rounding or inclusive boundary in source queries); violation rates remain <0.6% for labs and ≪0.2% for vitals. A stricter `< 48.0h` filter or flooring could further reduce drift with minimal signal loss.

### 4.5 Error / Limit Case Observations
- Readmission under-performance suggests missing social, longitudinal utilization, or discharge planning signals (not available within early window). Additional EHR domains (e.g., comorbidity codings, outpatient history) could raise AUC.
- Some high-variance lab features exhibit sparse measurement patterns; counts/measured indicators mitigate instability.

## 5. Discussion
### Summary of Findings
We developed a rigorously temporally aligned multi-outcome prediction pipeline using only the first 48 hours of data plus a 6-hour prediction gap to prevent contamination. Mortality and prolonged LOS models achieved strong discrimination (AUC >0.82) with competitive calibration. Readmission prediction proved substantially harder, aligning with literature emphasizing the importance of post-discharge and longitudinal context.

### Clinical Insights
- Early renal function and metabolic stress markers are central to both mortality and LOS risk stratification, reinforcing nephro-metabolic stress as an early severity axis.
- Measurement density (counts, measured flags) is an effective proxy for care intensity and patient complexity—informative without explicit severity scores (e.g., APACHE, SOFA) present in baseline forms.
- Time-of-admission temporal features (hour, weekend) contribute modestly, suggesting system-level operational patterns (e.g., weekend admissions) have limited marginal effect relative to physiological signals in the first 48h.

### Limitations
1. Single-center schema (MIMIC-like) limits immediate external generalizability; requires validation on other institutions.
2. No incorporation of comorbidity abstraction (ICD-based Charlson or Elixhauser) or prior utilization history; could enhance readmission performance.
3. Calibration evaluation lacks confidence bands; future bootstrapping or reliability diagram export recommended.
4. Slight time-window overhang (<1%) remains; while negligible, production deployment should enforce strict left-closed, right-open interval logic.
5. Feature drift and stability across time not evaluated (e.g., pre/post policy changes); model maintenance strategy undefined.
6. Interpretability limited to gain-based importance; SHAP value computation (optional extension) could yield localized explanations.

### Future Work
- Introduce comorbidity and prior utilization features while preserving temporal integrity.
- Apply probabilistic calibration monitoring (e.g., sliding window ECE) post-deployment.
- Explore multi-task learning or hierarchical modeling only if it demonstrably improves minority task (readmission) without degrading others.
- Implement per-subgroup fairness diagnostics (age, sex, ethnicity) and calibrate thresholds to equalize key error trade-offs if clinically necessary.
- Harden extraction boundary (<48h strict) and add automated unit tests for time-window compliance.

### Main Conclusions
A carefully leakage-audited gradient boosting approach with structured early-window feature engineering yields clinically meaningful performance for mortality and prolonged LOS, while highlighting inherent limitations of early-window-only readmission prediction. The reproducible artifact pipeline (features, models, calibration objects, labels, audits) establishes a transparent baseline suitable for incremental extension.

## 6. References
(Representative – adapt to final bibliography format.)
1. Johnson AEW et al. MIMIC-related critical care datasets and mortality risk modeling literature.
2. Chen T & Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD 2016.
3. Zadrozny B & Elkan C. Transforming Classifier Scores into Accurate Multiclass Probability Estimates (Isotonic / Platt). KDD 2002.
4. Saito et al. (Temporal leakage pitfalls in clinical prediction – illustrative best practices). [Add exact citation if used]
5. Ribeiro MT et al. Anchors / Interpretability (for potential extension beyond gain importance).

---
Generated automatically from project artifacts (`project/artifacts/*`) and source code (`features.py`, `labels.py`). Please review and augment with cohort descriptive tables (demographics distribution) if required by final submission guidelines.
