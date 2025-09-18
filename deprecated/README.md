# Deprecated Scripts (Pending Removal)

These scripts are superseded by the unified notebook + core helper modules:

Planned for deprecation:
- train.py (legacy full CLI) – Keep temporarily for reference
- refit_xgb_best.py
- xgb_optuna_fast.py
- xgb_optuna_tune.py
- shap_explain.py (merged into explain.py)
- group_shap.py (merged into explain.py)
- feature_schema_diff.py (can be a notebook cell)
- metrics_diff.py (can be a notebook cell)

Rationale: Consolidated into `train_core.py`, `explain.py`, and final pipeline notebook.
Do not modify these further; migrate any needed logic first.

## Move Log (2025-09-13)
- refit_xgb_best.py -> deprecated (replaced by notebook final training section)
- xgb_optuna_fast.py -> deprecated (Optuna tuning now inline in notebook)
- xgb_optuna_tune.py -> deprecated (legacy CLI path removed)
- shap_explain.py -> deprecated (SHAP via explain.py / notebook cell)
- group_shap.py -> deprecated (group aggregation integrated)
- feature_schema_diff.py -> deprecated (quick diff can be a one-off cell)
- metrics_diff.py -> deprecated (metrics comparison trivial in notebook)

Retained temporarily: train.py (still a comprehensive CLI; can be slimmed or removed once notebook fully replaces external entrypoints.)
