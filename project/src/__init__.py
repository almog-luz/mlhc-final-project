# src package initializer
# Provides backward-compatible imports if code elsewhere still uses 'project.<module>'

from importlib import import_module as _im
__all__ = []

_modules = [
    'audit_features','comorbidity_features','eda_quick_report','evaluate_unseen','evaluation',
    'explain','extract','features','hparam_tune','labels','load_cohort','medication_features',
    'metrics_utils','phenotype_clustering','precision_recall_table','prepare_splits','train',
    'train_core','unseen_data_evaluation','utilization_features','variability_features'
]
for m in _modules:
    try:
        _im(f'project.src.{m}')
        __all__.append(m)
    except Exception:
        pass
