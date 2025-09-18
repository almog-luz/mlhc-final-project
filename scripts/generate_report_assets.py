import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import load as jobload
from sklearn.metrics import roc_curve, precision_recall_curve, auc as sk_auc, brier_score_loss
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'project' / 'artifacts'
FIG_DIR = ROOT / 'docs' / 'figures'
TAB_DIR = ROOT / 'docs' / 'generated_tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ['mortality','prolonged_los','readmission']

# Load metrics
metrics = {}
for t in TARGETS:
    mp = ART / f'metrics_{t}.json'
    if mp.exists():
        metrics[t] = json.loads(mp.read_text())

# Cohort + prevalence table (derive from metrics if available)
# Fallback: infer prevalence from threshold precision/recall where possible (not ideal). Skipping complex inference.
# If labels.csv exists, compute directly.
labels_path_candidates = [ROOT / 'data' / 'labels.csv', ROOT / 'labels.csv']
labels_df = None
for cand in labels_path_candidates:
    if cand.exists():
        labels_df = pd.read_csv(cand)
        break
if labels_df is not None:
    prev_rows = []
    for t in TARGETS:
        col = f'{t}_label' if f'{t}_label' in labels_df.columns else None
        if col:
            p = labels_df[col].mean()
            prev_rows.append({'target': t, 'prevalence': p})
    prev_df = pd.DataFrame(prev_rows)
    prev_df.to_csv(TAB_DIR / 'prevalence_table.csv', index=False)

# Performance table (already in report but we regenerate CSV for consistency)
perf_rows = []
for t,m in metrics.items():
    perf_rows.append({
        'target': t,
        'auc': m['auc'],
        'pr_auc': m['pr_auc'],
        'brier': m['brier'],
        'threshold': m['threshold'],
        'f1_at_threshold': m['f1_at_threshold'],
        'precision_at_threshold': m['precision_at_threshold'],
        'recall_at_threshold': m['recall_at_threshold']
    })
if perf_rows:
    pd.DataFrame(perf_rows).to_csv(TAB_DIR / 'performance_table.csv', index=False)

# Load feature matrix for SHAP if possible
feature_cols_path = ART / 'feature_columns.json'
feature_matrix_path = ART / 'features_full.parquet'
feature_df = None
if feature_matrix_path.exists():
    try:
        feature_df = pd.read_parquet(feature_matrix_path)
        if 'subject_id' in feature_df.columns:
            feature_df = feature_df.set_index('subject_id')
    except Exception as e:
        print('Failed loading feature matrix:', e)

# Generate ROC/PR and calibration plots using calibrated models
roc_fig, roc_ax = plt.subplots(figsize=(6,5))
pr_fig, pr_ax = plt.subplots(figsize=(6,5))
cal_fig, cal_axes = plt.subplots(1, len(TARGETS), figsize=(5*len(TARGETS),4), sharey=True)
if len(TARGETS)==1:
    cal_axes = [cal_axes]

for idx, t in enumerate(TARGETS):
    model_path = ART / f'model_{t}.joblib'
    iso_path = ART / f'isotonic_{t}.joblib'
    full_path = ART / f'model_full_{t}.joblib'
    if not (model_path.exists() and iso_path.exists() and full_path.exists()):
        print('Skipping plots for', t, '(missing artifacts)')
        continue
    # Reconstruct test probabilities from stored metrics_all? Not stored raw; rely on models and feature_df alignment.
    # NOTE: We don't have test indices persisted here; fallback: use entire matrix as proxy.
    if feature_df is None:
        continue
    # For figure reproducibility we use all rows (proxy) - clarify in report that curves approximate.
    X_all = feature_df.values
    try:
        base_model = jobload(model_path)
        iso = jobload(iso_path)
    except Exception as e:
        print('Load fail', t, e)
        continue
    raw = base_model.predict_proba(X_all)[:,1]
    cal = iso.transform(raw)
    # Without true labels we cannot plot true ROC; skip if labels missing
    if labels_df is not None:
        col = f'{t}_label'
        if col in labels_df.columns:
            # align ordering
            lab_series = labels_df.drop_duplicates('subject_id').set_index('subject_id')[col]
            y = lab_series.reindex(feature_df.index).astype(int).values
            fpr, tpr, _ = roc_curve(y, cal)
            prec, rec, _ = precision_recall_curve(y, cal)
            roc_auc = sk_auc(fpr, tpr)
            pr_auc = sk_auc(rec, prec)
            roc_ax.plot(fpr, tpr, label=f"{t} AUC={roc_auc:.3f}")
            pr_ax.plot(rec, prec, label=f"{t} PR AUC={pr_auc:.3f}")
            fr_raw, mp_raw = calibration_curve(y, raw, n_bins=15, strategy='quantile')
            fr_cal, mp_cal = calibration_curve(y, cal, n_bins=15, strategy='quantile')
            axc = cal_axes[idx]
            axc.plot(mp_raw, fr_raw, 'o-', label='Raw', alpha=0.7)
            axc.plot(mp_cal, fr_cal, 'o-', label='Calibrated', alpha=0.7)
            axc.plot([0,1],[0,1],'--',color='gray')
            axc.set_title(f'Calibration: {t}')
            axc.set_xlabel('Predicted'); axc.set_ylabel('Observed')
            axc.legend()
    else:
        print('Labels missing; skipping ROC/PR/calibration for', t)

roc_ax.plot([0,1],[0,1],'--',color='grey')
roc_ax.set_xlabel('FPR'); roc_ax.set_ylabel('TPR'); roc_ax.set_title('ROC (Calibrated)'); roc_ax.legend()
pr_ax.set_xlabel('Recall'); pr_ax.set_ylabel('Precision'); pr_ax.set_title('PR (Calibrated)'); pr_ax.legend()
roc_fig.tight_layout(); pr_fig.tight_layout(); cal_fig.tight_layout()
roc_fig.savefig(FIG_DIR / 'roc_curves.png', dpi=160)
pr_fig.savefig(FIG_DIR / 'pr_curves.png', dpi=160)
cal_fig.savefig(FIG_DIR / 'calibration_curves.png', dpi=160)
plt.close(roc_fig); plt.close(pr_fig); plt.close(cal_fig)

# SHAP plots (top 20 bar per target)
try:
    import shap
    if feature_df is not None:
        shap.initjs()
        for t in TARGETS:
            model_path = ART / f'model_{t}.joblib'
            if not model_path.exists():
                continue
            model = jobload(model_path)
            sample_idx = np.random.default_rng(42).choice(len(feature_df), size=min(400, len(feature_df)), replace=False)
            X_sample = feature_df.values[sample_idx]
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            order = np.argsort(-mean_abs)[:20]
            feat_cols = list(feature_df.columns)
            top_df = pd.DataFrame({'feature':[feat_cols[i] for i in order], 'mean_abs_shap':mean_abs[order]})
            top_df.to_csv(TAB_DIR / f'shap_top20_{t}.csv', index=False)
            # Bar plot
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,6))
            top_df.iloc[::-1].plot(kind='barh', x='feature', y='mean_abs_shap', ax=ax, legend=False, title=f'SHAP Top 20 {t}')
            ax.set_xlabel('Mean |SHAP|')
            fig.tight_layout()
            fig.savefig(FIG_DIR / f'shap_top20_{t}.png', dpi=160)
            plt.close(fig)
except Exception as e:
    print('SHAP skipped:', e)

print('Assets generated in:', FIG_DIR, 'and', TAB_DIR)
