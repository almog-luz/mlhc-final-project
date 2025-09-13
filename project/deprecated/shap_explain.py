"""Generate SHAP explanations for the readmission model.

Usage (PowerShell):
  python -m project.shap_explain --model-dir project/models --target readmission \
      --max-samples 2000 --output-dir project/models/shap_readmission

Assumptions:
  - model_<target>.joblib exists in model-dir (e.g. model_readmission.joblib)
  - feature_columns.json + preprocessor.joblib in model-dir
  - features_full.parquet available (or pass --features)

Outputs:
  - shap_values.npz: compressed array (n_samples x n_features)
  - shap_summary.csv: per-feature mean |SHAP| & other stats
  - shap_top20.txt: textual top-20 importance list
  - optional beeswarm & bar plots if matplotlib present
"""
from __future__ import annotations
import argparse, os, json, sys, math
import numpy as np
import pandas as pd
import joblib

try:
    import shap  # type: ignore
except Exception as e:  # pragma: no cover
    print("ERROR: shap is not installed. Add it to requirements and pip install.")
    raise


def _load_features(path: str, feature_cols):
    df = pd.read_parquet(path)
    if 'subject_id' in df.columns:
        df = df.set_index('subject_id')
    # Reindex columns to expected order, fill missing with 0
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} expected feature columns missing in provided features parquet; filling with 0.")
        for m in missing:
            df[m] = 0.0
    return df[feature_cols]


def compute_shap(model, preprocessor, X: pd.DataFrame, model_type: str, background_size: int = 200):
    # Obtain transformed matrix as model sees it
    X_t = preprocessor.transform(X)
    # Choose explainer strategy
    if model_type == 'xgb' and hasattr(model, 'get_booster'):
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_t)
            return sv, X_t
        except Exception as e:
            print(f"TreeExplainer failure; falling back to KernelExplainer: {e}")
    # Generic fallback: KernelExplainer on a small background subset for speed
    ns = min(background_size, X_t.shape[0])
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(X_t.shape[0], size=ns, replace=False)
    background = X_t[bg_idx]
    def _predict(p):
        return model.predict_proba(p)[:,1]
    explainer = shap.KernelExplainer(_predict, background)
    sv = explainer.shap_values(X_t, nsamples=min(100 * X_t.shape[1], 500))  # limit nsamples for memory
    return sv, X_t


def summarize_shap(shap_vals: np.ndarray, feature_cols: list[str]):
    # If model is multiclass, shap_vals may be list; assume binary logistic => array
    if isinstance(shap_vals, list):  # pragma: no cover
        shap_vals = shap_vals[0]
    abs_mean = np.abs(shap_vals).mean(axis=0)
    std_vals = shap_vals.std(axis=0)
    df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': abs_mean,
        'shap_std': std_vals,
    }).sort_values('mean_abs_shap', ascending=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--target', default='readmission', help='Target name (readmission|mortality|prolonged_los)')
    ap.add_argument('--features', default=None, help='Explicit path to features_full.parquet (optional)')
    ap.add_argument('--max-samples', type=int, default=2000, help='Sample up to this many rows for SHAP (speed)')
    ap.add_argument('--background-size', type=int, default=200, help='Background size for KernelExplainer fallback')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--model-type', default='xgb', help='Force model type if auto detection fails (xgb|logreg|hgb)')
    args = ap.parse_args()

    model_path = os.path.join(args.model_dir, f'model_{args.target}.joblib')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    preproc_path = os.path.join(args.model_dir, 'preprocessor.joblib')
    if not os.path.exists(preproc_path):
        print(f"Preprocessor not found: {preproc_path}")
        sys.exit(1)
    feat_cols_path = os.path.join(args.model_dir, 'feature_columns.json')
    if not os.path.exists(feat_cols_path):
        print(f"feature_columns.json not found in {args.model_dir}")
        sys.exit(1)

    with open(feat_cols_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)

    model = joblib.load(model_path)
    pre = joblib.load(preproc_path)

    # Auto-detect model type if possible
    model_type = args.model_type
    name = type(model).__name__.lower()
    if 'xgb' in name:
        model_type = 'xgb'
    elif 'logistic' in name:
        model_type = 'logreg'
    elif 'histgradient' in name:
        model_type = 'hgb'

    # Load features
    feat_file = args.features or os.path.join(args.model_dir, 'features_full.parquet')
    X_full = _load_features(feat_file, feature_cols)
    if X_full.shape[0] > args.max_samples:
        # Stratified sample by approximate quantile on a stable column (first feature)
        rng = np.random.default_rng(42)
        idx = rng.choice(X_full.index, size=args.max_samples, replace=False)
        X = X_full.loc[idx]
    else:
        X = X_full

    print(f"Computing SHAP for {X.shape[0]} samples x {X.shape[1]} features using model type {model_type} ...")
    shap_vals, X_t = compute_shap(model, pre, X, model_type=model_type, background_size=args.background_size)

    os.makedirs(args.output_dir, exist_ok=True)
    # Save SHAP values compressed
    np.savez_compressed(os.path.join(args.output_dir, 'shap_values.npz'), shap_values=shap_vals, feature_order=np.array(feature_cols))

    summary_df = summarize_shap(shap_vals, feature_cols)
    summary_df.to_csv(os.path.join(args.output_dir, 'shap_summary.csv'), index=False)
    with open(os.path.join(args.output_dir, 'shap_top20.txt'), 'w', encoding='utf-8') as f:
        for _, row in summary_df.head(20).iterrows():
            f.write(f"{row.feature}\t{row.mean_abs_shap:.6f}\n")

    # Optional plots
    try:  # pragma: no cover
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_vals, features=X_t, show=False, max_display=25, feature_names=feature_cols)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'shap_beeswarm.png'), dpi=150)
        plt.close()
        shap.summary_plot(shap_vals, features=X_t, show=False, plot_type='bar', max_display=25, feature_names=feature_cols)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'shap_bar.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Plot generation skipped: {e}")

    print("Done. Outputs written to", args.output_dir)

if __name__ == '__main__':
    main()
