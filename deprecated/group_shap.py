"""Aggregate SHAP importance by feature groups.

Usage:
  python -m project.group_shap \
    --shap-summary project/models/shap_readmission/shap_summary.csv \
    --feature-provenance project/models/feature_provenance.json \
    --out project/models/shap_readmission/group_shap_summary.csv

If provenance not supplied, groups inferred by prefix/suffix heuristics:
  var_ -> variability
  med_ -> medication
  stack_ -> stacking
  pheno_cluster* -> phenotype
  *(__measured|__count) -> missingness
  cmb_*, cmbgrp_* -> comorbidity
  utl_* -> utilization
  else -> baseline

Outputs CSV with columns:
  group, feature_count, total_mean_abs_shap, mean_mean_abs_shap, pct_of_total
"""
from __future__ import annotations
import argparse, json, os, sys
import pandas as pd

def infer_group(feat: str) -> str:
    if feat.startswith('var_'): return 'variability'
    if feat.startswith('med_'): return 'medication'
    if feat.startswith('stack_'): return 'stacking'
    if feat.startswith('pheno_cluster'): return 'phenotype'
    if feat.startswith('cmb_') or feat.startswith('cmbgrp_'): return 'comorbidity'
    if feat.startswith('utl_'): return 'utilization'
    if feat.endswith('__measured') or feat.endswith('__count'): return 'missingness'
    return 'baseline'

def load_provenance(path: str | None) -> dict | None:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"Warning: provenance file not found: {path}; falling back to heuristic groups")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        try:
            prov = json.load(f)
        except Exception as e:
            print(f"Warning: failed to parse provenance JSON: {e}; using heuristics")
            return None
    return prov

def main():
    ap = argparse.ArgumentParser(description='Group-level SHAP aggregation')
    ap.add_argument('--shap-summary', required=True, help='Path to shap_summary.csv')
    ap.add_argument('--feature-provenance', default=None, help='Optional feature_provenance.json for groups')
    ap.add_argument('--out', required=True, help='Output CSV path')
    ap.add_argument('--top-n', type=int, default=0, help='If >0, restrict to top-N features before grouping')
    args = ap.parse_args()

    if not os.path.exists(args.shap_summary):
        print(f"Missing shap summary: {args.shap_summary}")
        sys.exit(1)
    df = pd.read_csv(args.shap_summary)
    if args.top_n and args.top_n > 0:
        df = df.sort_values('mean_abs_shap', ascending=False).head(args.top_n)
    prov = load_provenance(args.feature_provenance)

    groups = []
    for _, row in df.iterrows():
        feat = row['feature']
        if prov and feat in prov:
            g = prov[feat].get('group', infer_group(feat))
        else:
            g = infer_group(feat)
        groups.append(g)
    df['group'] = groups
    agg = df.groupby('group', as_index=False).agg(
        feature_count=('feature','count'),
        total_mean_abs_shap=('mean_abs_shap','sum'),
        mean_mean_abs_shap=('mean_abs_shap','mean'),
    )
    total = agg['total_mean_abs_shap'].sum()
    if total > 0:
        agg['pct_of_total'] = agg['total_mean_abs_shap'] / total
    agg = agg.sort_values('total_mean_abs_shap', ascending=False)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    agg.to_csv(args.out, index=False)
    print("Group SHAP summary written to", args.out)
    print(agg.to_string(index=False, justify='left', col_space=12))

if __name__ == '__main__':
    main()
