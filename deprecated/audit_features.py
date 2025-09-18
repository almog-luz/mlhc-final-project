import os, json, argparse
import pandas as pd
import numpy as np

"""Audit feature matrix coverage and sparsity by persistent splits.
Outputs JSON + printed summary:
- counts of subjects present in features
- zero-only row counts
- per-split prevalence of zero-only rows
- per-feature non-zero fraction (optional top/bottom listing)
"""

def parse_args():
    p = argparse.ArgumentParser(description='Audit feature coverage and sparsity by split.')
    p.add_argument('--labels', required=True, help='Path to labels.csv (must include subject_id).')
    p.add_argument('--features', required=True, help='Path to features_full.parquet.')
    p.add_argument('--splits', required=True, help='Path to splits.json.')
    p.add_argument('--output', default='audit_features.json', help='Output JSON path.')
    p.add_argument('--topn', type=int, default=15, help='Show top N sparse & dense features in stdout.')
    return p.parse_args()


def main():
    args = parse_args()
    labels = pd.read_csv(args.labels)
    feat = pd.read_parquet(args.features)
    if 'subject_id' in feat.columns and feat.index.name != 'subject_id':
        try:
            feat = feat.set_index('subject_id')
        except Exception:
            pass
    with open(args.splits, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    all_ids = labels['subject_id']
    aligned = feat.reindex(all_ids).fillna(0.0)
    zero_row_mask = (aligned == 0).all(axis=1)
    zero_rows_total = int(zero_row_mask.sum())

    def split_stats(name, id_list):
        mask = all_ids.isin(id_list)
        mask_arr = mask.to_numpy(dtype=bool)
        idx = np.where(mask_arr)[0]
        subset_zero = int(zero_row_mask.iloc[idx].sum())
        subset_total = int(len(idx))
        return {
            'total': subset_total,
            'zero_rows': subset_zero,
            'zero_row_pct': float(subset_zero / subset_total) if subset_total else 0.0,
        }

    coverage = {
        'total_subjects_labels': int(len(all_ids)),
        'feature_rows_original': int(feat.shape[0]),
        'zero_rows_total': zero_rows_total,
        'zero_row_pct_total': float(zero_rows_total / len(all_ids)),
        'splits': {
            k: split_stats(k, splits[k]) for k in ['train','val','test'] if k in splits
        }
    }

    non_zero_counts = (aligned != 0).sum(axis=0)
    non_zero_frac = (non_zero_counts / len(aligned)).sort_values()
    sparse_features = non_zero_frac.head(args.topn).to_dict()
    dense_features = non_zero_frac.tail(args.topn).to_dict()

    out = {
        'coverage': coverage,
        'sparse_features_fraction': sparse_features,
        'dense_features_fraction': dense_features,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print('=== Feature Coverage Summary ===')
    print(json.dumps(coverage, indent=2))
    print('\nMost Sparse Features (fraction non-zero):')
    for k,v in sparse_features.items():
        print(f'  {k}: {v:.4f}')
    print('\nMost Dense Features (fraction non-zero):')
    for k,v in dense_features.items():
        print(f'  {k}: {v:.4f}')
    print(f'Wrote detailed audit to {args.output}')

if __name__ == '__main__':
    main()
