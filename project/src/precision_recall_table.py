import argparse, pandas as pd, numpy as np
from typing import List

"""Compute precision/recall/threshold at specified recall targets given proba.csv."""

def parse_args():
    p = argparse.ArgumentParser(description='Precision/Recall table for given probability file.')
    p.add_argument('--proba', required=True, help='Path to proba.csv (subject_id, readmission_label, proba)')
    p.add_argument('--recall-targets', nargs='+', type=float, default=[0.2,0.3,0.4,0.5], help='Recall targets (0-1)')
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.proba)
    df = df.sort_values('proba', ascending=False).reset_index(drop=True)
    y = df['readmission_label'].to_numpy()
    p = df['proba'].to_numpy()
    total_pos = y.sum()
    results = []
    # cumulative recall as threshold moves down
    cum_tp = 0
    last_thr = None
    for i, (lab, prob) in enumerate(zip(y, p)):
        if lab == 1:
            cum_tp += 1
        recall = cum_tp / total_pos if total_pos else 0
        precision = cum_tp / (i+1)
        # capture when passing targets
        for target in list(args.recall_targets):
            if recall >= target and not any(r['recall_target']==target for r in results):
                results.append({
                    'recall_target': target,
                    'achieved_recall': recall,
                    'precision': precision,
                    'threshold': prob,
                    'n_selected': i+1,
                })
        last_thr = prob
    # Ensure all targets present (if not reached)
    for t in args.recall_targets:
        if not any(r['recall_target']==t for r in results):
            results.append({'recall_target': t, 'achieved_recall': None, 'precision': None, 'threshold': None, 'n_selected': None})
    results = sorted(results, key=lambda x: x['recall_target'])
    print('recall_target\tachieved_recall\tprecision\tthreshold\tn_selected')
    for r in results:
        print(f"{r['recall_target']}\t{r['achieved_recall']}\t{r['precision']}\t{r['threshold']}\t{r['n_selected']}")

if __name__ == '__main__':
    main()
