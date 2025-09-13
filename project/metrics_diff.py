"""Compare two metrics.json files and print deltas per target.

Usage:
  python -m project.metrics_diff --a path/to/runA/metrics.json --b path/to/runB/metrics.json

If directories are provided, will look for metrics.json inside each.
Outputs a table-like text with absolute and relative deltas for key metrics.
"""
from __future__ import annotations
import argparse, os, json, math, sys
from typing import Dict, Any, List

KEYS = ["roc_auc","pr_auc","brier","ece","f1","precision","recall","threshold"]


def _load_metrics(p: str) -> Dict[str, Dict[str, Any]]:
    if os.path.isdir(p):
        p = os.path.join(p, 'metrics.json')
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("metrics file must contain object mapping target->metrics")
    return data


def _fmt(v):
    if v is None:
        return 'NA'
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def diff_metrics(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    all_targets = sorted(set(a.keys()) | set(b.keys()))
    lines.append("=== METRICS DIFF ===")
    for tgt in all_targets:
        ma = a.get(tgt, {})
        mb = b.get(tgt, {})
        lines.append(f"\nTarget: {tgt}")
        for k in KEYS:
            va = ma.get(k)
            vb = mb.get(k)
            if va is None and vb is None:
                continue
            abs_delta = None
            rel_delta = None
            try:
                if va is not None and vb is not None and all(isinstance(x,(int,float)) for x in [va,vb]):
                    abs_delta = vb - va
                    if va != 0:
                        rel_delta = abs_delta / abs(va)
            except Exception:
                pass
            lines.append(
                f"  {k:10s} A={_fmt(va)}  B={_fmt(vb)}  Δ={_fmt(abs_delta) if abs_delta is not None else 'NA'}  Δ%={_fmt(rel_delta) if rel_delta is not None else 'NA'}"
            )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Diff two metrics.json files")
    ap.add_argument('--a', required=True, help='Baseline metrics.json (or directory)')
    ap.add_argument('--b', required=True, help='Updated metrics.json (or directory)')
    args = ap.parse_args()
    try:
        m_a = _load_metrics(args.a)
        m_b = _load_metrics(args.b)
    except Exception as e:
        print(f"Error loading metrics: {e}")
        sys.exit(1)
    print(diff_metrics(m_a, m_b))

if __name__ == '__main__':
    main()
