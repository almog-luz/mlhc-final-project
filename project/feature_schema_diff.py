"""Compare two feature matrices (parquet or feature_columns.json) and report differences.

Usage (PowerShell):
  python -m project.feature_schema_diff --a path/to/runA --b path/to/runB \
      [--features-a features_full.parquet] [--features-b features_full.parquet]

The --a/--b can point either to directories containing feature_columns.json or directly
at .parquet/.json files. The tool outputs:
  - Counts of total / shared / added / removed columns
  - List of newly added columns (optionally truncated unless --full)
  - List of removed columns
  - Grouped summary by suffix patterns (__measured, __count, etc.)
  - Simple Jaccard similarity
Exit code is 0 unless a critical error (e.g., cannot load either side).
"""
from __future__ import annotations
import argparse, os, json, sys
from typing import List, Set

import pandas as pd

def _load_feature_list(path: str, explicit_file: str | None) -> List[str]:
    # Priority: explicit file arg; else attempt feature_columns.json; else load parquet columns.
    if explicit_file:
        fp = explicit_file
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        if fp.endswith('.json'):
            with open(fp, 'r', encoding='utf-8') as f:
                cols = json.load(f)
                if isinstance(cols, dict):
                    # If provenance mapping provided, keys are feature names
                    return [c for c in cols.keys() if not c.startswith('__')]
                return list(cols)
        elif fp.endswith('.parquet'):
            df = pd.read_parquet(fp, columns=None)
            return [c for c in df.columns if c != 'subject_id']
        else:
            raise ValueError(f"Unsupported file extension for {fp}")
    # Directory mode
    if os.path.isdir(path):
        json_path = os.path.join(path, 'feature_columns.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict):
                return list(raw.keys())
        # Fallback: look for features_full.parquet
        pq_path = os.path.join(path, 'features_full.parquet')
        if os.path.exists(pq_path):
            df = pd.read_parquet(pq_path, columns=None)
            return [c for c in df.columns if c != 'subject_id']
    # Single file path
    if os.path.isfile(path):
        return _load_feature_list(path, explicit_file=path)
    raise FileNotFoundError(path)


def summarize(a: List[str], b: List[str], full: bool = False) -> str:
    set_a: Set[str] = set(a)
    set_b: Set[str] = set(b)
    added = sorted(list(set_b - set_a))
    removed = sorted(list(set_a - set_b))
    shared = sorted(list(set_a & set_b))
    lines: List[str] = []
    lines.append("=== FEATURE SCHEMA DIFF ===")
    lines.append(f"A count: {len(set_a)}  B count: {len(set_b)}  Shared: {len(shared)}")
    union = len(set_a | set_b)
    jaccard = (len(shared) / union) if union else 1.0
    lines.append(f"Jaccard similarity: {jaccard:.4f}")
    lines.append(f"Added (in B not A): {len(added)}")
    if added:
        preview = added if full else added[:50]
        lines.append("  + " + ", ".join(preview) + (" ..." if (not full and len(added) > 50) else ""))
    lines.append(f"Removed (in A not B): {len(removed)}")
    if removed:
        preview = removed if full else removed[:50]
        lines.append("  - " + ", ".join(preview) + (" ..." if (not full and len(removed) > 50) else ""))
    # Group by common suffixes
    def _suffix_counts(cols: List[str]):
        suffixes = ['__measured','__count','__mean','__std','__min','__max','__range','__last']
        counts = {s: 0 for s in suffixes}
        for c in cols:
            for s in suffixes:
                if c.endswith(s):
                    counts[s] += 1
        return counts
    lines.append("\n-- Added Suffix Breakdown --")
    for s, v in _suffix_counts(added).items():
        if v:
            lines.append(f"  {s}: {v}")
    lines.append("-- Removed Suffix Breakdown --")
    for s, v in _suffix_counts(removed).items():
        if v:
            lines.append(f"  {s}: {v}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Diff two feature schema sources.")
    ap.add_argument('--a', required=True, help='Directory or file for baseline (before) run')
    ap.add_argument('--b', required=True, help='Directory or file for updated (after) run')
    ap.add_argument('--features-a', default=None, help='Explicit feature file for A (json or parquet)')
    ap.add_argument('--features-b', default=None, help='Explicit feature file for B (json or parquet)')
    ap.add_argument('--full', action='store_true', help='Print full lists (not truncated)')
    args = ap.parse_args()
    try:
        a_cols = _load_feature_list(args.a, args.features_a)
        b_cols = _load_feature_list(args.b, args.features_b)
    except Exception as e:
        print(f"Error loading feature lists: {e}")
        sys.exit(1)
    print(summarize(a_cols, b_cols, full=args.full))

if __name__ == '__main__':
    main()
