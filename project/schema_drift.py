import json, pathlib, sys
import pandas as pd

"""Utility to compare two feature schema JSON files and (optionally) a feature parquet.
Usage:
  python -m project.schema_drift current_schema.json legacy_schema.json [features_full.parquet]
Outputs JSON report to stdout.
"""

def load_json(path: str):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else []

def report(current_path: str, legacy_path: str, features_parquet: str | None = None):
    cur = load_json(current_path)
    leg = load_json(legacy_path)
    cur_set, leg_set = set(cur), set(leg)
    inter = cur_set & leg_set
    missing = list(leg_set - cur_set)
    new = list(cur_set - leg_set)
    numeric_only = [c for c in cur if str(c).isdigit()]
    rep = {
        'current_count': len(cur),
        'legacy_count': len(leg),
        'intersection': len(inter),
        'jaccard': (len(inter) / len(cur_set | leg_set)) if (cur_set or leg_set) else 1.0,
        'missing_expected_count': len(missing),
        'unexpected_new_count': len(new),
        'numeric_only_columns': numeric_only,
        'missing_sample': sorted(missing)[:25],
        'new_sample': sorted(new)[:25]
    }
    if features_parquet and pathlib.Path(features_parquet).exists():
        try:
            df = pd.read_parquet(features_parquet, columns=[c for c in cur if not str(c).isdigit()])
            const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
            rep['constant_columns_count'] = len(const_cols)
            rep['constant_columns_sample'] = const_cols[:25]
        except Exception as e:
            rep['features_load_error'] = str(e)
    print(json.dumps(rep, indent=2))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python -m project.schema_drift current_schema.json legacy_schema.json [features_full.parquet]', file=sys.stderr)
        sys.exit(1)
    report(*sys.argv[1:4])
