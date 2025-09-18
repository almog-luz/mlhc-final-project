"""Utility to reserialize existing joblib artifacts with an older pickle protocol
for broader compatibility (e.g., Google Colab older Python versions).

Usage (from repo root):

    python -m project.compat_reserialize \
        --models-dir project/models \
        --protocol 4

This will create a backup copy of every *.joblib file (suffix .bak) and then
rewrite the original file using the requested protocol (default=4).

Protocol 4 is readable by Python >=3.4 so it is a safe baseline for Colab
instances that may lag behind latest CPython releases used locally.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import joblib


def reserialize_directory(models_dir: Path, protocol: int = 4, dry_run: bool = False) -> None:
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    joblib_files = list(models_dir.glob('*.joblib'))
    if not joblib_files:
        print(f"No .joblib artifacts found under {models_dir}")
        return

    print(f"Found {len(joblib_files)} joblib artifacts. Target protocol={protocol} dry_run={dry_run}")

    for fp in joblib_files:
        backup = fp.with_suffix(fp.suffix + '.bak')
        print(f"Processing {fp.name} -> backup {backup.name}")
        obj = joblib.load(fp)
        if dry_run:
            print("  (dry-run) would backup and reserialize")
            continue
        if not backup.exists():
            shutil.copy2(fp, backup)
        # Re-save with requested protocol (use simple gzip level int for broad support)
        joblib.dump(obj, fp, compress=3, protocol=protocol)
        print("  Reserialized with protocol", protocol)

    print("Done. You can now copy artifacts to Colab environment.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reserialize joblib artifacts with older pickle protocol")
    p.add_argument('--models-dir', type=Path, default=Path('project') / 'models', help='Directory containing joblib artifacts')
    p.add_argument('--protocol', type=int, default=4, help='Pickle protocol to use (<=4 for maximum compatibility)')
    p.add_argument('--dry-run', action='store_true', help='List actions without modifying artifacts')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reserialize_directory(args.models_dir, protocol=args.protocol, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
