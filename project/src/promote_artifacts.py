import argparse, os, shutil, json, time, hashlib, sys, glob, subprocess
from pathlib import Path

PROMOTED_DIR = Path('project/artifacts')
RUNS_ROOT = Path('project/runs')

ALLOWED_FILENAMES = {
    'features_full.parquet',
    'feature_columns.json',
    'feature_provenance.json',
    'preprocessor.joblib',
    'model_readmission.joblib',
    'metrics.json',
    'threshold.txt',
    'run_metadata.json',
    'audit_features.json',
}

def hash_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            h.update(chunk)
    return h.hexdigest()[:16]

def gather_artifacts(run_artifacts_dir: Path):
    files = []
    for fn in sorted(run_artifacts_dir.iterdir()):
        if fn.is_file() and fn.name in ALLOWED_FILENAMES:
            files.append(fn)
    return files

def detect_run_artifacts(run_path: Path) -> Path:
    # Accept either run/<ts>/artifacts or run/<ts>/ directly containing artifacts
    artifacts = run_path / 'artifacts'
    if artifacts.is_dir():
        return artifacts
    return run_path

def get_git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'UNKNOWN'

def promote(source_run: str, force: bool=False, include_extra: bool=False, dry: bool=False) -> dict:
    run_dir = RUNS_ROOT / source_run
    if not run_dir.exists():
        raise FileNotFoundError(f'Run directory not found: {run_dir}')
    artifact_root = detect_run_artifacts(run_dir)
    if not artifact_root.exists():
        raise FileNotFoundError(f'Artifact directory missing inside run: {artifact_root}')
    files = gather_artifacts(artifact_root)
    if not files:
        raise RuntimeError(f'No recognized artifact files in {artifact_root}')

    PROMOTED_DIR.mkdir(parents=True, exist_ok=True)

    # Backup existing promoted set (single timestamp snapshot)
    backup_dir = None
    if any(PROMOTED_DIR.iterdir()):
        backup_dir = PROMOTED_DIR.parent / f"artifacts_backup_{int(time.time())}"
        if not dry:
            shutil.copytree(PROMOTED_DIR, backup_dir)

    copied = []
    for f in files:
        dest = PROMOTED_DIR / f.name
        if dest.exists() and not force:
            raise FileExistsError(f'Destination file already exists (use --force): {dest}')
        if not dry:
            shutil.copy2(f, dest)
        copied.append({'name': f.name, 'sha256_16': hash_file(f) if f.is_file() else None})

    # Optional: include entire directory (e.g., shap_readmission) if requested
    extras = []
    if include_extra:
        for candidate in ['shap_readmission']:
            cpath = artifact_root / candidate
            if cpath.is_dir():
                dest_dir = PROMOTED_DIR / candidate
                if dest_dir.exists() and force:
                    if not dry:
                        shutil.rmtree(dest_dir)
                if not dest_dir.exists():
                    if not dry:
                        shutil.copytree(cpath, dest_dir)
                    extras.append(candidate)

    metadata_path = PROMOTED_DIR / 'METADATA.json'
    meta = {
        'source_run': source_run,
        'source_path': str(artifact_root),
        'promoted_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'git_commit': get_git_commit(),
        'files': copied,
        'included_extras': extras,
        'backup_dir': str(backup_dir) if backup_dir else None,
        'dry_run': dry,
    }
    if not dry:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
    return meta


def find_latest_run() -> str:
    if not RUNS_ROOT.exists():
        raise FileNotFoundError('No runs/ directory present')
    candidates = []
    for d in RUNS_ROOT.iterdir():
        if d.is_dir():
            try:
                candidates.append((d.stat().st_mtime, d.name))
            except FileNotFoundError:
                pass
    if not candidates:
        raise RuntimeError('No run subdirectories found')
    candidates.sort(reverse=True)
    return candidates[0][1]


def parse_args():
    p = argparse.ArgumentParser(description='Promote a run artifacts set to project/artifacts (canonical).')
    p.add_argument('--run', help='Run directory name under project/runs (e.g. 20250913_153045). If omitted, uses latest run.')
    p.add_argument('--force', action='store_true', help='Overwrite existing promoted files without error.')
    p.add_argument('--include-extra', action='store_true', help='Also copy supported extra directories (e.g. shap_readmission).')
    p.add_argument('--dry-run', action='store_true', help='Show actions without copying.')
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.run or find_latest_run()
    try:
        meta = promote(run_name, force=args.force, include_extra=args.include_extra, dry=args.dry_run)
        print(json.dumps(meta, indent=2))
    except Exception as e:
        print('ERROR:', type(e).__name__, str(e), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
