import os
import json
import shutil
import unittest
from pathlib import Path

# Ensure project package importable
import sys
root = Path(__file__).resolve().parents[0].parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from project.src.promote_artifacts import promote, PROMOTED_DIR, RUNS_ROOT  # type: ignore

class TestPromoteArtifacts(unittest.TestCase):
    def setUp(self):
        # Create a fake run with minimal artifacts
        self.run_name = 'TEST_RUN_0001'
        self.run_dir = RUNS_ROOT / self.run_name / 'artifacts'
        shutil.rmtree(RUNS_ROOT / self.run_name, ignore_errors=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Minimal required files
        (self.run_dir / 'model_readmission.joblib').write_bytes(b'123')
        (self.run_dir / 'metrics.json').write_text('{"auc":0.5}')
        (self.run_dir / 'feature_columns.json').write_text('[]')
        (self.run_dir / 'preprocessor.joblib').write_bytes(b'456')
        (self.run_dir / 'feature_provenance.json').write_text('{}')
        (self.run_dir / 'features_full.parquet').write_bytes(b'PARQ')
        (self.run_dir / 'threshold.txt').write_text('0.2')

        PROMOTED_DIR.mkdir(parents=True, exist_ok=True)
        # Clean promoted dir contents (only for test isolation)
        for item in PROMOTED_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    def tearDown(self):
        shutil.rmtree(RUNS_ROOT / self.run_name, ignore_errors=True)
        # Leave promoted dir as-is (not deleting full tree to avoid accidental real artifacts removal)

    def test_dry_run_metadata(self):
        meta = promote(self.run_name, dry=True)
        self.assertEqual(meta['source_run'], self.run_name)
        self.assertTrue(meta['dry_run'])
        # No files should be copied in dry run
        promoted_files = list(PROMOTED_DIR.glob('*'))
        self.assertEqual(promoted_files, [])
        self.assertIn('files', meta)
        self.assertGreater(len(meta['files']), 0)

    def test_real_promotion(self):
        meta = promote(self.run_name, dry=False, force=True)
        self.assertFalse(meta['dry_run'])
        self.assertTrue((PROMOTED_DIR / 'model_readmission.joblib').exists())
        self.assertTrue((PROMOTED_DIR / 'metrics.json').exists())
        self.assertTrue((PROMOTED_DIR / 'METADATA.json').exists())
        with open(PROMOTED_DIR / 'METADATA.json','r',encoding='utf-8') as f:
            recorded = json.load(f)
        self.assertEqual(recorded['source_run'], self.run_name)

    # Negative: Missing run directory
    def test_missing_run(self):
        with self.assertRaises(FileNotFoundError):
            promote('NON_EXISTENT_RUN_9999', dry=True)

    # Negative: Empty artifacts (remove files then attempt)
    def test_empty_artifacts(self):
        # Clear files from run artifacts directory
        for fp in list(self.run_dir.iterdir()):
            if fp.is_file():
                fp.unlink()
        with self.assertRaises(RuntimeError):
            promote(self.run_name, dry=True)
        # Recreate one file to restore environment for other tests (not strictly needed post-call)
        (self.run_dir / 'model_readmission.joblib').write_bytes(b'123')

    # Negative: Conflict without force
    def test_conflict_no_force(self):
        # First promote (real)
        promote(self.run_name, dry=False, force=True)
        # Modify one source file to ensure conflict detection would matter
        (self.run_dir / 'threshold.txt').write_text('0.25')
        with self.assertRaises(FileExistsError):
            promote(self.run_name, dry=False, force=False)

if __name__ == '__main__':
    unittest.main()