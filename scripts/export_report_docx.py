import subprocess, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
md_path = ROOT / 'docs' / 'project_report_draft.md'
out_docx = ROOT / 'docs' / 'project_report_draft.docx'

if not md_path.exists():
    print('Markdown report not found:', md_path)
    sys.exit(1)

# Prefer pypandoc if installed, else fallback to system pandoc
try:
    import pypandoc  # type: ignore
    pypandoc.convert_file(str(md_path), 'docx', outputfile=str(out_docx))
    print('DOCX written ->', out_docx)
except Exception as e:
    print('pypandoc not available or failed:', e)
    pandoc = shutil.which('pandoc')
    if pandoc is None:
        print('Pandoc not found. Install via: pip install pypandoc or system package.')
        sys.exit(2)
    cmd = [pandoc, str(md_path), '-o', str(out_docx)]
    subprocess.check_call(cmd)
    print('DOCX written via pandoc ->', out_docx)
