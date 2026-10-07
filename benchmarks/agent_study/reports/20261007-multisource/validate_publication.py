"""Validate this dated public snapshot against its original Beacon records.

Run on a Slurm compute node after extracting the candidate repository archive.
The private inputs and re-exported error examples stay on Beacon.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
EXPECTED_SHA = '80d3677eb5050e00ee4faa3d6be016d0e78741ae743dfed6b844c2700731fde0'
LABELS = {
    '7b': 'Qwen2.5 Coder 7B', '14b': 'Qwen2.5 Coder 14B',
    '32b': 'Qwen2.5 Coder 32B', 'qwen-general-7b': 'Qwen2.5 7B Instruct',
    'qwen3-4b': 'Qwen3 4B Instruct 2507', 'granite-8b': 'Granite 3.3 8B',
    'phi4-mini': 'Phi-4 Mini', 'smollm3-3b': 'SmolLM3 3B',
    'mistral12b': 'Mistral Nemo 12B', 'olmo3-7b': 'OLMo 3 7B',
    'falcon3-10b': 'Falcon3 10B',
}


def validate(archive_path, export_dir):
    snapshot_path = HERE / 'snapshot.json'
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == EXPECTED_SHA
    current = json.loads(snapshot_path.read_text())
    assert (current['completed_models'], current['verified_pairs']) == (11, 33)
    spec = importlib.util.spec_from_file_location(
        'snapshot_publisher', HERE.parent / '20261006' / 'publish_snapshot.py')
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    publisher.validate_summary(dict(current, rows=current['seed_results'], failures=[]))

    # Re-export from the fixed verifier/diagnostic files; check original receipts again.
    export_dir.mkdir(exist_ok=False)
    provenance = current['provenance']
    verification = BASE / provenance['verification_file']
    diagnostics = BASE / provenance['diagnostics_file']
    assert publisher.sha(verification) == provenance['verification_sha256']
    assert publisher.sha(diagnostics) == provenance['diagnostics_sha256']
    recreated = export_dir / 'snapshot.json'
    publisher.export(BASE, verification, diagnostics, recreated, export_dir / 'private-errors.json')
    assert recreated.read_bytes() == snapshot_path.read_bytes()

    # The extension must preserve every previously published paired result.
    previous = json.loads((HERE.parent / '20261006' / 'snapshot.json').read_text())
    new_rows = {row['pair']: row for row in current['seed_results']}
    assert len(new_rows) == current['verified_pairs']
    for row in previous['seed_results']:
        assert new_rows[row['pair']] == row, row['pair']
    for family, summary in previous['summary'].items():
        if summary['complete']:
            assert current['summary'][family] == summary, family

    # Check the reader-facing table against the full-precision public data.
    readme = (HERE / 'README.md').read_text()
    assert set(LABELS) == set(current['summary'])
    for family, label in LABELS.items():
        summary = current['summary'][family]
        returns = summary['return_rate_pct']
        row = (f"| {label} | {returns['raw']:.6f}% | {returns['library']:.6f}% | "
               f"{summary['difference_pp']:+.6f} |")
        assert row in readme, row

    # Regeneration is required before commit, but must not alter generated content.
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if name.endswith('/'):
                continue
            if name in ('README.md', 'catalog/index.json') or name.startswith('fin_skills/'):
                before = archive.read(name).decode('utf-8').replace('\r\n', '\n')
                assert (ROOT / name).read_text() == before, name
    print(json.dumps(dict(public_snapshot_reproduced=True, previous_pairs_preserved=True,
        readme_table_matches=True, generated_content_unchanged=True,
        completed_models=current['completed_models'], verified_pairs=current['verified_pairs'])))


if __name__ == '__main__':
    validate(Path(sys.argv[1]), Path(sys.argv[2]))
