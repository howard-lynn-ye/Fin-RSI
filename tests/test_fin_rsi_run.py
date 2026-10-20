"""Fin-RSI study runner (fin_rsi_run.py): CLI surface, and a replay that pins the frozen decisions.

The replay test reruns 8 saved Qwen2.5 14B/32B rows (job attempt 20261018-...general14b32b,
C_unchanged, 4 failure-selected units x 2 models) through the runner on CPU and compares a
replay-stable projection of every decision (per-turn parse/tool/ok/error, submit, target to
6 dp) with the projection the frozen I0-I21 runner (with the outcome bank) produced on the
same inputs in this tree. It needs the Beacon result files and the Fin-RSI environment, so it
is marked slow and skips elsewhere.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / 'benchmarks' / 'agent_study' / 'fin_rsi_run.py'
DATA = Path(os.environ.get('FIN_RSI_DATA_ROOT', '/beacon-projects/traumallm/shwaihe/fin-skills'))
PYTHON = Path(os.environ.get('FIN_RSI_PYTHON', DATA / '.fin-rsi-envs' / 'gpu' / 'bin' / 'python'))
ROWS = 'results/fin_rsi/reproduction/20261018-current-v7-submit-source-certificate-evaluation-001/inputs/row-bindings-v1.json'
ATTEMPT = 'results/fin_rsi/live_model_experiments/runs/20261018-current-v7-submit-source-certificate-general14b32b/attempt-01'

# Frozen runner (outcome bank included) on the same inputs, 2026-10-20.
FROZEN = {
    'tolerant': {
        'algorithm_contract_failure__seed23__general14b__tolerant__s23': '9baaf2eb00708146',
        'algorithm_contract_failure__seed23__general32b__tolerant__s23': 'e2f279cf09ae7ad7',
        'repeated_tiny_skill_page__seed11__general14b__tolerant__s11': 'ba457917a6b88413',
        'repeated_tiny_skill_page__seed11__general32b__tolerant__s11': '0e824cac10d536bf',
        'skill_page_then_submit__seed11__general14b__tolerant__s11': '0d09cc115adb4221',
        'skill_page_then_submit__seed11__general32b__tolerant__s11': 'adf7219105fc6dd0',
        'worker_submit_retention__seed23__general14b__tolerant__s23': '2f2f58f7f091d4b5',
        'worker_submit_retention__seed23__general32b__tolerant__s23': '97e73176f21c064b'},
    'strict': {
        'algorithm_contract_failure__seed23__general14b__strict__s23': '8105c8c4d37623cc',
        'algorithm_contract_failure__seed23__general32b__strict__s23': 'e2f279cf09ae7ad7',
        'repeated_tiny_skill_page__seed11__general14b__strict__s11': 'b253406dbced6ea9',
        'repeated_tiny_skill_page__seed11__general32b__strict__s11': '7fbd4ce6a053d536',
        'skill_page_then_submit__seed11__general14b__strict__s11': 'bf61382e7810dde7',
        'skill_page_then_submit__seed11__general32b__strict__s11': '07b9d3f1e6cef059',
        'worker_submit_retention__seed23__general14b__strict__s23': '2ef79e111749531e',
        'worker_submit_retention__seed23__general32b__strict__s23': 'ca08f554504f28f5'},
}


def projection_digest(decision):
    turns = [(t['parse'], t['tool'], t['ok'], (t['error'] or '')[:200]) for t in decision['turns']]
    target = None if decision['target'] is None else {
        k: round(v, 6) for k, v in sorted(decision['target'].items())}
    blob = json.dumps(dict(turns=turns, submitted=decision['submitted'], target=target), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def test_cli_flags_and_defaults():
    help_text = subprocess.run([sys.executable, str(RUNNER), '--help'], cwd=REPO, capture_output=True,
                               text=True, check=True, env=dict(os.environ, PYTHONPATH=str(REPO))).stdout
    for flag in ('--interface', '--backend', '--system-variant', '--task-variant', '--arm',
                 '--algorithm-hint', '--skill-search', '--skill-overlay', '--snapshot-tool-log',
                 '--skill-name-hint', '--persistent-python', '--python-only-hint', '--flag-arms',
                 '--strip-history-thinking', '--sampling', '--decision-channel'):
        assert flag in help_text, flag
    assert '--outcome-bank' not in help_text and '--bank-k' not in help_text  # rejected in I10


@pytest.mark.slow
@pytest.mark.parametrize('interface', ['tolerant', 'strict'])
def test_replay_reproduces_frozen_decisions(interface, tmp_path):
    if not (DATA / ROWS).exists() or not (DATA / ATTEMPT).exists() or not PYTHON.exists():
        pytest.skip('needs the Beacon Fin-RSI result files and environment')
    out = tmp_path / interface
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1')
    subprocess.run([str(PYTHON), str(RUNNER), '--rows', ROWS, '--condition', 'C_unchanged',
                    '--model', 'general14b=x', '--model', 'general32b=x', '--interface', interface,
                    '--backend', 'replay:' + ATTEMPT, '--out', str(out)],
                   cwd=DATA, env=env, check=True, capture_output=True, text=True, timeout=1800)
    got = {p.name: projection_digest(json.loads((p / 'decision.json').read_text()))
           for p in sorted((out / 'rows').iterdir())}
    assert got == FROZEN[interface]
