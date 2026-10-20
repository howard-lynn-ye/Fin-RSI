"""Fin-RSI study runner (fin_rsi_run.py): CLI surface, and a replay that pins the frozen decisions.

The replay test reruns 8 saved Qwen2.5 14B/32B rows (job attempt 20261018-...general14b32b,
C_unchanged, 4 failure-selected units x 2 models) through the runner on CPU and compares a
replay-stable projection of every decision (per-turn parse/tool/ok/error, submit, target to
6 dp) with the projection the frozen I0-I21 runner (with the outcome bank) produced on the
same inputs. The runner starts its tool worker with `python -m`, which puts the current
directory ahead of PYTHONPATH; the subprocess therefore runs with cwd = this repository and
absolute data paths, so the worker imports this tree's code. Absolute repository, data and venv
paths that appear in error text are normalised. It needs the Beacon result files and the Fin-RSI
environment, so it is marked slow and skips elsewhere.
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

# Frozen tree (main checkout, I0-I21 code, outcome bank included) replayed with cwd = that tree,
# 2026-10-20; the integration tree gives the same digests.
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
        'repeated_tiny_skill_page__seed11__general32b__strict__s11': '7f4666043d6a6485',
        'skill_page_then_submit__seed11__general14b__strict__s11': 'bf61382e7810dde7',
        'skill_page_then_submit__seed11__general32b__strict__s11': '1665ebd1ce2ef49c',
        'worker_submit_retention__seed23__general14b__strict__s23': '2ef79e111749531e',
        'worker_submit_retention__seed23__general32b__strict__s23': 'ca08f554504f28f5'},
}


def normalise(text, repo=REPO):
    """Drop install locations from error text (venv, repository, data root, in that order)."""
    return (text.replace(str(DATA / '.fin-rsi-envs'), '<ENV>').replace(str(repo), '<REPO>')
            .replace(str(DATA), '<DATA>'))


def projection_digest(decision, repo=REPO):
    turns = [(t['parse'], t['tool'], t['ok'], normalise(t['error'] or '', repo)[:200]) for t in decision['turns']]
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


def replay(interface, out, *extra):
    """Replay the 8 saved rows through this tree's runner; cwd = this repository (see docstring)."""
    if not (DATA / ROWS).exists() or not (DATA / ATTEMPT).exists() or not PYTHON.exists():
        pytest.skip('needs the Beacon Fin-RSI result files and environment')
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1')
    subprocess.run([str(PYTHON), str(RUNNER), '--rows', str(DATA / ROWS), '--condition', 'C_unchanged',
                    '--model', 'general14b=x', '--model', 'general32b=x', '--interface', interface,
                    '--backend', 'replay:' + str(DATA / ATTEMPT), '--out', str(out), *extra],
                   cwd=REPO, env=env, check=True, capture_output=True, text=True, timeout=1800)
    decisions = {p.name: json.loads((p / 'decision.json').read_text()) for p in sorted((out / 'rows').iterdir())}
    if REPO.resolve() != DATA.resolve():  # the worker must not have imported the data checkout's code
        foreign = [str(DATA / 'fin_skills'), str(DATA / 'benchmarks')]
        assert not [r for r, d in decisions.items() for t in d['turns']
                    if any(f in (t.get('feedback') or '') + (t.get('error') or '') for f in foreign)], \
            'replay worker imported code from the data checkout'
    return {name: projection_digest(d) for name, d in decisions.items()}


def test_normalise_strips_install_locations():
    text = f"ImportError: from 'fin_skills' ({REPO}/fin_skills/__init__.py); {DATA}/.fin-rsi-envs/gpu/x.py"
    assert normalise(text) == "ImportError: from 'fin_skills' (<REPO>/fin_skills/__init__.py); <ENV>/gpu/x.py"


@pytest.mark.slow
@pytest.mark.parametrize('interface', ['tolerant', 'strict'])
def test_replay_reproduces_frozen_decisions(interface, tmp_path):
    assert replay(interface, tmp_path / interface) == FROZEN[interface]
