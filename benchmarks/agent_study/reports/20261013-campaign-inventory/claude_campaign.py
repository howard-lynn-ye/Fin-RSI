"""Frozen native Claude campaign; one post-reset attempt, no account fallback."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.agent_study import multisource_model_study as study

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
M = Path('/beacon-projects/radfm/wy891/fin-manual-multisource-20261005')
S = B/'probe-20261013-candidate11/source'
R = B/'native-matrix-20261014-claude-r6'
FAMILIES = ('opus55', 'sonnet55', 'fable51')
RELEASE = datetime(2026, 10, 14, 18, 5, tzinfo=timezone.utc)
assert os.environ.get('SLURM_JOB_ID')
assert Path(study.__file__).resolve().is_relative_to(S)
assert (S.parent/'passed.txt').exists()
assert sys.argv[1:] in (['prepare'], ['run'])


def prepare():
    assert not R.exists(), 'Never replace an existing protocol or result'
    R.mkdir()
    protocols = []
    for family in FAMILIES:
        for seed in (11, 23, 37):
            root = R/'pairs'/f'{family}-{seed}'
            study.freeze(root, family, seed, M/'evidence.json', allow_retrospective=True,
                         batch='matrix_native', interface='v9')
            protocols.append(dict(family=family, seed=seed, root=str(root),
                protocol_sha256=study.old.sha(root/'protocol.json')))
    study.old.write(R/'relay-allowlist.json', dict(protocols=protocols,
        scope='Three distinct Claude model IDs, three paired repeats each. Frozen before inference.',
        not_before_utc=RELEASE.isoformat(),
        quota_rule='One attempt after declared weekly reset; a renewed quota failure stops that family. No route or account fallback.'))
    print('Prepared nine frozen protocols; no provider calls made.', flush=True)


def run_family(family):
    events = []
    for seed in (11, 23, 37):
        root = R/'pairs'/f'{family}-{seed}'
        tmp = R/'tmp'/f'{family}-{seed}'
        tmp.mkdir(parents=True)
        env = dict(os.environ, TMPDIR=str(tmp), HF_HOME=str(tmp/'hf'),
            HF_HUB_CACHE='/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub',
            XDG_CACHE_HOME=str(tmp/'xdg'), MPLCONFIGDIR=str(tmp/'matplotlib'))
        for phase in ('qualify', 'run', 'score'):
            log = R/'logs'/f'{family}-{seed}-{phase}.log'
            log.parent.mkdir(exist_ok=True)
            with log.open('x') as stream:
                code = subprocess.run([sys.executable, '-m', 'benchmarks.agent_study.multisource_model_study',
                    phase, str(root)], cwd=S, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
            row = dict(family=family, seed=seed, phase=phase, exit_code=code,
                time_utc=datetime.now(timezone.utc).isoformat(), job=os.environ['SLURM_JOB_ID'],
                driver_sha256=study.old.sha(Path(__file__)))
            study.old.write(R/'events'/f'{family}-{seed}-{phase}.json', row)
            print(json.dumps(row), flush=True)
            events.append(row)
            if code:
                return dict(family=family, status='stopped_after_failure', events=events)
    return dict(family=family, status='completed_three_pairs', events=events)


if sys.argv[1] == 'prepare':
    prepare()
else:
    assert datetime.now(timezone.utc) >= RELEASE, 'No native model call before quota reset'
    allowlist = study.old.load(R/'relay-allowlist.json')
    assert allowlist['not_before_utc'] == RELEASE.isoformat()
    expected = {(f, s) for f in FAMILIES for s in (11, 23, 37)}
    assert {(p['family'], p['seed']) for p in allowlist['protocols']} == expected
    assert len(allowlist['protocols']) == len(expected)
    for p in allowlist['protocols']:
        root = R/'pairs'/f"{p['family']}-{p['seed']}"
        assert str(root) == p['root']
        assert study.old.sha(root/'protocol.json') == p['protocol_sha256']
        study.verify(root)
    study.old.write(R/'started.json', dict(job=os.environ['SLURM_JOB_ID'],
        time_utc=datetime.now(timezone.utc).isoformat(), driver_sha256=study.old.sha(Path(__file__))))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run_family, FAMILIES))
    study.old.write(R/'campaign-status.json', dict(results=results,
        complete=all(r['status'] == 'completed_three_pairs' for r in results)))
    print('Consult individual results; no aggregate completion implied.', flush=True)
