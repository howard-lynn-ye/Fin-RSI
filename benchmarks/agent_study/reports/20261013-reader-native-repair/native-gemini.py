"""Beacon-only native model replay. The Windows relay supplies model text only."""
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
R = B/'native-matrix-20261013-r8'
S = B/'native-input-20261013-candidate17/source'
FAMILIES = ('gemini38-flash',)
assert os.environ.get('SLURM_JOB_ID')
assert Path(study.__file__).resolve().is_relative_to(S)
assert (S.parent/'passed.txt').exists()
mode = sys.argv[1]
assert mode in ('prepare', 'run')
if mode == 'prepare':
    assert not R.exists(), 'preserve existing experiment; no implicit reissue'
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
        scope='One Gemini model ID, three paired repeats; one text stdin prompt after observed stream-json multi-turn failure and successful nonfinancial input checks. Prior failures preserved. No completion implied.'))
    print('Frozen native protocol allowlist is ready', flush=True)
    raise SystemExit(0)
assert R.exists(), 'prepare first'
allowlist = study.old.load(R/'relay-allowlist.json')['protocols']
assert len(allowlist) == 3
for row in allowlist:
    p = study.verify(Path(row['root']))
    assert p['model'][0] == 'gemini38-flash' and p['seed'] == row['seed']
    assert study.old.sha(Path(row['root'])/'protocol.json') == row['protocol_sha256']
study.old.write(R/'run-started.json', dict(job=os.environ['SLURM_JOB_ID'], time_utc=datetime.now(timezone.utc).isoformat()))


def run_family(family):
    rows = []
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
            event = dict(family=family, seed=seed, phase=phase, exit_code=code,
                time_utc=datetime.now(timezone.utc).isoformat(), job=os.environ['SLURM_JOB_ID'])
            study.old.write(R/'events'/f'{family}-{seed}-{phase}.json', event)
            print(json.dumps(event), flush=True)
            rows.append(event)
            if code:
                return dict(family=family, status='stopped_after_failure', events=rows)
    return dict(family=family, status='completed_three_pairs', events=rows)


with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(run_family, FAMILIES))
study.old.write(R/'campaign-status.json', dict(results=results,
    complete=all(r['status'] == 'completed_three_pairs' for r in results)))
print('Native roster finished processing; consult every preserved result/failure.', flush=True)
