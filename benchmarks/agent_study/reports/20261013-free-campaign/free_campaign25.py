"""Pinned no-paid campaign. Gate on 7B/32B first, retain every model-authored failure."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_preflight_v8 as preflight

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
M = Path('/beacon-projects/radfm/wy891/fin-manual-multisource-20261005')
S = B/'reader-recovery-20261013-candidate24/source'
R = B/'free-campaign-20261013-r25'
W = B/'free-campaign-serial-weights25'
GROUPS = {
    'small': ['qwen-general-7b','7b','qwen3-4b','granite-8b','phi4-mini',
              'olmo3-7b','smollm3-3b','falcon3-10b','mistral12b'],
    'large': ['32b','14b','mistral-small-24b','deepseek-r1-distill-14b'],
    'new-large': ['qwen-general-32b','qwen3-30b-a3b'],
    'new-other': ['qwen-general-14b','phi4-14b','mistral-7b','falcon3-7b','granite-2b'],
}
group = sys.argv[1]
assert group in GROUPS
assert Path(study.__file__).resolve().is_relative_to(S)
assert (S.parent/'passed.txt').exists()
assert study.old.load(B/'reader-canary-20261013-r24/campaign-gate.json')['passed']
for family in ('qwen-general-7b','32b'):
    preflight.require(B/'reader-canary-20261013-r24/pairs'/f'{family}-11')
rows = study.matrix()
active = {r['model']: r for r in rows if r.get('execution_enabled')}
declared = [f for values in GROUPS.values() for f in values]
assert len(declared) == len(set(declared)) == len(active) == 20
assert all(r['transport']=='hf' for r in active.values())
assert set(declared) == set(study.BATCHES['matrix_hf']+study.BATCHES['matrix_reasoning'])
(R/'events').mkdir(parents=True,exist_ok=True)
W.mkdir(exist_ok=True)
job = os.environ['SLURM_JOB_ID']
with (R/f'started-{group}-{job}.json').open('x') as out:
    json.dump(dict(group=group, models=GROUPS[group], job=job, source=str(S),
        time_utc=datetime.now(timezone.utc).isoformat(), paid_calls=False),out)


def event(family, seed, phase, code, **extra):
    out = dict(family=family,seed=seed,phase=phase,exit_code=code,job=job,
        time_utc=datetime.now(timezone.utc).isoformat(),**extra)
    path = R/'events'/f'{family}-{seed}-{phase}-{job}.json'
    with path.open('x') as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out),flush=True)


def cached_weights(family):
    if family in ('7b','14b','32b'):
        return Path('/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub')
    if family == 'mistral12b':
        return Path('/beacon-projects/radfm/wy891/fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub')
    return B/('matrix-weights-v1' if family in ('phi4-mini','granite-8b','smollm3-3b')
              else 'matrix-weights-v2')/'cache/hf/hub'


def stage_public_weights(row):
    # Only the new-model groups use this exclusive cache. Other project/old caches are never cleaned.
    from huggingface_hub import HfApi, snapshot_download
    cache = W/row['id']/'hub'
    cache.mkdir(parents=True,exist_ok=True)
    info = HfApi(token=False).model_info(row['model'],revision=row['revision'],files_metadata=True)
    assert info.sha == row['revision'] and not info.gated and not info.disabled
    files = [x for x in info.siblings if '/' not in x.rfilename and
             x.rfilename.endswith(('.safetensors','.json','.model','.jinja','.tiktoken','.txt'))]
    assert any(x.rfilename.endswith('.safetensors') for x in files)
    assert all(isinstance(x.size,int) for x in files)
    required = sum(x.size for x in files)
    # Reserve space for other jobs and results. Download only one model at a time.
    assert shutil.disk_usage(W).free >= required + 12*1024**3, 'storage gate: insufficient free space'
    snapshot = Path(snapshot_download(row['model'],revision=row['revision'],token=False,
        cache_dir=str(cache),allow_patterns=[x.rfilename for x in files],max_workers=2))
    assert snapshot.resolve().is_relative_to(cache.resolve())
    weights = []
    for name in snapshot.glob('*.safetensors'):
        blob = name.resolve()
        assert blob.is_relative_to(cache.resolve()) and blob.is_file()
        weights.append((name,blob))
    assert weights
    event(row['id'],0,'download',0,revision=row['revision'],bytes=required,
          cache=str(cache),policy='exclusive new-model cache; retain tokenizer and all experiment records')
    return cache, weights


def model_cycles(family, cache):
    batch = 'matrix_reasoning' if family in study.REASONING_FAMILIES else 'matrix_hf'
    for seed in (11,23,37):
        root = R/'pairs'/f'{family}-{seed}'
        tmp = R/'tmp'/f'{family}-{seed}'
        tmp.mkdir(parents=True,exist_ok=True)
        env = dict(os.environ,HF_HUB_CACHE=str(cache),HF_HOME=str(tmp/'hf'),TMPDIR=str(tmp),
            XDG_CACHE_HOME=str(tmp/'xdg'),MPLCONFIGDIR=str(tmp/'matplotlib'),TORCH_HOME=str(tmp/'torch'),
            TORCHINDUCTOR_CACHE_DIR=str(tmp/'inductor'),TRITON_CACHE_DIR=str(tmp/'triton'),
            CUDA_CACHE_PATH=str(tmp/'cuda'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1')
        base = [sys.executable,'-m','benchmarks.agent_study.multisource_model_study']
        if root.exists():
            p = study.verify(root)
            assert p['model'][0] == family and p['seed'] == seed
            assert not list((root/'infrastructure-failures').glob('*.json')), 'Inspect infrastructure failure before resuming'
            if (root/'completed.json').exists():
                study.validate_receipt(root)
                preflight.require(root)
                continue
            # Explicit resume only after actual preflight completed; never silently retry failed preflight.
            preflight.require(root)
            phases = [('resume_cycle',base+['run',str(root)]),('score',base+['score',str(root)])]
        else:
            phases = [('freeze',base+['freeze',str(root),'--batch',batch,'--family',family,
                '--seed',str(seed),'--evidence',str(M/'evidence.json'),'--allow-retrospective','--interface','v9']),
                ('qualify',base+['qualify',str(root)]),
                ('model_preflight',[sys.executable,str(M/'fin-context-reader-preflight24.py'),str(root)]),
                ('run',base+['run',str(root)]),('score',base+['score',str(root)])]
        for phase,command in phases:
            result = subprocess.run(command,cwd=S,env=env)
            event(family,seed,phase,result.returncode,root=str(root))
            if result.returncode:
                return False
    return True


os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1')
for family in GROUPS[group]:
    batch = 'matrix_reasoning' if family in study.REASONING_FAMILIES else 'matrix_hf'
    row = active[study.model_spec(family,11,batch)[1]]
    lock = None
    try:
        weights = []
        if row.get('metadata_checked_utc'):
            lock = (W/'new-model.lock').open('a')
            # A dependency serializes new-model groups. Fail rather than consume a GPU waiting for a lock.
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            cache,weights = stage_public_weights(row)
        else:
            cache = cached_weights(family)
        success = model_cycles(family,cache)
        if weights:
            if not success:
                raise RuntimeError('New-model run failed; keep weights and original evidence for diagnosis')
            # All three cycles have independent accounting receipts. Remove only these new weight blobs.
            for name,blob in weights:
                assert blob.is_relative_to((W/row['id']).resolve())
                if name.is_symlink():
                    name.unlink()
                if blob.exists():
                    blob.unlink()
            event(family,0,'release_own_weights',0,files=len(weights))
        event(family,0,'model_finished',0 if success else 1,complete_three_seeds=success)
    except Exception as exc:
        event(family,0,'campaign_failure',1,error=f'{type(exc).__name__}: {exc}')
        if row.get('metadata_checked_utc'):
            raise  # Preserve limited storage after a failed new model; no blind next download.
    finally:
        if lock is not None:
            lock.close()
print('Assigned group processed; inspect success/failure events. No claim of twenty completed models.',flush=True)
