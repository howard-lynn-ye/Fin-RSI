"""One pinned free HF model; stages may run one pair or all three paired seeds.

No scheduler submissions, paid transport, downloads, retries or cache eviction.
Existing complete pairs are verified; incomplete pairs require reviewed recovery.
Every successful pair exports the full derived ledger without private source text.
"""
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

SEEDS = (11, 23, 37)


def completion_summary(spec, scores):
    """A short scheduling stage must never be counted as a complete model."""
    if sorted(r['seed'] for r in scores) != list(SEEDS):
        return None
    if any(r['family'] != spec[0] for r in scores):
        raise ValueError('results belong to another model')
    return dict(model=spec, seeds=SEEDS, complete_pairs=len(scores),
        counts_toward_twenty_models=True,
        return_rate_pct_mean={a:sum(r['paths'][a]['return_rate_pct'] for r in scores)/len(SEEDS)
                              for a in ('raw','library')}, historical_availability_verified=False)


def selection(family):
    from benchmarks.agent_study import multisource_model_study as study
    batch = 'matrix_reasoning' if family in study.REASONING_FAMILIES else 'matrix_hf'
    spec = study.model_spec(family, SEEDS[0], batch)
    row = next(r for r in study.matrix() if r['model'] == spec[1])
    if row['transport'] != 'hf' or not row.get('execution_enabled') or row['revision'] != spec[2]:
        raise ValueError('only enabled pinned open-weight HF models are allowed')
    return batch, spec, row


def check_cached_snapshot(cache, model, revision):
    """Metadata-only cache gate. Model loading and real acceptance remain mandatory."""
    snapshot = Path(cache)/('models--'+model.replace('/', '--'))/'snapshots'/revision
    config = snapshot/'config.json'
    if not config.is_file():
        raise ValueError('pinned model config missing from offline cache')
    index = snapshot/'model.safetensors.index.json'
    names = (set(json.loads(index.read_text())['weight_map'].values()) if index.is_file()
             else {'model.safetensors'})
    if not names or any(Path(n).name != n or not (snapshot/n).is_file() for n in names):
        raise ValueError('pinned weight shard missing from offline cache')
    if not (snapshot/'tokenizer_config.json').is_file():
        raise ValueError('pinned tokenizer config missing from offline cache')
    return dict(model=model, revision=revision, weight_shards=len(names),
                weight_bytes=sum((snapshot/n).stat().st_size for n in names),
                scope='metadata and file existence; not a weight hash or model qualification')


def export_pair(root, destination):
    from benchmarks.agent_study import multisource_model_study as study
    from benchmarks.agent_study.trading_preflight_v8 import observations
    from scripts.frozen_cycle_resume import write_once
    p = study.verify(root)
    inputs = study.validate_receipt(root)
    complete = study.old.load(root/'completed.json')
    for file, key in [('scores.json','scores_sha256'),
                      ('independent-model-audit.json','independent_audit_sha256')]:
        if complete[key] != study.old.sha(root/file):
            raise ValueError('completion receipt mismatch')
    if not study.old.load(root/'independent-model-audit.json')['passed']:
        raise ValueError('independent accounting failed')
    ledgers = []
    for arm in inputs['arms']:
        decisions = []
        for i in range(len(inputs['picks'])):
            path = root/'decisions'/arm/f'{i:02d}.json'
            r = study.old.load(path)
            decisions.append(dict(date=r['date'], action=r['decision_action'], target=r['target'],
                holdings_before=r['holdings_before'], observation=observations(r),
                record_sha256=study.old.sha(path)))
        ledgers.append(dict(pair=root.name, arm=arm, model=p['model'], seed=p['seed'],
            interaction_revision=p['interaction_revision'], decisions=decisions,
            nav=study.old.load(root/'nav'/f'{arm}.json')))
    write_once(destination/'ledgers.json', ledgers)
    for file in ('protocol.json','scores.json','inference-receipt.json',
                 'independent-model-audit.json','completed.json'):
        write_once(destination/file, study.old.load(root/file))
    return study.old.load(root/'scores.json')


@contextmanager
def model_lock(destination):
    import fcntl
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'model.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(family, output, evidence, cache, seed=None):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    if seed is not None and seed not in SEEDS:
        raise ValueError('seed is outside the declared repeats')
    from benchmarks.agent_study import multisource_model_study as study
    from scripts.frozen_cycle_resume import write_once
    from benchmarks.agent_study.trading_runtime_v8 import INTERACTION_REVISION
    if INTERACTION_REVISION != '20261017-research-environment-v11':
        raise ValueError('wrong frozen interaction revision')
    batch, spec, row = selection(family)
    cached = check_cached_snapshot(cache, spec[1], spec[2])
    output, evidence = Path(output), Path(evidence)
    os.environ.update(HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1')
    with model_lock(output/family):
        write_once(output/family/'plan.json', dict(model=spec, family=row['family'],
            seeds=SEEDS, paired_arms=['raw','library'], paid_calls=False, cache_check=cached,
            interaction_revision=INTERACTION_REVISION,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        scores = []
        # Reuse the tested atomic writer. It preserves identical scoring artifacts and
        # refuses different contents; killed writes cannot leave half a decision JSON.
        with patch.object(study.old, 'write', write_once):
            for current_seed in SEEDS if seed is None else (seed,):
                root = output/'pairs'/f'{family}-{current_seed}'
                if root.exists():
                    study.verify(root)
                    if not (root/'completed.json').is_file():
                        raise ValueError('incomplete existing pair requires reviewed recovery; no automatic retry')
                else:
                    study.freeze(root, family, current_seed, evidence, allow_retrospective=True,
                                 batch=batch, interface='v9')
                    subprocess.run([sys.executable, '-m',
                        'benchmarks.agent_study.multisource_model_study', 'qualify', str(root)], check=True)
                    study.run(root)  # All existing real-tool, decoder and research gates stay enabled.
                    study.score(root)
                scores.append(export_pair(root, output/'public'/root.name))
                print(json.dumps(dict(event='paired_cycle_complete', model=spec[1], seed=current_seed,
                    return_rate_pct={a:scores[-1]['paths'][a]['return_rate_pct'] for a in ('raw','library')})), flush=True)
                gc.collect()
                import torch
                torch.cuda.empty_cache()
        # Revalidate all completed repeats, including those produced in earlier allocations.
        scores = [export_pair(output/'pairs'/f'{family}-{s}', output/'public'/f'{family}-{s}')
                  for s in SEEDS if (output/'pairs'/f'{family}-{s}'/'completed.json').is_file()]
        summary = completion_summary(spec, scores)
        if summary is not None:
            write_once(output/family/'complete.json', summary)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--seed', type=int, choices=SEEDS, help='one full pair; all repeats remain required')
    a = parser.parse_args()
    run(a.family, a.output, a.evidence, a.cache, a.seed)
