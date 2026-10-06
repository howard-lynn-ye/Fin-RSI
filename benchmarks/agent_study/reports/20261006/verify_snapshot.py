"""Read-only validation of frozen model runs; execute on a Slurm compute node."""
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
SEEDS = {11, 23, 37}

def worker(source, parent):
    sys.path.insert(0, str(source))
    from benchmarks.agent_study import multisource_model_study as study
    rows, incomplete, failures = [], [], []
    for root in sorted(parent.iterdir()):
        if not (root / 'protocol.json').is_file():
            continue
        if not (root / 'completed.json').is_file():
            incomplete.append(dict(pair=root.name, decisions={
                arm: len(list((root / 'decisions' / arm).glob('*.json')))
                for arm in ('raw', 'library')}))
            continue
        try:
            p = study.verify(root)
            inputs = study.validate_receipt(root)
            load, sha = study.old.load, study.old.sha
            done, audit, score = [load(root / f) for f in (
                'completed.json', 'independent-model-audit.json', 'scores.json')]
            assert done['scores_sha256'] == sha(root / 'scores.json')
            assert done['independent_audit_sha256'] == sha(root / 'independent-model-audit.json')
            assert audit['passed'] is True
            assert audit['protocol_sha256'] == sha(root / 'protocol.json')
            assert audit['scores_sha256'] == sha(root / 'scores.json')
            assert audit['inference_receipt_sha256'] == sha(root / 'inference-receipt.json')
            assert score['inference_receipt_sha256'] == sha(root / 'inference-receipt.json')
            assert (score['family'], score['seed']) == (p['model'][0], p['seed'])
            assert set(inputs['arms']) == {'raw', 'library'}
            for arm in inputs['arms']:
                assert score['paths'][arm]['decisions'] == len(inputs['picks']) == 44
                assert abs(audit['results'][arm]['return_rate_pct'] -
                           score['paths'][arm]['return_rate_pct']) < 1e-10
            rows.append(dict(pair=root.name, family=score['family'], seed=score['seed'],
                model=p['model'][1:], source=str(source), window=score['window'],
                paths=score['paths'], difference_pp=score['return_difference_pp'],
                scores_sha256=done['scores_sha256'], audit_sha256=done['independent_audit_sha256'],
                common={k: p[k] for k in ('version','source_sha256','data_sha256',
                    'raw_download_sha256','universe','max_turns','max_tokens','temperature',
                    'top_p','execution','cost_bps','cash_interest','objective','primary_report',
                    'window','decisions_per_arm','treatment')},
                evidence_sha256=p['input_hashes']['evidence.json'],
                packets_sha256=p['input_hashes']['evidence-packets.json']))
        except Exception as error:
            failures.append(dict(pair=root.name, error=f'{type(error).__name__}: {error}'))
    print(json.dumps(dict(rows=rows, incomplete=incomplete, failures=failures)))

def main():
    if len(sys.argv) == 4 and sys.argv[1] == 'worker':
        worker(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    sets = [('v7-models-v1','batch'), ('matrix-replay-v1','pairs'), ('matrix-replay-v2','pairs')]
    all_rows, incomplete, failures = [], [], []
    for directory, pairs in sets:
        source, parent = B / directory / 'source', B / directory / pairs
        env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, __file__, 'worker', str(source), str(parent)],
                                cwd=source, env=env, text=True, capture_output=True, check=True)
        parsed = json.loads(result.stdout)
        all_rows.extend(parsed['rows'])
        incomplete.extend(parsed['incomplete'])
        failures.extend(parsed['failures'])
    groups = {}
    for row in all_rows:
        groups.setdefault(row['family'], []).append(row)
    summary = {}
    for family, rows in groups.items():
        assert len({r['seed'] for r in rows}) == len(rows), 'duplicate seed'
        for r in rows[1:]:
            for key in ('common','model','evidence_sha256','packets_sha256'):
                assert r[key] == rows[0][key], f'mixed {key} in {family}'
        returns = {arm: statistics.mean(r['paths'][arm]['return_rate_pct'] for r in rows)
                   for arm in ('raw','library')}
        summary[family] = dict(seeds=sorted(r['seed'] for r in rows),
            complete={r['seed'] for r in rows} == SEEDS,
            return_rate_pct=returns, difference_pp=returns['library'] - returns['raw'],
            submitted={arm: sum(r['paths'][arm]['submitted'] for r in rows) for arm in returns},
            decisions_per_arm=44*len(rows),
            failed_turns={arm: sum(r['paths'][arm]['failed_turns'] for r in rows) for arm in returns},
            seed_returns=[dict(seed=r['seed'], raw=r['paths']['raw']['return_rate_pct'],
                library=r['paths']['library']['return_rate_pct'], difference_pp=r['difference_pp'])
                for r in rows])
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        completed_models=sum(s['complete'] for s in summary.values()),
        verified_pairs=len(all_rows), summary=summary, incomplete=incomplete, failures=failures,
        rows=all_rows, limits='Development path; incomplete evidence vintages. Means only within model, not a cross-model causal estimate.')
    destination = B / f'verified-status-{os.environ["SLURM_JOB_ID"]}.json'
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))
    print(f'SAVED {destination}', flush=True)
    if failures:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
