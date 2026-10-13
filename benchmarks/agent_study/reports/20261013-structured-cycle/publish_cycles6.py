"""Public-ledger verification and Markdown rendering only; no inference or new scoring."""
import hashlib
import json
import os
from pathlib import Path
import sys

from benchmarks.agent_study import multisource_model_study as study

out, batch = map(Path, sys.argv[1:])
summary = study.old.load(out / 'summary.json')
ledgers = study.old.load(out / 'ledgers.json')
expected = {'7b-11', 'granite-8b-11', 'qwen3-4b-11'}
assert summary['all_expected_pairs_complete_and_valid']
assert {row['pair'] for row in summary['pairs']} == expected
assert {(row['pair'], row['arm']) for row in ledgers} == {(p, a) for p in expected for a in ('raw', 'library')}
reference = None
for row in summary['pairs']:
    p = study.verify(batch / row['pair'])
    common = {k: p[k] for k in ('source_sha256', 'data_sha256', 'raw_download_sha256', 'universe',
        'max_turns', 'max_tokens', 'temperature', 'top_p', 'cost_bps', 'cash_interest', 'objective',
        'window', 'initial_capital', 'research_responses', 'final_decision_responses', 'final_output')}
    common['input_hashes'] = p['input_hashes']
    if reference is None:
        reference = common
    assert common == reference, 'models differ in frozen input or execution conditions'
    for arm in ('raw', 'library'):
        ledger, = [r for r in ledgers if r['pair'] == row['pair'] and r['arm'] == arm]
        account = row['paths'][arm]
        assert len(ledger['decisions']) == account['decisions'] == p['decisions_per_arm']
        assert sum(d['action'] == 'failed' for d in ledger['decisions']) == account['states'].get('failed', 0)
        assert sum(d['observation']['library_execution_observed'] for d in ledger['decisions']) == (
            account['tool_usage']['decisions_with_library_execution'])
        assert abs(account['return_rate_pct'] - 100 * (account['ending_capital'] / account['initial_capital'] - 1)) < 1e-10

lines = [
    '# Repaired structured-output full-cycle results', '',
    'All three planned seed-11 pairs completed the 2025-01-03 to 2026-09-25 replay.',
    'Each arm starts with USD 100,000 and has 44 decision dates. Return Rate is cumulative,',
    'after modeled trading costs: `100 * (ending capital / initial capital - 1)`; it is not annualized.',
    'Slurm array 2124818 produced decisions and scores; independent export audit 2124823 rechecked',
    'every original record, model receipt, decoder gate and cash-and-units replay. No decisions were replaced.', '',
    '| Model | No library Return Rate | Library Return Rate | Difference (pp) | Failed dates: raw / library | Library execution dates |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
]
for row in summary['pairs']:
    raw, lib = row['paths']['raw'], row['paths']['library']
    lines.append(f"| {row['model'][1].split('/')[-1]} | {raw['return_rate_pct']:.2f}% | "
        f"{lib['return_rate_pct']:.2f}% | {row['return_difference_pp']:+.2f} | "
        f"{raw['states'].get('failed',0)}/44 / {lib['states'].get('failed',0)}/44 | "
        f"{lib['tool_usage']['decisions_with_library_execution']}/44 |")
lines += [
    '', '## What these results show', '',
    '- Both positive and negative differences are retained; no model, date or failure was dropped.',
    '- A library receipt establishes tool/knowledge execution, not understanding or causal attribution.',
    '  In particular, 7B made no research/library calls, so its return difference cannot establish tool benefit.',
    '- Research-request errors remain in the records even when the final decision succeeds.',
    '  A failed decision is separate from explicit HOLD and never receives a fallback portfolio.',
    '- This is one seed on one development market path for three distinct models, not the required >=20-model study.',
    '  Old protocol runs and other model sizes are not silently added to this repaired cohort.',
    '- Historical archive availability and model training exposure remain unverified. Reference skills are',
    '  current installed knowledge. These are retrospective case results, not unseen-market evidence.', '',
    '## Reproducibility and public records', '',
    '`summary.json` contains full-precision Return Rates, ending capital, completion counts and independent audits.',
    '`ledgers.json` contains every dated target/action, prior holdings, research/receipt observations, daily NAV',
    'and trades. Original-record SHA256 values bind those summaries to private Beacon originals. The export',
    'omits raw model responses, full prompts and evidence text. No original dataset or response was edited.', '',
    'Executable source: `v9-context-20261013-candidate6/source`, code committed at',
    '`9faf01fda23b04ca077d502e36f71c2fee5b05d3`, merged through PR #31. Interaction revision:',
    '`20261013-structured-final-v1`. Frozen run roots remain in',
    '`/beacon-projects/radfm/wy891/fin-multisource-models-20261005/v9-context-model-preflight-20261013-r6`.',
    'Earlier failed interface attempts remain in the sibling `20261013-context-repair` report.', '',
]
(out / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
manifest = dict(slurm_job=os.environ['SLURM_JOB_ID'], passed=True,
    checks=['build_index', 'build_package', 'validate', 'equal_frozen_conditions',
            'all_planned_pairs_present', 'all_public_ledger_dates_present', 'return_identity', 'library_usage_identity'],
    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (out/'summary.json', out/'ledgers.json', out/'README.md')})
study.old.write(out / 'publication-checks.json', manifest)
print(json.dumps(manifest), flush=True)
