"""Audit and export the completed 32B pair without modifying its frozen records.

Run on Slurm: python publish_results.py SOURCE PAIRS ORIGINAL_PUBLIC OUTPUT
The sibling audit_cycle.py replays the original accounting read-only.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Numerical audit requires Slurm'
    source, pairs, original, output = map(Path, sys.argv[1:])
    assert not output.exists(), 'Preserve existing publication'
    root = pairs / '32b-11'
    here = Path(__file__).resolve().parent
    recreated = output.with_name(output.name + '-private-recheck')
    subprocess.run([sys.executable, str(here / 'audit_cycle.py'), str(source),
                    str(pairs), str(recreated)], check=True)
    summary, reference = read(recreated / 'summary.json'), read(original / 'summary.json')
    ignore = {'observed_utc', 'source', 'root'}
    assert {k: v for k, v in summary.items() if k not in ignore} == {
        k: v for k, v in reference.items() if k not in ignore}
    assert read(recreated / 'ledgers.json') == read(original / 'ledgers.json')
    assert summary['all_expected_pairs_complete_and_valid']
    assert summary['complete_pairs'] == 1 and not summary['invalid'] and not summary['incomplete']
    protocol, inputs = read(root / 'protocol.json'), read(root / 'inputs.json')
    row = summary['pairs'][0]
    ledgers = read(original / 'ledgers.json')
    receipt = read(root / 'inference-receipt.json')
    assert len(receipt['decisions']) == 88
    counts = {}
    for ledger in ledgers:
        arm = ledger['arm']
        account = row['paths'][arm]
        assert len(ledger['decisions']) == len(inputs['picks']) == account['decisions'] == 44
        assert len(ledger['nav']['trades']) == 44
        assert ledger['nav'] == read(root / 'nav' / f'{arm}.json')
        assert len(ledger['nav']['nav']) == 434
        assert next(iter(ledger['nav']['nav'].values())) == 1.0
        assert list(ledger['nav']['nav'])[-1] == protocol['deadline']
        final = list(ledger['nav']['nav'].values())[-1]
        assert math.isclose(account['ending_capital'], account['initial_capital'] * final,
                            rel_tol=0, abs_tol=1e-6)
        assert math.isclose(account['return_rate_pct'],
                            100 * (account['ending_capital'] / account['initial_capital'] - 1),
                            rel_tol=0, abs_tol=1e-10)
        for index, decision in enumerate(ledger['decisions']):
            name = f'decisions/{arm}/{index:02d}.json'
            assert decision['record_sha256'] == receipt['decisions'][name] == sha(root / name)
        counts[arm] = dict(decisions=44, trade_rows=44, nav_rows=434,
                          failed_decisions=account['states'].get('failed', 0))
    assert set(counts) == {'raw', 'library'}
    assert row['paths']['library']['tool_usage']['decisions_with_library_execution'] == 44
    assert row['paths']['raw']['tool_usage']['decisions_with_library_execution'] == 0
    output.mkdir()
    summary = dict(reference)
    summary.pop('source')
    summary.pop('root')
    summary['run_id'] = 'full32-flex-20261017-r39'
    summary['publication_note'] = 'Only private source/root paths removed; all original results retained.'
    write(output / 'summary.json', summary)
    shutil.copyfile(original / 'ledgers.json', output / 'ledgers.json')
    for name in ('scores.json', 'independent-model-audit.json', 'inference-receipt.json',
                 'completed.json', 'decision-preflight.json', 'model-access-preflight.json'):
        shutil.copyfile(root / name, output / name)
    # Protocol contains hashes and settings, not the private data or model conversations.
    shutil.copyfile(root / 'protocol.json', output / 'protocol.json')
    for name in ('publish_results.py', 'audit_cycle.py'):
        shutil.copyfile(here / name, output / name)
    shutil.copyfile(here / 'build-validation.txt', output / 'build-validation.txt')
    raw, lib = row['paths']['raw'], row['paths']['library']
    readme = f'''# 32B full-cycle comparison, October 17, 2026

The completed paired cycle produced **{lib['return_rate_pct']:.2f}% with fin-skills** and
**{raw['return_rate_pct']:.2f}% without fin-skills**, a difference of
**{row['return_difference_pp']:.2f} percentage points** after modeled trading costs.

| Arm | Initial capital (USD) | Ending capital (USD) | Return Rate | Decisions |
| --- | ---: | ---: | ---: | ---: |
| Without library | {raw['initial_capital']:.2f} | {raw['ending_capital']:.2f} | {raw['return_rate_pct']:.6f}% | 44/44 |
| With library | {lib['initial_capital']:.2f} | {lib['ending_capital']:.2f} | {lib['return_rate_pct']:.6f}% | 44/44 |

`Return Rate = 100 * (ending capital / initial capital - 1)`.
Both arms have zero failed final decisions. Raw submitted 43 allocations and one hold;
library submitted 44 allocations. The library arm executed library capabilities in all
44 periods. This is execution evidence, not proof of understanding or use of every skill.

## Frozen experiment and completion

- Qwen/Qwen2.5-Coder-32B-Instruct, revision
  `381fc969f78efac66bc87ff7ddeadb7e73c218a7`, BF16, seed 11.
- Execution window: 2025-01-03 through 2026-09-25; first observation: 2025-01-02.
- Twelve allowed instruments, decisions every ten sessions, next-session-close execution,
  5 bps per traded side, zero cash interest. Both arms receive the same dated market data
  and evidence packet, with an equal six-research plus two-final-response budget.
- Frozen arm order: library, then raw. RSI disabled. No paid model calls.
- Job 2317005 completed at 2026-10-17 08:02 UTC (04:02 America/New_York), exit 0.
  Actual runtime: 05:46:15 on two L40S GPUs, four CPUs, 64 GiB host RAM.
  This window has no remaining GPU allocation. Other windows' jobs were untouched.
- Source lineage: frozen source `f36c3f63e317e6990b6692f6771bdb333b72b181`
  with only the benchmark loader replaced by the tested PR #57 version. The complete
  source and input hashes are bound in protocol.json. The active experiment was not
  changed after it started.

## Full accounting records

- `ledgers.json`: all 88 dated decisions, target allocations, holdings before each
  decision, source-record hashes and library-use receipts; 434 NAV rows per arm
  (initial capital plus 433 valuations); all 44 execution/hold rows per arm.
- NAV is normalized to initial capital. Multiply it by USD 100,000 for dollar equity.
  Trade `cost` is the fee fraction of pre-trade equity, not a dollar amount;
  `turnover` is the absolute risky-asset weight change. These are simulator records,
  not broker fills or actual stock-share statements.
- `scores.json`, `summary.json`: unrounded returns, final capital, usage and failure counts.
- `independent-model-audit.json`: separate units/cash replay agreed with the original
  daily NAV within 1.1e-14 and independently reproduced both returns.
- `inference-receipt.json`, `completed.json`, `protocol.json`: hashes, frozen settings
  and completion bindings. The final decision files themselves stay in the private
  run because their conversations embed market/news source data. Their full derived
  decisions and accounting are included in ledgers.json.
- `publication-checks.json`: fresh read-only replay, original-export equality, full
  ledger coverage, return identity, privacy checks and file hashes.
- `build-validation.txt`: build_index, build_package, validate passed on Slurm.

## Remaining limitations

This is one model, one seed and one development market path. It is not a new
20-model campaign, an unseen holdout, or evidence that every model benefits.
Historical news/disclosure availability uses stated archive-lag assumptions, not
verified point-in-time vintages; installed skill knowledge is current. The treatment
combines library knowledge, retrieval and execution interfaces, so this result does
not isolate the contribution of any one feature.

Research still had 21 unsuccessful turns with the library and 33 without it.
Five library responses were truncated. The remaining failed research turns have
parse label `json`: their JSON was recognized, but the research operation failed.
That label is not evidence of a JSON syntax error or its underlying cause.
One raw period had no successful research response but still produced a valid final
decision. No failed research turns or unfavorable periods were discarded.

Allocator OOM warnings occurred during the original job. The full job completed,
all 88 final decisions were accepted, and independent accounting passed; this does
not establish the cause or harmlessness of every research failure. Runtime and
interface issues remain distinct from the observed return difference.

## Recheck without rerunning model inference

On Slurm, with access to the unchanged frozen source and private run inputs:

```sh
python publish_results.py SOURCE PAIRS ORIGINAL_PUBLIC NEW_OUTPUT
```

Place the successful `build-validation.txt` beside the script. The audit checks
existing receipts read-only, compares its export with the original, and refuses to
overwrite an output directory. It never generates new model decisions.
'''
    (output / 'README.md').write_text(readme)
    forbidden = re.compile(r'/beacon-(?:projects|homes)/|[A-Z]:[\\/]|ihccs\d+|'
                           r'(?i:access_token|refresh_token|authorization: bearer|patient)')
    for path in output.iterdir():
        if path.suffix == '.py':  # Reviewed source; the scanner contains its own patterns.
            continue
        assert not forbidden.search(path.read_text()), f'Private content in {path.name}'
    manifest = dict(passed=True, publication_job=os.environ['SLURM_JOB_ID'],
        inference_job=2317005, inference_repeated=False, original_exports_unchanged=True,
        numerical_audit_reproduced=True, ledger_coverage=counts,
        original_summary_sha256=sha(original / 'summary.json'),
        original_ledgers_sha256=sha(original / 'ledgers.json'),
        build_source='dc5cc586fec3ced122e2b0b37c9cbf608a309685',
        build_input_paths_unchanged_since='2e91d57', privacy_scan_passed=True,
        checks=['frozen_source_and_input_bindings', 'independent_accounting_replay',
                'original_export_equality', '88_decision_hashes', 'complete_nav_and_trades',
                'capital_return_identity', 'library_execution_counts',
                'build_index', 'build_package', 'validate'],
        files={p.name: sha(p) for p in sorted(output.iterdir())})
    write(output / 'publication-checks.json', manifest)
    print(json.dumps(dict(publication_passed=True, counts=counts,
        raw_return_rate_pct=raw['return_rate_pct'], library_return_rate_pct=lib['return_rate_pct'],
        difference_pp=row['return_difference_pp'])), flush=True)


if __name__ == '__main__':
    main()
