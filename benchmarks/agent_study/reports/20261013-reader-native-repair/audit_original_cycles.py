"""Read-only audit using each batch's immutable original source, on Slurm."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('root', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--families', nargs='+', required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source))
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study.audit_model_multisource import audit
from benchmarks.agent_study import trading_preflight_v8 as preflight


def confirm_existing(path, value):
    if not path.exists() or study.old.load(path) != value:
        raise ValueError('independent audit differs from stored receipt: ' + str(path))


pairs, incomplete, invalid = [], [], []
ledgers = []
for family, seed in ((f,s) for f in args.families for s in (11,23,37)):
    root = args.root / f'{family}-{seed}'
    if not (root / 'completed.json').exists() or not (root / 'scores.json').exists() or not (root / 'inference-receipt.json').exists():
        incomplete.append(dict(pair=root.name, reason='no complete score and inference receipt'))
        continue
    try:
        protocol = study.verify(root)
        study.validate_receipt(root)
        preflight.require(root)
        assert protocol['interaction_revision'] in ('20261013-structured-final-v1',
            '20261013-structured-research-v3')
        completed = study.old.load(root / 'completed.json')
        assert completed['scores_sha256'] == study.old.sha(root / 'scores.json')
        assert completed['independent_audit_sha256'] == study.old.sha(root / 'independent-model-audit.json')
        with patch.object(study.old, 'write', confirm_existing):
            verified = audit(root)
        scores = study.old.load(root / 'scores.json')
        public = study.old.load(root / 'public-result.json') if (root / 'public-result.json').exists() else None
        if public:
            assert public['scores_sha256'] == study.old.sha(root / 'scores.json')
            assert public['independent_audit_sha256'] == study.old.sha(root / 'independent-model-audit.json')
        paths = {}
        for arm in ('raw', 'library'):
            states, parse_failures, failures, tools, research_errors = (Counter() for _ in range(5))
            decisions = []
            for path in sorted((root / 'decisions' / arm).glob('*.json')):
                record = study.old.load(path)
                states[record['decision_action']] += 1
                for call in record['tool_calls']:
                    if call.get('ok'):
                        tools[call['tool']] += 1
                for turn in record['turns']:
                    if turn.get('phase') == 'research' and not turn['result'].get('ok'):
                        research_errors[turn['parse']] += 1
                last = record['turns'][-1]
                if record['decision_action'] == 'failed':
                    parse_failures[last['parse']] += 1
                    message = str(last['result'].get('error', ''))
                    kind = ('generation_truncated' if last['finish_reason'] == 'length' else
                            'over_budget' if 'weights sum to' in message else
                            'multiple_actions' if 'multiple-actions' in last['parse'] else
                            'missing_cash' if 'explicit CASH' in message else
                            'final_research' if 'Final stage accepts' in message else
                            'other_validation_or_format')
                    failures[kind] += 1
                decisions.append(dict(date=record['date'], record_sha256=study.old.sha(path),
                    action=record['decision_action'], target=record['target'],
                    holdings_before=record['holdings_before'], observation=preflight.observations(record),
                    response_count=len(record['turns']), final_parse=last['parse'],
                    final_finish_reason=last['finish_reason']))
            account = scores['paths'][arm]
            assert sum(states.values()) == account['decisions']
            assert states['failed'] == account['failed_decisions']
            paths[arm] = dict(initial_capital=account['initial_capital'],
                ending_capital=account['ending_capital'], return_rate_pct=account['return_rate_pct'],
                decisions=account['decisions'], states=states, final_failure_categories=failures,
                final_parse_failures=parse_failures, research_error_parse=research_errors,
                successful_tool_calls=tools, tool_usage=account['tool_usage'])
            ledgers.append(dict(pair=root.name, arm=arm, model=protocol['model'], seed=protocol['seed'],
                interaction_revision=protocol['interaction_revision'], batch=protocol['batch'],
                max_tokens=protocol['max_tokens'], final_output=protocol['final_output'],
                research_output=protocol.get('research_output'), decisions=decisions,
                nav=study.old.load(root / 'nav' / f'{arm}.json')))
        pairs.append(dict(pair=root.name, model=protocol['model'], seed=protocol['seed'],
            interaction_revision=protocol['interaction_revision'], batch=protocol['batch'],
            max_tokens=protocol['max_tokens'], final_output=protocol['final_output'],
            research_output=protocol.get('research_output'), window=protocol['window'],
            paths=paths, return_difference_pp=scores['return_difference_pp'],
            protocol_sha256=study.old.sha(root / 'protocol.json'),
            inference_receipt_sha256=study.old.sha(root / 'inference-receipt.json'),
            scores_sha256=study.old.sha(root / 'scores.json'), independent_audit=verified))
    except Exception as exc:
        invalid.append(dict(pair=root.name, error=f'{type(exc).__name__}: {exc}'))
grouped = defaultdict(list)
for row in pairs:
    grouped[row['model'][0]].append(row)
means = []
for family, rows in sorted(grouped.items()):
    if sorted(row['seed'] for row in rows) != [11, 23, 37]:
        continue
    arms = {}
    for arm in ('raw', 'library'):
        arms[arm] = dict(return_rate_pct=sum(row['paths'][arm]['return_rate_pct'] for row in rows) / 3,
                         failed_decisions=sum(row['paths'][arm]['states'].get('failed', 0) for row in rows),
                         decisions=sum(row['paths'][arm]['decisions'] for row in rows))
    means.append(dict(family=family, seeds=[11,23,37], paths=arms,
                      difference_pp=arms['library']['return_rate_pct']-arms['raw']['return_rate_pct']))
out = dict(observed_utc=datetime.now(timezone.utc).isoformat(), source=str(args.source),
    root=str(args.root), complete_pairs=len(pairs), three_seed_models=len(means),
    means=means, pairs=pairs, incomplete=incomplete, invalid=invalid,
    distinct_models=len({row['model'][1] for row in pairs}),
    all_expected_pairs_complete_and_valid=len(pairs)==3*len(args.families) and not incomplete and not invalid,
    limitation='Original frozen protocols, including failed decisions. Independent accounting passed '
               'only for listed pairs. Historical availability is assumed, not verified vintage. '
               'No return loss is causally attributed to a bug; no counterfactual trades were inserted.')
args.output.mkdir(parents=True, exist_ok=False)
(args.output / 'summary.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
(args.output / 'ledgers.json').write_text(json.dumps(ledgers, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:out[k] for k in ('complete_pairs','three_seed_models','means','incomplete','invalid')}), flush=True)
if invalid:
    raise SystemExit(1)
