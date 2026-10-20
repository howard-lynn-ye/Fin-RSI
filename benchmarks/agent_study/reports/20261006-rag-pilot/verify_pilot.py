"""Verify a completed v9 first-date pilot using its frozen source; export no raw text.

Run on a Slurm compute node with PYTHONPATH set to the pilot's source directory.
No inference, rescoring, or edits to original run files are performed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_runtime_v9 as runtime
from benchmarks.agent_study.trading_capabilities import encoded
from benchmarks.agent_study.trading_tools_v6 import execution_status, validate_weights


FAMILIES = ('qwen3-4b', 'granite-8b', 'smollm3-3b')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failure_kind(turn):
    if turn['turn_ok']:
        return None
    if not turn['delivery_ok']:
        return 'output-delivery-failed'
    if turn['parse'] != 'json':
        return turn['parse']
    error = str(turn['result'].get('error', ''))
    if error.startswith('Final stage accepts'):
        return 'research-at-final-stage'
    if 'unexpected keyword argument' in error:
        return 'unsupported-argument'
    if 'required positional argument' in error:
        return 'missing-argument'
    match = re.match(r'([A-Za-z]+(?:Error|Exception)):', error)
    if match:
        return match.group(1)
    return 'other-rejected-call'


def verify(root, runner):
    check((root / 'completed.txt').read_text().strip() == 'V9_RAG_USABILITY_PILOT_COMPLETE',
          'pilot completion marker missing')
    check(Path(study.__file__).resolve().is_relative_to((root / 'source').resolve()),
          'must import the frozen source')
    rows = []
    for family in FAMILIES:
        pair = root / 'pairs' / (family + '-11')
        protocol = study.verify(pair)
        check(protocol['interface'] == 'v9' and protocol['seed'] == 11, 'unexpected protocol')
        qualification = runtime.require_qualification(pair)
        probe = pair / 'gpu-pilot'
        receipt = load(probe / 'completed.json')
        started = load(probe / 'started.json')
        day = load(pair / 'inputs.json')['dates'][0]
        check(receipt['model'] == protocol['model'] and receipt['model'][0] == family,
              'model mismatch')
        check(receipt['date'] == day and receipt['seed'] == 11, 'date/seed mismatch')
        check(receipt['counts_toward_twenty_models'] is False and
              started['counts_toward_twenty_models'] is False, 'pilot mislabelled')
        for binding in (receipt, started):
            check(binding['protocol_sha256'] == sha(pair / 'protocol.json'), 'protocol changed')
            check(binding['script_sha256'] == sha(runner), 'runner changed')
        check(set(receipt['probe']) == {'raw', 'library'}, 'arm inventory differs')
        arms = {}
        for arm in ('raw', 'library'):
            path = probe / (arm + '.json')
            record = load(path)
            runtime.validate_decision(record)
            check(record['interface'] == 'v9', 'wrong decision interface')
            check(record['backend_responses'] == len(record['turns']), 'response count differs')
            check(record['menu_seed'] == 11000, 'menu seed differs')
            holdings = dict.fromkeys(study.md.TICKERS, 0.)
            task = runtime.task(day, holdings, protocol['deadline']) + study.evidence_text(
                load(pair / 'evidence-packets.json')[day])
            history = [dict(role='system', content=runtime.system_prompt(arm)),
                       dict(role='user', content=task)]
            check(record['initial_request'] == history, 'initial task or information differs')
            for index, turn in enumerate(record['turns']):
                history[0]['content'] = runtime.system_prompt(arm, index)
                fingerprint = hashlib.sha256(encoded(history).encode()).hexdigest()
                check(turn['input_messages_sha256'] == fingerprint, 'prompt chain differs')
                status = execution_status(turn['result'])
                check(all(turn[key] == value for key, value in status.items()),
                      'execution/delivery receipt differs')
                check(turn['turn_index'] == index + 1 and turn['calls_remaining'] == 8 - index,
                      'response budget differs')
                check(turn['phase'] == ('final' if index == 7 else 'research'), 'phase differs')
                history += [dict(role='assistant', content=turn['response']),
                            dict(role='user', content=turn['model_feedback'])]
            if record['submitted']:
                normalized, error = validate_weights(record['target'])
                check(error is None and normalized == record['target'], 'invalid accepted weights')
            if not record['decision_completed']:
                check(len(record['turns']) == 8, 'failure before the full response budget')
            summary = dict(action=record['decision_action'], completed=record['decision_completed'],
                           turns=len(record['turns']),
                           failed_turns=sum(not t['turn_ok'] for t in record['turns']),
                           tool_calls=[t['tool'] for t in record['turns']], record_sha256=sha(path))
            check(summary == receipt['probe'][arm], 'derived outcome differs from receipt')
            counts = Counter(filter(None, (failure_kind(t) for t in record['turns'])))
            controller_calls = record.get('tool_calls', [])
            arms[arm] = dict(**summary, failure_categories=dict(counts),
                generation_seconds=sum(t.get('generation_seconds', 0) for t in record['turns']),
                direct_rag_turns=sum(t['tool'] == 'research_context' for t in record['turns']),
                recorded_rag_executions=sum(t.get('tool') == 'research_context'
                                            for t in controller_calls),
                python_turns=sum(t['tool'] == 'run_python' for t in record['turns']),
                final_parse=record['turns'][-1]['parse'],
                final_finish_reason=record['turns'][-1]['finish_reason'])
        rows.append(dict(model=receipt['model'], date=day, seed=11, gpu=receipt['gpu'], arms=arms,
                         protocol_sha256=receipt['protocol_sha256'],
                         completed_receipt_sha256=sha(probe / 'completed.json'),
                         rag_qualification_sha256=sha(pair / 'rag-qualification.json'),
                         rag_qualification_checks=qualification['checks']))
    return dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
                inference_job='1910117', interface='v9', source_commit='ad5a9c790aeb6a3f464bbc06d395e67dd722e159',
                runner_sha256=sha(runner), verifier_sha256=sha(Path(__file__)),
                models=len(rows), decision_opportunities=2 * len(rows), rows=rows,
                counts_toward_twenty_models=False, return_rate_percent=None,
                scope='First-date usability only; failed decisions retained; no returns computed.',
                limitations=['One seed and one date per family, not a return comparison.',
                             'Historical evidence uses declared retrospective availability assumptions.',
                             'No inference rerun or counterfactual trades during verification.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('runner', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = verify(args.root.resolve(), args.runner.resolve())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(report, indent=2))
