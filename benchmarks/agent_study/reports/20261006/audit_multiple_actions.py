"""Count ignored JSON submissions without executing code or recomputing returns."""
import argparse
from collections import Counter
import json
from pathlib import Path

from benchmarks.agent_study.trading_runtime_v6 import extract_call
from benchmarks.agent_study.trading_tools_v6 import validate_weights


def json_calls(text):
    # The qualified decoder retains balanced reasoning markers. Ignore that draft.
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    decoder, position, calls = json.JSONDecoder(), 0, []
    while (start := text.find('{', position)) != -1:
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            position = start + 1
            continue
        position = end
        if isinstance(value, dict) and 'tool' in value:
            calls.append(value)
    return calls


def main(verification, output):
    groups = {}
    for row in json.loads(verification.read_text())['rows']:
        parent = Path(row['source']).parent
        root = parent / ('batch' if parent.name == 'v7-models-v1' else 'pairs') / row['pair']
        for arm in ('raw', 'library'):
            counts = groups.setdefault(f"{row['family']}/{arm}", Counter())
            for path in (root/'decisions'/arm).glob('*.json'):
                record = json.loads(path.read_text())
                for turn in record['turns']:
                    text = turn.get('response', '')
                    if turn.get('tool') is None or extract_call(text)[1] != 'multiple-actions':
                        continue
                    counts['accepted_multiple_action_turns'] += 1
                    calls = json_calls(text)
                    # Count only later standalone JSON submit objects. Do not infer
                    # submissions from Python text, nor execute hypothetical code.
                    for call in calls[1:]:
                        if call.get('tool') != 'submit':
                            continue
                        counts['later_json_submit_objects'] += 1
                        args = call.get('arguments')
                        weights = args.get('weights', args) if isinstance(args, dict) else None
                        target, _ = validate_weights(weights)
                        counts['later_valid_json_submit_objects'] += target is not None
    report = dict(groups=groups, verification=verification.name,
        scope='Static JSON candidates in rejected multi-action responses. Not executed trades, counterfactual returns, or exhaustive Python-submission detection.')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verification', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    main(args.verification, args.output)
