"""Static comparison of frozen parser outputs and a candidate parser; no calls execute."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from benchmarks.agent_study.trading_runtime_v6 import extract_call as frozen_parse


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_objects(text):
    # Static candidates only. A JSON object here is not necessarily an executable
    # response: another action or a generation-length stop may invalidate it.
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    elif '<think>' in text:
        return []
    decoder, position, objects = json.JSONDecoder(), 0, []
    while (start := text.find('{', position)) >= 0:
        try:
            value, position = decoder.raw_decode(text, start)
        except ValueError:
            position = start + 1
            continue
        if isinstance(value, dict) and 'tool' in value:
            objects.append(value)
    return objects


def audit(root, candidate):
    spec = importlib.util.spec_from_file_location('candidate_parser', candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = {}
    for family in ('qwen3-4b', 'granite-8b', 'smollm3-3b'):
        for arm in ('raw', 'library'):
            path = root / 'pairs' / (family + '-11') / 'gpu-pilot' / (arm + '.json')
            record = json.loads(path.read_text())
            counts, changed = Counter(), []
            for turn in record['turns']:
                text = turn['response']
                old_call, old_status = frozen_parse(text)
                new_call, new_status = module.extract_call(text)
                objects = complete_objects(text)
                counts['responses'] += 1
                counts['responses_containing_complete_rag_object'] += any(
                    obj['tool'] == 'research_context' for obj in objects)
                counts['responses_containing_complete_submit_object'] += any(
                    obj['tool'] == 'submit' for obj in objects)
                counts['responses_containing_submit_text'] += bool(
                    re.search(r'"tool"\s*:\s*"submit"', text))
                counts['length_stopped_responses'] += turn['finish_reason'] == 'length'
                counts['empty_think_block_responses'] += bool(re.search(r'<think>\s*</think>', text))
                counts['length_stopped_with_single_parseable_call'] += (
                    turn['finish_reason'] == 'length' and old_call is not None)
                if old_call is not None and new_call is None:
                    changed.append(dict(turn=turn['turn_index'], old_tool=old_call['tool'],
                                        old_parse=old_status, candidate_parse=new_status,
                                        executed_in_original=bool(turn['execution_ok'])))
            groups[family + '/' + arm] = dict(counts=counts, candidate_rejections=changed,
                                              record_sha256=sha(path))
    return dict(verified_at_utc=datetime.now(timezone.utc).isoformat(),
                candidate_parser_sha256=sha(candidate), auditor_sha256=sha(Path(__file__)),
                groups=groups, scope='Static response inspection; no hypothetical trades or returns.',
                definitions={
                    'complete_rag_object': 'Syntactically complete request inside a response; not execution.',
                    'submit_text': 'A textual tool=submit marker, possibly invalid JSON; not a decision.',
                    'candidate_rejections': 'Originally parseable responses rejected by candidate; not reruns.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.candidate)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
