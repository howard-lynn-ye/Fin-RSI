"""Post-hoc, target-blind parsing sensitivity analysis of frozen FinQA final answers.

Legacy official execution scores are reproduced unchanged. Numeric prose recovery is a
separate final-answer metric; it is NOT program accuracy, citation correctness or a rerun.
"""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent))
from finqa_rag_ablation import parsed_answer, ANSWER_FORMAT

NUMBER = r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?'


def numeric_final(text):
    """One explicit answer value; no labels, tool receipts or expected answer are inputs."""
    if not isinstance(text, str):
        return {'accepted': False, 'reason': 'missing_final'}
    text = text.strip()
    # Refusal/uncertainty is not converted to a convenient number elsewhere in the prose.
    if re.search(r'cannot|can\s*not|unable|not possible|insufficient|unfortunately|not (?:provided|available)', text, re.I):
        return {'accepted': False, 'reason': 'abstention'}
    patterns = [
        rf'^\s*\$?({NUMBER})\s*(%|percent|million|billion|:1)?\s*\.?\s*$',
        rf'(?:\bis\b|\bwas\b|\bequals\b|\bapproximately\b|\babout\b|=)\s*'
        rf'(?:approximately\s+|about\s+)?\$?({NUMBER})\s*(%|percent|million|billion|:1)?',
        rf'^\s*({NUMBER})\s*(%|percent)\s+of\b',
    ]
    matches = []
    for pattern in patterns:
        for found in re.finditer(pattern, text, re.I):
            value, unit = found.group(1), (found.group(2) or '').lower()
            # A calendar year alone is not an answer, absent an explicit answer marker.
            if not unit and '.' not in value and value.isdigit() and 1900 <= int(value) <= 2100:
                continue
            numeric = Decimal(value.replace(',', ''))
            if unit in ('%', 'percent'):
                numeric /= 100
            elif unit in ('million', 'billion'):
                # Report table denomination cannot be established from the final sentence alone.
                return {'accepted': False, 'reason': 'ambiguous_currency_scale'}
            matches.append((numeric, found.group(0)))
    unique = {value for value, _ in matches}
    if len(unique) != 1:
        return {'accepted': False, 'reason': 'ambiguous_or_no_explicit_numeric_answer',
                'candidate_count': len(unique)}
    return {'accepted': True, 'value': str(next(iter(unique))), 'spans': [s for _, s in matches]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def freeze(inputs, output):
    """Freeze target-blind extraction before loading any gold file in the scorer."""
    output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], {}
    for family in ('qwen', 'mistral'):
        for study in ('rag', 'rerank'):
            root = inputs/f'fin-skills-campaign-finqa-{study}-{family}-20260923-v1'
            receipt_path = root/'inference-receipt.json'
            receipt = json.loads(receipt_path.read_text())
            assert sha(root/'protocol.json') == receipt['protocol_sha256']
            assert len(receipt['rows']) == receipt['planned']
            sources[str(root.name)] = dict(inference_receipt_sha256=sha(receipt_path),
                                          legacy_scores_sha256=sha(root/'scores.json'))
            for item in receipt['rows']:
                path = root/item['file']
                assert sha(path) == item['sha256']
                row = json.loads(path.read_text())
                if study == 'rag' and row['arm'] != 'rag_api':
                    continue
                result = dict(family=family, arm=row['arm'], id=row['id'], file=str(path.relative_to(inputs)),
                    source_sha256=sha(path), final=row['final'], episode_error=row.get('error'),
                    numeric=numeric_final(row['final']), strict=None, fence_v2=None)
                for name, version in (('strict','strict-json-v1'), ('fence_v2',ANSWER_FORMAT)):
                    try:
                        result[name] = parsed_answer(row['final'], answer_format=version)
                    except (ValueError, TypeError):
                        pass
                rows.append(result)
    assert len(rows) == 192
    write(output/'extractions.json', rows)
    write(output/'extraction-receipt.json', dict(rows=192, sha256=sha(output/'extractions.json'),
        parser_sha256=sha(Path(__file__)), sources=sources, numeric_parser='explicit-final-number-v1',
        limits=['Post-hoc parser sensitivity, development-exposed public questions.',
                'Numeric final-answer correctness does not establish program or citation correctness.',
                'Missing finals stay missing; no last-tool-output substitution or model regeneration.',
                'Percentage markers are divided by 100; ambiguous monetary scales are rejected.']))


def score(inputs, output):
    receipt = json.loads((output/'extraction-receipt.json').read_text())
    assert receipt['sha256'] == sha(output/'extractions.json')
    upstream = inputs/'fin-skills-campaign-reuse-bootstrap-20260923-v2'
    evaluator_path = upstream/'upstream/finqa/code/evaluate/evaluate.py'
    manifest = json.loads((upstream/'upstream-manifest.json').read_text())
    assert sha(evaluator_path) == manifest['sources']['finqa']['files']['code/evaluate/evaluate.py']['sha256']
    spec = importlib.util.spec_from_file_location('finqa_official_reanalysis', evaluator_path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    goldpath = upstream/'upstream/finqa/dataset/test.json'
    gold = {r['id']:r for r in json.loads(goldpath.read_text())}
    rows = json.loads((output/'extractions.json').read_text())
    for row in rows:
        assert sha(inputs/row['file']) == row['source_sha256']
        row.update(strict_correct=False, fence_v2_correct=False, numeric_correct=False)
        for name in ('strict', 'fence_v2'):
            if row[name]:
                try:
                    program = official.program_tokenization(row[name]['program'])
                    if not row[name]['program'].strip() or len(program)>81:
                        continue
                    invalid, value = official.eval_program(program, gold[row['id']]['table'])
                    row[name+'_correct'] = bool(invalid == 0 and value == gold[row['id']]['qa']['exe_ans'])
                except Exception:
                    pass
        if row['numeric']['accepted']:
            target = gold[row['id']]['qa']['exe_ans']
            try:
                # FinQA executor rounds numerical outputs to five decimal places.
                row['numeric_correct'] = round(float(row['numeric']['value']),5) == round(float(target),5)
            except (ValueError, TypeError):
                pass
        row['recoverable_final_correct'] = bool(row['fence_v2_correct'] or row['numeric_correct'])
    aggregate = {}
    for family in ('qwen','mistral'):
        for arm in ('rag_api','bge','kev'):
            group = [r for r in rows if r['family']==family and r['arm']==arm]
            assert len(group)==32
            study = 'rag' if arm=='rag_api' else 'rerank'
            legacy = json.loads((inputs/f'fin-skills-campaign-finqa-{study}-{family}-20260923-v1/scores.json').read_text())
            old_correct = sum(r['execution_correct'] for r in legacy['rows'] if r['arm']==arm)
            assert old_correct == sum(r['strict_correct'] for r in group)
            aggregate[family+'-'+arm] = dict(n=32, legacy_correct=old_correct,
                fence_v2_correct=sum(r['fence_v2_correct'] for r in group),
                numeric_parsed=sum(r['numeric']['accepted'] for r in group),
                numeric_correct=sum(r['numeric_correct'] for r in group),
                recoverable_final_correct=sum(r['recoverable_final_correct'] for r in group),
                missing_final=sum(r['final'] is None for r in group),
                parse_rejections=dict(Counter(r['numeric'].get('reason','accepted') for r in group)))
    write(output/'scores.json', dict(rows=rows, aggregate=aggregate,
        extraction_receipt_sha256=sha(output/'extraction-receipt.json'), gold_sha256=sha(goldpath)))
    print(json.dumps(aggregate,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','score'])
    p.add_argument('inputs',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();globals()[a.mode](a.inputs,a.output)
