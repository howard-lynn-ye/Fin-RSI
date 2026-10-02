"""Post-hoc, arm-blind code-extraction sensitivity analysis of the frozen code-generation study.

The pre-registered parser accepts either the whole response or exactly one Markdown fence that
spans the whole response; prose after the fence therefore compiles as Python and fails. Those
official scores are reproduced unchanged. This analysis separately takes the first complete
fence and discards surrounding prose. It never edits code, regenerates a response, chooses
among programs or looks at a grade before the extraction is bound by hash. Execution reuses
the frozen worker, sandbox and grader; only the host environment differs, and the programs
that the official parser already accepted are re-run as an environment consistency check.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.agent_study.codegen_utility import (  # noqa: E402
    MODELS, audit, dataset, execute, parse_code)

EXTRACTOR = 'first-complete-fence-v1'
FENCE = re.compile(r'```[^\n`]*\r?\n(.*?)\r?\n[ \t]*```', re.DOTALL)


def first_fence(text):
    """Code in the first complete fence; unfenced text is kept as-is, like the official parser."""
    if not isinstance(text, str) or not text.strip():
        return '', 'empty'
    found = FENCE.search(text.strip())
    if not found:
        return text.strip(), 'no_fence'
    return found.group(1), 'first_fence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as out:
        json.dump(value, out, indent=2, allow_nan=False)


def load_batch(fetched):
    """Verify the downloaded batch end to end before anything is extracted or executed."""
    receipt = json.loads((fetched/'fetch-receipt.json').read_text())
    assert receipt['complete'] and receipt['all_match']
    protocol_sha = sha(fetched/'protocol.json')
    protocol = json.loads((fetched/'protocol.json').read_text())
    assert sha(fetched/'inputs.json') == protocol['input_sha256']
    inference = json.loads((fetched/'inference-receipt.json').read_text())
    assert inference['protocol_sha256'] == protocol_sha
    assert len(inference['rows']) == inference['planned'] == protocol['planned'] == 48
    items = {r['id']: r for r in json.loads((fetched/'inputs.json').read_text())}
    assert set(items) == {r['id'] for r in inference['rows']}
    official = json.loads((fetched/'scores.json').read_text())
    assert official['inference_receipt_sha256'] == sha(fetched/'inference-receipt.json')
    for rec in inference['rows']:
        assert sha(fetched/rec['path']) == rec['sha256']
        assert sha(fetched/'audits'/(rec['id']+'.json')) == official['audits'][rec['id']]
    return inference, items


def freeze(fetched, output):
    inference, items = load_batch(fetched)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for rec in inference['rows']:
        item = items[rec['id']]
        text = json.loads((fetched/rec['path']).read_text())['response']
        text = text['choices'][0]['message']['content']
        official, (code, status) = parse_code(text), first_fence(text)
        stripped = text.strip()
        if FENCE.fullmatch(stripped) or status == 'no_fence':
            # Where the official parser already isolated code, the two extractions coincide.
            assert code == official, rec['id']
        found = FENCE.search(stripped)
        rows.append(dict(id=rec['id'], family=item['family'], arm=item['arm'], task=item['task'],
            seed=item['seed'], response_sha256=rec['sha256'], status=status,
            official_code_sha256=digest(official), code_sha256=digest(code),
            changed=code != official,
            leading_prose_chars=len(stripped[:found.start()].strip()) if found else 0,
            trailing_prose_chars=len(stripped[found.end():].strip()) if found else 0,
            mentions_fin_skills='fin_skills' in text, code=code))
    write(output/'extractions.json', rows)
    write(output/'extraction-receipt.json', dict(rows=len(rows), extractor=EXTRACTOR,
        sha256=sha(output/'extractions.json'), script_sha256=sha(Path(__file__)),
        fetch_receipt_sha256=sha(fetched/'fetch-receipt.json'),
        inference_receipt_sha256=sha(fetched/'inference-receipt.json'),
        official_scores_sha256=sha(fetched/'scores.json'),
        changed=sum(r['changed'] for r in rows),
        limits=['Post-hoc parser sensitivity; the pre-registered endpoint is the official one.',
                'Only prose outside the first complete fence is discarded; code is never edited.',
                'Bound by hash before execution; no response is regenerated or selected.',
                'Executed on another Linux host with the same worker, sandbox and grader.']))


def _strip(result):
    """Drop weight matrices from saved rows but keep a digest so equality stays checkable."""
    if isinstance(result, dict):
        out = {}
        for key, value in result.items():
            if key == 'weights':
                out['weights_sha256'] = digest(json.dumps(value))
            else:
                out[key] = _strip(value)
        return out
    if isinstance(result, list):
        return [_strip(v) for v in result]
    return result


def score(fetched, output, workdir, budget=None):
    receipt = json.loads((output/'extraction-receipt.json').read_text())
    assert receipt['sha256'] == sha(output/'extractions.json')
    assert receipt['fetch_receipt_sha256'] == sha(fetched/'fetch-receipt.json')
    load_batch(fetched)
    (workdir/'tmp').mkdir(parents=True, exist_ok=True)
    # Per-row checkpoints outside the evidence folder let a time-limited host resume.
    done = workdir/'rows'/receipt['sha256']
    done.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = []
    for row in json.loads((output/'extractions.json').read_text()):
        if (done/(row['id']+'.json')).exists():
            results.append(json.loads((done/(row['id']+'.json')).read_text()))
            continue
        if budget and time.monotonic() - started > budget:
            print(json.dumps(dict(phase='posthoc-audit', paused=len(results), planned=48)))
            return
        data = dataset(row['seed'])
        initial = execute(row['code'], dict(data, audit_guard=True), workdir)
        if initial.get('infrastructure_error'):
            raise RuntimeError(f"Confinement/runtime unavailable for {row['id']}: {initial}")
        check = audit(row['code'], data, row['task'], initial, workdir)
        official = json.loads((fetched/'audits'/(row['id']+'.json')).read_text())
        same = None
        if official['execution']['executed'] and not row['changed']:
            # Environment check: identical code must reproduce the official Beacon grades.
            oa = official['audit']
            same = bool(initial['executed'] and np.allclose(
                initial['weights'], official['execution']['weights'], atol=1e-12, rtol=0)
                and check['split_invariant'] == oa['split_invariant']
                and check['future_invariant'] == oa['future_invariant']
                and check['numerically_correct'] == oa['numerically_correct'])
        results.append(dict(id=row['id'], family=row['family'], arm=row['arm'],
            task=row['task'], seed=row['seed'], changed=row['changed'],
            official_executed=official['execution']['executed'],
            official_error=official['execution'].get('error', {}).get('type', 'none'),
            official_financially_valid=official['audit'].get('financially_valid', False),
            reproduces_official=same, execution=_strip(initial), audit=_strip(check)))
        write(done/(row['id']+'.json'), results[-1])
        print(json.dumps(dict(phase='posthoc-audit', completed=len(results), planned=48)),
              flush=True)
    summary = {}
    for family, _, _ in MODELS:
        for arm in ('raw', 'library'):
            group = [r for r in results if r['family'] == family and r['arm'] == arm]
            summary[family+'-'+arm] = dict(planned=len(group),
                official_executed=sum(r['official_executed'] for r in group),
                official_financially_valid=sum(r['official_financially_valid'] for r in group),
                extraction_changed=sum(r['changed'] for r in group),
                executed=sum(r['execution']['executed'] for r in group),
                numerical=sum(r['audit']['numerically_correct'] for r in group),
                future_failures=sum(r['audit']['future_invariant'] is False for r in group),
                split_failures=sum(r['audit']['split_invariant'] is False for r in group),
                financial_valid=sum(r['audit'].get('financially_valid', False) for r in group),
                algorithm_users=sum(any(t['kind'] == 'algorithm'
                                        for t in r['execution'].get('trace', [])) for r in group),
                agent_guard_users=sum(any(t['kind'] == 'agent_guard'
                                          for t in r['execution'].get('trace', []))
                                      for r in group),
                posthoc_guard_rejections=sum(r['execution'].get(
                    'posthoc_causality_guard', {}).get('passed') is False for r in group),
                errors=dict(Counter(r['execution'].get('error', {}).get('type', 'none')
                                    for r in group)))
    checked = [r['reproduces_official'] for r in results if r['reproduces_official'] is not None]
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy')}
    write(output/'scores.json', dict(aggregate=summary, rows=results,
        extraction_receipt_sha256=sha(output/'extraction-receipt.json'),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         packages=versions, host='local Linux VM, not the Beacon node'),
        environment_check=dict(compared=len(checked), reproduced=sum(checked))))
    print(json.dumps(dict(aggregate=summary, environment_check=dict(
        compared=len(checked), reproduced=sum(checked))), indent=1))


def diagnose(output, workdir):
    """Locate the earliest weight row that moves under each failed future-price probe."""
    scores = json.loads((output/'scores.json').read_text())
    codes = {r['id']: r for r in json.loads((output/'extractions.json').read_text())}
    (workdir/'tmp').mkdir(parents=True, exist_ok=True)
    rows = []
    for row in scores['rows']:
        if row['audit'].get('future_invariant') is not False:
            continue
        item = codes[row['id']]
        data = dataset(item['seed'])
        base = np.array(execute(item['code'], data, workdir)['weights'])
        probes = []
        for k in (32, 64):
            changed = np.array(data['prices'])
            changed[k:] *= np.linspace(1.15, 1.9, len(changed)-k)[:, None]
            moved = np.array(execute(item['code'], dict(data, prices=changed.tolist()),
                                     workdir)['weights'])
            bad = np.flatnonzero(np.abs(moved[:k+1] - base[:k+1]).max(axis=1) > 1e-8)
            first = int(bad[0]) if len(bad) else None
            probes.append(dict(k=k, first_changed_row=first,
                               kind=None if first is None else
                               'same_session' if first == k else 'strictly_future'))
        kinds = {p['kind'] for p in probes if p['kind']}
        guard = row['execution'].get('posthoc_causality_guard', {}).get('passed')
        rows.append(dict(id=row['id'], probes=probes, posthoc_guard_passed=guard,
                         kind=('strictly_future' if 'strictly_future' in kinds
                               else 'same_session')))
    summary = Counter(f"{r['kind']}|guard_passed={r['posthoc_guard_passed']}" for r in rows)
    # Static convention check: split_factors are already cumulative in the prompt.
    audits = {r['id']: r['audit'] for r in scores['rows']}
    recumulated = Counter()
    for item in codes.values():
        if re.search(r'split_factors\s*\.\s*cumprod', item['code']):
            recumulated[f"{item['family']}-{item['arm']}|split_invariant="
                        f"{audits[item['id']]['split_invariant']}"] += 1
    write(output/'diagnostics.json', dict(timing_rows=rows,
        scores_sha256=sha(output/'scores.json'),
        timing_summary=dict(summary), recumulated_split_factors=dict(recumulated),
        note='Timing: rows up to and including k are compared, as in the grader; the library '
             'guard compares rows strictly before k, so it tests feature causality, not the '
             'one-bar execution lag. Split: counts code that applies cumprod to factors that '
             'the prompt defines as already cumulative.'))
    print(json.dumps(dict(timing=summary, recumulated=recumulated)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'score', 'diagnose'])
    parser.add_argument('fetched', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--budget-seconds', type=float,
                        help='pause after this many seconds; rerun to resume from checkpoints')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.fetched, args.output)
    elif args.mode == 'score':
        score(args.fetched, args.output, args.workdir, args.budget_seconds)
    else:
        diagnose(args.output, args.workdir)
