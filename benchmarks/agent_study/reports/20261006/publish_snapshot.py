"""Export verified summary numbers; never publish decision text or market data.

Run on a compute node. The two input reports and original run trees stay on Beacon.
Use exclusive output files so a later snapshot cannot overwrite this one.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def validate_summary(report):
    if report['failures']:
        raise ValueError('verification failures must be resolved before publication')
    groups = {}
    for row in report['rows']:
        groups.setdefault(row['family'], []).append(row)
    if set(groups) != set(report['summary']):
        raise ValueError('model inventory differs')
    if sum(len(rows) for rows in groups.values()) != report['verified_pairs']:
        raise ValueError('pair count differs')
    for family, rows in groups.items():
        expected = report['summary'][family]
        seeds = sorted(row['seed'] for row in rows)
        if len(set(seeds)) != len(seeds) or seeds != expected['seeds']:
            raise ValueError('duplicate or mismatched seeds')
        if expected['complete'] != (seeds == [11, 23, 37]):
            raise ValueError('partial model marked complete')
        for arm in ('raw', 'library'):
            values = [row['paths'][arm]['return_rate_pct'] for row in rows]
            if not all(math.isfinite(v) for v in values + [expected['return_rate_pct'][arm]]):
                raise ValueError('nonfinite return')
            if abs(statistics.mean(values) - expected['return_rate_pct'][arm]) > 1e-10:
                raise ValueError('mean return differs')
        difference = expected['return_rate_pct']['library'] - expected['return_rate_pct']['raw']
        if not math.isfinite(expected['difference_pp']) or abs(difference - expected['difference_pp']) > 1e-10:
            raise ValueError('percentage-point difference differs')
    if report['completed_models'] != sum(s['complete'] for s in report['summary'].values()):
        raise ValueError('completed model count differs')


def error_class(turn):
    if turn.get('result', {}).get('ok'):
        return None
    status = turn.get('parse')
    if status in ('generation-truncated', 'unparsed', 'reasoning-incomplete', 'reasoning-malformed'):
        return status
    error = str(turn.get('result', {}).get('error', ''))
    match = re.match(r'([A-Za-z]+(?:Error|Exception)):', error)
    if match:
        return match.group(1)
    if 'unknown tool' in error or 'unavailable in' in error:
        return 'unavailable-tool'
    if 'weights' in error or 'weight' in error:
        return 'invalid-weights'
    if 'timeout' in error:
        return 'tool-timeout'
    return 'other-rejected-call'


def export(base, verification, diagnostics, output, private_errors):
    report, counts = load(verification), load(diagnostics)
    validate_summary(report)
    if counts['based_on'] != verification.name:
        raise ValueError('diagnostics refer to another snapshot')
    if set(counts['groups']) != set(report['summary']):
        raise ValueError('diagnostic model inventory differs')
    public_rows, categories, private = [], {}, {}
    for row in report['rows']:
        parent = Path(row['source']).parent
        parent.relative_to(base.resolve())
        root = parent / ('batch' if parent.name == 'v7-models-v1' else 'pairs') / row['pair']
        score_hash, audit_hash = sha(root/'scores.json'), sha(root/'independent-model-audit.json')
        if (score_hash, audit_hash) != (row['scores_sha256'], row['audit_sha256']):
            raise ValueError('result changed since verification')
        receipt = load(root/'inference-receipt.json')
        audit = load(root/'independent-model-audit.json')
        if sha(root/'inference-receipt.json') != audit['inference_receipt_sha256']:
            raise ValueError('receipt changed since audit')
        for arm in ('raw', 'library'):
            key = f"{row['family']}/{arm}"
            errors, examples = categories.setdefault(key, Counter()), private.setdefault(key, Counter())
            for name, digest in receipt['decisions'].items():
                if Path(name).parts[:2] != ('decisions', arm):
                    continue
                if sha(root/name) != digest:
                    raise ValueError('decision changed since receipt')
                record = load(root/name)
                for turn in record['turns']:
                    category = error_class(turn)
                    if category:
                        errors[category] += 1
                        examples[str(turn.get('result', {}).get('error', ''))[:500]] += 1
        public_rows.append({k: row[k] for k in ('pair', 'family', 'seed', 'model', 'window',
            'paths', 'difference_pp', 'scores_sha256', 'audit_sha256',
            'evidence_sha256', 'packets_sha256')})
        public_rows[-1]['protocol_sha256'] = sha(root/'protocol.json')
        public_rows[-1]['inference_receipt_sha256'] = sha(root/'inference-receipt.json')
        public_rows[-1]['source_manifest_sha256'] = hashlib.sha256(json.dumps(
            row['common']['source_sha256'], sort_keys=True).encode()).hexdigest()
    for family, summary in report['summary'].items():
        for arm in ('raw', 'library'):
            count = counts['groups'][family][arm]
            if (count['decisions'], count['submitted']) != (
                    summary['decisions_per_arm'], summary['submitted'][arm]):
                raise ValueError('diagnostic decisions or submissions differ')
    published = dict(schema='fin-skills-multisource-snapshot-v1',
        timestamp_utc=report['timestamp_utc'], completed_models=report['completed_models'],
        verified_pairs=report['verified_pairs'], summary=report['summary'],
        seed_results=public_rows, response_counts=counts['groups'],
        error_classes=categories, incomplete=report['incomplete'],
        provenance=dict(verification_file=verification.name, verification_sha256=sha(verification),
            diagnostics_file=diagnostics.name, diagnostics_sha256=sha(diagnostics)),
        limits=['One development market path; three seeds are not independent markets.',
            'Accounting identity passed; historical evidence vintages remain unverified.',
            'Incomplete models are reported separately; personal and legacy v6 runs are excluded.',
            'Error classes are observations, not causal attribution of return differences.',
            'This file deliberately omits model responses, evidence text and raw market data.'])
    with output.open('x') as stream:
        json.dump(published, stream, indent=2, allow_nan=False)
    with private_errors.open('x') as stream:
        json.dump({key: counter.most_common(12) for key, counter in private.items()}, stream, indent=2)
    print(json.dumps(dict(output=str(output), sha256=sha(output),
        completed_models=published['completed_models'], verified_pairs=published['verified_pairs'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'verification', 'diagnostics', 'output', 'private-errors'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    export(args.base, args.verification, args.diagnostics, args.output, args.private_errors)
