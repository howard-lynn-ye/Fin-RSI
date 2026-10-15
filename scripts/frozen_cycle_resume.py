"""Continue an immutable HF pair on Slurm; never submits jobs or edits frozen source.

The host owns this driver. The frozen runner still makes every investment decision,
validates its output and computes its original accounting. Existing artifacts are
verified and retained, including genuine failed decisions. Corrupt/partial JSON and
recorded infrastructure failures require investigation, not an automatic retry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
from unittest.mock import patch


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'nonfinite JSON: {value}')
    def finite(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid(value)
        return parsed
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=unique, parse_constant=invalid, parse_float=finite)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(root, path):
    root, path = Path(root).resolve(), Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f'artifact outside expected directory: {path}')
    return path


def write_once(path, value):
    """Publish complete JSON atomically, never replace different existing evidence."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, allow_nan=False).encode('utf-8')
    if path.exists():
        if load(path) != json.loads(encoded):
            raise ValueError(f'existing artifact differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.resume-write-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes complete bytes without overwriting a concurrent writer.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if load(path) != json.loads(encoded):
                raise ValueError(f'concurrent artifact differs: {path}') from None
    finally:
        Path(temporary).unlink()


@contextmanager
def locked(directory):
    """Kernel lock is released on process death; the retained file is not a stale lock."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = within(directory, directory / '.resume.lock')
    with path.open('a+b') as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise RuntimeError('another recovery writer is active') from exc
        try:
            yield
        finally:
            stream.seek(0)
            release()


def sole_job(rows, current, previous):
    jobs = [line.split('|') for line in rows.splitlines() if line.strip()]
    if any(len(row) != 3 for row in jobs):
        raise ValueError('unrecognized scheduler response')
    if not current.isdigit() or not previous.isdigit() or current == previous:
        raise ValueError('recovery requires a new allocated Slurm job')
    if any(row[0] == previous for row in jobs):
        raise ValueError('previous job is still queued or running')
    owned = [row for row in jobs if row[1].startswith('fin-')]
    if len(owned) != 1 or owned[0][0] != current or owned[0][2] != 'RUNNING':
        raise ValueError('exactly one running fin-skills allocation is required')


def binding(root, protocol):
    files = dict(protocol_sha256='protocol.json',
                 qualification_sha256='qualification/qualification.json',
                 capability_qualification_sha256='capability-qualification.json',
                 decision_qualification_sha256='decision-qualification.json')
    if protocol['interface'] == 'v9':
        files['rag_qualification_sha256'] = 'rag-qualification.json'
    return {key: sha(root / name) for key, name in files.items()}


def inventory(root):
    names = ('inference-started.json', 'inference-receipt.json', 'scores.json',
             'independent-model-audit.json', 'completed.json')
    paths = [root / name for name in names if (root / name).exists()]
    paths += list((root / 'decisions').glob('*/*.json'))
    paths += list((root / 'nav').glob('*.json'))
    return {within(root, p).relative_to(root).as_posix(): sha(p) for p in sorted(paths)}


def preserve(root, hashes):
    for name, expected in hashes.items():
        path = within(root, root / name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'previously recorded artifact changed: {name}')


def validate_partial(study, root, protocol):
    """Check the same chronology, targets, output contracts and evidence before loading."""
    if protocol.get('batch') != 'matrix_hf' or protocol.get('interface') not in ('v8', 'v9'):
        raise ValueError('recovery supports frozen structured HF pairs only')
    if list((root / 'infrastructure-failures').glob('*.json')):
        raise ValueError('recorded infrastructure failure needs investigation; no automatic retry')
    inputs, packets = load(root / 'inputs.json'), load(root / 'evidence-packets.json')
    if set(inputs['arms']) != {'raw', 'library'} or len(inputs['arms']) != 2:
        raise ValueError('expected exactly the two declared arms')
    picks = inputs['picks']
    if len(picks) != protocol['decisions_per_arm'] or picks != sorted(set(picks)):
        raise ValueError('decision schedule differs')
    expected = [f'decisions/{arm}/{i:02d}.json' for arm in inputs['arms'] for i in range(len(picks))]
    # A process killed before publication may leave our incomplete temporary file.
    # It is retained for inspection, never interpreted as a completed decision.
    actual = {p.relative_to(root).as_posix() for p in (root / 'decisions').glob('*/*')
              if not (p.is_file() and p.name.startswith('.resume-write-'))}
    if actual != set(expected[:len(actual)]):
        raise ValueError('decisions must form a contiguous prefix of the frozen arm order')
    if len(actual) < len(expected) and (
        any((root / name).exists() for name in ('inference-receipt.json', 'scores.json',
            'independent-model-audit.json', 'completed.json')) or list((root / 'nav').glob('*'))
    ):
        raise ValueError('scoring artifacts exist before all decisions are complete')
    binds = binding(root, protocol)
    started = root / 'inference-started.json'
    if started.exists():
        record = load(started)
        if any(record.get(key) != value for key, value in binds.items()):
            raise ValueError('inference start has different frozen bindings')
    elif actual:
        raise ValueError('decisions exist without their inference start receipt')
    prices = study.pd.read_csv(root / 'data' / study.md.HIDDEN, index_col=0)
    returns = prices.pct_change().fillna(0.)
    evidence = load(root / 'evidence.json')
    runtime = study.terminal_runtime(protocol['interface'])
    holdings = {arm: dict.fromkeys(protocol['universe'], 0.) for arm in inputs['arms']}
    hashes = {}
    for name in expected[:len(actual)]:
        path = within(root, root / name)
        record = load(path)
        arm, i = path.parent.name, int(path.stem)
        pick, day = picks[i], str(prices.index[picks[i]])
        if (record['index'], record['date'], record['interface']) != (pick, day, protocol['interface']):
            raise ValueError('decision date/index/interface differs')
        runtime.validate_decision(record, expected_final_output=protocol['final_output'],
            expected_research_output=study.research_schema.arm_contract(protocol, arm))
        study.check_backend(record, protocol)
        before = record['holdings_before']
        if set(before) != set(holdings[arm]) or any(
            not math.isfinite(before[t]) or not math.isclose(before[t], holdings[arm][t], abs_tol=1e-12, rel_tol=0)
            for t in holdings[arm]
        ):
            raise ValueError('holdings before decision differ from original path')
        packet = packets[day]
        snapshot = runtime.evidence_snapshot(evidence, packet)
        if (record['evidence_packet_sha256'] != study.digest(packet) or
            record['evidence_snapshot_sha256'] != hashlib.sha256(runtime.encoded(snapshot).encode()).hexdigest() or
            not record['initial_request'][1]['content'].endswith(study.evidence_text(packet))):
            raise ValueError('decision evidence differs from frozen input')
        target = record['target']
        if target is not None and set(target) != set(protocol['universe']):
            raise ValueError('target does not cover the frozen universe')
        end = picks[i + 1] + 1 if i + 1 < len(picks) else pick + 1
        for k in range(pick + 1, end):
            holdings[arm], _ = study.old.drift(holdings[arm], returns.iloc[k])
            if k == pick + 1 and target is not None:
                holdings[arm] = dict(target)
        hashes[name] = sha(path)
    return hashes, binds, len(expected)


def advance(study, root, protocol):
    """Skip inference entirely when all original decisions already exist."""
    hashes, binds, count = validate_partial(study, root, protocol)
    if len(hashes) == count:
        write_once(root / 'inference-receipt.json', dict(binds, decisions=hashes))
    elif (root / 'inference-receipt.json').exists():
        raise ValueError('complete inference receipt with missing decisions')
    else:
        study.run(root)
    preserve(root, hashes)
    complete, _, _ = validate_partial(study, root, protocol)
    if len(complete) != count:
        raise ValueError('runner did not complete both arms')
    study.validate_receipt(root)
    study.score(root)
    return complete


def recover(study, root, output, protocol, *, job, previous, export, resources=None):
    """Single-writer recovery, including crashes between score and public export."""
    root, output = Path(root), Path(output)
    with locked(output):
        journal = output / 'recovery'
        journal.mkdir(exist_ok=True)
        for path in sorted(journal.glob('*.json')):
            event = load(path)
            if event['protocol_sha256'] != sha(root / 'protocol.json'):
                raise ValueError('recovery journal belongs to a different protocol')
            preserve(root, event['artifacts'])
            for name, expected in event.get('public_sha256', {}).items():
                if sha(within(output, output / 'public' / (name + '.json'))) != expected:
                    raise ValueError('previously published export changed')
        before = inventory(root)
        attempt = f'{job}-{uuid.uuid4().hex}'
        common = dict(job=job, previous_job=previous, protocol_sha256=sha(root / 'protocol.json'),
                      driver_sha256=sha(Path(__file__)), as_of=datetime.now(timezone.utc).isoformat(),
                      resources=resources)
        write_once(journal / f'{attempt}-started.json', dict(common, phase='started', artifacts=before))
        def immutable_writer(path, value):
            write_once(within(root, path), value)
        try:
            with patch.object(study.old, 'write', immutable_writer):
                decisions = advance(study, root, protocol)
            preserve(root, before)
            public = export(journal / f'{attempt}-public')
            if not public['summary']['all_expected_pairs_complete_and_valid'] or len(public['ledgers']) != 2:
                raise ValueError('public export is incomplete or invalid')
            for name, value in public.items():
                destination = within(output, output / 'public' / (name + '.json'))
                if destination.exists() and name == 'summary':
                    # A fresh audit time alone must not force overwriting an already valid export.
                    old = load(destination)
                    if {k: v for k, v in old.items() if k != 'observed_utc'} != {
                        k: v for k, v in value.items() if k != 'observed_utc'
                    }:
                        raise ValueError('existing public summary differs from independent audit')
                else:
                    write_once(destination, value)
            write_once(journal / f'{attempt}-completed.json', dict(common, phase='completed',
                artifacts=inventory(root),
                preserved_decisions=sum(name.startswith('decisions/') for name in before),
                decisions=len(decisions),
                public_sha256={name: sha(output / 'public' / (name + '.json')) for name in public}))
            return public['summary']
        except BaseException as exc:
            preserve(root, before)
            write_once(journal / f'{attempt}-interrupted.json', dict(common, phase='interrupted',
                artifacts=inventory(root), error_type=type(exc).__name__, error=str(exc)))
            raise


def allocation(job, needs_inference):
    resources = dict(slurm=subprocess.check_output(
        ['scontrol', 'show', 'job', '-o', job], text=True, timeout=15).strip(),
        needs_inference=needs_inference, devices=[])
    if needs_inference:
        import torch
        for index in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(index)
            resources['devices'].append(dict(index=index, name=device.name,
                                            total_memory_bytes=device.total_memory))
        if len(resources['devices']) != 1 or resources['devices'][0]['total_memory_bytes'] < 78 * 1024**3:
            raise ValueError('frozen single-device bf16 runner requires one compatible >=80GB GPU')
    return resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--previous-job', required=True)
    args = parser.parse_args()
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit():
        raise RuntimeError('this driver must run inside an allocated Slurm job')
    rows = subprocess.check_output(['squeue', '--me', '-h', '-o', '%i|%j|%T'], text=True, timeout=15)
    sole_job(rows, job, args.previous_job)
    config = load(args.config)
    root, source, output = (Path(config[k]).resolve() for k in ('root', 'source', 'output'))
    known_jobs = {str(config['original_job'])}
    for path in (output / 'recovery').glob('*-started.json'):
        known_jobs.add(str(load(path)['job']))
    if args.previous_job not in known_jobs:
        raise ValueError('previous job is not a recorded owner of this experiment')
    if sha(root / 'protocol.json') != config['protocol_sha256']:
        raise ValueError('protocol differs from the independently pinned configuration')
    verification = Path(config['verification'])
    if sha(verification) != config['verification_sha256']:
        raise ValueError('source verification manifest differs')
    for name, expected in load(verification)['sha256'].items():
        if sha(within(source, source / name)) != expected:
            raise ValueError(f'frozen source fingerprint differs: {name}')
    exporter = Path(config['exporter'])
    if sha(exporter) != config['exporter_sha256']:
        raise ValueError('independent exporter fingerprint differs')
    sys.path.insert(0, str(source))
    study = importlib.import_module('benchmarks.agent_study.multisource_model_study')
    if not Path(study.__file__).resolve().is_relative_to(source):
        raise ValueError('study imported outside the frozen source')
    protocol = study.verify(root)
    preflight = importlib.import_module('benchmarks.agent_study.trading_preflight_v8')
    preflight.require_campaign_ready(root, require_library_use=True)
    if (protocol['model'], protocol['seed'], protocol['interaction_revision']) != (
        config['model'], config['seed'], config['interaction_revision']
    ):
        raise ValueError('model, seed or interaction revision differs')
    if (output / 'started.json').exists():
        origin = load(output / 'started.json')
        if origin['protocol_sha256'] != config['protocol_sha256'] or Path(origin['source']).resolve() != source:
            raise ValueError('original launcher receipt differs')
    # A score/export-only continuation needs no GPU and must not load the model.
    existing, _, expected = validate_partial(study, root, protocol)
    resources = allocation(job, needs_inference=len(existing) < expected)
    def export(destination):
        subprocess.run([sys.executable, str(exporter), str(source), str(root.parent), str(destination)],
                       check=True, env=dict(os.environ, PYTHONPATH=str(source) + os.pathsep +
                                            os.environ.get('PYTHONPATH', '')))
        return {name: load(destination / (name + '.json')) for name in ('summary', 'ledgers')}
    summary = recover(study, root, output, protocol, job=job, previous=args.previous_job,
                      export=export, resources=resources)
    print(json.dumps(dict(status='complete_audited_pair', pairs=summary['pairs']), allow_nan=False))


if __name__ == '__main__':
    main()
