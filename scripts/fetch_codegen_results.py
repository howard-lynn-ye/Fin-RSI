"""Read-only status check and result download for Beacon job 1733007 (codegen utility v2).

Never submits, cancels or modifies anything on Beacon: it runs squeue/sacct/ls/tail/sha256sum
and SFTP reads only. The credential file is read only inside scripts.beacon_campaign.connect()
and is never printed or copied.

Run from D:\\fin_skill in PowerShell:
  python scripts\\fetch_codegen_results.py status           # print and save job status
  python scripts\\fetch_codegen_results.py fetch            # download once completion.txt exists
  python scripts\\fetch_codegen_results.py fetch --partial  # snapshot an unfinished job
Downloads go to a NEW folder runs\\codegen-utility-v2-fetch-<UTC stamp>\\ with fetch-receipt.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paramiko  # noqa: E402
from scripts.beacon_campaign import connect  # noqa: E402
from scripts.deploy_beacon_followup import remote_command  # noqa: E402

CRED = os.environ.get('FIN_BEACON_CREDENTIAL', 'D:/ct_agent_vqa/.secrets/beacon_password.txt')
JOB = '1733007'
REMOTE = '/beacon-projects/radfm/wy891/fin-skills-campaign-codegen-utility-20260928-v2'
TOP = ('protocol.json', 'inputs.json', 'qualification.json', 'inference-started.json',
       'inference-receipt.json', 'scores.json', 'completion.txt', 'job.sh', 'manifest.json',
       'plan.json')
DIRS = ('responses', 'audits', 'logs')
SOURCE = ('source/inputs.json', 'source/library-and-runner.tar.gz', 'source/protocol.json',
          'source/protocol.md', 'job.sh', 'source/submit_model_followup.py')


def open_transport():
    logging.getLogger('paramiko').setLevel(logging.CRITICAL)
    for attempt in range(4):
        try:
            return connect(CRED)
        except Exception as exc:  # banner EOF is a transport hiccup, not a job failure
            print('connect attempt', attempt + 1, type(exc).__name__, flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit('Beacon unreachable after 4 attempts; nothing was changed')


def status_text(t):
    q = shlex.quote(REMOTE)
    cmd = '; '.join([
        'echo "== utc $(date -u +%FT%TZ)"',
        f'echo "== squeue"; squeue -j {JOB} -o "%i %T %M %L %R" 2>&1',
        f'echo "== sacct"; sacct -j {JOB} -X -n -P '
        '-o JobID,State,Start,End,Elapsed,ExitCode 2>&1',
        f'cd {q}',
        'echo "== responses $(ls -1 responses 2>/dev/null | wc -l)'
        ' audits $(ls -1 audits 2>/dev/null | wc -l)"',
        'echo "== completion"; cat completion.txt 2>/dev/null || echo none',
        f'echo "== stdout tail"; tail -n 4 logs/slurm-{JOB}.out 2>&1',
        f'echo "== stderr tail"; tail -n 15 logs/slurm-{JOB}.err 2>&1',
        'true'])
    return remote_command(t, cmd)


def remote_hashes(t, names):
    out = {}
    for i in range(0, len(names), 40):
        chunk = names[i:i + 40]
        text = remote_command(t, 'cd ' + shlex.quote(REMOTE) + ' && sha256sum -- '
                              + ' '.join(shlex.quote(n) for n in chunk))
        for line in text.splitlines():
            digest, name = line.split(None, 1)
            out[name.lstrip('*')] = digest
    return out


def fetch(t, partial):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    status = status_text(t)
    with paramiko.SFTPClient.from_transport(t) as s:
        def exists(name):
            try:
                s.stat(REMOTE + '/' + name)
                return True
            except FileNotFoundError:
                return False
        complete = exists('completion.txt')
        if not complete and not partial:
            print(status)
            raise SystemExit('completion.txt not present yet; rerun later or pass --partial')
        names = [n for n in TOP if exists(n)]
        for d in DIRS:
            try:
                names += [d + '/' + a.filename for a in s.listdir_attr(REMOTE + '/' + d)
                          if stat.S_ISREG(a.st_mode)]
            except FileNotFoundError:
                pass
        names = sorted(set(names))
        before = remote_hashes(t, names)
        source = remote_hashes(t, [n for n in SOURCE if exists(n)])
        out = ROOT / 'runs' / ('codegen-utility-v2-fetch-' + stamp)
        out.mkdir(parents=True, exist_ok=False)
        files = {}
        for name in names:
            dest = out.joinpath(*PurePosixPath(name).parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            s.get(REMOTE + '/' + name, str(dest))
            local = hashlib.sha256(dest.read_bytes()).hexdigest()
            files[name] = dict(bytes=dest.stat().st_size, sha256=local,
                               remote_sha256=before.get(name), match=local == before.get(name))
    receipt = dict(fetched_utc=stamp, job_id=JOB, remote_root=REMOTE, complete=complete,
                   read_only=True, status=status, source_remote_sha256=source,
                   n_files=len(files), all_match=all(f['match'] for f in files.values()),
                   files=files)
    (out / 'fetch-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(status)
    print(json.dumps(dict(folder=str(out), complete=complete, n_files=len(files),
                          all_match=receipt['all_match'])))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['status', 'fetch'])
    p.add_argument('--partial', action='store_true')
    args = p.parse_args()
    with open_transport() as t:
        if args.action == 'status':
            text = status_text(t)
            print(text)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            out = ROOT / 'runs' / f'codegen-status-{stamp}.txt'
            out.write_text(text + '\n', encoding='utf-8')
        else:
            fetch(t, args.partial)


if __name__ == '__main__':
    main()
