"""Stage a frozen code-generation bundle on Beacon and submit it exactly once.

Refuses if the remote root already exists or the bundle was deployed before. Uploads with
exclusive-create, re-checks every SHA-256 against manifest.json, then submits through
submit_model_followup._dispatch, which persists a marker before sbatch so an interrupted call
cannot silently create a duplicate job. The credential file is read only by connect().

    python scripts/deploy_codegen_bundle.py runs/<bundle> --credential-file <path>
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paramiko  # noqa: E402
from scripts.beacon_campaign import connect  # noqa: E402
from scripts.deploy_beacon_followup import remote_command  # noqa: E402

RUNTIME = '/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--credential-file', required=True)
    args = parser.parse_args()
    bundle = args.bundle
    manifest = json.loads((bundle/'manifest.json').read_text())
    plan = manifest['plan']
    assert plan == json.loads((bundle/'plan.json').read_text())
    root = plan['remote_root']
    assert root.startswith(('/beacon-projects/radfm/wy891/fin-skills-campaign-codegen-',
                            '/beacon-projects/radfm/wy891/fin-skills-campaign-trading-'))
    for marker in ('deployment-started.json', 'deployment-receipt.json'):
        if (bundle/marker).exists():
            raise SystemExit(f'{marker} exists; inspect the remote state instead of redeploying')
    files = {name: (bundle/name).read_bytes() for name in manifest['sha256']}
    assert all(hashlib.sha256(b).hexdigest() == manifest['sha256'][n] for n, b in files.items())
    files['manifest.json'] = (bundle/'manifest.json').read_bytes()
    logging.getLogger('paramiko').setLevel(logging.CRITICAL)
    for attempt in range(3):
        try:
            transport = connect(args.credential_file)
            break
        except Exception as exc:
            print('connect attempt', attempt + 1, type(exc).__name__)
            time.sleep(5)
    else:
        raise SystemExit('Beacon unreachable; nothing was staged')
    with transport:
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            try:
                sftp.stat(root)
                raise SystemExit('remote root already exists; inspect before continuing')
            except FileNotFoundError:
                pass
            (bundle/'deployment-started.json').write_text(json.dumps(
                dict(remote_root=root, status='transport_started')))
            sftp.mkdir(root, mode=0o700)
            for sub in ('logs', 'tmp', 'cache', 'source'):
                sftp.mkdir(root + '/' + sub, mode=0o700)
            made = set()
            for name, blob in files.items():
                parent = root + '/' + name.rsplit('/', 1)[0] if '/' in name else root
                if parent not in made and parent != root:
                    try:
                        sftp.stat(parent)
                    except FileNotFoundError:
                        sftp.mkdir(parent, mode=0o700)
                    made.add(parent)
                with sftp.open(root + '/' + name, 'wx') as out:
                    out.write(blob)
                with sftp.open(root + '/' + name, 'rb') as back:
                    if hashlib.sha256(back.read()).digest() != hashlib.sha256(blob).digest():
                        raise RuntimeError(f'upload hash mismatch: {name}')
            sftp.chmod(root + '/job.sh', 0o700)
        command = ['sbatch', '--parsable', '--account=angliece', '--partition=beacon',
                   '--qos=medium', '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
                   '--gres=' + plan.get('gres', 'gpu:1'), '--mem=' + plan['memory'],
                   '--time=' + plan['time'],
                   '--job-name=' + plan['name'], '--chdir=' + root,
                   '--output=' + root + '/logs/slurm-%j.out',
                   '--error=' + root + '/logs/slurm-%j.err', root + '/job.sh', root]
        script = ('import sys,json;from pathlib import Path;r=Path(' + repr(root) + ');'
                  'sys.path.insert(0,str(r/"source"));'
                  'from submit_model_followup import _dispatch;'
                  'print(json.dumps(_dispatch(r,' + repr(command) + ')))')
        result = remote_command(transport, shlex.join([RUNTIME, '-B', '-c', script]))
    (bundle/'deployment-receipt.json').write_text(result)
    print(result)


if __name__ == '__main__':
    main()
