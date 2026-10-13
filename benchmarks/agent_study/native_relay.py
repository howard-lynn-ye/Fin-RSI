"""Windows transport for a declared Beacon native-model queue.

No market data is evaluated here. Requests, responses and provider events are
transported in memory; persistent experiment artifacts are written on Beacon.
Uses only the already authenticated accounts and an existing PuTTY master.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import threading

WRAPPER = ('This is one stateless response in a historical research simulation. '
    'Do not use native CLI tools, browsing, files, delegation, or remembered conversations. '
    'All permitted research uses the JSON tool protocol shown inside the supplied messages; '
    'Beacon will execute that JSON and return its result in a later stateless request. '
    'Read the ordered system/user/assistant messages below as the complete simulated conversation. '
    'Return only the next assistant response: one compact JSON object, at most 1024 reference tokens. '
    'Choose your own decisions; do not invent a tool result.\n\n')
GEMINI_AGENT_NAME = 'fin-replay-response'
GEMINI_AGENT = '''---
name: fin-replay-response
description: Stateless text responder for a historical simulation with external tool execution.
tools: []
mainAgent: true
subagent: false
model: inherit
commandExecutionPolicy: off
inheritCustomizations: false
mcpServers: []
skills: []
plugins: []
---
Read only the supplied ordered conversation. Return its next assistant response as text.
All research tools are simulated JSON requests executed by the caller, never native tools.
Do not access files, browse, delegate, or perform workspace tasks.
'''

QUEUE = '/beacon-projects/radfm/wy891/fin-multisource-models-20261005/native-queue-20261013-v1'
REMOTE_PAUSE = '/beacon-projects/radfm/wy891/BEACON_CONNECTION_PAUSED.md'
LOCAL_PAUSE = Path(r'D:\ct_agent_vqa\.codex-beacon-work\CONNECTION_PAUSED.md')
HOSTKEY = 'ssh-ed25519 255 SHA256:F0/SPeAvRCv/rg28Tx5Z+uKmtcU5DSwW8T5iwbQzKmo'


def source_digest(path=None):
    # Source semantics are identical under Git's LF/CRLF checkout conversion.
    raw = Path(path or __file__).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def recovered_codex_retry(events, index):
    """Only the observed 503 reconnect notice, before any answer, may recover."""
    event = events[index]
    match = re.fullmatch(r'Reconnecting\.\.\. ([1-9][0-9]*)/([1-9][0-9]*) '
        r'\(unexpected status 503 Service Unavailable: [\s\S]+\)', event.get('message', ''))
    before, after = events[:index], events[index+1:]
    return bool(match and int(match[1]) <= int(match[2])
        and sum(e.get('type') == 'turn.started' for e in events) == 1
        and any(e.get('type') == 'turn.started' for e in before)
        and not any(e.get('type') in ('turn.failed', 'turn.completed') or
            e.get('item', {}).get('type') == 'agent_message' for e in before)
        and sum(e.get('type') == 'turn.completed' for e in after) == 1
        and not any(e.get('type') == 'turn.failed' for e in events))

def parse_events(transport, stdout, exit_code):
    events, malformed = [], []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            malformed.append(line)
    tools, models, text, usage = [], [], '', None
    diagnostics, errors = [], []
    if transport == 'claude_subscription':
        for event in events:
            if event.get('type') == 'assistant':
                message = event.get('message', {})
                models.append(message.get('model'))
                tools.extend(x for x in message.get('content', []) if x.get('type') == 'tool_use')
        result = next((x for x in reversed(events) if x.get('type') == 'result'), {})
        text, usage = result.get('result', ''), result.get('usage')
        ok = result.get('subtype') == 'success' and not result.get('is_error')
    elif transport == 'antigravity_subscription':
        init = next((x for x in events if x.get('event') == 'init'), {})
        models = [init.get('init', {}).get('model')]
        result = next((x.get('result', {}) for x in reversed(events) if x.get('event') == 'result'), {})
        for event in events:
            if event.get('event') == 'step_update':
                step = event.get('step_update', {})
                if step.get('step_type') not in ('user_input', 'agent_response', 'agent_thought'):
                    tools.append(event)
            elif event.get('event') not in ('init', 'result'):
                tools.append(event)
        text, usage = result.get('response', ''), result.get('usage')
        ok = (result.get('status') == 'SUCCESS' and not result.get('denied_actions')
              and init.get('init', {}).get('agent') == GEMINI_AGENT_NAME
              and isinstance(init.get('init', {}).get('tools'), list))
    elif transport == 'codex_subscription':
        turn_started = False
        known_startup = ('Under-development features enabled: skip_host_skill_discovery.',
                         'Code Mode is unavailable because code-mode host is disabled.')
        for index, event in enumerate(events):
            turn_started |= event.get('type') == 'turn.started'
            if event.get('type') == 'error':
                if recovered_codex_retry(events, index):
                    diagnostics.append(event)
                else:
                    errors.append(event)
            item = event.get('item', {})
            if item.get('type') == 'error':
                # Retain observed client notices; never reclassify an in-turn error.
                if not turn_started and item.get('message', '').startswith(known_startup):
                    diagnostics.append(event)
                else:
                    errors.append(event)
            elif item and item.get('type') not in ('agent_message', 'reasoning'):
                tools.append(event)
        messages = [x['item'].get('text', '') for x in events
                    if x.get('type') == 'item.completed' and x.get('item', {}).get('type') == 'agent_message']
        text = '\n'.join(messages)
        done = [x for x in events if x.get('type') == 'turn.completed']
        usage = done[-1].get('usage') if done else None
        ok = len(done) == 1 and not any(x.get('type') == 'turn.failed' for x in events)
    else:
        raise ValueError('undeclared transport')
    return dict(success=bool(ok and exit_code == 0 and text and not tools and not malformed and not errors),
                text=text, reported_models=models, native_tool_events=tools, usage=usage,
                identity_level={'codex_subscription': 'configured_request_only', 'antigravity_subscription': 'client_init_model',
                                'claude_subscription': 'response_model'}[transport],
                events=events, diagnostic_events=diagnostics, provider_error_events=errors, malformed_event_lines=malformed,
                client_advertised_tools=(init.get('init', {}).get('tools')
                                         if transport == 'antigravity_subscription' else None))


def invoke(request, *, timeout=900):
    """Official native clients only; a failure is returned, never reissued."""
    if os.name != 'nt':
        raise RuntimeError('existing desktop account route requires Windows')
    spec = request['native_policy']
    if spec['prompt_wrapper'] != WRAPPER:
        raise ValueError('frozen prompt wrapper differs')
    env, token = os.environ.copy(), None
    # Never let a pre-existing API-key environment silently select paid API billing.
    for key in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL', 'AZURE_OPENAI_API_KEY',
                'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS'):
        env.pop(key, None)
    prompt = WRAPPER + json.dumps(request['messages'], ensure_ascii=False)
    transport, model, effort = spec['transport'], spec['model'], spec['effort']
    with tempfile.TemporaryDirectory(prefix='fin-native-empty-') as isolated:
        if transport == 'claude_subscription':
            sys.path.insert(0, str(Path.home()/'.codex/shared/agent-backends'))
            from desktop_auth import APP, inference_token
            token, _ = inference_token()
            if not token:
                raise RuntimeError('existing Desktop inference credential unavailable; no account fallback')
            binary = APP/'claude-code/2.1.284/claude.exe'
            if not binary.exists():
                raise RuntimeError('validated Claude client unavailable')
            env.update(CLAUDE_CODE_OAUTH_TOKEN=token, ANTHROPIC_BASE_URL='https://api.anthropic.com',
                       CLAUDE_CONFIG_DIR=isolated)
            for name in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_PROFILE', 'CLAUDE_CODE_SIMPLE'):
                env.pop(name, None)
            command = [str(binary), '-p', '--model', model, '--effort', effort,
                '--strict-mcp-config', '--setting-sources', '', '--settings', '{"disableAllHooks":true}',
                '--tools', '', '--permission-mode', 'plan', '--no-session-persistence',
                '--output-format', 'stream-json', '--verbose']
            payload = prompt
        elif transport == 'antigravity_subscription':
            agent_path = Path(isolated)/'.agents/agents'/GEMINI_AGENT_NAME/'agent.md'
            agent_path.parent.mkdir(parents=True)
            agent_path.write_text(GEMINI_AGENT, encoding='utf-8')
            binary = Path.home()/'AppData/Local/agy/bin/agy.exe'
            command = [str(binary), '--model', model, '--effort', effort, '--mode', 'plan',
                       '--agent', GEMINI_AGENT_NAME,
                       '--input-format', 'stream-json', '--output-format', 'stream-json', '--print-timeout', '10m',
                       '--sandbox', '--log-file', 'NUL']
            payload = json.dumps({'event': 'user', 'message': {'content': prompt}}, ensure_ascii=False)+'\n'
        elif transport == 'codex_subscription':
            binary = Path.home()/'AppData/Local/OpenAI/Codex/bin/2e5e00daee91c61d/codex.exe'
            command = [str(binary), 'exec', '--model', model, '--json', '--ephemeral', '--sandbox', 'read-only',
                '--ignore-user-config', '--skip-git-repo-check', '-C', isolated,
                '-c', 'model_reasoning_effort='+json.dumps(effort), '-c', 'web_search="disabled"']
            for feature in ('shell_tool', 'unified_exec', 'apps', 'plugins', 'hooks', 'multi_agent',
                            'computer_use', 'browser_use', 'in_app_browser', 'view_image',
                            'image_generation', 'workspace_dependencies', 'skill_search', 'code_mode_host'):
                command += ['-c', f'features.{feature}=false']
            command += ['-c', 'features.skip_host_skill_discovery=true', '-']
            payload = prompt
        else:
            raise ValueError('undeclared transport')
        try:
            proc = subprocess.run(command, input=payload, text=True, encoding='utf-8',
                capture_output=True, cwd=isolated, env=env, timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW)
            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, exit_code = exc.stdout or '', exc.stderr or '', -1
            if isinstance(stdout, bytes):
                stdout = stdout.decode('utf-8', errors='replace')
            if isinstance(stderr, bytes):
                stderr = stderr.decode('utf-8', errors='replace')
            stderr += '\nNative request timed out; not retried.'
    if token:
        stdout, stderr = stdout.replace(token, '[REDACTED]'), stderr.replace(token, '[REDACTED]')
    result = parse_events(transport, stdout, exit_code)
    return dict(result, request_sha256=request['request_sha256'], requested_model=model,
        requested_effort=effort, transport=transport, exit_code=exit_code, stderr=stderr,
        completed_at=datetime.now(timezone.utc).isoformat(),
        relay_sha256=source_digest())


class SharedBeacon:
    def __init__(self):
        self.lock = threading.Lock()
        self.binary = r'C:\Program Files\PuTTY\plink.exe'
        self.common = ['-batch', '-noagent', '-hostkey', HOSTKEY, '-l', 'wy891', 'beaconsub01.ihc.umd.edu']

    def run(self, command, payload=None):
        with self.lock:
            return self._run(command, payload)

    def _run(self, command, payload=None):
        if LOCAL_PAUSE.exists():
            raise RuntimeError('shared Beacon connection pause; relay stopped')
        try:
            exists = subprocess.run([self.binary, '-shareexists', *self.common],
                capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            if exists.returncode:
                raise RuntimeError('existing SSH master unavailable')
            proc = subprocess.run([self.binary, '-share', *self.common,
                'test ! -f '+REMOTE_PAUSE+' && '+command], input=payload,
                text=True, encoding='utf-8', capture_output=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if proc.returncode:
                raise RuntimeError('shared SSH operation failed; no retry')
            return proc.stdout
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            try:
                with LOCAL_PAUSE.open('x', encoding='utf-8') as stream:
                    stream.write('Native experiment relay stopped at '+datetime.now(timezone.utc).isoformat()+
                        '. Existing PuTTY master check or shared operation failed. '
                        'Recovery owner must establish a concrete fix or recovery before another attempt.\n')
            except FileExistsError:
                pass
            raise RuntimeError('Beacon relay paused; inspect connection before any retry') from exc

    def pending(self):
        output = self.run(f"if test -d {QUEUE}; then find {QUEUE} -mindepth 2 -maxdepth 2 -name request.json -print; fi")
        return [line.split('/')[-2] for line in output.splitlines() if re.fullmatch(re.escape(QUEUE)+r'/[0-9a-f]{64}/request.json', line)]

    def read(self, digest):
        if not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid queue id')
        base = QUEUE+'/'+digest
        output = self.run(f'if test -f {base}/response.json || test -d {base}/relay-claim; then printf DONE; else cat {base}/request.json; fi')
        return None if output == 'DONE' else json.loads(output)

    def claim(self, digest):
        claim = f'{QUEUE}/{digest}/relay-claim'
        result = self.run(f'if mkdir {claim} 2>/dev/null; then printf CLAIMED; '
                          f'elif test -d {claim}; then printf BUSY; else exit 1; fi')
        return result == 'CLAIMED'

    def publish(self, digest, response):
        base = QUEUE+'/'+digest
        encoded = base64.b64encode(json.dumps(response, ensure_ascii=False, allow_nan=False).encode()).decode()
        self.run(f'(set -e; test ! -e {base}/response.json; base64 -d > {base}/relay-claim/response.tmp; '
                 f'ln {base}/relay-claim/response.tmp {base}/response.json)', encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-protocol', action='append', required=True)
    parser.add_argument('--hours', type=float, default=12)
    args = parser.parse_args()
    allowed = set(args.allow_protocol)
    if not all(re.fullmatch('[0-9a-f]{64}', value) for value in allowed):
        raise ValueError('explicit frozen protocol hashes required')
    ssh, seen, active = SharedBeacon(), set(), {}
    deadline = time.monotonic()+args.hours*3600
    def process(digest, request):
        try:
            response = invoke(request)
        except Exception as exc:
            response = dict(request_sha256=digest, requested_model=request['native_policy']['model'],
                requested_effort=request['native_policy']['effort'], transport=request['native_policy']['transport'],
                success=False, exit_code=-1, text='', native_tool_events=[],
                error=type(exc).__name__+'; native call failed before a validated result; no retry')
        ssh.publish(digest, response)
        return dict(request=digest, model=request['native_policy']['model'],
                    success=response['success'], response_chars=len(response.get('text', '')))
    with ThreadPoolExecutor(max_workers=3) as pool:
        while time.monotonic() < deadline:
            for digest, future in list(active.items()):
                if future.done():
                    print(json.dumps(future.result()), flush=True)
                    del active[digest]
            if len(active) < 3:
                for digest in ssh.pending():
                    if digest in seen:
                        continue
                    request = ssh.read(digest)
                    seen.add(digest)
                    if request is None or request['protocol_sha256'] not in allowed:
                        continue
                    if (request['request_sha256'] != digest or
                            fingerprint({k:v for k,v in request.items() if k != 'request_sha256'}) != digest or
                            request['native_policy']['relay_sha256'] != source_digest()):
                        raise ValueError('queue request or declared relay source changed')
                    if not ssh.claim(digest):
                        continue
                    active[digest] = pool.submit(process, digest, request)
                    if len(active) == 3:
                        break
            time.sleep(5)


if __name__ == '__main__':
    main()
