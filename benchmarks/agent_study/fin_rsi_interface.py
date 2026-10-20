"""Action parsing and worker variants for the single Fin-RSI runner.

Two interfaces, selected per run:

- ``strict``: the frozen current-v7 rule (``trading_runtime_v6.extract_call``) and worker.
- ``tolerant``: accepts what instruction models actually emit. 2026-10-18 diagnosis over 268
  non-final turns of jobs 2367319/2367443/2367577 (Qwen2.5 14B/32B): 31% were rejected for
  containing more than one fence/call (code restated, or code plus a JSON echo of the same
  action), 22% died on ``from fin_skills import <bound tool>``, 13% hit the 1024-token cap
  after restating code. Only 5% executed. The tolerant interface runs the LAST complete Python
  fence (or the last JSON call when there is no Python), also when generation was truncated
  after a complete fence, and resolves agent-level ``fin_skills`` imports of bound tool names
  to those bound functions. Tool implementations, sandbox, validators and budgets are unchanged.
"""
import builtins
import json
import os
import re
import types

from benchmarks.agent_study import trading_runtime_v6 as runtime

INTERFACES = ('strict', 'tolerant')
FENCE_ANY = re.compile(r'```([^\n`]*)\n(.*?)```', re.DOTALL)
TOOL_START = re.compile(r'\{\s*"tool"\s*:')


def _strip_reasoning(text):
    markers = list(re.finditer(r'</?think>', text))
    if not markers:
        return text, None
    thinking = False
    for marker in markers:
        opening = marker.group() == '<think>'
        if opening == thinking:
            return None, 'reasoning-malformed'
        thinking = opening
    if thinking:
        return None, 'reasoning-incomplete'
    return text[markers[-1].end():], None


def _json_calls(text):
    decoder, calls, position = json.JSONDecoder(), [], 0
    while (match := TOOL_START.search(text, position)) is not None:
        try:
            value, end = decoder.raw_decode(text, match.start())
        except ValueError:
            position = match.start() + 1
            continue
        if isinstance(value, dict) and isinstance(value.get('tool'), str):
            calls.append(value)
        position = end
    return calls


def extract_call_tolerant(text, truncated=False):
    """Return (call, status, n_candidates). Last complete action wins."""
    text, problem = _strip_reasoning(text)
    if problem:
        return None, problem, 0
    fences = [(lang.strip().lower(), body) for lang, body in FENCE_ANY.findall(text)]
    python = [body for lang, body in fences
              if lang in ('python', 'py', 'python3') or
              (lang == '' and not TOOL_START.match(body.strip()))]
    if python:
        status = 'python-last-of-%d' % len(python) if len(python) > 1 else 'python-fence'
        if truncated:
            status += '-before-truncation'
        return {'tool': 'run_python', 'arguments': {'code': python[-1]}}, status, len(python)
    calls = _json_calls(text)
    if calls and not truncated:
        call = calls[-1]
        if not isinstance(call.get('arguments'), dict):
            call = {'tool': call['tool'], 'arguments': {}}
        return call, ('json-last-of-%d' % len(calls) if len(calls) > 1 else 'json'), len(calls)
    return None, ('generation-truncated' if truncated else 'no-action'), 0


def extract_call(interface, text, truncated):
    if interface == 'strict':
        if truncated:
            return None, 'generation-truncated', 0
        call, status = runtime.extract_call(text)
        return call, status, int(call is not None)
    return extract_call_tolerant(text, truncated)


def tolerant_note(status, n):
    if n > 1:
        return (f'[interface] Your reply contained {n} actions; only the last one was executed. '
                'Send one action per turn.')
    if status.endswith('-before-truncation'):
        return ('[interface] Your reply hit the output-token limit; the last complete Python '
                'fence before the cut was executed. Keep replies shorter.')
    return None


# ---------------------------------------------------------------- worker side

def _agent_import(bindings, real_import=builtins.__import__):
    def agent_import(name, globals=None, locals=None, fromlist=(), level=0):
        wanted = [n for n in (fromlist or ()) if n in bindings]
        if level == 0 and wanted and (name == 'fin_skills' or name.startswith('fin_skills.')):
            module = types.ModuleType(name)
            try:
                real = real_import(name, globals, locals, fromlist, level)
                module.__dict__.update(vars(real))
            except ImportError:
                pass
            for n in wanted:
                setattr(module, n, bindings[n])
            return module
        return real_import(name, globals, locals, fromlist, level)
    return agent_import


def run_python_tolerant(code, tools, state=None, decision=None):
    """trading_worker_v6.run_python with agent-level imports of bound tools resolved."""
    import contextlib
    import io
    import traceback
    from benchmarks.agent_study.trading_worker_v6 import Submitted
    from benchmarks.agent_study.trading_tools_v6 import validate_weights
    out, err, target = io.StringIO(), io.StringIO(), None

    def submit(weights):
        nonlocal target
        target, problem = validate_weights(weights)
        if problem:
            raise ValueError(problem)
        raise Submitted()
    bindings = channel_bindings(dict(tools.python_bindings(), submit=submit), tools, decision)
    namespace = {'__name__': '__main__', **(restore_namespace(state) if state else {}), **bindings}
    namespace['__builtins__'] = dict(vars(builtins), __import__=_agent_import(bindings))
    ok, error = True, None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, 'agent_code.py', 'exec'), namespace)
        except Submitted:
            pass
        except BaseException as exc:
            error = f'{type(exc).__name__}: {str(exc)[:500]}'
            ok, target = False, None
            traceback.print_exc(limit=4)
    text = out.getvalue() + ('\n[stderr]\n' + err.getvalue() if err.getvalue() else '')
    result = dict(ok=ok, error=error, output=text, truncated=False,
                  submission=target, tool_calls=tools.calls)
    if state is not None:
        result[STATE_OUT] = capture_namespace(namespace, set(bindings))
    return result


# ------------------------------------------------- opt-in: persistent Python namespace
# Each run_python call is a fresh confined worker process, and the sealed prompt says state does
# not persist; 14B/32B tolerant still reused earlier variables (12 NameError turns in
# 2367730/2367732). With --persistent-python the controller carries the previous namespace,
# pickled inside the sandbox, into the next run_python request. The parent never unpickles it.
# run_python_tolerant(state='') on the first call; the namespace after the run, including
# assignments made before an exception, comes back in STATE_OUT. Keep new code below
# run_python_tolerant: agent tracebacks show its line numbers, so moving them changes feedback.
STATE_IN, STATE_OUT = '_fin_rsi_state', '_fin_rsi_state_out'
STATE_VALUE_LIMIT, STATE_TOTAL_LIMIT = 16 << 20, 64 << 20


def capture_namespace(namespace, bound):
    """Picklable user variables; modules by name; functions/unpicklable values are dropped."""
    import base64
    import pickle
    kept, total = {}, 0
    for name, value in namespace.items():
        if name.startswith('__') or name in bound:
            continue
        if isinstance(value, types.ModuleType):
            kept[name] = ('module', value.__name__)
            continue
        try:
            blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.loads(blob)  # agent-defined classes/functions pickle but do not load later
        except Exception:
            continue
        if len(blob) > STATE_VALUE_LIMIT or total + len(blob) > STATE_TOTAL_LIMIT:
            continue
        kept[name], total = ('value', blob), total + len(blob)
    return base64.b64encode(pickle.dumps(kept)).decode('ascii')


def restore_namespace(state):
    import base64
    import importlib
    import pickle
    restored = {}
    for name, (kind, payload) in pickle.loads(base64.b64decode(state)).items():
        try:
            restored[name] = (importlib.import_module(payload) if kind == 'module'
                              else pickle.loads(payload))
        except Exception:
            continue
    return restored


def unknown_algorithm_message(algorithm_id):
    """Unknown-id error that names the closest ready ids and lists every ready id by task."""
    import difflib
    from collections import defaultdict
    from fin_skills.algorithms import catalog
    by_task = defaultdict(list)
    for card in catalog():
        if card['status'] == 'ready':
            by_task[card['task']].append(card['id'])
    ready = sorted(i for ids in by_task.values() for i in ids)
    close = difflib.get_close_matches(str(algorithm_id), ready, n=2, cutoff=0.55)
    # Portfolio weights first: run_python truncates uncaught errors at 500 characters.
    portfolio = ', '.join(sorted(by_task.pop('portfolio', [])))
    others = '; '.join(f'{task}: ' + ', '.join(sorted(ids)) for task, ids in by_task.items())
    return (f'unknown algorithm_id {algorithm_id!r}. '
            + (f'Did you mean: {", ".join(close)}? ' if close else '')
            + f'Ready portfolio-weight ids: {portfolio}. Other ready ids: {others}.')


# Opt-in (--algorithm-hint, tolerant only): unknown algorithm_id errors from run_algorithm and
# describe_algorithm name the valid ids. Everything else is the current Tools.
from benchmarks.agent_study.trading_capabilities import Tools as _Tools  # noqa: E402


class HintTools(_Tools):
    # Same signatures as the parent: the Python bindings introspect them.
    def run_algorithm(self, algorithm_id, tickers=None, lookback=252, parameters=None):
        try:
            return super().run_algorithm(algorithm_id, tickers, lookback, parameters)
        except ValueError as exc:
            if str(exc).startswith('unknown algorithm_id'):
                raise ValueError(unknown_algorithm_message(algorithm_id)) from None
            raise

    def describe_algorithm(self, algorithm_id=''):
        try:
            return super().describe_algorithm(algorithm_id)
        except ValueError as exc:
            if str(exc).startswith('unknown algorithm_id'):
                raise ValueError(unknown_algorithm_message(algorithm_id)) from None
            raise


SKILL_STOPWORDS = frozenset('a an and for in of on or the to with'.split())
WORD = re.compile(r'[a-z0-9]+')


def rank_skills(cards, query):
    """Ranked skill search (opt-in ``--skill-search ranked``).

    The frozen list_skills matches the whole query as one substring and returns catalog
    (alphabetical) order with full descriptions, 3-4 per 4000-character page; "min variance"
    or "portfolio optimizer" match nothing and "portfolio" puts portfolio-optimizers on page 7.
    Here the query is split into words; a word matches a skill when some word of its name or
    description starts with it. Skills matching any word are ranked by matched words, then
    words matched in the name, then occurrences; ties keep catalog order. Rows carry the
    name and a short description (<= 120 characters).
    """
    words = [w for w in dict.fromkeys(WORD.findall(str(query).lower())) if w not in SKILL_STOPWORDS]
    ranked = []
    for index, card in enumerate(cards):
        name = WORD.findall(card['name'].lower())
        text = name + WORD.findall(card.get('description', '').lower())
        hits = [sum(t.startswith(w) for t in text) for w in words]
        matched = sum(h > 0 for h in hits)
        if words and not matched:
            continue
        in_name = sum(any(t.startswith(w) for t in name) for w in words)
        ranked.append(((-matched, -in_name, -sum(hits), index), card))
    ranked.sort(key=lambda item: item[0])
    rows = []
    for _, card in ranked:
        short = ' '.join(str(card.get('summary') or card.get('description', '')).split())
        rows.append(dict(name=card['name'],
                         description=short if len(short) <= 120 else short[:117].rstrip() + '...'))
    return rows


class RankedSkillTools(_Tools):
    def list_skills(self, query="", offset=0, limit=8):
        import fin_skills
        from benchmarks.agent_study.trading_tools_v5 import rows_page
        return rows_page(rank_skills(fin_skills.catalog(), query), offset, limit, "skills",
                         query=query, usage='read_skill(name) for the full text')


class SkillNameHintTools(_Tools):
    """Opt-in (--skill-name-hint): read_skill(<algorithm_id>) names the algorithm's own skill.

    2367852: all 8 failed read_skill calls passed an algorithm id (equal_weight, min_variance,
    inverse_volatility) and got only "Unknown skill; use list_skills(query=...)".
    """
    # Same signature as the parent: the Python bindings introspect it.
    def read_skill(self, name="", offset=0, limit=4000, reference=None):
        try:
            return super().read_skill(name, offset, limit, reference)
        except ValueError as exc:
            if not str(exc).startswith('Unknown skill'):
                raise
            from fin_skills.algorithms import catalog
            card = next((c for c in catalog() if c['id'] == name), None)
            if card is None or not card.get('skill'):
                raise
            raise ValueError(f'{name!r} is an algorithm_id, not a skill name; its skill is '
                             f'{card["skill"]!r} (read_skill(name={card["skill"]!r})).') from None


_OVERLAYS = {}


def load_skill_overlay(spec):
    """``<skill>=<path>`` -> {skill: text}; cached. Call before sandbox confinement (the worker
    cannot read repository files afterwards), then ``read_skill`` serves the cached text."""
    import fin_skills
    from pathlib import Path
    name, sep, path = spec.partition('=')
    if not sep or not name or not path:
        raise ValueError('--skill-overlay expects <skill>=<path>')
    if name not in fin_skills.names():
        raise ValueError(f'skill overlay: unknown skill {name!r}')
    if (spec) not in _OVERLAYS:
        _OVERLAYS[spec] = (name, Path(path).read_text(encoding='utf-8'))
    return _OVERLAYS[spec]


class SkillOverlayTools(_Tools):
    """Opt-in (--skill-overlay <skill>=<path>): read_skill(<skill>) serves the candidate text.

    Pagination (offset/limit/next_offset/total_chars) is computed on the overlay text;
    references, list_skills (name and description) and all other tools are unchanged.
    """
    overlay_spec = None  # set from FIN_RSI_SKILL_OVERLAY by env_tools_class

    # Same signature as the parent: the Python bindings introspect it.
    def read_skill(self, name="", offset=0, limit=4000, reference=None):
        if self.overlay_spec and reference is None:
            skill, text = load_skill_overlay(self.overlay_spec)
            if name == skill:
                import fin_skills
                from benchmarks.agent_study.trading_tools_v5 import text_page
                return text_page(text, offset, limit, name=name,
                                 references=list(fin_skills.references(name)))
        return super().read_skill(name, offset, limit, reference)


class SnapshotLogTools(_Tools):
    """Opt-in (--snapshot-tool-log): log a copy of each tool reply, not the live object.

    trading_tools_v5 logs ``weights=result["result"].get("weights")``, the same dict the agent's
    Python receives; code that edits ``reply["result"]["weights"]`` in place rewrote the logged
    algorithm output (② 2026-10-19, job 2368333). The log is not model-visible, so behaviour is
    unchanged; only tool_calls in decision.json become faithful to what the tool returned.
    """
    def call(self, name, arguments):
        import copy
        start = len(self.calls)
        result = super().call(name, arguments)
        self.calls[start:] = [copy.deepcopy(c) for c in self.calls[start:]]
        return result


def tools_class(algorithm_hint=False, skill_search='substring', skill_name_hint=False,
                skill_overlay=None, snapshot_tool_log=False):
    """Tools for the chosen opt-in interface fixes; with none, the frozen current Tools."""
    bases = (([HintTools] if algorithm_hint else []) +
             ([RankedSkillTools] if skill_search == 'ranked' else []) +
             ([SkillNameHintTools] if skill_name_hint else []) +
             ([SnapshotLogTools] if snapshot_tool_log else []))
    if skill_overlay:
        bases.append(type('ConfiguredSkillOverlayTools', (SkillOverlayTools,),
                          dict(overlay_spec=skill_overlay)))
    if not bases:
        return _Tools
    if len(bases) == 1:
        return bases[0]
    return type('InterfaceTools', tuple(bases), {})


def env_tools_class():
    return tools_class(os.environ.get('FIN_RSI_ALGORITHM_HINT') == '1',
                       os.environ.get('FIN_RSI_SKILL_SEARCH', 'substring'),
                       os.environ.get('FIN_RSI_SKILL_NAME_HINT') == '1',
                       os.environ.get('FIN_RSI_SKILL_OVERLAY') or None,
                       os.environ.get('FIN_RSI_SNAPSHOT_TOOL_LOG') == '1')


def python_only_message(tool):
    """Error for a JSON call to a function that exists only inside Python (opt-in)."""
    example = ("```python\nh = load_history(lookback=252)\nreturns = h['returns']\n```"
               if tool == 'load_history' else f'```python\nreply = {tool}(...)\n```')
    message = (f'{tool!r} is a Python-only function, not a JSON tool. Call it inside one Python '
               f'fence, e.g. {example}')
    if tool == 'load_history':
        message += '\nOr call run_algorithm(algorithm_id, ...) directly: it loads the same history internally.'
    return message


class InterfaceControllerMixin:
    """Controller-side opt-in fixes; mixed into the runner's Controller only when enabled."""
    persistent_python = False
    python_only_hint = False

    def call(self, tool, arguments):
        persistent = self.persistent_python and tool == 'run_python' and isinstance(arguments, dict)
        if persistent:
            arguments = dict(arguments, **{STATE_IN: getattr(self, 'python_state', '')})
        result = super().call(tool, arguments)
        state = result.pop(STATE_OUT, None)
        if persistent and state is not None:
            self.python_state = state
        if (self.python_only_hint and result.get('error') == f'unknown tool {tool!r}' and
                tool in self.tools.python_bindings()):
            result['error'] = python_only_message(tool)
        return result


def worker_run_python():
    """run_python for this worker process; reads carried state / decision channel before confinement."""
    import sys
    from pathlib import Path
    persistent = os.environ.get('FIN_RSI_PERSISTENT_PYTHON') == '1'
    request = json.loads((Path(sys.argv[1]) / 'request.json').read_text())
    arguments = request.get('arguments') or {}
    decision = arguments.get(DECISION_IN) if isinstance(arguments, dict) else None
    if not persistent and decision is None:
        return run_python_tolerant
    state = arguments.get(STATE_IN, '') if persistent else None
    return lambda code, tools: run_python_tolerant(code, tools, state=state, decision=decision)


# ------------------------------------------- opt-in: structured decision channel (I18, 2026-10-19)
# --decision-channel structured: the decision must be allocate(method, max_weight, max_turnover).
# The tool runs one of the five menu methods point-in-time (as run_algorithm, lookback 252, default
# parameters), caps every weight at max_weight (water-filling: excess pro rata to assets below the
# cap, equally if those hold nothing), then limits one-way turnover from the current holdings to
# max_turnover by moving part of the way (convex combination, so budget, long-only and the cap
# still hold), and submits. submit(weights) is disabled in this channel (JSON and Python).
DECISION_IN = '_fin_rsi_decision'
MENU_METHODS = ('equal_weight', 'inverse_volatility', 'min_variance', 'hrp', 'cross_sectional_momentum')
MAX_WEIGHT_RANGE, MAX_TURNOVER_RANGE = (0.10, 1.0), (0.0, 1.0)
SUBMIT_REDIRECT = ('submit(weights) is not available in this task; finish with '
                   'allocate(method, max_weight, max_turnover).')


def check_allocation(method, max_weight=1.0, max_turnover=1.0):
    if method not in MENU_METHODS:
        raise ValueError('method must be one of: ' + ', '.join(MENU_METHODS))
    values = []
    for name, value, (lo, hi) in (('max_weight', max_weight, MAX_WEIGHT_RANGE),
                                  ('max_turnover', max_turnover, MAX_TURNOVER_RANGE)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a number between {lo} and {hi}')
        if not lo <= float(value) <= hi:
            raise ValueError(f'{name} must be between {lo} and {hi}')
        values.append(float(value))
    return method, values[0], values[1]


def cap_weights(w, cap):
    """Water-filling cap on a normalized long-only vector (feasible for cap >= 1/len(w))."""
    import numpy as np
    w = np.asarray(w, float) / np.sum(w)
    for _ in range(200):
        over = w > cap + 1e-12
        excess = float((w[over] - cap).sum()) if over.any() else 0.0
        w[over] = cap
        if excess <= 1e-15:
            break
        room = w < cap - 1e-12
        base = w[room]
        share = base / base.sum() if base.sum() > 1e-12 else np.ones(int(room.sum())) / room.sum()
        w[room] += excess * share
    return w


def allocate_weights(method_weights, holdings, max_weight, max_turnover):
    """Deterministic allocate(): method output -> cap -> turnover limit. Returns (weights, stats)."""
    import numpy as np
    tickers = list(holdings)
    w = np.array([float(method_weights.get(t, 0.0)) for t in tickers])
    if w.sum() <= 0:
        raise ValueError('method returned no weights')
    w = cap_weights(w, max_weight) if max_weight < 1.0 else w / w.sum()
    h = np.array([float(holdings[t]) for t in tickers])
    h = h / h.sum()
    need = 0.5 * float(np.abs(w - h).sum())
    if need > max_turnover:
        w = h + (max_turnover / need) * (w - h)
    weights = {t: round(float(x), 6) for t, x in zip(tickers, w)}
    stats = dict(max_weight_achieved=round(float(w.max()), 6),
                 turnover_achieved=round(0.5 * float(np.abs(w - h).sum()), 6),
                 method_turnover=round(need, 6))
    return weights, stats


def channel_bindings(bindings, tools, decision):
    """Python bindings for the decision channel; unchanged unless the channel is structured."""
    if not decision or decision.get('channel') != 'structured':
        return bindings
    real_submit, run_algorithm, holdings = bindings['submit'], bindings['run_algorithm'], decision['holdings']

    def submit(*args, **kwargs):
        raise ValueError(SUBMIT_REDIRECT)

    def allocate(method, max_weight=1.0, max_turnover=1.0):
        method, max_weight, max_turnover = check_allocation(method, max_weight, max_turnover)
        start = len(tools.calls)
        reply = run_algorithm(algorithm_id=method, lookback=252)
        for c in tools.calls[start:]:
            c['via'] = 'allocate'
        if not reply.get('ok'):
            raise ValueError(f"allocate could not run {method}: {reply.get('error')}")
        weights, stats = allocate_weights(reply['result']['weights'], holdings, max_weight, max_turnover)
        tools.calls.append(dict(tool='allocate', ok=True, method=method, max_weight=max_weight,
                                max_turnover=max_turnover, weights=dict(weights), **stats))
        real_submit(weights)
    return dict(bindings, submit=submit, allocate=allocate)


class DecisionChannelMixin:
    """Controller side of the structured channel: JSON allocate, and the channel for run_python."""
    holdings = None

    def call(self, tool, arguments):
        if tool == 'run_python' and isinstance(arguments, dict):
            arguments = dict(arguments, **{DECISION_IN: dict(channel='structured', holdings=self.holdings)})
            return super().call(tool, arguments)
        if tool != 'allocate':
            return super().call(tool, arguments)
        args = arguments if isinstance(arguments, dict) else {}
        try:
            if set(args) - {'method', 'max_weight', 'max_turnover'}:
                raise ValueError('allocate takes method, max_weight, max_turnover')
            method, max_weight, max_turnover = check_allocation(
                args.get('method'), args.get('max_weight', 1.0), args.get('max_turnover', 1.0))
        except ValueError as exc:
            self.calls.append(dict(tool='allocate', ok=False, error=str(exc)))
            return dict(ok=False, error=f'ValueError: {exc}')
        start = len(self.calls)
        reply = super().call('run_algorithm', dict(algorithm_id=method, lookback=252))
        for c in self.calls[start:]:
            c['via'] = 'allocate'
        if not reply.get('ok'):
            self.calls.append(dict(tool='allocate', ok=False, method=method, error=reply.get('error')))
            return dict(ok=False, error=f"allocate could not run {method}: {reply.get('error')}")
        weights, stats = allocate_weights(reply['result']['weights'], self.holdings, max_weight, max_turnover)
        self.calls.append(dict(tool='allocate', ok=True, method=method, max_weight=max_weight,
                               max_turnover=max_turnover, weights=dict(weights), **stats))
        return dict(ok=True, submission=weights,
                    allocation=dict(method=method, max_weight=max_weight, max_turnover=max_turnover, **stats))


if __name__ == '__main__':
    # Worker entry: same confinement and Tools as trading_worker_v7; only run_python differs.
    from benchmarks.agent_study import trading_worker_v6
    if os.environ.get('FIN_RSI_SKILL_OVERLAY'):
        load_skill_overlay(os.environ['FIN_RSI_SKILL_OVERLAY'])  # before confinement
    if os.environ.get('FIN_RSI_INTERFACE') == 'tolerant':
        trading_worker_v6.run_python = worker_run_python()
    trading_worker_v6.main(env_tools_class())
