"""Frozen out-of-sample trading study on real US daily data: agent with vs without fin-skills.

One path = one (model, seed, arm). Every STEP sessions from START the agent sees the visible
data through the decision session, may use up to MAX_TURNS tool calls, and submits long-only
target weights that execute at the next session's close and are held until the next trade.
The hidden ledger prices the path with total-return closes and COST_BPS per side. The arms
differ only in the library tools and documentation. Nothing about the model's outputs is
repaired or selected; an invalid or missing submission leaves holdings unchanged.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from benchmarks.agent_study.market_data import (
    HIDDEN, TICKERS, VISIBLE, adjust, load_visible, truncate, write_dataset)

ARMS = ('raw', 'library')
SEEDS = (11, 23, 37)
MODELS = (
    ('7b', 'Qwen/Qwen2.5-Coder-7B-Instruct', 'c03e6d358207e414f1eca0bb1891e29f1db0e242'),
    ('14b', 'Qwen/Qwen2.5-Coder-14B-Instruct', 'aedcc2d42b622764e023cf882b6652e646b95671'),
    ('32b', 'Qwen/Qwen2.5-Coder-32B-Instruct', '381fc969f78efac66bc87ff7ddeadb7e73c218a7'))
START, STEP, COST_BPS = '2025-01-02', 10, 5.0
MAX_TURNS, MAX_TOKENS, TOOL_TIMEOUT = 8, 1024, 60
READY_ALGORITHMS = ('equal_weight', 'inverse_volatility', 'min_variance', 'hrp',
                    'cross_sectional_momentum', 'momentum', 'ma_crossover', 'macd',
                    'rsi_reversion', 'bollinger_reversion', 'donchian_breakout',
                    'vol_target_momentum', 'rolling_market_state', 'historical_volatility',
                    'ewma_volatility', 'historical_var_es', 'normal_var_es', 'naive', 'drift',
                    'mean', 'arima')
SKILLS = ('portfolio-optimizers', 'trend-following-models', 'signal-construction',
          'corporate-actions-processing', 'risk-measures-var-cvar', 'regime-detection')

SYSTEM = """You manage a long-only portfolio of 12 US-listed instruments: {tickers}.
Workspace files (read-only): quotes.csv, columns date,ticker,open,high,low,close,volume,
one row per ticker and session through today's close; quotes are RAW, not adjusted for
splits or dividends. corporate_actions.csv, columns date,ticker,kind,value,note (kind is
split or dividend). README.md. Today is {date} (session {index} of the
history). Target weights you submit are executed at the NEXT session's close, held for
{step} sessions, then you decide again. Trading costs {cost:g} basis points per side on the
traded notional. Objective: maximize net risk-adjusted return over the holding period; there
is no benchmark to track and cash earns zero. Current holdings after price drift: {holdings}.
Each turn, call exactly one tool with a JSON object {{"tool": <name>, "arguments": {{...}}}}.
For run_python you may instead write {{"tool": "run_python"}} followed by the code in one
```python fenced block, which avoids escaping code inside JSON. Tools:
- read_file(path, offset=0, limit=6000): part of a workspace file as text.
- run_python(code): run Python (numpy, pandas, scipy) with the workspace as the working
  directory; print what you need to see; {timeout} second limit; no network or writes.
- submit(weights): {{ticker: weight}} with every weight >= 0 and the total <= 1; the
  remainder is cash. Omitted tickers get weight 0. This ends the decision.
You have at most {turns} turns. If you do not submit, holdings stay unchanged."""

LIBRARY = """The fin-skills financial library is available in this condition.
- run_python may `import fin_skills` (fin_skills.algorithms.run(algorithm_id, data, **params),
  fin_skills.api.get(guard).run(...)).
- list_algorithms(): catalog of executable algorithms. Ready and price-compatible:
  {algorithms}.
- describe_algorithm(algorithm_id): the catalog card (inputs, parameters, caveats).
- run_algorithm(algorithm_id, tickers=[...], lookback=252, parameters={{}}): the library
  builds split- and dividend-adjusted inputs from the workspace files and runs the method.
  Portfolio methods (input asset_returns) return target weights over the given tickers;
  signal and regime methods return the latest values per ticker, already lagged one session;
  volatility, risk and forecast methods return their statistics per ticker.
- run_guard(name, ticker): adjustment_check (does a close series jump at a split?) or
  data_quality (bar-level anomalies).
- read_skill(name): a dated skill document; names: {skills}.
Guards diagnose; they do not repair data. Weights returned by run_algorithm are a valid
submit argument."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as out:
        json.dump(value, out, indent=2, allow_nan=False)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def schedule(sessions):
    """Decision sessions: every STEP sessions from the first session on/after START, such
    that the trade session (next) and at least one holding session exist."""
    first = next(i for i, d in enumerate(sessions) if d >= START)
    picks = [i for i in range(first, len(sessions) - 2, STEP)]
    return picks


FENCE = re.compile(r'```(?:python|py)[^\n]*\n(.*?)\n?```', re.DOTALL)
TRIPLE = re.compile(r'"code"\s*:\s*(?:\"\"\"|\'\'\')(.*?)(?:\"\"\"|\'\'\')', re.DOTALL)
TOOL = re.compile(r'"tool"\s*:\s*"([a-z_]+)"')


def extract_call(text):
    """Tool call from a response; the same rule for both arms.

    1. The first complete JSON object with a "tool" key. For run_python without a code
       string, the first ```python fence supplies the code.
    2. Otherwise, a run_python call whose code is in a ```python fence or in a Python
       triple-quoted "code" value (invalid JSON that models often write): status lenient.
    Nothing else is repaired; code and weights are never edited.
    """
    call, status = extract_json(text)
    fence = FENCE.search(text)
    if call is not None:
        arguments = call.get('arguments') if isinstance(call.get('arguments'), dict) else {}
        if call.get('tool') == 'run_python' and not arguments.get('code') and fence:
            call = dict(call, arguments=dict(arguments, code=fence.group(1)))
            status = 'json+fence'
        return call, status
    tool = TOOL.search(text)
    if tool and tool.group(1) == 'run_python':
        triple = TRIPLE.search(text)
        code = fence.group(1) if fence else (triple.group(1) if triple else None)
        if code:
            return {'tool': 'run_python', 'arguments': {'code': code}}, 'lenient'
    return None, 'unparsed'


def extract_json(text):
    """First complete JSON object in the response; same rule for both arms."""
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = stripped.strip('`')
        if stripped.startswith('json'):
            stripped = stripped[4:]
    start = stripped.find('{')
    while start != -1:
        try:
            value, _ = decoder.raw_decode(stripped[start:])
            if isinstance(value, dict) and 'tool' in value:
                return value, 'json'
        except ValueError:
            pass
        start = stripped.find('{', start + 1)
    return None, 'unparsed'


def validate_weights(weights):
    if not isinstance(weights, dict) or not weights:
        return None, 'weights must be a non-empty object'
    clean = {}
    for ticker, value in weights.items():
        if ticker not in TICKERS:
            return None, f'unknown ticker {ticker!r}'
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None, f'weight for {ticker} is not a number'
        if not np.isfinite(value) or value < -1e-9:
            return None, f'weight for {ticker} must be finite and >= 0'
        clean[ticker] = max(value, 0.0)
    total = sum(clean.values())
    if total > 1 + 1e-4:
        return None, f'weights sum to {total:.4f} > 1'
    return {t: clean.get(t, 0.0) for t in TICKERS}, None


class Controller:
    """Tools that need no sandbox plus dispatch of sandboxed ones; per-decision workspace."""

    def __init__(self, root, arm, workspace):
        self.root, self.arm, self.workspace = root, arm, Path(workspace)
        self.calls = []

    def read_file(self, path='', offset=0, limit=6000):
        name = Path(str(path)).name
        target = self.workspace/name
        if name not in VISIBLE or not target.exists():
            return dict(ok=False, error=f'files: {", ".join(VISIBLE)}')
        text = target.read_text(encoding='utf8')
        offset, limit = max(int(offset), 0), min(max(int(limit), 1), 6000)
        return dict(ok=True, path=name, offset=offset, total_chars=len(text),
                    text=text[offset:offset+limit])

    def list_algorithms(self):
        import fin_skills.algorithms as algorithms
        cards = [c for c in algorithms.catalog() if c['id'] in READY_ALGORITHMS]
        return dict(ok=True, algorithms=[dict(id=c['id'], task=c['task'], inputs=c['inputs'],
                                              name=c['name']) for c in cards])

    def describe_algorithm(self, algorithm_id=''):
        import fin_skills.algorithms as algorithms
        for card in algorithms.catalog():
            if card['id'] == algorithm_id:
                return dict(ok=True, card={k: v for k, v in card.items() if k != 'module'})
        return dict(ok=False, error=f'unknown algorithm_id {algorithm_id!r}')

    def read_skill(self, name=''):
        import fin_skills
        if name not in SKILLS:
            return dict(ok=False, error=f'skills: {", ".join(SKILLS)}')
        return dict(ok=True, name=name, text=fin_skills.load(name)[:6000])

    def sandboxed(self, tool, arguments):
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='trade-', dir=self.root/'tmp') as td:
            box = Path(td)/'box'
            box.mkdir()
            (box/'request.json').write_text(json.dumps(dict(arm=self.arm, tool=tool,
                                                             arguments=arguments)))
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', TMPDIR=str(box), MPLCONFIGDIR=str(box))
            package_root = str(Path(__file__).resolve().parents[2])
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [package_root,
                                                              env.get('PYTHONPATH')]))
            try:
                process = subprocess.run([sys.executable, '-B', '-m',
                    'benchmarks.agent_study.trading_worker', str(box), str(self.workspace)],
                    capture_output=True, text=True, timeout=TOOL_TIMEOUT + 30, env=env)
                if (box/'result.json').exists():
                    result = json.loads((box/'result.json').read_text())
                else:
                    result = dict(ok=False, error='worker crashed: '
                                  + process.stderr[-1500:])
            except subprocess.TimeoutExpired:
                result = dict(ok=False, error=f'tool exceeded {TOOL_TIMEOUT} seconds')
        result['seconds'] = round(time.perf_counter() - started, 3)
        return result

    def call(self, tool, arguments):
        arguments = arguments if isinstance(arguments, dict) else {}
        library_only = ('list_algorithms', 'describe_algorithm', 'read_skill',
                        'run_algorithm', 'run_guard')
        if tool in library_only and self.arm != 'library':
            result = dict(ok=False, error=f'tool {tool!r} is not available in this condition')
        elif tool == 'read_file':
            result = self.read_file(**{k: v for k, v in arguments.items()
                                       if k in ('path', 'offset', 'limit')})
        elif tool in ('list_algorithms', 'describe_algorithm', 'read_skill'):
            try:
                result = getattr(self, tool)(**{k: v for k, v in arguments.items()
                                                if k in ('algorithm_id', 'name')})
            except TypeError as exc:
                result = dict(ok=False, error=str(exc))
        elif tool in ('run_python', 'run_algorithm', 'run_guard'):
            result = self.sandboxed(tool, arguments)
        else:
            result = dict(ok=False, error=f'unknown tool {tool!r}')
        self.calls.append(dict(tool=tool, ok=bool(result.get('ok'))))
        return result


def holdings_text(weights):
    held = {t: round(w, 4) for t, w in weights.items() if w > 1e-6}
    return json.dumps(held) if held else 'all cash'


def system_prompt(arm, date, index, holdings):
    text = SYSTEM.format(tickers=', '.join(TICKERS), date=date, index=index, step=STEP,
                         cost=COST_BPS, holdings=holdings_text(holdings), timeout=TOOL_TIMEOUT,
                         turns=MAX_TURNS)
    if arm == 'library':
        text += '\n' + LIBRARY.format(algorithms=', '.join(READY_ALGORITHMS),
                                      skills=', '.join(SKILLS))
    return text


def decide(backend, controller, arm, date, index, holdings):
    """One decision: multi-turn tool loop; returns record with target weights or None."""
    history = [dict(role='system', content=system_prompt(arm, date, index, holdings)),
               dict(role='user', content='Decide the target weights for the next holding '
                                         'period. Use the tools, then submit.')]
    turns, usage, seconds, target, error = [], Counter(), 0.0, None, None
    for turn in range(1, MAX_TURNS + 1):
        try:
            response = backend(history)
        except ValueError as exc:  # context budget exceeded: the decision ends
            error = f'context: {exc}'
            break
        text = response['choices'][0]['message']['content']
        usage['prompt_tokens'] += response['usage']['prompt_tokens']
        usage['completion_tokens'] += response['usage']['completion_tokens']
        seconds += response['generation_seconds']
        call, status = extract_call(text)
        record = dict(turn=turn, response=text, parse=status,
                      finish_reason=response['choices'][0]['finish_reason'])
        if call is None:
            result = dict(ok=False, error='No tool call could be parsed. Send one JSON '
                          'object {"tool": name, "arguments": {...}}; for run_python, send '
                          '{"tool": "run_python"} followed by the code in one ```python '
                          'fenced block. Python triple quotes are not valid JSON.')
            record['tool'] = None
        elif call.get('tool') == 'submit':
            weights = (call.get('arguments') or {}).get('weights', call.get('arguments'))
            target, problem = validate_weights(weights)
            record['tool'] = 'submit'
            if target is not None:
                record['accepted'] = True
                turns.append(record)
                break
            result = dict(ok=False, error=f'submit rejected: {problem}')
            record['accepted'] = False
        else:
            record['tool'] = call.get('tool')
            result = controller.call(call.get('tool'), call.get('arguments'))
        record['result'] = result
        turns.append(record)
        history += [dict(role='assistant', content=text),
                    dict(role='user', content=json.dumps(result)[:6000])]
    return dict(date=date, index=index, holdings_before=holdings, target=target,
                submitted=target is not None, turns=turns, n_turns=len(turns),
                tool_calls=controller.calls, prompt_tokens=usage['prompt_tokens'],
                completion_tokens=usage['completion_tokens'], generation_seconds=seconds,
                error=error)


def drift(weights, returns):
    """Holdings after one session of price moves; cash is the remainder."""
    grown = {t: weights[t] * (1 + returns[t]) for t in TICKERS}
    cash = 1 - sum(weights.values())
    nav = cash + sum(grown.values())
    return {t: grown[t] / nav for t in TICKERS}, nav


def ledger(total, picks, targets):
    """Price a path. `targets[i]` is the target after decision at session picks[i] or None.

    Trades happen at the close of session picks[i]+1 after that session's return accrues.
    Returns daily NAV (indexed by session date from the first trade session) and trades.
    """
    returns = total.pct_change().fillna(0.0)
    holdings, nav, navs, trades = {t: 0.0 for t in TICKERS}, 1.0, {}, []
    trade_sessions = {p + 1: i for i, p in enumerate(picks)}
    for k in range(picks[0] + 1, len(total)):
        holdings, growth = drift(holdings, returns.iloc[k])
        nav *= growth
        if k in trade_sessions:
            target = targets[trade_sessions[k]]
            if target is not None:
                turnover = sum(abs(target[t] - holdings[t]) for t in TICKERS)
                cost = turnover * COST_BPS / 1e4
                nav *= 1 - cost
                trades.append(dict(session=k, date=total.index[k], turnover=turnover,
                                   cost=cost))
                holdings = dict(target)
            else:
                trades.append(dict(session=k, date=total.index[k], turnover=0.0, cost=0.0,
                                   unchanged=True))
        navs[total.index[k]] = nav
    return pd.Series(navs), trades


def metrics(nav):
    daily = nav.pct_change().dropna()
    years = len(daily) / 252
    total = nav.iloc[-1] / nav.iloc[0] - 1
    vol = float(daily.std(ddof=1) * np.sqrt(252)) if len(daily) > 1 else float('nan')
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else float('nan')
    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std() > 0 else 0.0
    dd = float((nav / nav.cummax() - 1).min())
    return dict(cumulative_return=float(total), annualized_return=float(ann),
                annualized_volatility=vol, sharpe=sharpe, max_drawdown=dd,
                sessions=int(len(nav)))


def baselines(total, picks):
    n = len(picks)
    ew = {t: 1 / len(TICKERS) for t in TICKERS}
    out = {}
    out['cash'] = metrics(ledger(total, picks, [None] * n)[0])
    out['equal_weight_buy_and_hold'] = metrics(ledger(total, picks, [ew] + [None]*(n-1))[0])
    out['equal_weight_rebalanced'] = metrics(ledger(total, picks, [ew] * n)[0])
    sixty = {t: 0.0 for t in TICKERS}
    sixty.update(SPY=0.6, IEF=0.4)
    out['sixty_forty_rebalanced'] = metrics(ledger(total, picks, [sixty] * n)[0])
    return out


def freeze(root, families=None):
    root.mkdir(parents=True, exist_ok=False)
    (root/'tmp').mkdir()
    hashes = write_dataset(root/'data')
    total = pd.read_csv(root/'data'/HIDDEN, index_col=0)
    picks = schedule(list(total.index))
    rows = []
    for family, model, revision in MODELS:
        if families and family not in families:
            continue
        for seed in SEEDS:
            for arm in ARMS:
                rows.append(dict(id=f'{family}-{seed}-{arm}', family=family, model=model,
                                 revision=revision, seed=seed, arm=arm))
    random.Random(20260929).shuffle(rows)
    write(root/'inputs.json', dict(paths=rows, decision_sessions=picks,
                                   decision_dates=[total.index[i] for i in picks]))
    write(root/'protocol.json', dict(version='trading-study-v2',
        amendment='v1 jobs never started (queue); v2 jobs were stopped after 7 minutes '
                  'because the one-JSON-object format made 7B fail every turn by writing '
                  'code in Python triple quotes; the code-fence option, the lenient '
                  'run_python parse, the parse-failure hint and the column list apply to '
                  'both arms. No return had been computed.',
        created_utc=datetime.now(timezone.utc).isoformat(), universe=TICKERS,
        start=START, step=STEP, cost_bps=COST_BPS, max_turns=MAX_TURNS,
        max_tokens=MAX_TOKENS, temperature=.1, top_p=1., seeds=SEEDS, arms=ARMS,
        models=[m for m in MODELS if not families or m[0] in families],
        decisions_per_path=len(picks), planned_paths=len(rows), data_sha256=hashes,
        raw_download_sha256=sha(Path(__file__).parent/'data'/'market-us-20260929'
                                /'yahoo_raw_20260929.json'),
        input_sha256=sha(root/'inputs.json'),
        primary_endpoint='net Sharpe ratio and net cumulative return of each path over '
                         f'{total.index[picks[0]+1]}..{total.index[-1]}, paired '
                         'library minus raw by (model, seed); block bootstrap over holding '
                         'periods for confidence intervals',
        secondary=['submission rate', 'tool use and library calls', 'turnover',
                   'max drawdown', 'tokens and seconds', 'parse failures'],
        baselines=['cash', 'equal-weight buy and hold', 'equal-weight rebalanced',
                   '60/40 SPY/IEF rebalanced'],
        limits=['One real market path (2025-01 to 2026-09); seeds vary model sampling, '
                'not the market, so paths are not independent samples of markets.',
                'Qwen2.5-Coder training data predates 2025 by public statements; '
                'contamination of the test window cannot be ruled out with certainty.',
                'Long-only, daily closes, no borrowing, zero cash yield, 5 bps per side; '
                'no intraday execution, slippage or capacity modelling.',
                'The library arm receives longer documentation; the arms differ in tools '
                'and text together.',
                'An unchanged holding after a failed decision is a design choice; '
                'results are also reported per submission rate.']))


def qualify(root):
    """Sandbox, worker, tools, ledger; no model. Seed-independent."""
    total = pd.read_csv(root/'data'/HIDDEN, index_col=0)
    picks = schedule(list(total.index))
    date = total.index[picks[0]]
    results, passed = {}, True
    with tempfile.TemporaryDirectory(dir=root/'tmp') as td:
        truncate(root/'data', Path(td)/'ws', date)
        quotes = pd.read_csv(Path(td)/'ws'/'quotes.csv')
        results['truncated'] = dict(last_date=quotes['date'].max(), expected=date)
        passed &= quotes['date'].max() == date
        raw, lib = Controller(root, 'raw', Path(td)/'ws'), Controller(root, 'library',
                                                                       Path(td)/'ws')
        results['read_file'] = raw.call('read_file', dict(path='README.md', limit=80))
        passed &= results['read_file']['ok']
        results['python_raw'] = raw.call('run_python', dict(
            code='import pandas as pd\nq=pd.read_csv("quotes.csv")\nprint(q.date.max())'))
        passed &= results['python_raw']['ok'] and date in results['python_raw']['output']
        results['python_raw_blocked_library'] = raw.call('run_python', dict(
            code='import fin_skills\nprint("imported")'))
        passed &= not results['python_raw_blocked_library']['ok']
        results['python_network'] = raw.call('run_python', dict(
            code='import socket\nsocket.socket()\nprint("open")'))
        passed &= 'PermissionError' in results['python_network']['output']
        results['python_write'] = raw.call('run_python', dict(
            code='open("quotes.csv","a").write("x")'))
        passed &= 'PermissionError' in results['python_write']['output']
        results['raw_denied_algorithm'] = raw.call('run_algorithm',
                                                    dict(algorithm_id='inverse_volatility'))
        passed &= not results['raw_denied_algorithm']['ok']
        results['library_algorithm'] = lib.call('run_algorithm', dict(
            algorithm_id='inverse_volatility', tickers=['SPY', 'TLT', 'GLD'], lookback=120))
        weights = results['library_algorithm'].get('result', {}).get('weights', {})
        passed &= results['library_algorithm']['ok'] and abs(sum(weights.values()) - 1) < 1e-4
        results['library_signal'] = lib.call('run_algorithm', dict(
            algorithm_id='ma_crossover', tickers=['SPY'], lookback=120))
        passed &= results['library_signal']['ok']
        results['library_guard_raw_orly'] = lib.call('run_guard', dict(
            name='adjustment_check', ticker='NVDA'))
        passed &= results['library_guard_raw_orly']['ok'] and \
            results['library_guard_raw_orly']['passed'] is False
        results['library_python'] = lib.call('run_python', dict(
            code='import fin_skills.algorithms as a\nprint(len(a.catalog()))'))
        passed &= results['library_python']['ok']
        results['describe'] = lib.call('describe_algorithm', dict(algorithm_id='hrp'))
        results['skill'] = lib.call('read_skill', dict(name='portfolio-optimizers'))
        passed &= results['describe']['ok'] and results['skill']['ok']
    base = baselines(total, picks)
    results['baselines'] = base
    passed &= all(np.isfinite(v['sharpe']) for v in base.values())
    passed &= base['cash']['cumulative_return'] == 0.0
    write(root/'qualification.json', dict(passed=passed, results=results,
        scope='Sandbox, tools and ledger on the frozen data; not model performance.'))
    if not passed:
        raise RuntimeError('trading study qualification failed')


def run(root, families=None, backend_factory=None):
    protocol = load(root/'protocol.json')
    assert sha(root/'inputs.json') == protocol['input_sha256']
    assert load(root/'qualification.json')['passed']
    inputs = load(root/'inputs.json')
    total = pd.read_csv(root/'data'/HIDDEN, index_col=0)
    picks = inputs['decision_sessions']
    if backend_factory is None:
        from benchmarks.agent_study.transformers_chat import TransformersChat
        import torch

        def backend_factory(model, revision):
            return TransformersChat(model, revision, max_tokens=MAX_TOKENS)
        device = dict(torch=torch.__version__, device=torch.cuda.get_device_name(0))
    else:
        device = dict(torch=None, device='test backend')
    started = root/'inference-started.json'
    if not started.exists():
        write(started, dict(protocol_sha256=sha(root/'protocol.json'),
                            job_id=os.environ.get('SLURM_JOB_ID'), **device))
    returns = total.pct_change().fillna(0.0)
    for family, model, revision in MODELS:
        paths = [p for p in inputs['paths'] if p['family'] == family
                 and (not families or family in families)]
        if not paths:
            continue
        backend = backend_factory(model, revision)
        for path in paths:
            folder = root/'decisions'/path['id']
            holdings = {t: 0.0 for t in TICKERS}
            for i, pick in enumerate(picks):
                file = folder/f'{i:02d}.json'
                if file.exists():
                    record = load(file)
                else:
                    date = total.index[pick]
                    with tempfile.TemporaryDirectory(dir=root/'tmp') as td:
                        truncate(root/'data', Path(td)/'ws', date)
                        controller = Controller(root, path['arm'], Path(td)/'ws')
                        backend.seed, backend.calls = path['seed'] * 1000 + i, 0
                        record = decide(backend, controller, path['arm'], date, pick, holdings)
                    write(file, record)
                    print(json.dumps(dict(phase='decision', path=path['id'], decision=i,
                                          planned=len(picks), submitted=record['submitted'])),
                          flush=True)
                # Advance holdings exactly as the ledger will: trade at pick+1, drift after.
                for k in range(pick + 1, picks[i + 1] + 1 if i + 1 < len(picks) else pick + 1):
                    holdings, _ = drift(holdings, returns.iloc[k])
                    if k == pick + 1 and record['target'] is not None:
                        holdings = dict(record['target'])
        del backend
        gc.collect()
        if device['torch']:
            import torch
            torch.cuda.empty_cache()
    inventory = []
    for path in inputs['paths']:
        folder = root/'decisions'/path['id']
        files = sorted(folder.glob('*.json')) if folder.exists() else []
        inventory.append(dict(id=path['id'], decisions=[dict(path=f.relative_to(root).as_posix(),
                                                              sha256=sha(f)) for f in files]))
    write(root/'inference-receipt.json', dict(rows=inventory, planned_paths=len(inputs['paths']),
                                            decisions_per_path=len(picks),
                                            protocol_sha256=sha(root/'protocol.json')))


def block_bootstrap(diffs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) == 0:
        return None
    means = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n)]
    return dict(mean=float(diffs.mean()), ci95=[float(np.percentile(means, 2.5)),
                                                  float(np.percentile(means, 97.5))],
                n_blocks=int(len(diffs)))


def score(root):
    protocol = load(root/'protocol.json')
    receipt = load(root/'inference-receipt.json')
    assert receipt['protocol_sha256'] == sha(root/'protocol.json')
    inputs = load(root/'inputs.json')
    total = pd.read_csv(root/'data'/HIDDEN, index_col=0)
    picks = inputs['decision_sessions']
    paths = {p['id']: p for p in inputs['paths']}
    results = {}
    for row in receipt['rows']:
        assert len(row['decisions']) == len(picks), row['id']
        records = []
        for item in row['decisions']:
            assert sha(root/item['path']) == item['sha256'], item['path']
            records.append(load(root/item['path']))
        targets = [r['target'] for r in records]
        nav, trades = ledger(total, picks, targets)
        # Holding-period net returns are the bootstrap blocks.
        marks = [t['session'] for t in trades] + [len(total) - 1]
        blocks = [float(nav.loc[total.index[b]] / nav.loc[total.index[a]] - 1)
                  for a, b in zip(marks[:-1], marks[1:])]
        info = paths[row['id']]
        tools = Counter(c['tool'] for r in records for c in r['tool_calls'])
        results[row['id']] = dict(**info, metrics=metrics(nav), blocks=blocks,
            submitted=sum(r['submitted'] for r in records), decisions=len(records),
            mean_turns=float(np.mean([r['n_turns'] for r in records])),
            tool_calls=dict(tools),
            library_calls=sum(tools[k] for k in ('run_algorithm', 'run_guard',
                                                   'list_algorithms', 'describe_algorithm',
                                                   'read_skill')),
            parse_failures=sum(t['parse'] == 'unparsed' for r in records for t in r['turns']),
            lenient_parses=sum(t['parse'] == 'lenient' for r in records for t in r['turns']),
            mean_turnover=float(np.mean([t['turnover'] for t in trades])),
            prompt_tokens=sum(r['prompt_tokens'] for r in records),
            completion_tokens=sum(r['completion_tokens'] for r in records),
            generation_seconds=sum(r['generation_seconds'] for r in records),
            nav={d: round(v, 6) for d, v in nav.items()})
    paired = {}
    for family, _, _ in MODELS:
        for seed in SEEDS:
            lib = results.get(f'{family}-{seed}-library')
            raw = results.get(f'{family}-{seed}-raw')
            if lib and raw:
                paired[f'{family}-{seed}'] = dict(
                    sharpe_diff=lib['metrics']['sharpe'] - raw['metrics']['sharpe'],
                    return_diff=lib['metrics']['cumulative_return']
                    - raw['metrics']['cumulative_return'],
                    block_bootstrap=block_bootstrap(np.array(lib['blocks'])
                                                    - np.array(raw['blocks'])))
    by_family = {}
    for family, _, _ in MODELS:
        for arm in ARMS:
            group = [r for r in results.values() if r['family'] == family and r['arm'] == arm]
            if group:
                by_family[f'{family}-{arm}'] = {
                    k: float(np.mean([g['metrics'][k] for g in group]))
                    for k in group[0]['metrics'] if k != 'sessions'}
                by_family[f'{family}-{arm}']['submission_rate'] = float(np.mean(
                    [g['submitted'] / g['decisions'] for g in group]))
                by_family[f'{family}-{arm}']['library_calls_per_decision'] = float(np.mean(
                    [g['library_calls'] / g['decisions'] for g in group]))
    pooled = [d for p in paired.values() for d in [p['return_diff']]]
    write(root/'scores.json', dict(paths=results, paired=paired, by_family=by_family,
        pooled_return_diff=block_bootstrap(pooled) if pooled else None,
        baselines=baselines(total, picks), window=[total.index[picks[0] + 1], total.index[-1]],
        inference_receipt_sha256=sha(root/'inference-receipt.json')))
    print(json.dumps(dict(by_family=by_family, paired={k: {kk: v[kk] for kk in
        ('sharpe_diff', 'return_diff')} for k, v in paired.items()}), indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'qualify', 'run', 'score'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--families', nargs='*')
    args = parser.parse_args()
    if args.mode in ('freeze', 'run'):
        globals()[args.mode](args.root, args.families)
    else:
        globals()[args.mode](args.root)
