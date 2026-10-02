"""Repaired paired-study tools and accounting; no changes to frozen v3 scoring."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd

from benchmarks.agent_study import trading_study as v3
from benchmarks.agent_study.trading_tools_v4 import OUTPUT_LIMIT, Tools, validate_weights

COMMON = """Use the visible market files to complete the task. Both arms have the same
long-only constraints and execution budget. quotes.csv has columns
date,ticker,open,high,low,close,volume; corporate_actions.csv has date,ticker,kind,value,note.
Quotes are raw; actions contain dated splits and dividends. Dates are YYYY-MM-DD.
Tools: read_file(path, offset=0, limit=4000), run_python(code), submit(weights).
Python has numpy, pandas and scipy; read_file and submit are also callable in Python.
Files are read-only, network is unavailable, Python state does not persist across calls.
Tool output is capped at 4000 characters. Read smaller slices when necessary.
Return one JSON object {"tool": "name", "arguments": {...}} per turn, or one complete
```python code fence for run_python. For example:
{"tool":"read_file","arguments":{"path":"README.md","limit":1000}}
The equivalent Python is print(read_file("README.md", limit=1000)).
To inspect the actual CSV in Python, use:
```python
import pandas as pd
quotes = pd.read_csv("quotes.csv")
print(quotes.columns.tolist())
```
submit(weights) ends the decision. Weights are a nonempty ticker:number mapping, finite,
nonnegative, total at most 1; omitted assets are zero and remainder is cash. Totals up to
1.001 caused by rounding are normalized to 1, and the normalized target is recorded.
You have at most 8 turns and 1024 output tokens per turn. Invalid submissions consume a
turn. Submit is a tool, not a fin_skills package function. No submission keeps holdings.
"""
LIBRARY = """The fin-skills library is available. These tool functions ALSO exist in Python:
list_algorithms(), describe_algorithm(algorithm_id), read_skill(name),
run_algorithm(algorithm_id, tickers=None, lookback=252, parameters=None), run_guard(name,ticker).
Use these study adapters with the SAME arguments in JSON and Python. Inspect the catalog
for methods and their inputs. The menu order is randomized and is not a recommendation.
run_algorithm constructs split/dividend-adjusted inputs from visible data. asset_returns
methods return result.weights; price/signal/statistic methods return result.per_ticker.
For the hrp adapter only, omitted linkage defaults to 'single'; pass parameters to override.
Example of the API shape (choose the method for the task):
card = describe_algorithm(algorithm_id="hrp"); print(card)
Direct package APIs differ: fin_skills.algorithms.run(id, {"asset_returns": dataframe},
**parameters) accepts a nonempty mapping, not a bare DataFrame. Guards are accessed by
fin_skills.api.get(name).run(**keyword_inputs); skill names are not guard names.
Study run_guard supports adjustment_check and data_quality; example:
print(run_guard(name="data_quality", ticker="SPY"))
Guards diagnose; they do not repair data. These examples do not prescribe an allocation.
"""


def extract_call(text):
    # A bare Python fence is code, even when its body contains a JSON tool example.
    stripped = text.strip()
    fences = list(v3.FENCE.finditer(stripped))
    match = v3.FENCE.fullmatch(stripped) if len(fences) == 1 else None
    if match:
        return {'tool': 'run_python', 'arguments': {'code': match.group(1)}}, 'python-fence'
    if len(fences) == 1:
        fence = fences[0]
        outside = stripped[:fence.start()] + stripped[fence.end():]
        if not v3.TOOL.search(outside):
            return {'tool': 'run_python', 'arguments': {'code': fence.group(1)}}, 'python-fence'
    return v3.extract_call(text)


class Controller:
    def __init__(self, root, workspace, arm, menu_seed=0):
        self.root, self.workspace = Path(root), Path(workspace)
        self.arm, self.menu_seed, self.calls = arm, menu_seed, []
        self.tools = Tools(workspace, arm, menu_seed)
        (self.root / 'tmp').mkdir(exist_ok=True, parents=True)

    def call(self, tool, arguments):
        arguments = {} if arguments is None else arguments
        if not isinstance(arguments, dict):
            result = dict(ok=False, error='arguments must be an object')
        elif tool in ('read_file', 'list_algorithms', 'describe_algorithm', 'read_skill'):
            result = self.tools.call(tool, arguments)
        elif tool in ('run_python', 'run_algorithm', 'run_guard'):
            with tempfile.TemporaryDirectory(dir=self.root / 'tmp') as temp:
                box = Path(temp) / 'box'
                box.mkdir()
                (box / 'request.json').write_text(json.dumps(dict(arm=self.arm, tool=tool,
                    arguments=arguments, menu_seed=self.menu_seed)))
                env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                    TMPDIR=str(box), MPLCONFIGDIR=str(box))
                try:
                    proc = subprocess.run([sys.executable, '-B', '-m',
                        'benchmarks.agent_study.trading_worker_v4', str(box),
                        str(self.workspace)], env=env, capture_output=True, text=True,
                        timeout=v3.TOOL_TIMEOUT)
                    result = (json.loads((box / 'result.json').read_text())
                              if (box / 'result.json').exists() else
                              dict(ok=False, error='worker failed: ' + proc.stderr[-1500:]))
                except subprocess.TimeoutExpired:
                    result = dict(ok=False, error='tool timeout')
        else:
            result = dict(ok=False, error=f'unknown tool {tool!r}')
        self.calls.append(dict(tool=tool, ok=bool(result.get('ok'))))
        self.calls.extend(result.get('tool_calls', []))
        return result


def decide(backend, controller, task):
    history = [dict(role='system', content=COMMON + (LIBRARY if controller.arm == 'library'
                                                     else '')),
               dict(role='user', content=task)]
    turns, target = [], None
    for _ in range(v3.MAX_TURNS):
        try:
            response = backend(history)
        except ValueError as exc:
            turns.append(dict(response='', parse='backend-error', tool=None, usage={},
                              result=dict(ok=False, error=str(exc))))
            break
        text = response['choices'][0]['message']['content']
        call, status = extract_call(text)
        row = dict(response=text, parse=status, usage=response.get('usage'),
                   generation_seconds=response.get('generation_seconds', 0.),
                   finish_reason=response['choices'][0]['finish_reason'])
        if call is None:
            result = dict(ok=False, error='Use one tool JSON object or a complete Python fence.')
        elif call.get('tool') == 'submit':
            args = call.get('arguments')
            weights = args.get('weights', args) if isinstance(args, dict) else None
            target, problem = validate_weights(weights)
            result = dict(ok=target is not None, submission=target, error=problem)
        else:
            result = controller.call(call.get('tool'), call.get('arguments'))
            if result.get('ok') and result.get('submission') is not None:
                target, problem = validate_weights(result['submission'])
                if problem:
                    result = dict(ok=False, error=problem)
        row.update(tool=call.get('tool') if call else None, result=result)
        turns.append(row)
        if target is not None:
            break
        history += [dict(role='assistant', content=text),
                    dict(role='user', content=json.dumps(result)[:OUTPUT_LIMIT])]
    return dict(target=target, submitted=target is not None, turns=turns,
                tool_calls=controller.calls, menu_seed=controller.menu_seed)


def ledger(total, picks, targets):
    nav, trades = v3.ledger(total, picks, targets)
    initial = pd.Series([1.0], index=[total.index[picks[0]]])
    return pd.concat([initial, nav]), trades


def metrics(nav):
    """Input includes capital BEFORE the initial trade; first fee counts everywhere."""
    return v3.metrics(nav)


def qualify(root, data_dir):
    """CPU acceptance checks, including reads of actual future data in both arms."""
    from benchmarks.agent_study import market_data as md
    root = Path(root)
    root.mkdir(exist_ok=False, parents=True)
    md.truncate(data_dir, root / 'visible', '2024-12-02')
    results = {}
    for arm in ('raw', 'library'):
        c = Controller(root, root / 'visible', arm)
        probes = {
            'visible': ('import pandas as pd\nq=pd.read_csv("quotes.csv")\n'
                        'assert q.date.max() <= "2024-12-02"\nprint("OK")'),
            'future_source': f'open({str(md.RAW)!r}).read()',
            'future_ledger': f'open({str(Path(data_dir).resolve() / md.HIDDEN)!r}).read()',
            'network': 'import socket; socket.socket()',
            'write': 'open("quotes.csv", "a").write("bad")',
            'submit': 'submit({"SPY": 0.5001, "IEF": 0.5001})',
            'raw_library': 'import fin_skills',
        }
        for name, code in probes.items():
            result = c.call('run_python', dict(code=code))
            if name in ('future_source', 'future_ledger', 'network', 'write'):
                passed = not result.get('ok') and 'PermissionError' in result.get('output', '')
            elif name == 'raw_library' and arm == 'raw':
                passed = not result.get('ok') and 'ImportError' in result.get('output', '')
            else:
                passed = bool(result.get('ok'))
                if name == 'submit':
                    passed &= result.get('submission', {}).get('SPY') == 0.5
            results[f'{arm}/{name}'] = dict(passed=passed, result=result)
        if arm == 'library':
            result = c.call('list_algorithms', None)
            results['library/empty_arguments'] = dict(passed=bool(result.get('ok')),
                                                     result=result)
            for name in ('inverse_volatility', 'hrp'):
                args = dict(algorithm_id=name, tickers=['SPY', 'TLT', 'GLD'], lookback=60)
                direct = c.call('run_algorithm', args)
                python = c.call('run_python', dict(code=f'print(run_algorithm(**{args!r}))'))
                results[f'{arm}/{name}'] = dict(passed=bool(direct.get('ok') and
                    python.get('ok') and "'ok': True" in python.get('output', '')),
                    direct=direct, python=python)
    report = dict(passed=all(r['passed'] for r in results.values()), results=results,
                  scope='CPU qualification only; no model performance or return claim',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob('trading_*v4.py')})
    (root / 'qualification.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    import argparse
    from benchmarks.agent_study import market_data as md
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--data-dir', type=Path, default=md.DATA_DIR)
    args = parser.parse_args()
    result = qualify(args.root, args.data_dir)
    print(json.dumps({k: v['passed'] for k, v in result['results'].items()}))
    sys.exit(0 if result['passed'] else 1)
