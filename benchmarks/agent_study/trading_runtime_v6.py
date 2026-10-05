"""Opt-in v6 history interface. Requires a new protocol; never resume old runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd

from benchmarks.agent_study import trading_study as v3
from benchmarks.agent_study.trading_tools_v6 import OUTPUT_LIMIT, Tools, validate_weights, wire, execution_status

COMMON = """Use only the visible market files for the task. Both arms have the same budget.
quotes.csv: date,ticker,open,high,low,close,volume (raw, NOT split/dividend adjusted).
corporate_actions.csv: date,ticker,kind,value,note. README.md explains both files.
Tools and Python functions:
read_file(path, offset=0, limit=4000)
read_market(file="quotes.csv", tickers=None, start=None, end=None, offset=0, limit=12)
read_snapshot(tickers=None): raw price levels for the last visible session, NOT history.
run_python(code), submit(weights)
read_market returns a DICTIONARY with keys ok, rows, as_of, next_offset, total_rows.
It has no "data" key. For inspection: page=read_market(tickers=["SPY"],limit=3);
assert page["ok"], page; print(page["rows"]). These are raw long-format records.
For complete RAW history in Python:
import pandas as pd
quotes = pd.read_csv("quotes.csv")
prices = quotes.pivot(index="date", columns="ticker", values="close").sort_index()
Raw prices still need split/dividend adjustment before computing investment returns.
read_market defaults to recent rows; optional dates are YYYY-MM-DD.
Both readers return a page, not the full history. Continue with next_offset until null.
For calculations load complete visible CSVs with pandas.read_csv in Python.
Python has numpy/pandas/scipy. Functions above are already bound; no imports for tools.
Python state does NOT persist between calls: load inputs and submit within the same call,
or copy numerical weights into a later submit. Files are read-only; no network.
Use one tool JSON object {"tool":"name","arguments":{...}} or ONE Python code fence per turn.
Example: {"tool":"read_market","arguments":{"tickers":["SPY"],"limit":3}}
Send ONLY one call or one complete Python fence, not a multi-step prose plan.
Each Python call is independent; code blocks cannot share variables across turns.
All tool replies are dictionaries; check reply["ok"]. Output has a 4000-character budget.
Large structured output is rejected. Long Python stdout is explicitly clipped while
preserving its ending and exception message; output_truncated flags incomplete delivery.
Print a compact summary. Failed or truncated feedback consumes the same turn budget.
submit(weights) ends the decision. weights is a nonempty ticker:number dictionary.
Weights must be finite, nonnegative, sum <=1; remainder is cash. Rounding totals <=1.001
are normalized to 1. Example of shape only: submit({"SPY":0.5,"IEF":0.5}).
Choose your own assets/weights. Invalid calls consume a turn. No submission keeps holdings.
At most 8 turns, 1024 output tokens each. Submit is a tool, not a package function.
"""
LIBRARY = """fin-skills tools are available as JSON tools AND already-bound Python functions:
read_history(tickers=None, lookback=252, offset=0, limit=12): inspection pages only.
load_history(tickers=None, lookback=252): PYTHON ONLY; full validated matrices.
For calculations inside one Python call:
h = load_history(lookback=252)
returns = h["returns"]
prices = h["prices"]
print(h["metadata"])
These prices/returns include visible splits and cash dividends. No forward fill is used.
lookback counts returns: 252 returns require 253 complete price sessions.
Do not apply corporate actions again. Missing data or insufficient history raises an error.
This is a new data convention: dividends reinvest at ex-date close. No v5 results apply.
list_algorithms(), describe_algorithm(algorithm_id)
list_skills(query="", offset=0, limit=8)
read_skill(name, offset=0, limit=4000, reference=None)
run_algorithm(algorithm_id, tickers=None, lookback=252, parameters=None)
run_guard(name, ticker)
The menu order is randomized, not a recommendation. Choose a method and describe it
for an executable example. All installed skill documents and their Markdown references
can be discovered and paged. next_offset=null means the document/page list is complete.
run_algorithm uses the same validated history as load_history internally.
Pass algorithm_id, NOT id; do NOT pass asset_returns, a DataFrame, or raw prices.
Outer lookback is the history length. Method parameters.lookback is a DIFFERENT window.
describe_algorithm lists actual method defaults; replies report effective_parameters.
The tool reply is a dictionary:
reply = run_algorithm(algorithm_id=chosen_method, lookback=252)
Check reply["ok"]; portfolio weights are reply["result"]["weights"], never reply.weights.
For non-portfolio methods inspect reply["result"]["per_ticker"]; signals are not weights.
Within one Python call, a successful portfolio reply can be submitted with
submit(reply["result"]["weights"]). Do not import run_algorithm from fin_skills.
The direct package API fin_skills.algorithms.run is separate from these study tools.
run_guard supports adjustment_check and data_quality. reply["ok"] means the guard ran;
reply["passed"] is the audit verdict. Inspect warnings in reply["summary"] even on pass.
Guards diagnose data; they do not repair it or guarantee a profitable allocation.
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
        elif tool in ('read_file', 'list_algorithms', 'describe_algorithm', 'read_skill', 'list_skills'):
            result = self.tools.call(tool, arguments)
        elif tool in ('run_python', 'run_algorithm', 'run_guard', 'read_market', 'read_snapshot', 'read_history'):
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
                        'benchmarks.agent_study.trading_worker_v6', str(box),
                        str(self.workspace)], env=env, capture_output=True, text=True,
                        timeout=v3.TOOL_TIMEOUT)
                    result = (json.loads((box / 'result.json').read_text())
                              if (box / 'result.json').exists() else
                              dict(ok=False, error='worker failed: ' + proc.stderr[-1500:]))
                except subprocess.TimeoutExpired:
                    result = dict(ok=False, error='tool timeout')
        else:
            result = dict(ok=False, error=f'unknown tool {tool!r}')
        receipt = dict(tool=tool, ok=bool(result.get('ok')))
        if tool == 'run_algorithm' and result.get('ok'):
            receipt.update(algorithm_id=result['algorithm_id'], history=result['history'],
                           weights=result['result'].get('weights'),
                           effective_parameters=result['effective_parameters'])
        self.calls.append(receipt)
        self.calls.extend(result.get('tool_calls', []))
        return result


def decide(backend, controller, task):
    history = [dict(role='system', content=COMMON + (LIBRARY if controller.arm == 'library'
                                                     else '')),
               dict(role='user', content=task)]
    turns, target = [], None
    base_system = history[0]["content"]
    for turn_index in range(v3.MAX_TURNS):
        remaining = v3.MAX_TURNS - turn_index
        history[0]["content"] = (base_system +
            f"\nCurrent call {turn_index + 1} of {v3.MAX_TURNS}; "
            f"{remaining} calls remain including this one. Finish with submit(weights).")
        if remaining == 1:
            history[0]["content"] += " This is the last call; no later response can execute."
        try:
            response = backend(history)
        except ValueError as exc:
            turns.append(dict(response='', parse='backend-error', tool=None, usage={},
                              result=dict(ok=False, error=str(exc))))
            break
        text = response['choices'][0]['message']['content']
        if response['choices'][0]['finish_reason'] == 'length':
            call, status = None, 'generation-truncated'
        else:
            call, status = extract_call(text)
        row = dict(response=text, parse=status, usage=response.get('usage'),
                   turn_index=turn_index + 1, calls_remaining=remaining,
                   generation_seconds=response.get('generation_seconds', 0.),
                   finish_reason=response['choices'][0]['finish_reason'])
        if call is None:
            message = ('Your response reached the 1024 output-token limit and was NOT executed. '
                       'Use one shorter JSON tool call or one concise complete Python fence; '
                       'omit explanations.' if status == 'generation-truncated' else
                       'Use exactly one tool JSON object or ONE complete Python fence. '
                       'Multiple code blocks are not executed; combine code into one fence. '
                       'Return the call, not a multi-step prose plan.')
            result = dict(ok=False, error=message)
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
        row.update(tool=call.get('tool') if call else None, result=result,
                   **execution_status(result))
        turns.append(row)
        if target is not None:
            break
        history += [dict(role='assistant', content=text),
                    dict(role='user', content=row['model_feedback'])]
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
                    for p in Path(__file__).parent.glob('trading_*v6.py')})
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
