"""Execute one agent tool call for the trading study inside the Linux confinement.

Arguments: the writable box (holds `request.json` {arm, tool, arguments}, receives
`result.json`) and the read-only workspace (visible data truncated at the decision
date). Tools:

  run_python(code)                       both arms; numpy/pandas/scipy; 60 s; stdout captured.
                                         The raw arm cannot import fin_skills.
  run_algorithm(algorithm_id, tickers,   library arm; the library prepares split- and
                lookback, parameters)    dividend-adjusted inputs from the visible files.
  run_guard(name, ticker)                library arm; adjustment_check or data_quality.

Nothing here knows the hidden total-return file or any future session.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

MAX_OUTPUT = 4000


def _round(value):
    import numpy as np
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        return {str(k): _round(v) for k, v in value.tail(5).to_dict(orient='index').items()}
    if isinstance(value, pd.Series):
        tail = value if len(value) <= 12 else value.tail(12)
        return {str(k): _round(v) for k, v in tail.to_dict().items()}
    if isinstance(value, dict):
        return {str(k): _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_round(v) for v in list(value)[-12:]]
    if isinstance(value, (np.floating, float)):
        return None if value != value else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    return value if isinstance(value, (int, str, bool)) or value is None else str(value)


def run_python(code, workspace, arm):
    if arm == 'raw':
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == 'fin_skills' or name.startswith('fin_skills.'):
                    raise ImportError('fin_skills is not available in this condition')
                return None
        sys.meta_path.insert(0, Blocker())
        sys.modules.pop('fin_skills', None)
    os.chdir(workspace)
    out, err = io.StringIO(), io.StringIO()
    started = time.perf_counter()
    ok = True
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, 'agent_code.py', 'exec'), {'__name__': '__main__'})
        except BaseException:
            ok = False
            traceback.print_exc(limit=4)
    text = out.getvalue()
    if err.getvalue():
        text += ('\n' if text else '') + '[stderr]\n' + err.getvalue()
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f'\n... [truncated to {MAX_OUTPUT} characters]'
    return dict(ok=ok, output=text, seconds=time.perf_counter()-started)


def run_algorithm(arguments, workspace):
    import numpy as np
    import pandas as pd
    import fin_skills.algorithms as algorithms
    from benchmarks.agent_study.market_data import adjust, load_visible
    ident = arguments.get('algorithm_id')
    cards = {c['id']: c for c in algorithms.catalog()}
    if ident not in cards:
        return dict(ok=False, error=f'unknown algorithm_id {ident!r}; use list_algorithms')
    card = cards[ident]
    quotes, actions = load_visible(workspace)
    prices = adjust(quotes, actions)
    tickers = arguments.get('tickers') or list(prices.columns)
    unknown = [t for t in tickers if t not in prices.columns]
    if unknown:
        return dict(ok=False, error=f'unknown tickers {unknown}')
    lookback = int(arguments.get('lookback') or 252)
    params = dict(arguments.get('parameters') or {})
    window = prices[tickers].tail(lookback + 1)
    returns = window.pct_change().dropna()
    inputs = tuple(card['inputs'])
    try:
        if inputs == ('asset_returns',):
            result = algorithms.run(ident, {'asset_returns': returns}, **params)
            value = {'weights': _round(pd.Series(np.asarray(result, dtype=float),
                                                index=tickers))}
        elif inputs in (('prices',), ('returns',), ('series',)):
            value = {}
            for t in tickers:
                data = {'prices': window[t], 'returns': returns[t],
                        'series': window[t]}[inputs[0]]
                result = algorithms.run(ident, {inputs[0]: data}, **params)
                value[t] = _round(result)
            value = {'per_ticker': value,
                     'note': 'signal/regime values are for the last sessions; the library '
                             'lags signal exposures by one session already'}
        else:
            return dict(ok=False, error=f'{ident} needs inputs {inputs}; not available '
                                        'through run_algorithm (use run_python)')
    except Exception as exc:
        return dict(ok=False, error=f'{type(exc).__name__}: {str(exc)[:400]}')
    return dict(ok=True, algorithm_id=ident, tickers=tickers, lookback=lookback,
                parameters=params, adjusted=True, result=value)


def run_guard(arguments, workspace):
    import pandas as pd
    from fin_skills.api import get
    from benchmarks.agent_study.market_data import load_visible
    name, ticker = arguments.get('name'), arguments.get('ticker')
    quotes, actions = load_visible(workspace)
    bars = quotes[quotes['ticker'] == ticker].set_index('date').sort_index()
    if bars.empty:
        return dict(ok=False, error=f'unknown ticker {ticker!r}')
    bars.index = pd.to_datetime(bars.index)
    acts = actions[(actions['ticker'] == ticker) & (actions['kind'] == 'split')]
    try:
        if name == 'adjustment_check':
            result = get(name).run(close=bars['close'],
                                   actions=[(pd.Timestamp(d), float(r))
                                            for d, r in zip(acts['date'], acts['value'])])
        elif name == 'data_quality':
            result = get(name).run(bars=bars[['open', 'high', 'low', 'close', 'volume']])
        else:
            return dict(ok=False, error='supported guards: adjustment_check, data_quality')
    except Exception as exc:
        return dict(ok=False, error=f'{type(exc).__name__}: {str(exc)[:400]}')
    return dict(ok=True, guard=name, ticker=ticker, passed=bool(result.passed),
                summary=result.summary()[:1500])


def main():
    box, workspace = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    request = json.loads((box/'request.json').read_text())
    import numpy  # noqa: F401  (import before confinement denies threads)
    import pandas  # noqa: F401
    import scipy.optimize  # noqa: F401
    import scipy.cluster.hierarchy  # noqa: F401
    import fin_skills
    import fin_skills.algorithms  # noqa: F401
    import fin_skills.api  # noqa: F401
    from benchmarks.agent_study.linux_sandbox import confine
    readonly = [Path(sys.prefix), Path(sys.base_prefix), Path('/usr'), Path('/lib'),
                Path('/lib64'), Path(fin_skills.__file__).parent,
                Path(__file__).resolve().parent, workspace]
    shared = os.environ.get('FIN_STUDY_SHARED_RUNTIME')
    if shared:
        readonly.append(Path(shared))
    confine(box, readonly)
    tool, args, arm = request['tool'], request.get('arguments') or {}, request['arm']
    if tool == 'run_python':
        result = run_python(str(args.get('code', '')), workspace, arm)
    elif tool == 'run_algorithm' and arm == 'library':
        result = run_algorithm(args, workspace)
    elif tool == 'run_guard' and arm == 'library':
        result = run_guard(args, workspace)
    else:
        result = dict(ok=False, error=f'tool {tool!r} is not available in this condition')
    (box/'result.json').write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
