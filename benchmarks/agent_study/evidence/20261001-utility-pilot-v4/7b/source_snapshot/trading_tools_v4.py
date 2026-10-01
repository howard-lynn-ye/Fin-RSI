"""Versioned trading interface; v3 inference and scoring remain unchanged."""
import math
import random
from pathlib import Path

from benchmarks.agent_study.market_data import TICKERS, VISIBLE

OUTPUT_LIMIT = 4000
WEIGHT_TOLERANCE = 0.001


def validate_weights(weights):
    if not isinstance(weights, dict) or not weights:
        return None, 'weights must be a non-empty object'
    clean = dict.fromkeys(TICKERS, 0.0)
    for ticker, value in weights.items():
        if ticker not in clean:
            return None, f'unknown ticker {ticker!r}'
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None, f'weight for {ticker} is not a number'
        if not math.isfinite(value) or value < 0:
            return None, f'weight for {ticker} must be finite and >= 0'
        clean[ticker] = value
    total = sum(clean.values())
    if total > 1 + WEIGHT_TOLERANCE + 1e-12:
        return None, f'weights sum to {total:.6f} > 1.001'
    # Accept rounding drift, never introduce leverage or negative cash.
    if total > 1:
        clean = {k: v / total for k, v in clean.items()}
    return clean, None


class Tools:
    """Identical signatures and results in JSON calls and in agent Python."""

    def __init__(self, workspace, arm, menu_seed=0):
        self.workspace, self.arm = Path(workspace), arm
        self.menu_seed, self.calls = menu_seed, []

    def read_file(self, path='', offset=0, limit=OUTPUT_LIMIT):
        if path not in VISIBLE:
            return dict(ok=False, error=f'files: {", ".join(VISIBLE)}')
        text = (self.workspace / path).read_text(encoding='utf-8')
        offset, limit = max(int(offset), 0), min(max(int(limit), 1), OUTPUT_LIMIT)
        return dict(ok=True, path=path, offset=offset, total_chars=len(text),
                    text=text[offset:offset + limit])

    def list_algorithms(self):
        import fin_skills.algorithms as algorithms
        from benchmarks.agent_study.trading_study import READY_ALGORITHMS
        cards = [c for c in algorithms.catalog() if c['id'] in READY_ALGORITHMS]
        cards.sort(key=lambda c: c['id'])
        random.Random(self.menu_seed).shuffle(cards)
        return dict(ok=True, algorithms=[dict(id=c['id'], task=c['task'], inputs=c['inputs'])
                                         for c in cards])

    def describe_algorithm(self, algorithm_id=''):
        import fin_skills.algorithms as algorithms
        for card in algorithms.catalog():
            if card['id'] == algorithm_id:
                result = {k: v for k, v in card.items() if k != 'module'}
                if algorithm_id == 'hrp':
                    result['study_adapter_defaults'] = {'linkage': 'single'}
                return dict(ok=True, card=result)
        return dict(ok=False, error=f'unknown algorithm_id {algorithm_id!r}')

    def read_skill(self, name=''):
        import fin_skills
        from benchmarks.agent_study.trading_study import SKILLS
        if name not in SKILLS:
            return dict(ok=False, error=f'skills: {", ".join(SKILLS)}')
        return dict(ok=True, name=name, text=fin_skills.load(name)[:OUTPUT_LIMIT])

    def run_algorithm(self, algorithm_id, tickers=None, lookback=252, parameters=None):
        from benchmarks.agent_study.trading_worker import run_algorithm
        parameters = dict(parameters or {})
        if algorithm_id == 'hrp':
            parameters.setdefault('linkage', 'single')
        return run_algorithm(dict(algorithm_id=algorithm_id, tickers=tickers,
                                  lookback=lookback, parameters=parameters), self.workspace)

    def run_guard(self, name, ticker):
        from benchmarks.agent_study.trading_worker import run_guard
        return run_guard(dict(name=name, ticker=ticker), self.workspace)

    def call(self, name, arguments):
        arguments = {} if arguments is None else arguments
        common = ('read_file',)
        library = ('list_algorithms', 'describe_algorithm', 'read_skill',
                   'run_algorithm', 'run_guard')
        if name not in common and not (self.arm == 'library' and name in library):
            result = dict(ok=False, error=f'tool {name!r} unavailable in {self.arm}')
        elif not isinstance(arguments, dict):
            result = dict(ok=False, error='arguments must be an object')
        else:
            try:
                result = getattr(self, name)(**arguments)
            except (TypeError, ValueError, KeyError, OSError) as exc:
                result = dict(ok=False, error=f'{type(exc).__name__}: {str(exc)[:500]}')
        self.calls.append(dict(tool=name, ok=bool(result.get('ok'))))
        return result

    def python_bindings(self):
        names = ['read_file']
        if self.arm == 'library':
            names += ['list_algorithms', 'describe_algorithm', 'read_skill',
                      'run_algorithm', 'run_guard']
        # Bind through the same signatures while preserving the recorded dispatch path.
        import inspect
        def binding(name):
            signature = inspect.signature(getattr(self, name))
            def call(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                return self.call(name, dict(bound.arguments))
            return call
        return {name: binding(name) for name in names}
