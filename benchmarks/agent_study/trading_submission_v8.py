"""Explicit portfolio instructions; conversion is arithmetic, never a strategy.

Relative allocations require a model-chosen CASH entry. Absolute weights keep
their original budget constraint; an over-budget order is never silently scaled.
"""
import json
import math
import re

from benchmarks.agent_study import trading_runtime_v6 as prior
from benchmarks.agent_study.market_data import TICKERS
from benchmarks.agent_study.trading_tools_v6 import validate_weights


def unique_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate key')
        out[key] = value
    return out


def extract_decision(text):
    call, status = prior.extract_call(text)
    if status.startswith('reasoning-') or status == 'multiple-actions':
        return call, status
    # prior has already checked that reasoning delimiters are balanced.
    answer = text.rsplit('</think>', 1)[-1].strip()
    if answer.lower() == 'hold':
        return {'tool': 'hold', 'arguments': {}}, 'json'
    # Count complete top-level intents, including the shorter decision envelopes.
    # Never select a trade from two competing instructions.
    decoder, position, intents = json.JSONDecoder(object_pairs_hook=unique_keys), 0, 0
    while (start := answer.find('{', position)) != -1:
        try:
            value, position = decoder.raw_decode(answer, start)
        except ValueError as exc:
            if str(exc) == 'duplicate key':
                return None, 'duplicate-keys'
            position = start + 1
            continue
        if isinstance(value, dict) and set(value) & {'tool', 'weights', 'allocation', 'action'}:
            intents += 1
    if intents > 1 and status != 'python-fence':
        return None, 'multiple-actions'
    if call is not None:
        return call, status
    fence = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n?```', answer, re.DOTALL)
    if fence:
        answer = fence.group(1).strip()
    try:
        value = json.loads(answer, object_pairs_hook=unique_keys)
    except ValueError:
        return None, status
    if value == {'action': 'hold'}:
        return {'tool': 'hold', 'arguments': {}}, 'json'
    if isinstance(value, dict) and set(value) & {'weights', 'allocation'}:
        # Validate the entire envelope later, including unknown order parameters.
        return {'tool': 'submit', 'arguments': value}, 'json'
    return None, status


def submission(arguments):
    if (not isinstance(arguments, dict) or
            len(set(arguments) & {'weights', 'allocation'}) != 1 or
            set(arguments) - {'weights', 'allocation', 'rationale'} or
            ('rationale' in arguments and not isinstance(arguments['rationale'], str))):
        return dict(ok=False, submission=None, error=(
            'submit requires exactly one of weights or allocation, plus optional string rationale. '
            'No other keys.'))
    if 'weights' in arguments:
        target, problem = validate_weights(arguments['weights'])
        result = dict(ok=target is not None, submission=target, error=problem)
        if problem:
            result['hint'] = ('Absolute weights must total at most 1. For relative parts instead, '
                'explicitly send allocation with your chosen assets AND CASH (zero is allowed). '
                'The executor then divides each part by the total. It never scales invalid weights.')
        return result
    allocation = arguments['allocation']
    problem = None
    if not isinstance(allocation, dict) or 'CASH' not in allocation:
        problem = 'allocation requires a dictionary with an explicit CASH part, including zero.'
    elif set(allocation) - set(TICKERS) - {'CASH'}:
        problem = 'allocation contains an unknown instrument.'
    else:
        try:
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in allocation.values()):
                problem = 'allocation parts must be finite nonnegative numbers, not strings or booleans.'
        except OverflowError:
            problem = 'allocation parts exceed the supported numeric range.'
    if problem:
        return dict(ok=False, submission=None, error=problem)
    scale = max(allocation.values())
    if scale == 0:
        return dict(ok=False, submission=None, error='At least one allocation part must be positive.')
    # Scaling first avoids overflow when otherwise valid finite parts are large.
    total = math.fsum(v / scale for v in allocation.values())
    fractions = {k: (v / scale) / total for k, v in allocation.items()}
    target, problem = validate_weights({t: fractions.get(t, 0.) for t in TICKERS})
    return dict(ok=target is not None, submission=target, error=problem,
                allocation_input=allocation, cash_weight=fractions['CASH'],
                conversion='each explicit asset/cash part divided by total parts')
