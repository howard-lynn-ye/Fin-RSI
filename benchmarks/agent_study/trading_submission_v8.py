"""Explicit portfolio instructions; conversion is arithmetic, never a strategy.

Relative allocations require a model-chosen CASH entry. Absolute weights keep
their original budget constraint; an over-budget order is never silently scaled.
"""
import ast
import json
import math
import re

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
    """Parse one complete response; never mine a nested or competing instruction.

    A single Markdown fence is only a transport wrapper. Its complete JSON body
    is still JSON even when a model labels the fence ``python``. Otherwise only
    an explicit Python fence or complete language-prefixed program is code;
    the runtime permits it only in research turns.
    """
    if not isinstance(text, str):
        return None, 'non-text-response'
    answer = text.strip()
    thinking, position = False, 0
    for marker in re.finditer(r'</?think>', answer):
        opening = marker.group() == '<think>'
        if opening == thinking:
            return None, 'reasoning-malformed'
        if opening and answer[position:marker.start()].strip():
            return None, 'reasoning-mixed-actions'
        thinking, position = opening, marker.end()
    if thinking:
        return None, 'reasoning-incomplete'
    answer = answer[position:].strip()
    if answer.lower() == 'hold':
        return {'tool': 'hold', 'arguments': {}}, 'json'
    # A language label may wrap JSON or a complete Python program. Never infer
    # Python from unlabelled prose, or choose between code and a separate order.
    label = re.fullmatch(r'(json|python|py)[ \t]*\r?\n(.*)', answer, re.DOTALL | re.IGNORECASE)
    if label:
        answer = label.group(2).strip()
    fence = None if label else re.fullmatch(
        r'```(json|python|py)?[ \t]*\r?\n(.*?)\n?```', answer, re.DOTALL | re.IGNORECASE)
    if fence:
        answer = fence.group(2).strip()
        if '```' in answer:
            return None, 'multiple-actions'
    try:
        value = json.loads(answer, object_pairs_hook=unique_keys)
    except RecursionError:
        return None, 'json-too-deep'
    except ValueError as exc:
        if str(exc) == 'duplicate key':
            return None, 'duplicate-keys'
        if label and label.group(1).casefold() in ('python', 'py'):
            try:
                program = ast.parse(answer)
            except (SyntaxError, ValueError, RecursionError):
                return None, 'invalid-python-label'
            # Bare literal statements can be a second JSON order, a second
            # data response, or commentary. Assignments inside real code are
            # ordinary data and retain the existing Python worker semantics.
            literals = (ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Constant)
            if not program.body or any(isinstance(node, ast.Expr) and
                    isinstance(node.value, literals) for node in program.body):
                return None, 'mixed-python-and-data'
            return {'tool': 'run_python', 'arguments': {'code': answer}}, 'python-label'
        if fence and (fence.group(1) or '').casefold() in ('python', 'py'):
            return {'tool': 'run_python', 'arguments': {'code': answer}}, 'python-fence'
        if isinstance(exc, json.JSONDecodeError) and exc.msg == 'Extra data':
            return None, 'trailing-content-or-multiple-actions'
        return None, 'unparsed-or-multiple-actions'
    if (isinstance(value, dict) and set(value) in ({'action'}, {'action', 'rationale'}) and
            isinstance(value.get('rationale', ''), str) and
            isinstance(value['action'], str) and value['action'].casefold() == 'hold'):
        return {'tool': 'hold', 'arguments': {}}, 'json'
    if isinstance(value, dict) and 'tool' in value:
        if (set(value) != {'tool', 'arguments'} or not isinstance(value['tool'], str)
                or not isinstance(value['arguments'], dict)):
            return None, 'invalid-tool-envelope'
        return value, 'json'
    if isinstance(value, dict) and set(value) & {'weights', 'allocation'}:
        # Validate the entire envelope later, including unknown order parameters.
        return {'tool': 'submit', 'arguments': value}, 'json'
    return None, 'unparsed'


def submission(arguments):
    if isinstance(arguments, dict) and 'action' in arguments and set(arguments) & {'weights', 'allocation'}:
        return dict(ok=False, submission=None, error=(
            'Conflicting decision: action cannot accompany allocation or weights. '
            'Choose ONE: an allocation/weights object to rebalance, or {"action":"hold"} '
            'to keep current units and cash. Nothing was executed.'))
    if (not isinstance(arguments, dict) or
            len(set(arguments) & {'weights', 'allocation'}) != 1 or
            set(arguments) - {'weights', 'allocation', 'rationale'} or
            ('rationale' in arguments and not isinstance(arguments['rationale'], str))):
        return dict(ok=False, submission=None, error=(
            'submit requires exactly one of weights or allocation, plus optional string rationale. '
            'No other keys.'))
    if 'weights' in arguments:
        weights = arguments['weights']
        if isinstance(weights, dict):
            try:
                valid = all(type(v) in (int, float) and math.isfinite(v) for v in weights.values())
            except OverflowError:
                valid = False
            if not valid:
                return dict(ok=False, submission=None, error='weights must be finite numbers, not strings or booleans.')
        target, problem = validate_weights(arguments['weights'])
        result = dict(ok=target is not None, submission=target, error=problem,
                      weights_input=weights,
                      budget_rounding_applied=target is not None and any(
                          target[t] != weights.get(t, 0.) for t in TICKERS))
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
        unknown = ', '.join(sorted(str(k) for k in set(allocation) - set(TICKERS) - {'CASH'}))
        problem = ('allocation contains an unknown instrument: ' + unknown + '. Allowed instruments: ' +
                   ', '.join((*TICKERS, 'CASH')) + '.')
    else:
        try:
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in allocation.values()):
                problem = 'allocation parts must be finite nonnegative numbers, not strings or booleans.'
        except OverflowError:
            problem = 'allocation parts exceed the supported numeric range.'
    if problem:
        result = dict(ok=False, submission=None, error=problem)
        if isinstance(allocation, dict):
            invalid = []
            for name, value in allocation.items():
                try:
                    valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
                except OverflowError:
                    valid = False
                if not valid:
                    invalid.append(str(name))
            if invalid:
                result['invalid_parts'] = invalid
                result['hint'] = ('Correct these allocation fields yourself: ' + ', '.join(invalid) +
                    '. Parts are relative amounts, not a budget remainder; they need not sum to 1. '
                    'Every asset and CASH must be nonnegative. No order was executed or changed.')
        return result
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
