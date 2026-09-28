import numpy as np
import pandas as pd
import pytest

from benchmarks.agent_study.codegen_utility import dataset, reference, TASKS, parse_code, messages
from fin_skills.algorithms import run


@pytest.mark.parametrize('seed', [11, 23, 37])
@pytest.mark.parametrize('task', TASKS)
def test_independent_oracle_matches_explicit_library_composition(seed, task):
    data = dataset(seed)
    p = pd.DataFrame(np.array(data['prices'])*np.array(data['factors']))
    ret = p.pct_change(fill_method=None)
    out = pd.DataFrame(0., index=p.index, columns=p.columns)
    if task in TASKS[:2]:
        for t in range(61, len(p)):
            name = 'inverse_volatility' if task == TASKS[0] else task
            kwargs = {} if task == TASKS[0] else dict(lookback=60, top_k=2)
            out.iloc[t] = run(name, {'asset_returns':ret.iloc[t-60:t]}, **kwargs)
    else:
        for c in p:
            kwargs = dict(fast=5, slow=20) if task == 'ma_crossover' else dict(
                lookback=20, target_vol=.1, periods_per_year=252, max_exposure=1.)
            out[c] = run(task, {'prices':p[c]}, **kwargs).fillna(0)/4
    np.testing.assert_allclose(reference(task, data), out, atol=1e-8, rtol=0)


def test_paired_inputs_differ_only_in_library_access():
    data = dataset(11)
    import json
    a = json.loads(messages(TASKS[0], data, 'raw', {})[1]['content'])
    b = json.loads(messages(TASKS[0], data, 'library', {'catalog':'actual'})[1]['content'])
    b.pop('library_discovery')
    assert a == b


def test_parser_does_not_extract_code_from_explanation():
    assert parse_code('```python\nx = 1\n```') == 'x = 1'
    raw = 'Here is code:\n```python\nx = 1\n```'
    assert parse_code(raw) == raw
