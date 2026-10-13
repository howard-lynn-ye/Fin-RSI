"""Exercise the advertised final envelope and actual trailing-prose regression."""
import json
import pytest
from benchmarks.agent_study import trading_runtime_v8 as v8
from test_trading_terminal_v8 import Controller, response


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_final_allocation_rationale_uses_real_submit_and_never_executes_trailing_text(arm):
    calls = []
    def backend(messages):
        calls.append(1)
        if len(calls) <= 6:
            return response('read_market')
        result = response('submit', {'allocation': {'SPY': 2, 'CASH': 1},
                                     'rationale': 'Model-authored fixture explanation.'})
        if len(calls) == 7:
            result['choices'][0]['message']['content'] += '\nOutside explanation.'
        return result
    controller = Controller(arm)
    record = v8.decide(backend, controller, 'Synthetic format check, not an investment outcome.')
    v8.validate_decision(record)
    assert len(calls) == 8 and len(controller.calls) == 6
    assert record['turns'][6]['parse'] == 'trailing-content-or-multiple-actions'
    assert not record['turns'][6]['result']['ok']
    assert record['turns'][7]['result']['ok']
    assert record['target']['SPY'] == pytest.approx(2/3)
    assert 'arguments.rationale' in record['turns'][6]['model_feedback']
