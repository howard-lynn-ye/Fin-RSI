"""Research output constraints preserve choices while preventing format ambiguity."""
import copy
import json
import pytest
from benchmarks.agent_study import trading_research_v9 as research
from benchmarks.agent_study import trading_runtime_v8 as runtime, trading_schema_v8 as final
from test_trading_terminal_v8 import Controller, response


class Backend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.research_calls = self.final_calls = 0

    def prepare_research_call(self, contract):
        return contract

    def prepare_final_decision(self):
        return final.output_contract(True)

    def research_call(self, messages, contract):
        self.research_calls += 1
        self.requests.append(copy.deepcopy(messages))
        reply = copy.deepcopy(next(self.replies))
        reply['backend_metadata'] = {'research_output': contract}
        return reply

    def final_decision(self, messages):
        self.final_calls += 1
        reply = copy.deepcopy(next(self.replies))
        reply['backend_metadata'] = {'final_output': final.output_contract(True)}
        return reply

    def __call__(self, messages):
        raise AssertionError('No unconstrained model call is declared in this test')


def decide(arm, replies):
    contract = research.output_contract(arm)
    backend, controller = Backend(replies), Controller(arm)
    record = runtime.decide(backend, controller, 'fixture', interface='v9',
        final_output=final.output_contract(True), research_output=contract)
    runtime.validate_decision(record, expected_final_output=final.output_contract(True),
                              expected_research_output=contract)
    return record, backend, controller


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_read_then_independently_chosen_allocation_uses_declared_transport(arm):
    chosen = {'SPY': 7, 'CASH': 3}
    record, backend, controller = decide(arm, [response('read_snapshot'),
        response('submit', {'allocation': chosen})])
    assert backend.research_calls == 2 and backend.final_calls == 0
    assert controller.calls == [{'tool': 'read_snapshot', 'arguments': {}}]
    assert record['target']['SPY'] == pytest.approx(.7)
    assert record['turns'][-1]['result']['allocation_input'] == chosen


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_model_can_choose_hold_without_any_research(arm):
    record, backend, controller = decide(arm, [response('hold')])
    assert record['decision_action'] == 'hold' and not controller.calls
    assert backend.research_calls == 1


def test_six_research_calls_leave_final_and_correction_budget_unchanged():
    record, backend, controller = decide('raw', [response('read_snapshot')]*6 + [
        response('submit', {'allocation': {'SPY': -1, 'CASH': 1}}), response('hold')])
    assert backend.research_calls == 6 and backend.final_calls == 2
    assert len(controller.calls) == 6 and record['decision_action'] == 'hold'
    assert not record['turns'][6]['result']['ok']


def test_raw_cannot_claim_the_library_tool_or_a_fabricated_reader():
    for tool in ('research_context', 'unscored_reader_check'):
        with pytest.raises(final.DecoderFailure):
            decide('raw', [response(tool)])


def test_library_reader_reaches_controller_without_a_forced_investment():
    record, _, controller = decide('library', [response('research_context', {'query': 'costs'}), response('hold')])
    assert controller.calls[0] == {'tool': 'research_context', 'arguments': {'query': 'costs'}}
    assert record['target'] is None


def test_truncated_research_is_preserved_not_executed():
    record, _, controller = decide('raw', [response('read_snapshot', finish='length'), response('hold')])
    assert not controller.calls and record['turns'][0]['parse'] == 'generation-truncated'


def test_missing_or_changed_receipt_stops_before_execution():
    contract = research.output_contract('raw')
    turn = dict(response='{"tool":"hold","arguments":{}}', finish_reason='stop',
                backend_metadata={'research_output': research.output_contract('library')})
    with pytest.raises(ValueError, match='receipt'):
        research.check_turn(turn, contract)
    changed = copy.deepcopy(contract)
    changed['tool_names'].append('secret_future_prices')
    with pytest.raises(ValueError, match='contract'):
        research.check_contract(changed)


@pytest.mark.parametrize('tool,args,allowed', [
    ('read_snapshot', {}, True),
    ('run_python', {'code': 'quotes = read_snapshot()\nprint(quotes)', 'nested': {'x': [1, None, True]}}, True),
    ('submit', {'allocation': {'SPY': -2, 'CASH': 3}}, True),
    ('run_python', {'data': list(range(40))}, True),
    ('hold', {}, True),
    ('unscored_reader_check', {}, False),
    ('research_context', {'query': 'news'}, False),
])
def test_actual_pinned_grammar_allows_arbitrary_arguments_and_choices_but_only_visible_names(tool, args, allowed):
    pytest.importorskip('lmformatenforcer')
    assert research.grammar_permits(json.dumps({'tool': tool, 'arguments': args}),
                                   research.output_contract('raw')) is allowed


@pytest.mark.parametrize('text', [
    'py\nprint(read_snapshot())',
    '{"tool":"read_snapshot","arguments":{}}\n{"tool":"hold","arguments":{}}',
    'tool: hold\narguments: {}',
])
def test_actual_grammar_rejects_observed_unstructured_and_multi_action_failures(text):
    pytest.importorskip('lmformatenforcer')
    assert not research.grammar_permits(text, research.output_contract('raw'))


def test_separate_native_reasoning_and_personal_tracks_remain_declared_text():
    for batch in ('matrix_native', 'matrix_reasoning', 'personal'):
        assert research.protocol_contract('v9', batch) == {
            'raw': {'transport': 'text'}, 'library': {'transport': 'text'}}


def test_full_receipt_validation_selects_each_arms_own_contract(tmp_path):
    pytest.importorskip('bs4')
    from benchmarks.agent_study import multisource_model_study as study
    old = study.old
    protocol = dict(interface='v9', batch='matrix_hf', model=['fixture', 'fixture/model', 'pinned'],
        max_tokens=1024, final_output=final.output_contract(True),
        research_output=research.protocol_contract('v9', 'matrix_hf'))
    old.write(tmp_path/'protocol.json', protocol)
    inputs = {'arms': ['raw', 'library'], 'picks': [0]}
    old.write(tmp_path/'inputs.json', inputs)
    receipt = {'protocol_sha256': old.sha(tmp_path/'protocol.json'), 'decisions': {}}
    for key, filename in [('capability', 'capability'), ('decision', 'decision'), ('rag', 'rag')]:
        path = tmp_path/(filename+'-qualification.json')
        old.write(path, {'fixture': True})
        receipt[key+'_qualification_sha256'] = old.sha(path)
    for arm in inputs['arms']:
        record, _, _ = decide(arm, [response('read_snapshot')]*6 + [
            response('submit', {'allocation': {'SPY': -1, 'CASH': 1}}), response('hold')])
        for turn in record['turns']:
            turn['backend_metadata']['identity'] = final.backend_identity(protocol)
        name = 'decisions/'+arm+'/00.json'
        old.write(tmp_path/name, record)
        receipt['decisions'][name] = old.sha(tmp_path/name)
    old.write(tmp_path/'inference-receipt.json', receipt)
    assert study.validate_receipt(tmp_path) == inputs


def test_research_preparation_rejects_text_before_touching_model_runtime():
    from benchmarks.agent_study.transformers_chat import TransformersChat
    backend = TransformersChat.__new__(TransformersChat)
    with pytest.raises(ValueError, match='structured research contract required'):
        backend.prepare_research_call({'transport': 'text'})


@pytest.mark.parametrize('value', ['01', '1.', '1e'])
def test_research_numeric_leniency_is_rejected_then_corrected_and_auditable(value):
    pytest.importorskip('lmformatenforcer')
    contract = research.output_contract('raw')
    text = '{"tool":"submit","arguments":{"allocation":{"SPY":' + value + ',"CASH":0}}}'
    bad = response('hold')
    bad['choices'][0]['message']['content'] = text
    backend, controller = Backend([bad, response('hold')]), Controller('raw')
    if not research.grammar_permits(text, contract):
        with pytest.raises(final.DecoderFailure):
            runtime.decide(backend, controller, 'fixture', research_output=contract)
        return
    record = runtime.decide(backend, controller, 'fixture', research_output=contract)
    runtime.validate_decision(record, expected_research_output=contract)
    assert record['turns'][0]['parse'] == 'structured-json-rejected'
    assert not record['turns'][0]['result']['ok'] and not controller.calls
    assert 'format filter allowed text rejected by strict JSON' in backend.requests[1][-1]['content']
    assert record['decision_action'] == 'hold' and record['target'] is None
