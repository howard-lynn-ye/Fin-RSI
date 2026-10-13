"""Check review concerns against actual helpers and the pinned parser."""
import json

import pytest

from benchmarks.agent_study import trading_schema_v8 as schema


def test_decoder_diagnostic_creates_its_directory_and_cannot_overwrite(tmp_path):
    response = {'choices': [{'message': {'content': 'raw invalid response'}, 'finish_reason': 'stop'}]}
    failure = schema.DecoderFailure(response, [], schema.output_contract(True))
    schema.persist_failure(tmp_path, 'preflight-raw-00', failure)
    path = tmp_path / 'infrastructure-failures' / 'preflight-raw-00.json'
    assert json.loads(path.read_text())['response'] == response
    with pytest.raises(FileExistsError):
        schema.persist_failure(tmp_path, 'preflight-raw-00', failure)


@pytest.mark.parametrize('length,expected', [(256, True), (257, False)])
def test_actual_config_and_pinned_parser_enforce_rationale_length(length, expected):
    lmfe = pytest.importorskip('lmformatenforcer')
    config = lmfe.CharacterLevelParserConfig(max_consecutive_whitespaces=12, force_json_field_order=False)
    parser = lmfe.JsonSchemaParser(schema.decision_schema(), config=config)
    text = json.dumps({'tool': 'submit', 'arguments': {'allocation': {'CASH': 1},
                                                    'rationale': 'a' * length}})
    accepted = True
    for character in text:
        if character not in parser.get_allowed_characters():
            accepted = False
            break
        parser = parser.add_character(character)
    assert (accepted and parser.can_end()) is expected


def test_preflight_and_execution_status_are_the_same_callable():
    from benchmarks.agent_study import trading_tools_v5, trading_tools_v6
    assert trading_tools_v5.execution_status is trading_tools_v6.execution_status



@pytest.mark.parametrize('bad_value', ['01', '1.', '-', '1e', '9' * 4301])
def test_pinned_numeric_leniency_is_rejected_without_silent_conversion(bad_value):
    pytest.importorskip('lmformatenforcer')
    from benchmarks.agent_study import trading_runtime_v8 as v8
    from test_trading_schema_v8 import StructuredBackend
    from test_trading_terminal_v8 import Controller, response
    text = '{"tool":"submit","arguments":{"allocation":{"SPY":' + bad_value + ',"CASH":0}}}'
    with pytest.raises(ValueError):
        schema.check_format(text)
    answer = response('hold')
    answer['choices'][0]['message']['content'] = text
    backend = StructuredBackend([answer, response('hold')])
    if schema.grammar_permits(text):
        record = v8.decide(backend, Controller('raw'), 'fixture', final_output=schema.output_contract(True))
        v8.validate_decision(record, expected_final_output=schema.output_contract(True))
        assert record['turns'][-2]['parse'] == 'structured-json-rejected'
        assert not record['turns'][-2]['result']['ok']
        assert record['decision_action'] == 'hold' and backend.final_calls == 2
    else:
        with pytest.raises(schema.DecoderFailure):
            v8.decide(backend, Controller('raw'), 'fixture', final_output=schema.output_contract(True))


def test_raw_newline_in_rationale_is_never_accepted_as_valid_json():
    pytest.importorskip('lmformatenforcer')
    text = '{"tool":"submit","arguments":{"allocation":{"CASH":1},"rationale":"a\nb"}}'
    with pytest.raises(ValueError):
        schema.check_format(text)
    turn = dict(response=text, finish_reason='stop', backend_metadata={'final_output': schema.output_contract(True)})
    if schema.grammar_permits(text):
        assert schema.check_turn(turn, schema.output_contract(True)) == 'structured-json-rejected'
    else:
        with pytest.raises(ValueError):
            schema.check_turn(turn, schema.output_contract(True))


@pytest.mark.parametrize('field,value', [('model','another/model'),('revision','other'),('max_tokens',2048)])
def test_backend_identity_checks_actual_records_not_only_declared_model(field,value):
    protocol = dict(model=['id','model/name','pinned'],max_tokens=1024)
    identity = schema.backend_identity(protocol)
    record = {'turns':[{'backend_metadata':{'identity':identity}}]}
    schema.check_backend(record,protocol)
    identity[field] = value
    with pytest.raises(ValueError,match='actual backend identity'):
        schema.check_backend(record,protocol)


def test_real_decoder_gate_cannot_be_satisfied_by_only_early_text_holds(tmp_path,monkeypatch):
    pytest.importorskip('bs4')
    from benchmarks.agent_study import trading_preflight_v8 as gate, trading_study as old
    from test_trading_schema_v8 import StructuredBackend
    from test_trading_terminal_v8 import response
    protocol = dict(interface='v8',seed=11,model=['id','model/name','pinned'],max_tokens=1024,
                    final_output=schema.output_contract(True))
    old.write(tmp_path/'protocol.json',protocol)
    messages = [{'role':'system','content':'FINAL DECISION'}, {'role':'user','content':'frozen fixture'}]
    monkeypatch.setattr(gate,'decoder_messages',lambda root,p: messages)
    with pytest.raises(FileNotFoundError):
        gate.require_decoder(tmp_path,protocol)
    class Backend(StructuredBackend):
        def final_decision(self,messages):
            reply = super().final_decision(messages)
            reply['backend_metadata']['identity'] = schema.backend_identity(protocol)
            return reply
    backend = Backend([response('hold')])
    report = gate.qualify_decoder(tmp_path,backend,protocol)
    assert report['passed'] and backend.final_calls == 1 and backend.research == 0
    assert gate.qualify_decoder(tmp_path,backend,protocol) == report
    assert backend.final_calls == 1  # resume uses the exact preserved response, no reselection
    messages[1]['content'] = 'different task'
    with pytest.raises(ValueError,match='qualification prompt'):
        gate.require_decoder(tmp_path,protocol)
