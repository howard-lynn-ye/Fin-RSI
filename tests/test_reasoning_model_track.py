"""Reasoning models keep publisher templates and a separately declared budget."""
import json
import pandas as pd
import pytest
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_runtime_v8 as runtime
from benchmarks.agent_study.trading_schema_v8 import output_contract
from test_trading_terminal_v8 import Controller, response


def test_reasoning_track_freezes_separate_output_budget_and_rejects_changes(tmp_path, monkeypatch):
    archive = tmp_path / 'market-source'
    archive.write_text('fixture')
    monkeypatch.setattr(study.md, 'RAW', archive)
    def dataset(path):
        path.mkdir(parents=True)
        pd.DataFrame({'SPY': [100., 101.]}, index=['2025-01-02', '2025-01-03']).to_csv(path/study.md.HIDDEN)
        return {study.md.HIDDEN: study.old.sha(path/study.md.HIDDEN)}
    monkeypatch.setattr(study.md, 'write_dataset', dataset)
    monkeypatch.setattr(study.old, 'schedule', lambda dates: [0])
    evidence = tmp_path/'evidence.json'
    evidence.write_text(json.dumps([dict(category=c, series=c, id=c, event_date='2025-01-01',
        eligible_date='2025-01-02', text='fixture') for c in ('news','psychology','behavior','officials')]))
    root = tmp_path/'run'
    study.freeze(root, 'deepseek-r1-distill-14b', 11, evidence, allow_retrospective=True,
                 batch='matrix_reasoning', interface='v9')
    original = study.verify(root)
    assert original['max_tokens'] == 8192
    assert original['final_output'] == output_contract(False)
    assert original['research_responses'] == 6 and original['final_decision_responses'] == 2
    assert original['model_access_probe']
    assert original['model'][1] == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'
    assert 'deepseek-r1-distill-14b' not in study.BATCHES['matrix_hf']
    for key, value in [('max_tokens',1024),('final_output',output_contract(True)),('reasoning_policy',{})]:
        changed = dict(original, **{key:value})
        (root/'protocol.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            study.verify(root)
    (root/'protocol.json').write_text(json.dumps(original))
    assert study.verify(root) == original


@pytest.mark.parametrize('arm',['raw','library'])
def test_only_balanced_reasoning_final_is_executed_and_original_text_is_retained(arm):
    text = '<think>{"tool":"submit","arguments":{"allocation":{"SPY":999,"CASH":0}}}</think>'
    text += '{"tool":"submit","arguments":{"allocation":{"GLD":2,"CASH":3}}}'
    def backend(messages):
        result = response('hold')
        result['choices'][0]['message']['content'] = text
        return result
    record = runtime.decide(backend, Controller(arm), 'fixture', final_output=output_contract(False))
    runtime.validate_decision(record, expected_final_output=output_contract(False))
    assert record['target']['GLD'] == .4 and record['target']['SPY'] == 0
    assert record['turns'][0]['response'] == text


@pytest.mark.parametrize('finish,text',[('length','<think>long reasoning'),
    ('stop','<think>{"tool":"hold","arguments":{}}'),
    ('stop','</think>{"tool":"hold","arguments":{}}')])
def test_truncated_or_unbalanced_reasoning_never_becomes_hold(finish,text):
    def backend(messages):
        result = response('hold', finish=finish)
        result['choices'][0]['message']['content'] = text
        return result
    record = runtime.decide(backend, Controller('library'), 'fixture', final_output=output_contract(False))
    runtime.validate_decision(record, expected_final_output=output_contract(False))
    assert record['decision_action'] == 'failed' and not record['decision_completed']
    assert record['target'] is None