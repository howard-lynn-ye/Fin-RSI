"""Boundary checks without local model inference or test-label access."""
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/agent_study'))
from finqa_rag_ablation import ANSWER_FORMAT, answer_diagnostics, documents, pack, parsed_answer, gate
sys.path.remove(str(ROOT/"benchmarks/agent_study"))
from fin_skills.rag import RAGIndex


def test_labels_and_budget_do_not_truncate_evidence():
    hits=[dict(id='a',source='s',text='a'*100),dict(id='b',source='s',text='cash')]
    result=pack(hits,budget=30)
    assert [p['id'] for p in result['passages']]==['b']
    assert '[S1]' in result['context'] and result['passages'][0]['text']=='cash'


@pytest.mark.parametrize('raw',['{"program":"x"}','{"program":1,"citations":"[S1]"}',
                              '```json\n{"program":"x","citations":"[S1]"}\n```'])
def test_strict_format_errors_remain_errors(raw):
    with pytest.raises((ValueError,TypeError)):
        parsed_answer(raw)


def test_real_gate_rejects_unknown_but_cannot_detect_wrong_arithmetic():
    index=RAGIndex([dict(id='a',text='cash was 10 then 12',source='report')])
    assert not gate(index,'cash','{"program":"add(10, 12)","citations":"[S99]"}')['citation_check']['valid']
    # Wrong arithmetic can pass. This check must not be sold as semantic correctness.
    assert gate(index,'cash','{"program":"add(10, 12)","citations":"[S1]"}')['citation_check']['valid']


def test_table_row_header_preserved_without_answer_labels():
    case=dict(id='c/y/q',pre_text=['cash'],post_text=[],table=[['row','2020'],['cash','12']],
              qa={'program':'private'})
    result=documents(case)
    assert '2020' in result[1]['text'] and '12' in result[1]['text']
    assert 'private' not in str(result)


@pytest.mark.parametrize('fence', ['```json', '```'])
def test_versioned_format_accepts_only_a_complete_fence(fence):
    raw = fence + '\n{"program":"subtract(12, 10)","citations":"[S1]"}\n```'
    with pytest.raises(ValueError):
        parsed_answer(raw)  # Historical protocols retain strict grading.
    assert parsed_answer(raw, answer_format=ANSWER_FORMAT) == {
        'program': 'subtract(12, 10)', 'citations': '[S1]'}
    assert not answer_diagnostics({'final': raw})['strict_json_compliant']


@pytest.mark.parametrize('raw', [
    'Here is the answer: {"program":"add(1, 1)","citations":"[S1]"}',
    '```json\n{"program":"add(1, 1)","citations":"[S1]"}\n```\nExtra prose',
    '{"program":"add(1, 1)","citations":"[S1]"} {}',
    '{"program":"add(1, 1)","program":"subtract(1, 1)","citations":"[S1]"}',
    '{"program":"add(1, 1)","citations":["S1"]}',
    '{"program":"add(1, 1)"}',
    '```python\n{"program":"add(1, 1)","citations":"[S1]"}\n```',
    None,
])
def test_v2_does_not_repair_or_select_an_answer(raw):
    with pytest.raises((ValueError, TypeError)):
        parsed_answer(raw, answer_format=ANSWER_FORMAT)


def test_generation_failure_is_not_hidden_as_a_json_error():
    failure = {'type': 'RuntimeError', 'message': 'PRESPECIFIED_MODEL_CALL_LIMIT'}
    row = {'final': None, 'error': failure, 'calls': [
        {'response': {'choices': [{'finish_reason': 'length'}]}},
        {'error': 'provider failure'},
    ]}
    assert answer_diagnostics(row) == {
        'episode_error': failure, 'missing_final': True,
        'strict_json_compliant': False, 'length_limited_responses': 1}


def test_unknown_format_version_fails_closed():
    with pytest.raises(ValueError, match='Unknown answer format'):
        parsed_answer('{}', answer_format='unknown')
