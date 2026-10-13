"""Qualification follows real RAG pagination without accepting missing content."""
import json

import pytest

from benchmarks.agent_study import trading_runtime_v9 as v9
from benchmarks.agent_study.trading_tools_v5 import OUTPUT_LIMIT, text_page, wire
from test_trading_terminal_v8 import workspace  # noqa: F401


def test_large_retrieval_qualifies_only_after_complete_delivered_json(workspace, monkeypatch):
    import fin_skills.rag.research as rag
    monkeypatch.setattr(rag, 'research_context', lambda *a, **kw: dict(
        knowledge={'context': 'dated financial context. ' * 260},
        evidence={'context': 'dated evidence'}, tool_contracts=[]))
    tools = v9.Tools(workspace, 'library')
    passed, pages = v9.retrieval_check(tools, 'library')
    assert passed and len(pages) > 1
    assert all(len(wire(page)) <= OUTPUT_LIMIT for page in pages)
    rebuilt = json.loads(''.join(page['text'] for page in pages))
    assert rebuilt['knowledge']['context'] == 'dated financial context. ' * 260
    assert rebuilt['evidence']['context'] == 'dated evidence'
    assert v9.retrieval_check(v9.Tools(workspace, 'raw'), 'raw')[0]


@pytest.mark.parametrize('fault', ['gap', 'missing_tail', 'size_changed', 'invalid_json', 'not_delivered'])
def test_broken_page_stream_fails_qualification(fault):
    body = json.dumps(dict(ok=True, knowledge={'context': 'x' * 4500}, evidence={}))
    if fault == 'invalid_json':
        body = body[:-1] + '!'
    class Controller:
        def call(self, name, arguments):
            result = text_page(body, arguments['offset'], 2000,
                               format='json', name='research_context')
            if fault == 'gap':
                result['next_offset'] = arguments['offset']
            elif fault == 'missing_tail':
                result['next_offset'] = None
            elif fault == 'size_changed' and arguments['offset']:
                result['total_chars'] += 1
            elif fault == 'not_delivered':
                result = dict(ok=True, unexpected='x' * 5000)
            return result
    assert not v9.retrieval_check(Controller(), 'library')[0]
