"""Retrieved evidence must update without allowing later revisions into past queries."""
import json
import sqlite3

import pytest

from fin_skills.collect import Batch, Event, Store, Watch
from fin_skills.rag.research import documents_from_events
from fin_skills.tools import call_tool, list_tools


def event(text, *, id='item', observed='2026-01-01T12:00:00Z', published=None):
    return Event(id, 'rss', 'news', 'public observation', 'https://example.com/news',
                 observed, published, data={'text': text})


def save(store, watch, text, **options):
    store.save(watch, Batch([event(text, **options)]), next_due=0)


def test_one_query_has_current_knowledge_dated_evidence_and_exact_contracts():
    result = call_tool('research_context', dict(query='retrieve_context',
        documents=[dict(id='past', text='retrieve_context reference evidence',
                        available_at='2020-01-01T00:00:00Z'),
                   dict(id='future', text='retrieve_context FUTURE_MARKER',
                        available_at='2030-01-01T00:00:00Z')],
        as_of='2025-01-01T00:00:00Z'))
    assert result['method'] == 'bm25' and result['content_is_untrusted']
    assert 'FUTURE_MARKER' not in result['evidence']['context']
    assert result['evidence']['citations'][0]['document_id'] == 'past'
    assert 'not a historical corpus' in result['reference_policy']
    specs = {s['name']: s for s in list_tools()}
    assert result['tool_contracts']
    for contract in result['tool_contracts']:
        assert contract['input_schema'] == specs[contract['name']]['input_schema']
        assert contract['template_requires_values'] is True
    assert list_tools()[0]['name'] == 'research_context'


def test_collection_query_sees_new_records_and_preserves_past_revision(tmp_path):
    path, watch = tmp_path / 'events.sqlite3', Watch('watch', 'rss', 'https://example.com/feed')
    with Store(path) as store:
        store.put_watch(watch)
        save(store, watch, 'NEBULA_OLD')
    first = call_tool('collection_search', dict(database=str(path), query='NEBULA_OLD',
                                               as_of='2026-01-01T13:00:00Z'))
    assert first['collection_records'] == 1
    assert 'NEBULA_OLD' in first['evidence']['context']
    with Store(path) as store:
        save(store, watch, 'NEBULA_NEW', observed='2026-01-02T12:00:00Z',
             published='2025-12-01T00:00:00Z')
        save(store, watch, 'NEBULA_NEW second item', id='second', observed='2026-01-02T12:00:00Z')
    historical = call_tool('collection_search', dict(database=str(path), query='NEBULA_OLD',
                                                    as_of='2026-01-01T13:00:00Z'))
    assert historical['evidence'] == first['evidence']
    assert historical['collection_health'] is None
    current = call_tool('collection_search', dict(database=str(path), query='NEBULA_NEW',
                                                 as_of='2026-01-02T13:00:00Z'))
    assert current['collection_records'] == 2
    assert 'NEBULA_NEW' in current['evidence']['context']
    assert 'NEBULA_OLD' not in current['evidence']['context']
    assert all(c['metadata']['observed_at'].startswith('2026-01-02')
               for c in current['evidence']['citations'])


def test_observation_publication_and_microseconds_are_separate(tmp_path):
    path, watch = tmp_path / 'events.sqlite3', Watch('watch', 'rss', 'https://example.com/feed')
    with Store(path) as store:
        store.put_watch(watch)
        save(store, watch, 'old observed value', observed='2026-01-01T12:00:00Z')
        save(store, watch, 'later observation', observed='2026-01-01T12:00:00.000001Z')
        save(store, watch, 'future publication', id='future', published='2026-01-02T00:00:00Z')
        rows = store.retrieval_events(as_of='2026-01-01T07:00:00-05:00')
        assert len(rows) == 1 and rows[0]['data']['text'] == 'old observed value'
        document = documents_from_events(rows)[0]
        assert document.metadata['published_at'] is None
        assert document.available_at == '2026-01-01T12:00:00+00:00'
        assert '2026-01-02' not in json.dumps(document.to_dict())


def test_collection_search_is_read_only_and_refuses_partial_corpus(tmp_path):
    missing = tmp_path / 'missing.sqlite3'
    with pytest.raises(FileNotFoundError):
        call_tool('collection_search', dict(database=str(missing), query='news'))
    assert not missing.exists()
    path, watch = tmp_path / 'events.sqlite3', Watch('watch', 'rss', 'https://example.com/feed')
    with Store(path) as store:
        store.put_watch(watch)
        save(store, watch, 'first')
        save(store, watch, 'second', id='other')
    with Store(path, read_only=True) as store:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            store.db.execute('DELETE FROM events')
    with pytest.raises(ValueError, match='exceeds max_records'):
        call_tool('collection_search', dict(database=str(path), query='news', max_records=1,
                                           as_of='2026-01-02T00:00:00Z'))


def test_unknown_dates_do_not_get_invented_and_naive_cutoff_is_rejected(tmp_path):
    row = event('transaction_date is not availability').to_dict()
    row.update(actor='Synthetic Official', symbol='EXAMPLE_TICKER')
    row['data']['transaction_date'] = '2000-01-01'
    document = documents_from_events([row])[0]
    assert 'Synthetic Official' in document.text and 'EXAMPLE_TICKER' in document.text
    assert document.available_at.startswith('2026-01-01')
    assert document.metadata['published_at'] is None
    row['observed_at'] = '2026-01-01T12:00:00'
    with pytest.raises(ValueError, match='timezone'):
        documents_from_events([row])
    with pytest.raises(ValueError, match='timezone'):
        call_tool('research_context', dict(query='news', as_of='2026-01-01'))
