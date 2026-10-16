"""Required information cannot be inferred from healthy polling or generic web pages."""
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from fin_skills.collect import Batch, Collector, Event, Store, Watch
from fin_skills.collect.__main__ import main
from fin_skills.tools.collection import collection_coverage, collection_search

NOW = '2026-01-10T12:00:00+00:00'
PAST = '2026-01-10T10:00:00+00:00'
KINDS = ('news', 'sentiment_survey', 'positioning_observation', 'congressional_transaction')


def save(store, kind='news', *, identity=None, published=PAST, observed=PAST, data=None):
    watch = Watch('feed', 'rss', 'https://example.com/feed')
    store.put_watch(watch)
    event = Event(identity or kind, 'fixture', kind, kind, 'https://example.com/' + kind,
                  observed, published, data={'parse_status': 'structured', 'metric': 'fixture_index',
                      'value': 12.5, 'unit': 'points'} if data is None else data)
    store.save(watch, Batch([event]), next_due=0)
    return event


def test_required_categories_remain_missing_with_only_news(tmp_path):
    with Store(tmp_path / 'events.db') as store:
        save(store)
        result = store.coverage(as_of=NOW)
        assert not result['ready']
        assert result['missing_categories'] == ['psychology', 'behavior', 'official_disclosures']
        assert result['categories'][0]['eligible_records'] == 1


def test_polling_healthy_is_not_full_readiness(tmp_path, capsys):
    path = tmp_path / 'events.db'
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    class Backend:
        def fetch(self, *_):
            return Batch([])
    with Store(path) as store:
        save(store, published=stamp, observed=stamp)
        def check(_):
            status = store.health()
            assert status['healthy'] and status['data_ready']
            assert not status['ready'] and not status['coverage_ready']
            assert status['status'] == 'degraded'
            assert main(['--database', str(path), 'health']) == 1
        Collector(store, backends={'rss': Backend()}).run(cycles=1, on_cycle=check)
    assert json.loads(capsys.readouterr().out)['coverage']['missing_categories']


def test_complete_declared_coverage_and_explicit_publication_lag(tmp_path):
    with Store(tmp_path / 'events.db') as store:
        for kind in KINDS:
            save(store, kind, observed=NOW)
        report = store.coverage(as_of=NOW)
        assert report['ready']
        assert all(c['examples'][0]['observation_lag_seconds'] == 7200 for c in report['categories'])
        assert report['categories'][0]['examples'][0]['published_at'] == PAST


@pytest.mark.parametrize('kind', ['web_page', 'social_post', 'institutional_holding'])
def test_generic_pages_posts_and_institutions_do_not_fill_other_categories(tmp_path, kind):
    with Store(tmp_path / 'events.db') as store:
        save(store, kind)
        report = store.coverage(as_of=NOW)
        assert not report['ready'] and all(c['eligible_records'] == 0 for c in report['categories'])


@pytest.mark.parametrize('change,excluded', [
    ({'published': None}, 'missing_publication'),
    ({'published': '2025-01-01T00:00:00Z'}, 'stale'),
    ({'data': {'parse_status': 'unparsed'}}, 'unstructured'),
    ({'data': {'parse_status': 'structured'}}, 'invalid_measurement'),
])
def test_recency_uses_publication_not_fetch_time(tmp_path, change, excluded):
    with Store(tmp_path / 'events.db') as store:
        save(store, 'sentiment_survey', observed=NOW, **change)
        row = store.coverage(as_of=NOW)['categories'][1]
        assert not row['ready'] and row['excluded'][excluded] == 1


def test_future_records_and_future_revision_do_not_change_past_coverage(tmp_path):
    with Store(tmp_path / 'events.db') as store:
        save(store)
        before = store.coverage(as_of=NOW)
        save(store, published='2030-01-01T00:00:00Z', observed=NOW)
        save(store, 'sentiment_survey', observed='2030-01-01T00:00:00Z')
        assert store.coverage(as_of=NOW) == before


def test_disabled_watch_only_affects_current_not_historical_coverage(tmp_path):
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with Store(tmp_path / 'events.db') as store:
        save(store, published=stamp, observed=stamp)
        store.put_watch(replace(store.watches()[0], enabled=False))
        rule = [{'category': 'news'}]
        current = store.coverage(requirements=rule)
        assert current['categories'][0]['excluded']['disabled'] == 1
        assert store.coverage(requirements=rule, as_of=stamp)['ready']


def test_explicit_sources_minimum_counts_and_record_cap(tmp_path):
    with Store(tmp_path / 'events.db') as store:
        save(store, identity='one')
        save(store, identity='two')
        rules = [{'category': 'news', 'min_records': 2, 'watch_ids': ['feed'], 'max_age_seconds': 7200}]
        assert store.coverage(requirements=rules, as_of=NOW)['ready']
        rules[0]['watch_ids'] = ['missing-source']
        assert not store.coverage(requirements=rules, as_of=NOW)['ready']
        limited = store.coverage(as_of=NOW, max_records=1)
        assert limited['status'] == 'unknown' and not limited['ready']


@pytest.mark.parametrize('rules', [[], [{'category': 'bad'}], [{'category': 'news', 'kinds': ['web_page']}],
    [{'category': 'news'}, {'category': 'news'}], [{'category': 'news', 'max_age_seconds': float('nan')}],
    [{'category': 'news', 'min_records': True}], [{'category': 'news', 'watch_ids': []}]])
def test_invalid_requirement_cannot_disable_check(tmp_path, rules):
    with Store(tmp_path / 'events.db') as store:
        with pytest.raises(ValueError):
            store.coverage(requirements=rules, as_of=NOW)


def test_registered_coverage_tool_validates_and_calls_the_same_contract(tmp_path):
    from fin_skills.tools import call_tool, list_tools
    path = tmp_path / 'events.db'
    with Store(path) as store:
        save(store, 'institutional_holding')
    report = call_tool('collection_coverage', dict(database=str(path), as_of=NOW,
        requirements=[{'category': 'institutional_disclosures'}]))
    assert report['ready']
    assert any(t['name'] == 'collection_coverage' for t in list_tools())


def test_tool_cli_and_rag_use_read_only_coverage(tmp_path, capsys):
    path = tmp_path / 'events.db'
    with Store(path) as store:
        save(store)
    original = path.read_bytes()
    assert not collection_coverage(str(path), as_of=NOW)['ready']
    assert collection_coverage(str(path), requirements=[{'category': 'news'}], as_of=NOW)['ready']
    assert main(['--database', str(path), 'coverage', '--as-of', NOW]) == 1
    assert json.loads(capsys.readouterr().out)['missing_categories']
    assert path.read_bytes() == original
    past = collection_search(str(path), 'news', as_of=NOW)
    assert past['collection_coverage'] is None
    assert not collection_search(str(path), 'news')['collection_readiness']['ready']
    missing = tmp_path / 'missing.db'
    with pytest.raises(FileNotFoundError):
        collection_coverage(str(missing))
    assert not missing.exists()
