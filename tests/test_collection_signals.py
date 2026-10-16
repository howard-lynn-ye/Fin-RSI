"""Live-source contracts: observed-time availability, revisions and scheduled refresh."""
import json
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZipFile

import pytest

from fin_skills.collect import Collector, FetchError, HttpClient, Response, Store, Watch
from fin_skills.collect.disclosures import HouseSource
from fin_skills.collect.signals import AAII_URL, AAIISource, CFTCSource, aaii_readings

NOW = '2026-10-16T12:00:00+00:00'
HTML = '''<html><body><h2>This week's results</h2><p>Week ending October 14, 2026</p>
<div>Bullish</div><div>40.0%</div><div>Avg 37.5%</div><div>Neutral</div><div>20.0%</div>
<div>Avg 31.0%</div><div>Bearish</div><div>40.0%</div><div>Avg 31.5%</div></body></html>'''


def http(sender):
    return HttpClient(sender=sender, min_interval=0, sleep=lambda _: None)


def row():
    r = dict(id='fixture', report_date_as_yyyy_mm_dd='2026-10-06T00:00:00.000',
             cftc_contract_market_code='13874A', futonly_or_combined='FutOnly',
             open_interest_all='1000', market_and_exchange_names='Fixture S&P 500',
             contract_units='fixture units')
    for long, short in CFTCSource.GROUPS.values():
        r.update({long: '120', short: '100'})
    return r


def test_aaii_extracts_current_panel_not_average_and_respects_robots(monkeypatch):
    calls = []
    def sender(url, *_):
        calls.append(url)
        return Response(200, (b'User-agent: *\nAllow: /' if url.endswith('robots.txt')
                              else HTML.encode()), {'content-type': 'text/html'})
    source = AAIISource(http(sender))
    sleeps = []
    source.pages.client.sleep = sleeps.append
    batch = source.fetch(Watch('a', 'aaii', 'sentiment'), {})
    assert [e.data['value'] for e in batch.events] == [40, 20, 40]
    assert all(e.published_at is None and e.data['availability_basis'] == 'first_observed'
               for e in batch.events)
    assert calls == ['https://www.aaii.com/robots.txt', AAII_URL]
    assert sleeps == [10, 10]
    source = AAIISource(http(lambda *_: Response(200, b'User-agent: *\nDisallow: /')))
    with pytest.raises(FetchError, match='disallows'):
        source.fetch(Watch('a', 'aaii', 'sentiment'), {})


@pytest.mark.parametrize('content', ['login required',
    "This week's results Week ending October 14, 2026 Bullish 120% Neutral 0% Bearish 0%",
    "This week's results Week ending October 14, 2026 Bullish 10% Neutral 10% Bearish 10%",
    "This week's results Week ending October 14, 2030 Bullish 40% Neutral 20% Bearish 40%"])
def test_aaii_schema_invalid_values_and_future_dates_fail_closed(content):
    with pytest.raises(ValueError):
        aaii_readings(content, NOW)


def test_cftc_returns_reported_groups_and_uninvented_publication(monkeypatch):
    monkeypatch.setattr('fin_skills.collect.signals.utcnow', lambda: NOW)
    calls = []
    def sender(url, *_):
        calls.append(url)
        return Response(200, json.dumps([row()]).encode())
    batch = CFTCSource(http(sender)).fetch(Watch('c', 'cftc', '13874A'), {})
    assert len(batch.events) == 5
    assert all(e.data['value'] == 20 and e.data['unit'] == 'contracts' for e in batch.events)
    assert all(e.published_at is None and e.data['report_date'] == '2026-10-06' for e in batch.events)
    assert '%24limit=4' in calls[0] and '13874A' in calls[0]


@pytest.mark.parametrize('mutate', ['empty', 'duplicate', 'missing', 'scope', 'negative', 'future', 'above_oi'])
def test_cftc_missing_bad_or_wrong_scope_never_becomes_zero(mutate, monkeypatch):
    monkeypatch.setattr('fin_skills.collect.signals.utcnow', lambda: NOW)
    item = row()
    payload = [item]
    if mutate == 'empty': payload = []
    elif mutate == 'duplicate': payload *= 2
    elif mutate == 'missing': del item['lev_money_positions_long']
    elif mutate == 'scope': item['futonly_or_combined'] = 'Combined'
    elif mutate == 'negative': item['lev_money_positions_long'] = '-1'
    elif mutate == 'above_oi': item['lev_money_positions_long'] = '1001'
    else: item['report_date_as_yyyy_mm_dd'] = '2030-10-06T00:00:00.000'
    with pytest.raises(ValueError):
        CFTCSource(http(lambda *_: Response(200, json.dumps(payload).encode()))).fetch(
            Watch('c', 'cftc', '13874A'), {})


def test_scheduled_cftc_refresh_revisions_and_restart_do_not_backdate(tmp_path, monkeypatch):
    monkeypatch.setattr('fin_skills.collect.signals.utcnow', lambda: NOW)
    payload, clock, calls = [row()], [0.], []
    def sender(url, *_):
        calls.append(url)
        return Response(200, json.dumps(payload).encode())
    path = tmp_path / 'events.db'
    with Store(path) as store:
        collector = Collector(store, client=http(sender), clock=lambda: clock[0])
        collector.add(Watch('c', 'cftc', '13874A', interval_seconds=60))
        assert collector.once()['new_records'] == 5
        assert not collector.once()['watches'] and len(calls) == 1
        baseline = store.coverage(requirements=[{'category': 'behavior'}], as_of=NOW)
        assert baseline['ready'] and not baseline['publication_time_complete']
        assert baseline['categories'][0]['examples'][0]['freshness_basis'] == 'report_date'
    with Store(path) as store:
        collector = Collector(store, client=http(sender), clock=lambda: clock[0])
        clock[0] = 61
        assert collector.once()['new_records'] == 0  # unchanged fetch does not create revisions
        payload[0]['lev_money_positions_long'] = '125'
        monkeypatch.setattr('fin_skills.collect.signals.utcnow', lambda: '2026-10-17T12:00:00+00:00')
        clock[0] = 122
        assert collector.once()['new_records'] == 1
        assert store.coverage(requirements=[{'category': 'behavior'}], as_of=NOW) == baseline
        past = store.coverage(requirements=[{'category': 'behavior'}], as_of='2026-10-15T12:00:00Z')
        assert not past['ready']


@pytest.mark.parametrize('change', [
    {'source': 'unverified'}, {'url': 'https://example.com/fake'},
    {'data': {'availability_basis': 'unknown'}},
    {'data': {'report_date': '2025-01-01', 'availability_basis': 'first_observed'}},
    {'data': {'report_date': '2030-01-01', 'availability_basis': 'first_observed'}},
])
def test_observed_time_requires_real_adapter_provenance_and_fresh_reference(tmp_path, change, monkeypatch):
    monkeypatch.setattr('fin_skills.collect.signals.utcnow', lambda: NOW)
    e = CFTCSource(http(lambda *_: Response(200, json.dumps([row()]).encode()))).fetch(
        Watch('c', 'cftc', '13874A'), {}).events[0]
    if 'data' in change:
        change = dict(change, data={**e.data, **change['data']})
    with Store(tmp_path / 'events.db') as store:
        from fin_skills.collect import Batch
        w = Watch('c', 'cftc', '13874A')
        store.put_watch(w)
        store.save(w, Batch([replace(e, **change)]), next_due=0)
        assert not store.coverage(requirements=[{'category': 'behavior'}], as_of=NOW)['ready']


def house_index(year, docs):
    raw = '<FinancialDisclosure>' + ''.join(
        f'<Member><Last>Example</Last><First>Person</First><FilingType>P</FilingType>'
        f'<Year>{year}</Year><FilingDate>{day}</FilingDate><DocID>{identity}</DocID></Member>'
        for identity, day in docs) + '</FinancialDisclosure>'
    stream = BytesIO()
    with ZipFile(stream, 'w') as archive:
        archive.writestr(f'{year}FD.xml', raw)
    return stream.getvalue()


def test_house_current_year_newest_first_and_unparsed_retry(monkeypatch):
    year = datetime.now(timezone.utc).year
    data = house_index(year, [('1', f'1/1/{year}'), ('2', f'2/1/{year}')])
    calls = []
    def sender(url, *_):
        calls.append(url)
        return Response(200, data if url.endswith('.ZIP') else b'pdf')
    source = HouseSource(http(sender))
    clock = [100.]
    monkeypatch.setattr('fin_skills.collect.disclosures.time.time', lambda: clock[0])
    monkeypatch.setattr('fin_skills.collect.disclosures.pdf_text', lambda _: 'unreadable')
    w = Watch('h', 'house', 'Example', 1, {'year': 'current', 'newest_first': True, 'max_filings': 1})
    first = source.fetch(w, {})
    assert calls[-1].endswith(f'/{year}/2.pdf') and first.warnings
    assert first.state['seen'] == [] and '2' in first.state['retry']
    assert first.state['remaining'] == 2
    monkeypatch.setattr('fin_skills.collect.disclosures.pdf_text', lambda _: (
        f'SP Example (EXM) P 01/01/{year} 01/02/{year} $1,001 - $15,000'))
    clock[0] = 1000
    second = source.fetch(w, first.state)
    assert second.state['seen'] == ['1']  # a failed newest filing cannot starve other work
    third = source.fetch(w, second.state)
    assert third.state['seen'] == ['1', '2']
    tx = third.events[-1]
    assert tx.published_at is None and tx.data['availability_basis'] == 'first_observed'


def test_four_kind_profile_and_registered_sources(tmp_path):
    from fin_skills.collect import sources
    from fin_skills.tools import call_tool
    assert {'rss', 'aaii', 'cftc', 'house'} <= {s['name'] for s in sources()}
    assert {'aaii', 'cftc'} <= {s['name'] for s in call_tool('collection_sources', {})['sources']}
    report = call_tool('collection_configure', {'database': str(tmp_path / 'tools.db'), 'watches': [
        {'id': 'a', 'source': 'aaii', 'target': 'sentiment'},
        {'id': 'c', 'source': 'cftc', 'target': '13874A'}]})
    assert report['started'] is False
