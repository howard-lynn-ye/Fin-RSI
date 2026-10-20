import json

import numpy as np
import pandas as pd
import pytest

import fin_skills
from fin_skills.algorithms import catalog
from fin_skills.tools import list_tools
from benchmarks.agent_study import trading_capabilities as cap
from benchmarks.agent_study.linux_sandbox import confinement_available
from benchmarks.agent_study.trading_tools_v5 import wire, OUTPUT_LIMIT
from benchmarks.agent_study.trading_worker_v6 import run_python


def pages(tools, name, arguments, key):
    offset, found = 0, []
    while True:
        reply = tools.call(name, dict(arguments, offset=offset))
        assert reply['ok'], reply
        assert json.loads(wire(reply)) == reply and len(wire(reply)) <= OUTPUT_LIMIT
        found.extend(reply[key] if isinstance(reply[key], list) else [reply[key]])
        if reply['next_offset'] is None:
            return found
        assert reply['next_offset'] > offset
        offset = reply['next_offset']


@pytest.fixture
def workspace(tmp_path):
    dates = pd.bdate_range('2024-01-01', periods=270).strftime('%Y-%m-%d')
    rows = []
    for j, ticker in enumerate(('SPY', 'IEF', 'GLD')):
        prices = 100 * np.cumprod(1 + .001 + .01 * np.sin(np.arange(270)/(3+j)))
        rows.extend(dict(date=d, ticker=ticker, close=p) for d, p in zip(dates, prices))
    pd.DataFrame(rows).to_csv(tmp_path / 'quotes.csv', index=False)
    pd.DataFrame(columns=['date', 'ticker', 'kind', 'value']).to_csv(
        tmp_path / 'corporate_actions.csv', index=False)
    (tmp_path / 'README.md').write_text('Synthetic fixture, no real market outcome.')
    packet = dict(as_of=dates[-1], records=[dict(id='news-1', category='news',
        eligible_date=dates[0], event_date=dates[0], text='short excerpt',
        full_text='Full synthetic source body. ' * 500)])
    (tmp_path / 'evidence.json').write_text(json.dumps(packet))
    return tmp_path


def test_entire_tool_registry_is_classified_and_discoverable(workspace):
    tools = cap.Tools(workspace, 'library')
    cards = pages(tools, 'list_library_tools', {}, 'tools')
    assert {r['name'] for r in cards} == {r['name'] for r in list_tools()}
    assert all(r['study_status'] != 'unreviewed' for r in cards)
    assert any(r['study_status'] == 'outside_replay' for r in cards)
    for card in cards:
        schema = json.loads(''.join(pages(tools, 'describe_library_tool',
                                         {'name': card['name']}, 'text')))
        assert schema['name'] == card['name'] and schema['input_schema']['type'] == 'object'


def test_complete_skill_and_reference_text_is_reachable(workspace):
    tools = cap.Tools(workspace, 'library')
    names = {r['name'] for r in pages(tools, 'list_skills', {}, 'skills')}
    assert names == set(fin_skills.names())
    for name in sorted(names):
        assert ''.join(pages(tools, 'read_skill', {'name': name}, 'text')) == fin_skills.load(name)
        for ref, text in fin_skills.references(name).items():
            assert ''.join(pages(tools, 'read_skill', {'name': name, 'reference': ref}, 'text')) == text


def test_algorithm_discovery_no_longer_hides_non_shortcut_methods(workspace, monkeypatch):
    import fin_skills.algorithms as algorithms
    tools = cap.Tools(workspace, 'library')
    cards = pages(tools, 'list_algorithms', {}, 'algorithms')
    assert {r['id'] for r in cards} == {r['id'] for r in catalog()}
    assert len(cards) == len({r['id'] for r in cards})
    assert {r['id']: r['status'] for r in cards} == {r['id']: r['status'] for r in catalog()}
    explicit = next(c for c in catalog() if tuple(c['inputs']) not in (
        ('asset_returns',), ('prices',), ('returns',), ('series',)))
    result = tools.call('describe_algorithm', {'algorithm_id': explicit['id']})
    assert result['ok'] and result['prepared_history_adapter'] is False
    assert 'call_library_tool' in result['usage']
    # Exercise missing dependencies even in an environment with every extra installed.
    simulated = [dict(r, status='missing_dependency') for r in catalog()]
    monkeypatch.setattr(algorithms, 'catalog', lambda task=None: simulated)
    unavailable = pages(tools, 'list_algorithms', {}, 'algorithms')
    assert {r['id'] for r in unavailable} == {r['id'] for r in cards}
    assert all(r['status'] == 'missing_dependency' for r in unavailable)


@pytest.mark.parametrize('rf,passed', [(0.05, True), (5.0, False)])
def test_guard_execution_success_does_not_hide_failed_audit(workspace, rf, passed):
    tools = cap.Tools(workspace, 'library')
    result = tools.call('call_library_tool', {'name': 'check_rf_convention',
        'arguments': {'returns': [.01, -.02, .005, .012], 'rf': rf}})
    assert result['ok'], result
    assert result['result']['passed'] is passed and result['passed'] is passed
    assert tools.calls[-2]['library_tool'] == 'check_rf_convention'
    assert tools.calls[-2]['passed'] is passed


def test_external_and_secondary_model_requests_are_explicitly_rejected(workspace):
    tools = cap.Tools(workspace, 'library')
    for name, arguments in (('search_news', {}), ('collect_once', {'database': 'x'}),
                            ('run_model', {'model_id': 'jev', 'data': {}})):
        result = tools.call('call_library_tool', {'name': name, 'arguments': arguments})
        assert not result['ok'], result
    assert not cap.Tools(workspace, 'raw').call('list_library_tools', {})['ok']


def test_both_arms_can_retrieve_full_current_evidence_without_future_records(workspace):
    expected = json.loads((workspace / 'evidence.json').read_text())['records'][0]['full_text']
    for arm in ('raw', 'library'):
        tools = cap.Tools(workspace, arm)
        assert len(pages(tools, 'read_evidence', {}, 'records')) == 1
        assert ''.join(pages(tools, 'read_evidence_document', {'id': 'news-1'}, 'text')) == expected
    packet = json.loads((workspace / 'evidence.json').read_text())
    packet['records'][0]['eligible_date'] = '2099-01-01'
    (workspace / 'evidence.json').write_text(json.dumps(packet))
    result = cap.Tools(workspace, 'library').call('read_evidence', {})
    assert not result['ok'] and 'future record' in result['error']


def test_model_still_chooses_weights_and_backend_failure_stops_inference(workspace):
    tools = cap.Tools(workspace, 'library')
    result = run_python("r=call_library_tool('check_rf_convention', "
                        "{'returns':[.01,-.02,.005,.012], 'rf':.05})\n"
                        "assert r['ok'] and r['result']['passed']\nprint('audit read')", tools)
    assert result['ok'] and result['submission'] is None
    guide = workspace / 'guide.md'
    guide.write_text('Synthetic orientation, not a strategy.')
    def failed_backend(messages):
        assert 'list_library_tools' in messages[0]['content']
        raise ValueError('context budget exceeded')
    with pytest.raises(RuntimeError, match='infrastructure qualification'):
        cap.decide(failed_backend, cap.Controller(workspace / 'run', workspace, 'library'),
                   'synthetic task', orientation_path=guide)


def test_python_full_result_has_same_external_policy_and_no_truncation(workspace):
    tools = cap.Tools(workspace, 'library')
    result = run_python("r=execute_library_tool('list_algorithms')\n"
                        "assert r['ok']\nprint(len(r['result']['algorithms']))\n"
                        "r=execute_library_tool('search_news')\n"
                        "assert not r['ok'] and 'Live network' in r['error']", tools)
    assert result['ok'] and result['output'].strip() == str(len(catalog())), result
    assert 'execute_library_tool' not in cap.Tools(workspace, 'raw').python_bindings()


def test_evidence_snapshot_matches_prompt_and_qualification_cannot_be_reused(tmp_path):
    pytest.importorskip('bs4')
    from benchmarks.agent_study import multisource_model_study as study
    records = [dict(category=c, series=c, id=c, event_date='2025-01-01',
        eligible_date='2025-01-02', text='excerpt', full_text='complete text')
        for c in ('news', 'psychology', 'behavior', 'officials')]
    packet = study.evidence_packet(records, '2025-01-03')
    cap.write_evidence_snapshot(records, packet, tmp_path)
    copied = json.loads((tmp_path / 'evidence.json').read_text())
    assert all(r['full_text'] == 'complete text' for r in copied['records'])
    bad = dict(packet, records=[])
    with pytest.raises(ValueError, match='does not match frozen'):
        cap.write_evidence_snapshot(records, bad, tmp_path)
    (tmp_path / 'protocol.json').write_text('{}')
    report = dict(passed=True, checks={'probe': {'passed': True}},
                  protocol_sha256=study.old.sha(tmp_path / 'protocol.json'),
                  registry_sha256=cap.hashlib.sha256(cap.encoded(cap.registry()).encode()).hexdigest(),
                  environment=cap.environment(),
                  evidence_snapshot_sha256=study.old.sha(tmp_path / 'evidence.json'))
    (tmp_path / 'capability-visible').mkdir()
    (tmp_path / 'capability-visible' / 'evidence.json').write_bytes((tmp_path / 'evidence.json').read_bytes())
    (tmp_path / 'capability-qualification.json').write_text(json.dumps(report))
    assert cap.require_qualification(tmp_path)['passed']
    (tmp_path / 'protocol.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='stale'):
        cap.require_qualification(tmp_path)


def test_pages_preserve_one_execution_and_dataframe_inputs_keep_labels(workspace, monkeypatch):
    import fin_skills.tools as ft
    calls = []
    def fake(name, args):
        calls.append(args)
        return {'nonce': len(calls), 'blob': 'x' * (3 * OUTPUT_LIMIT)}
    monkeypatch.setattr(ft, 'call_tool', fake)
    tools = cap.Tools(workspace, 'library')
    text = ''.join(pages(tools, 'call_library_tool', {'name': 'check_rf_convention'}, 'text'))
    assert len(calls) == 1 and json.loads(text)['nonce'] == 1
    assert sum(c['tool'] == 'library_execution' for c in tools.calls) == 1
    h = tools.load_history(lookback=60)
    reply = tools.python_bindings()['execute_library_tool']('run_model',
        {'model_id': 'ewma_covariance', 'data': {'asset_returns': h['returns']}})
    assert reply['ok'], reply
    frame = calls[-1]['data']['asset_returns']
    assert frame['columns'] == list(h['returns'].columns)
    assert len(frame['records']) == 60


def test_guard_flag_is_not_permission_and_failed_results_are_explicit(workspace, monkeypatch):
    import fin_skills.tools as ft
    previous = ft.list_tools
    monkeypatch.setattr(ft, 'list_tools', lambda: previous() + [dict(name='check_new', guard=True,
        description='new unreviewed guard', input_schema={'type': 'object'})])
    assert next(r for r in cap.registry() if r['name'] == 'check_new')['study_status'] == 'unreviewed'
    tools = cap.Tools(workspace, 'library')
    assert not tools.call('call_library_tool', {'name': 'check_new'})['ok']
    def broken(*args):
        raise RuntimeError('solver failure')
    monkeypatch.setattr(ft, 'call_tool', broken)
    reply = tools.call('call_library_tool', {'name': 'check_rf_convention'})
    assert not reply['ok'] and 'RuntimeError' in reply['error']
    monkeypatch.setattr(ft, 'call_tool', lambda *args: {'invalid': float('nan')})
    reply = tools.call('call_library_tool', {'name': 'check_rf_convention'})
    assert not reply['ok'] and 'unrepresentable library result' in reply['error']


def test_full_text_excerpt_and_numeric_record_are_not_confused(workspace):
    packet = json.loads((workspace / 'evidence.json').read_text())
    for name, fields in [('excerpt', {'full_text': '', 'text': 'excerpt only'}),
                         ('numeric', {'open_interest': 123})]:
        packet['records'].append(dict(id=name, category='behavior', eligible_date=packet['as_of'],
                                      event_date=packet['as_of'], **fields))
    (workspace / 'evidence.json').write_text(json.dumps(packet))
    tools = cap.Tools(workspace, 'raw')
    for name, source, complete in [('news-1', 'full_text', True), ('excerpt', 'text', False),
                                   ('numeric', 'record', False)]:
        result = tools.call('read_evidence_document', {'id': name})
        assert (result['source'], result['complete_source']) == (source, complete)


@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_actual_sandbox_can_use_new_tools_and_still_blocks_future_files(workspace, tmp_path):
    hidden = tmp_path / 'private'
    hidden.mkdir()
    (hidden / 'future.json').write_text('future sentinel')
    visible = tmp_path / 'visible'
    visible.mkdir()
    for name in ('quotes.csv', 'corporate_actions.csv', 'README.md', 'evidence.json'):
        (visible / name).write_bytes((workspace / name).read_bytes())
    controller = cap.Controller(tmp_path / 'runtime', visible, 'library')
    code = """r=call_library_tool('check_rf_convention', {'returns':[.01,-.02,.005,.012],'rf':5})
assert r['ok'] and not r['result']['passed']
r=call_library_tool('get_quant_method', {'method_id':'not-a-real-method'})
assert not r['ok']
h=load_history(tickers=['SPY','IEF','GLD'],lookback=60)
r=run_algorithm('inverse_volatility',tickers=['SPY','IEF','GLD'],lookback=60)
assert r['ok']
print('audited without auto-submitting')
"""
    result = controller.call('run_python', {'code': code})
    assert result['ok'] and result['submission'] is None, result
    for code in (f"open({str(hidden / 'future.json')!r}).read()",
                 'import socket; socket.socket()', "open('quotes.csv','w').write('bad')"):
        result = controller.call('run_python', {'code': code})
        assert not result['ok'] and 'PermissionError' in result['output'], result
    result = controller.call('call_library_tool', {'name': 'retrieve_context',
        'arguments': {'query': 'reporting delay', 'skills': ['congressional-trading-disclosures'],
                      'top_k': 1, 'max_context_chars': 1200}})
    assert result['ok'], result
    before = sum(c.get('library_tool') == 'list_algorithms' and c['tool'] == 'library_execution'
                 for c in controller.calls)
    offset, chunks, fingerprints = 0, [], set()
    while True:
        reply = controller.call('call_library_tool', {'name': 'list_algorithms', 'offset': offset,
                                                       'limit': 1800})
        assert reply['ok'], reply
        assert '_cached_library_result' not in reply
        chunks.append(reply['text'])
        fingerprints.add(reply['result_sha256'])
        if reply['next_offset'] is None:
            break
        offset = reply['next_offset']
    assert len(fingerprints) == 1
    assert len(json.loads(''.join(chunks))['algorithms']) == len(catalog())
    after = sum(c.get('library_tool') == 'list_algorithms' and c['tool'] == 'library_execution'
                for c in controller.calls)
    assert after - before == 1
    # Cache hits must still obey the argument contract, and metadata must not double-count calls.
    assert not controller.call('call_library_tool', {'name': 'list_algorithms', 'arguments': []})['ok']
    assert not controller.call('call_library_tool', {'name': 'list_algorithms', 'limit': 0})['ok']
    before = sum(c['tool'] == 'read_evidence' for c in controller.calls)
    assert controller.call('read_evidence', {})['ok']
    assert sum(c['tool'] == 'read_evidence' for c in controller.calls) - before == 1
    for arm in ('raw', 'library'):
        controller = cap.Controller(tmp_path / arm, visible, arm)
        assert controller.call('read_evidence', {})['ok']
    assert not controller.tools.call('call_library_tool', {'name': 'search_news'})['ok']
