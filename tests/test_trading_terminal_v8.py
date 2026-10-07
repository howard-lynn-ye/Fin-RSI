"""Behavioral checks for terminal-return instructions and the new decision boundary."""
import json
import numpy as np
import pandas as pd
import pytest

from benchmarks.agent_study import trading_runtime_v8 as v8
from benchmarks.agent_study.linux_sandbox import confinement_available
from benchmarks.agent_study.trading_worker_v6 import run_python


def response(tool, arguments=None, finish='stop'):
    return {'choices': [{'message': {'content': json.dumps(
        {'tool': tool, 'arguments': {} if arguments is None else arguments})},
        'finish_reason': finish}]}


class Controller:
    def __init__(self, arm):
        self.arm, self.menu_seed, self.calls = arm, 7, []

    def call(self, tool, args):
        self.calls.append({'tool': tool, 'arguments': args})
        return {'ok': True, 'rows': []}


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_seven_reads_cannot_consume_the_reserved_eighth_decision(arm):
    c, requests = Controller(arm), []
    def backend(messages):
        requests.append([dict(m) for m in messages])
        if len(requests) == 8:
            assert 'FINAL DECISION' in messages[0]['content']
            assert messages[-1]['content'].endswith(v8.FINAL_DECISION)
            assert 'list_algorithms' not in messages[0]['content']
            return response('submit', {'weights': {'SPY': .75, 'IEF': .25}})
        return response('read_market')
    record = v8.decide(backend, c, v8.task('2025-01-03', {}, '2026-09-25'))
    v8.validate_decision(record)
    assert record['submitted'] and record['target']['SPY'] == .75
    assert len(c.calls) == 7 and len(record['turns']) == 8
    assert record['turns'][-1]['phase'] == 'final'
    assert 'risk-adjusted' not in requests[0][0]['content'] + requests[0][1]['content']
    assert '2026-09-25' in requests[0][1]['content']
    assert ('fin-skills' in requests[0][0]['content']) == (arm == 'library')
    assert record['turns'][-2]['model_feedback'] == requests[-1][-1]['content']


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_submit_can_carry_explanation_without_changing_weights(arm):
    record = v8.decide(lambda _: response('submit', {
        'weights': {'SPY': .6}, 'rationale': 'Model-authored explanation, not an order.'}),
        Controller(arm), 'fixture')
    v8.validate_decision(record)
    assert record['decision_action'] == 'submit' and record['target']['SPY'] == .6
    assert sum(record['target'].values()) == .6 and len(record['turns']) == 1


@pytest.mark.parametrize('extra', [{'leverage': 2}, {'rationale': {'SPY': 1}}])
def test_submit_does_not_ignore_unrecognized_or_invalid_metadata(extra):
    record = v8.decide(lambda _: response('submit', {'weights': {'SPY': .6}, **extra}),
                       Controller('raw'), 'fixture')
    assert record['decision_action'] == 'failed' and record['target'] is None
    assert all('No other keys' in t['result']['error'] for t in record['turns'])


@pytest.mark.parametrize('last', [response('read_market'), response('run_python', {'code': 'submit({"SPY":1})'}),
    response('submit', {'weights': {'SPY': -1}}), response('hold', {'weights': {}}),
    response('submit', {'weights': {'SPY': 1}}, finish='length')])
def test_invalid_final_action_is_failure_without_ninth_call_or_fallback(last):
    c, calls = Controller('library'), []
    def backend(messages):
        calls.append(1)
        return last if len(calls) == 8 else response('read_market')
    record = v8.decide(backend, c, 'fixture')
    v8.validate_decision(record)
    assert len(calls) == 8 and len(c.calls) == 7
    assert record['target'] is None and record['decision_action'] == 'failed'
    assert not record['decision_completed'] and not record['submitted']


def test_explicit_hold_is_completed_but_never_rebalances_drifted_weights():
    record = v8.decide(lambda _: response('hold'), Controller('raw'), 'fixture')
    v8.validate_decision(record)
    assert record['decision_action'] == 'hold' and record['decision_completed']
    assert record['target'] is None and not record['submitted']
    prices = pd.DataFrame({'SPY': [100., 100., 120., 132., 140.],
                           'IEF': [100., 100., 100., 100., 100.]},
                          index=pd.bdate_range('2025-01-01', periods=5).strftime('%Y-%m-%d'))
    for ticker in v8.TICKERS:
        if ticker not in prices:
            prices[ticker] = 100.
    weights = dict.fromkeys(v8.TICKERS, 0.)
    weights.update(SPY=.5, IEF=.5)
    held, held_trades = v8.prior.ledger(prices, [0, 2], [weights, record['target']])
    replaced, replaced_trades = v8.prior.ledger(prices, [0, 2], [weights, weights])
    assert held_trades[-1]['cost'] == 0 and replaced_trades[-1]['cost'] > 0
    assert held.iloc[-1] != replaced.iloc[-1]


def test_hold_cannot_be_fabricated_from_an_unanswered_turn():
    record = v8.decide(lambda _: response('hold'), Controller('raw'), 'fixture')
    record['turns'][-1]['response'] = 'I might hold.'
    with pytest.raises(ValueError, match='explicit accepted'):
        v8.validate_decision(record)


def test_multi_action_never_silently_selects_a_trade():
    answer = response('hold')
    answer['choices'][0]['message']['content'] += '\n' + response('submit', {'weights': {'SPY': 1}})['choices'][0]['message']['content']
    c = Controller('raw')
    record = v8.decide(lambda _: answer, c, 'fixture')
    assert record['decision_action'] == 'failed' and not c.calls
    assert all(t['parse'] == 'multiple-actions' for t in record['turns'])


def test_backend_failure_is_not_a_zero_return_model_decision():
    def broken(_):
        raise ValueError('context exceeded')
    with pytest.raises(RuntimeError, match='infrastructure qualification'):
        v8.decide(broken, Controller('raw'), 'fixture')


@pytest.fixture
def workspace(tmp_path):
    dates = pd.bdate_range('2024-01-01', periods=270).strftime('%Y-%m-%d')
    rows = []
    for j, ticker in enumerate(('SPY', 'IEF', 'GLD')):
        prices = 100 * np.cumprod(1 + .001 + .01 * np.sin(np.arange(270) / (3 + j)))
        rows.extend(dict(date=d, ticker=ticker, close=p) for d, p in zip(dates, prices))
    pd.DataFrame(rows).to_csv(tmp_path / 'quotes.csv', index=False)
    pd.DataFrame(columns=['date', 'ticker', 'kind', 'value']).to_csv(tmp_path / 'corporate_actions.csv', index=False)
    (tmp_path / 'evidence.json').write_text(json.dumps(dict(as_of=dates[-1], records=[dict(
        id='fixture', category='news', eligible_date=dates[0], text='synthetic observation')])))
    (tmp_path / 'README.md').write_text('Synthetic market fixture.')
    return tmp_path


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_help_examples_execute_and_do_not_assume_persistent_evidence(arm, workspace):
    tools = v8.Tools(workspace, arm)
    help_reply = tools.call('help_tool', {'name': 'run_python'})
    reply = run_python(help_reply['python_example'], tools)
    assert reply['ok'] and 'fixture' in reply['output']
    assert tools.call('help_tool', {'name': 'load_history'})['ok'] == (arm == 'library')
    assert tools.call('help_tool', {'name': 'list_library_tools'})['ok'] == (arm == 'library')


def test_wrong_arguments_get_exact_contract_without_silent_algorithm_substitution(workspace):
    tools = v8.Tools(workspace, 'library')
    reply = tools.call('list_algorithms', {'query': 'quantum'})
    assert not reply['ok'] and 'no query' in reply['contract']['note']
    assert reply['contract']['signature'] == 'list_algorithms(task=None, offset=0, limit=8)'
    reply = tools.call('run_algorithm', {'algorithm_id': 'invented_algo'})
    assert not reply['ok'] and 'describe_algorithm' in reply['contract']['note']
    assert not any(r['ok'] and r['tool'] == 'run_algorithm' for r in tools.calls)


@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_v8_real_worker_contract_submission_and_raw_isolation(workspace):
    for arm in ('raw', 'library'):
        c = v8.Controller(workspace / ('runtime-' + arm), workspace, arm)
        help_reply = c.call('help_tool', {'name': 'run_python'})
        reply = c.call('run_python', {'code': help_reply['python_example'] + '\nsubmit({"SPY":.8})'})
        assert reply['ok'] and reply['submission']['SPY'] == .8, reply
        reply = c.call('run_python', {'code': 'print(help_tool("read_market")["signature"])'})
        assert reply['ok'] and 'read_market(' in reply['output'], reply
        reply = c.call('run_python', {'code': 'print(evidence["records"])'})
        assert not reply['ok'] and 'json.load' in reply['hint']
        reply = c.call('run_python', {'code': 'import fin_skills'})
        assert reply['ok'] == (arm == 'library')
        assert not c.call('run_python', {'code': 'open("quotes.csv","w").write("bad")'})['ok']


@pytest.mark.parametrize('interface', ['v8', 'v9'])
def test_freeze_uses_terminal_deadline_and_qualification_uses_identical_prompt(tmp_path, monkeypatch, interface):
    pytest.importorskip('bs4')
    from benchmarks.agent_study import multisource_model_study as study
    from benchmarks.agent_study.qualify_model_matrix import prompts
    data = pd.DataFrame({'SPY': [100., 101., 102.]}, index=['2025-01-02', '2025-01-03', '2025-01-06'])
    archive = tmp_path / 'market-source'
    archive.write_text('fixture')
    monkeypatch.setattr(study.md, 'RAW', archive)
    def write_dataset(path):
        path.mkdir(parents=True)
        data.to_csv(path / study.md.HIDDEN)
        return {study.md.HIDDEN: study.old.sha(path / study.md.HIDDEN)}
    monkeypatch.setattr(study.md, 'write_dataset', write_dataset)
    monkeypatch.setattr(study.old, 'schedule', lambda dates: [0])
    records = [dict(category=c, series=c, id=c, event_date='2025-01-01', eligible_date='2025-01-02',
                    text='fixture observation') for c in ('news', 'psychology', 'behavior', 'officials')]
    evidence = tmp_path / 'evidence.json'
    evidence.write_text(json.dumps(records))
    root = tmp_path / 'new-run'
    study.freeze(root, '7b', 11, evidence, allow_retrospective=True, interface=interface)
    protocol = study.verify(root)
    assert protocol['version'] == study.VERSIONS[interface] and protocol['interface'] == interface
    assert protocol['objective'] == v8.OBJECTIVE and protocol['deadline'] == '2025-01-06'
    for arm, messages in prompts(root):
        assert messages[0]['content'] == study.terminal_runtime(interface).system_prompt(arm)
        assert '2025-01-06' in messages[1]['content'] and 'risk-adjusted' not in messages[1]['content']
    protocol['deadline'] = '2099-01-01'
    (root / 'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='deadline'):
        study.verify(root)


@pytest.mark.parametrize('interface', ['v8', 'v9'])
def test_reported_capital_and_return_reconcile_with_independent_holdings_replay(tmp_path, monkeypatch, interface):
    pytest.importorskip('bs4')
    from benchmarks.agent_study import multisource_model_study as study
    from benchmarks.agent_study.audit_model_multisource import audit
    dates = pd.bdate_range('2025-01-02', periods=6).strftime('%Y-%m-%d')
    prices = pd.DataFrame({ticker: [100., 100., 110., 121., 120., 132.]
                           for ticker in study.md.TICKERS}, index=dates)
    root = tmp_path / 'case'
    (root / 'data').mkdir(parents=True)
    prices.to_csv(root / 'data' / study.md.HIDDEN)
    protocol = dict(interface=interface, model=['fixture', 'fixture', 'fixture'], seed=11,
                    initial_capital=v8.INITIAL_CAPITAL, capital_currency='USD', cost_bps=5,
                    universe=list(study.md.TICKERS), window=[dates[1], dates[-1]], limits=[])
    inputs = dict(arms=['raw', 'library'], picks=[0, 2])
    study.old.write(root / 'protocol.json', protocol)
    study.old.write(root / 'inputs.json', inputs)
    study.old.write(root / 'capability-qualification.json', {'fixture': True})
    study.old.write(root / 'decision-qualification.json', {'fixture': True})
    receipt = dict(protocol_sha256=study.old.sha(root / 'protocol.json'), decisions={},
        capability_qualification_sha256=study.old.sha(root / 'capability-qualification.json'),
        decision_qualification_sha256=study.old.sha(root / 'decision-qualification.json'))
    if interface == 'v9':
        study.old.write(root / 'rag-qualification.json', {'fixture': True})
        receipt['rag_qualification_sha256'] = study.old.sha(root / 'rag-qualification.json')
    for arm in inputs['arms']:
        for index, pick in enumerate(inputs['picks']):
            record = study.terminal_runtime(interface).decide(
                               lambda _: response('hold') if index else response('submit', {'weights': {'SPY': 1}}),
                               Controller(arm), 'synthetic test')
            record.update(date=dates[pick], index=pick)
            name = f'decisions/{arm}/{index:02d}.json'
            study.old.write(root / name, record)
            receipt['decisions'][name] = study.old.sha(root / name)
    study.old.write(root / 'inference-receipt.json', receipt)
    monkeypatch.setattr(study, 'verify', lambda path: protocol)
    scores = study.score(root)  # Uses the real receipt checks, ledger and independent audit.
    for arm in inputs['arms']:
        account = scores['paths'][arm]
        assert account['initial_capital'] == 100000
        assert account['ending_capital'] == pytest.approx(100000 * (1 - .0005) * 1.32)
        assert account['return_rate_pct'] == pytest.approx(100 * (account['ending_capital'] / 100000 - 1))
        assert account['explicit_holds'] == 1 and account['failed_decisions'] == 0
    bad_scores = study.old.load(root / 'scores.json')
    bad_scores['paths']['library']['ending_capital'] += 100
    (root / 'scores.json').write_text(json.dumps(bad_scores))
    with pytest.raises(AssertionError):
        audit(root)
