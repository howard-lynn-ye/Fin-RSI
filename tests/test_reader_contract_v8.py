"""The displayed file schemas and examples must match actual visible inputs."""
import json
import pandas as pd
import pytest

from benchmarks.agent_study import trading_runtime_v8 as v8
from benchmarks.agent_study import market_data as md
from benchmarks.agent_study.linux_sandbox import confinement_available


@pytest.fixture
def workspace(tmp_path):
    # Exercise the real vendor-to-visible-data builder, rather than the older
    # close-only unit fixture that intentionally omits OHLCV and action notes.
    vendor = {}
    for ticker in md.TICKERS:
        vendor[ticker] = {'chart': {'result': [{'timestamp': [1735833600, 1735920000],
            'indicators': {'quote': [{'open': [100, 101], 'high': [101, 102],
                'low': [99, 100], 'close': [100, 101], 'volume': [1000, 1100]}],
                'adjclose': [{'adjclose': [100, 101]}]},
            'events': {'dividends': {'one': {'date': 1735920000, 'amount': 0.1}}}}]}}
    vendor_path = tmp_path/'vendor.json'
    vendor_path.write_text(json.dumps(vendor))
    quotes, actions, _ = md.build(vendor_path)
    visible = tmp_path/'visible'
    visible.mkdir()
    quotes.to_csv(visible/'quotes.csv', index=False)
    actions.to_csv(visible/'corporate_actions.csv', index=False)
    (visible/'README.md').write_text(md.README)
    (visible/'evidence.json').write_text(json.dumps({'as_of': str(quotes.date.max()), 'records': []}))
    return visible


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_visible_csv_headers_match_documented_contracts(arm, workspace):
    tools = v8.Tools(workspace, arm)
    contract = tools.help_tool('run_python')
    for name, fields in contract['input_files'].items():
        assert pd.read_csv(workspace/name, nrows=0).columns.tolist() == fields
    prompt = v8.system_prompt(arm)
    assert 'date,ticker,open,high,low,close,volume' in prompt
    assert 'use rows (a list of records)' in prompt
    assert 'do not recreate supplied rows' in prompt
    assert 'Raw prices, not adj_close' in contract['input_convention']


@pytest.mark.parametrize('arm', ['raw', 'library'])
@pytest.mark.parametrize('reader', ['read_market', 'read_snapshot'])
@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_reader_examples_run_through_real_confined_worker(arm, reader, workspace):
    controller = v8.Controller(workspace/('contract-'+arm+'-'+reader), workspace, arm)
    contract = controller.tools.help_tool(reader)
    page = controller.call(reader, {})
    assert page['ok'] and page['rows'] and 'data' not in page
    assert set(page['rows'][0]) == set(contract['row_fields_by_file']['quotes.csv'])
    result = controller.call('run_python', {'code': contract['python_example']})
    assert result['ok'], result
    assert 'close' in result['output'] and 'ticker' in result['output']
    assert 'submission' not in result or result['submission'] is None


@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_common_data_contracts_and_python_failure_feedback_are_equal(workspace):
    raw = v8.Controller(workspace/'contract-raw-error', workspace, 'raw')
    library = v8.Controller(workspace/'contract-library-error', workspace, 'library')
    for name in ('run_python', 'read_market', 'read_snapshot'):
        assert raw.tools.help_tool(name) == library.tools.help_tool(name)
    for controller in (raw, library):
        result = controller.call('run_python', {'code': 'import pandas as pd\nprint(pd.read_csv("quotes.csv")["Date"])'})
        assert not result['ok'] and 'KeyError' in result['error']
        assert result['contract']['input_files']['quotes.csv'][0] == 'date'
        assert 'Date' not in result['contract']['input_files']['quotes.csv']
        assert 'submission' not in result or result['submission'] is None
