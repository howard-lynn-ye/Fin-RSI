import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('bs4')
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study.audit_model_multisource import audit


def test_nested_model_audit_reproduces_fees_hold_and_rebalance_and_detects_corruption(tmp_path, monkeypatch):
    old = study.old
    dates = pd.bdate_range('2025-01-01', periods=8).strftime('%Y-%m-%d')
    prices = pd.DataFrame({t: 100 * np.cumprod(1 + .01 * np.sin(np.arange(8) + j))
                           for j, t in enumerate(study.md.TICKERS)}, index=dates)
    (tmp_path / 'data').mkdir()
    prices.to_csv(tmp_path / 'data' / study.md.HIDDEN)
    inputs = dict(picks=[0, 2, 4], arms=['raw', 'library'])
    old.write(tmp_path / 'inputs.json', inputs)
    protocol = dict(universe=list(study.md.TICKERS), cost_bps=old.COST_BPS)
    old.write(tmp_path / 'protocol.json', protocol)
    monkeypatch.setattr(study, 'verify', lambda root: protocol)
    receipts, scores = {}, {'paths': {}}
    for arm in inputs['arms']:
        a = dict.fromkeys(study.md.TICKERS, 0.)
        a['SPY'] = .5
        b = dict(a, SPY=.2, IEF=.6)
        targets = [a, None, b]
        for i, (pick, target) in enumerate(zip(inputs['picks'], targets)):
            path = tmp_path / 'decisions' / arm / f'{i:02d}.json'
            old.write(path, dict(index=pick, date=dates[pick], target=target))
            receipts[path.relative_to(tmp_path).as_posix()] = old.sha(path)
        nav, trades = study.runtime.ledger(prices, inputs['picks'], targets)
        old.write(tmp_path / 'nav' / f'{arm}.json', dict(nav=nav.to_dict(), trades=trades))
        scores['paths'][arm] = dict(return_rate_pct=100 * (nav.iloc[-1] - 1))
    old.write(tmp_path / 'scores.json', scores)
    old.write(tmp_path / 'inference-receipt.json', dict(decisions=receipts,
        protocol_sha256=old.sha(tmp_path / 'protocol.json')))
    report = audit(tmp_path)
    assert report['passed']
    assert report['results']['raw']['daily_valuations'] == 7
    path = tmp_path / 'nav' / 'library.json'
    value = json.loads(path.read_text())
    value['nav'][dates[3]] += .001
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        audit(tmp_path)
