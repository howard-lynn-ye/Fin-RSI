"""Fin-RSI dev protocol: scorer accounting, dev22 rebuild, and a stored full-cycle path.

The scorer must price exactly like the frozen trading_study ledger (weights execute at the
close of pick+1 after that session's return, 5 bps per side, cash earns 0, a missing
submission keeps drifted holdings at no cost). The slow tests need the Beacon result files.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.agent_study import fin_rsi_score as score
from benchmarks.agent_study import trading_study as ledger

DATA = Path(os.environ.get('FIN_RSI_DATA_ROOT', '/beacon-projects/traumallm/shwaihe/fin-skills'))
DEV22 = DATA / 'results/fin_rsi/devsets/20261018-fin-dev22-v1'
I20 = DATA / 'results/fin_rsi/runs/20261019-i20-full-root'


def synthetic(n=80, seed=0):
    rng = np.random.default_rng(seed)
    tickers = list(ledger.TICKERS)
    days = pd.date_range('2025-01-01', periods=n, freq='B').strftime('%Y-%m-%d')
    total = pd.DataFrame(np.cumprod(1 + rng.normal(0, .01, (n, len(tickers))), axis=0),
                         index=days, columns=tickers)
    picks = list(range(0, 70, 10))
    targets = [None if i == 3 else dict(zip(tickers, rng.dirichlet(np.ones(len(tickers))) * (.9 if i % 2 else 1)))
               for i in range(len(picks))]
    return total, picks, targets


def test_path_equals_frozen_ledger():
    total, picks, targets = synthetic()
    nav, periods = score.path(total, picks, targets)
    reference, trades = ledger.ledger(total, picks, targets)
    assert abs(nav.iloc[-1] - reference.iloc[-1]) < 1e-12
    assert all(abs(p['turnover'] - t['turnover']) < 1e-12 for p, t in zip(periods, trades))
    common = reference.index.intersection(nav.index)
    assert np.allclose(nav.loc[common].values, reference.loc[common].values, atol=1e-12)


def test_metrics_equal_frozen_metrics():
    total, picks, targets = synthetic(seed=1)
    nav, _ = score.path(total, picks, targets)
    reference, _ = ledger.ledger(total, picks, targets)
    ours = score.metrics(nav.loc[reference.index])
    frozen = ledger.metrics(reference)
    assert all(abs(ours[k] - frozen[k]) < 1e-12 for k in ('cumulative_return', 'sharpe', 'max_drawdown'))


def test_missing_submission_keeps_drifted_holdings_at_no_cost():
    total, picks, _ = synthetic(seed=2)
    held = {t: 1 / len(total.columns) for t in total.columns}
    p = score.period(total, picks[0], picks[1], held, None)
    assert not p['traded'] and p['cost'] == 0 and p['turnover'] == 0
    q = score.period(total, picks[0], picks[1], held, dict(held))
    assert q['traded'] and q['cost'] == q['turnover'] * score.COST_BPS / 1e4


@pytest.mark.slow
def test_dev22_rebuild_matches_stored_devset(tmp_path):
    from benchmarks.agent_study import fin_rsi_devset as devset
    source = DATA / devset.SOURCE.relative_to(devset.REPO)
    if not (DEV22 / 'rows-v1.json').exists() or not source.exists():
        pytest.skip('needs the Beacon Fin-RSI result files')
    out = tmp_path / 'dev22'
    devset.build(out, source=source)
    stored = (DEV22 / 'rows-v1.json').read_text().replace(str(DEV22), '<ROOT>')
    rebuilt = (out / 'rows-v1.json').read_text().replace(str(out), '<ROOT>')
    assert json.loads(rebuilt) == json.loads(stored)

    def hashes(root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob('*')) if p.is_file()}
    assert hashes(out / 'pit-workspaces') == hashes(DEV22 / 'pit-workspaces')


@pytest.mark.slow
def test_scorer_reproduces_i20_full_cycle_returns():
    from benchmarks.agent_study import market_data as md
    if not (I20 / 'scores.json').exists():
        pytest.skip('needs the Beacon Fin-RSI result files')
    total = score.load_total(I20 / 'data' / md.HIDDEN)
    stored = json.loads((I20 / 'scores.json').read_text())['paths']
    for arm in ('library', 'raw'):
        out = score.score_decision_dir(total, I20 / 'decisions' / arm)
        assert abs(out['net_return_pct'] - stored[arm]['return_rate_pct']) < 1e-9
        assert out['holdings_before_max_abs_diff'] < 1e-12
