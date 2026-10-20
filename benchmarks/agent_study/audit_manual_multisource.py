"""Independent dollar/units accounting after all personal decisions are scored.

This audit never selects, changes or replays a decision for a different outcome.
It does not call the study's ledger or drift implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def replay(prices, execution_targets, *, cost_bps=5):
    """Rebalance synthetic total-return units from net capital at execution close."""
    units = pd.Series(0., index=prices.columns)
    cash, fees = 1., 0.
    pnl = pd.Series(0., index=prices.columns)
    navs, equity_weights, trades = {}, [], []
    equity = [t for t in ("SPY", "QQQ", "EFA", "EEM", "NVDA", "AVGO", "ORLY", "FAST")
              if t in prices]
    previous = prices.iloc[0]
    for day, price in prices.iterrows():
        previous_nav = cash + float((units * previous).sum())
        equity_weights.append(float((units * previous).loc[equity].sum()) / previous_nav)
        pnl += units * (price - previous)
        values = units * price
        capital = cash + float(values.sum())
        target = execution_targets.get(day)
        if target is not None:
            w = pd.Series(target, index=prices.columns, dtype=float)
            turnover = float((w - values / capital).abs().sum())
            fee = capital * turnover * cost_bps / 10_000
            fees += fee
            capital -= fee
            units = capital * w / price
            cash = capital * (1 - float(w.sum()))
            trades.append(dict(date=day, fee_initial_capital_units=fee, turnover=turnover))
        navs[day] = capital
        previous = price
    nav = pd.Series(navs)
    np.testing.assert_allclose(pnl.sum() - fees, nav.iloc[-1] - 1, atol=1e-12, rtol=0)
    return nav, dict(
        return_rate_pct=100 * (float(nav.iloc[-1]) - 1),
        fee_drag_initial_capital_pp=100 * fees,
        pnl_initial_capital_pp={t: 100 * float(v) for t, v in pnl.items()},
        mean_start_of_day_equity_weight_pct=100 * float(np.mean(equity_weights)),
        rebalances=len(trades), trades=trades,
    )


def audit(root, data_dir):
    root, data_dir = Path(root), Path(data_dir)
    protocol = json.loads((root / "protocol.json").read_text())
    results = json.loads((root / "results.json").read_text())  # refuses unscored paths
    days, tickers = protocol["dates"], protocol["universe"]
    expected = {f"{i:02d}-{arm}.json" for i in range(len(days)) for arm in ("raw", "library")}
    assert expected == {p.name for p in (root / "decisions").glob("*.json")}
    hidden = data_dir / "total_return_close.csv"
    assert hashlib.sha256(hidden.read_bytes()).hexdigest() == protocol["market_hashes"][hidden.name]
    prices = pd.read_csv(hidden, index_col=0)[tickers]
    assert str(prices.index[-1]) == protocol["valuation_end"]
    executions = [prices.index[prices.index.get_loc(day) + 1] for day in days]
    prices = prices.loc[executions[0]:]
    out = {}
    for arm in ("raw", "library"):
        targets = {}
        for i, execution in enumerate(executions):
            row = json.loads((root / "decisions" / f"{i:02d}-{arm}.json").read_text())
            assert (row["index"], row["date"], row["arm"]) == (i, days[i], arm)
            targets[execution] = row["weights"]
        nav, detail = replay(prices, targets, cost_bps=protocol["cost_bps_per_side"])
        reference = pd.read_csv(root / f"nav-{arm}.csv", index_col=0).iloc[:, 0]
        assert list(nav.index) == list(reference.index)
        np.testing.assert_allclose(nav, reference, rtol=0, atol=1e-12)
        np.testing.assert_allclose(detail["return_rate_pct"], results[arm]["return_rate_pct"],
                                   rtol=0, atol=1e-10)
        detail["max_daily_nav_difference"] = float((nav - reference).abs().max())
        detail["daily_valuations"] = len(nav)
        out[arm] = detail
    out["difference_percentage_points"] = out["library"]["return_rate_pct"] - out["raw"]["return_rate_pct"]
    out["attribution_note"] = "Dollar P&L identity, not a causal allocation-effect decomposition."
    out["decision_files"] = len(expected)
    with (root / "independent-audit.json").open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps({k: ({a: b for a, b in v.items() if a != "trades"}
                         if isinstance(v, dict) else v) for k, v in out.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    audit(args.root, args.data_dir)
