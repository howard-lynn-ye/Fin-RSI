"""Run and output the paired 44-decision study: Control (Raw) vs. Treatment (Library).

Reproduces the exact results recorded in snapshot.json under the same protocol:
- Window: 2025-01-02 to 2026-09-22 (44 decision dates, held 10 sessions each)
- Initial Capital: USD 100,000.00
- Cost: 5.0 bps per traded side
- Ledger: Exact total-return closes with cash-and-drift accounting
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.agent_study.market_data import adjust, load_visible, TICKERS
from benchmarks.agent_study.trading_study import ledger, schedule, metrics, COST_BPS
import fin_skills.algorithms as algorithms

INITIAL_CAPITAL = 100_000.0
DATA_DIR = ROOT / "benchmarks/agent_study/data/market-us-20260929"


def run_paired_study():
    quotes, actions = load_visible(DATA_DIR)
    total = pd.read_csv(DATA_DIR / "total_return_close.csv", index_col=0)
    picks = schedule(list(total.index))

    raw_targets = []
    lib_targets = []
    decisions_log = []

    for i, p in enumerate(picks):
        dec_date = total.index[p]
        v_quotes = quotes[quotes["date"] <= dec_date]

        # 1. Control Group (Raw): Unassisted, uses raw unadjusted close prices
        raw_piv = v_quotes.pivot(index="date", columns="ticker", values="close")
        raw_mom = raw_piv.iloc[-1] / raw_piv.iloc[-60] - 1
        raw_top4 = raw_mom.nlargest(4).index.tolist()
        w_raw = {t: (0.25 if t in raw_top4 else 0.0) for t in TICKERS}
        raw_targets.append(w_raw)

        # 2. Treatment Group (Library): With fin_skills data adjustment + momentum filter + inverse vol
        v_actions = actions[actions["date"] <= dec_date]
        adj_prices = adjust(v_quotes, v_actions)
        adj_window = adj_prices.tail(253)
        rets = adj_window.pct_change().dropna()

        mom_126 = adj_window.iloc[-1] / adj_window.iloc[-126] - 1
        pos_mom = mom_126[mom_126 > 0]
        if len(pos_mom) >= 3:
            lib_candidates = pos_mom.nlargest(4).index.tolist()
        else:
            lib_candidates = mom_126.nlargest(4).index.tolist()

        sub_rets = rets[lib_candidates]
        w_algo = algorithms.run("inverse_volatility", {"asset_returns": sub_rets})
        w_lib = {t: float(w_algo.get(t, 0.0)) for t in TICKERS}
        lib_targets.append(w_lib)

        decisions_log.append({
            "decision_index": i + 1,
            "date": dec_date,
            "raw_top4": raw_top4,
            "raw_weights": {k: v for k, v in w_raw.items() if v > 0},
            "lib_candidates": lib_candidates,
            "lib_weights": {k: float(v) for k, v in w_lib.items() if v > 0}
        })

    raw_nav, raw_trades = ledger(total, picks, raw_targets)
    lib_nav, lib_trades = ledger(total, picks, lib_targets)

    raw_m = metrics(raw_nav)
    lib_m = metrics(lib_nav)

    raw_ending = INITIAL_CAPITAL * float(raw_nav.iloc[-1])
    lib_ending = INITIAL_CAPITAL * float(lib_nav.iloc[-1])
    raw_ret = (raw_ending / INITIAL_CAPITAL - 1) * 100.0
    lib_ret = (lib_ending / INITIAL_CAPITAL - 1) * 100.0

    raw_turnover = sum(t["turnover"] for t in raw_trades)
    lib_turnover = sum(t["turnover"] for t in lib_trades)

    raw_fees = sum(t["cost"] for t in raw_trades) * INITIAL_CAPITAL
    lib_fees = sum(t["cost"] for t in lib_trades) * INITIAL_CAPITAL

    return {
        "protocol": "terminal-return-v8-paired-simulation",
        "window": [total.index[picks[0]], total.index[picks[-1]]],
        "decisions_count": len(picks),
        "initial_capital": INITIAL_CAPITAL,
        "cost_bps": COST_BPS,
        "results": {
            "raw": {
                "ending_capital": raw_ending,
                "return_rate_pct": raw_ret,
                "annualized_return_pct": raw_m["annualized_return"] * 100,
                "annualized_volatility_pct": raw_m["annualized_volatility"] * 100,
                "sharpe": raw_m["sharpe"],
                "max_drawdown_pct": raw_m["max_drawdown"] * 100,
                "total_turnover": raw_turnover,
                "fees_paid_usd": raw_fees
            },
            "library": {
                "ending_capital": lib_ending,
                "return_rate_pct": lib_ret,
                "annualized_return_pct": lib_m["annualized_return"] * 100,
                "annualized_volatility_pct": lib_m["annualized_volatility"] * 100,
                "sharpe": lib_m["sharpe"],
                "max_drawdown_pct": lib_m["max_drawdown"] * 100,
                "total_turnover": lib_turnover,
                "fees_paid_usd": lib_fees
            },
            "difference": {
                "return_rate_diff_pp": lib_ret - raw_ret,
                "ending_capital_diff_usd": lib_ending - raw_ending,
                "sharpe_diff": lib_m["sharpe"] - raw_m["sharpe"],
                "max_drawdown_diff_pp": (lib_m["max_drawdown"] - raw_m["max_drawdown"]) * 100
            }
        },
        "decisions_log": decisions_log
    }


if __name__ == "__main__":
    res = run_paired_study()
    out = Path(__file__).resolve().parent / "snapshot.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Paired study completed. Wrote snapshot to {out}")
    print(f"Raw Return: {res['results']['raw']['return_rate_pct']:.2f}% | Library Return: {res['results']['library']['return_rate_pct']:.2f}%")
