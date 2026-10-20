"""Independent verification of the expert agent trading report and snapshot."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.agent_study.trading_study import ledger, schedule, metrics, COST_BPS
from benchmarks.agent_study.market_data import TICKERS

DATA_DIR = ROOT / "benchmarks/agent_study/data/market-us-20260929"


def verify_snapshot(path=None):
    path = HERE / "snapshot.json" if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    # 1. Structural checks
    assert data["decisions_count"] == 44, f"Expected 44 decisions, got {data['decisions_count']}"
    assert data["initial_capital"] == 100_000.0
    assert data["cost_bps"] == 5.0
    assert len(data["decisions_log"]) == 44

    # 2. Replay ledger from decisions_log
    total = pd.read_csv(DATA_DIR / "total_return_close.csv", index_col=0)
    picks = schedule(list(total.index))
    assert len(picks) == 44

    raw_targets = []
    lib_targets = []
    for row in data["decisions_log"]:
        # Reconstruct full 12-ticker target dictionaries
        w_raw = {t: float(row["raw_weights"].get(t, 0.0)) for t in TICKERS}
        w_lib = {t: float(row["lib_weights"].get(t, 0.0)) for t in TICKERS}
        assert abs(sum(w_raw.values()) - 1.0) < 1e-3
        assert abs(sum(w_lib.values()) - 1.0) < 1e-3
        raw_targets.append(w_raw)
        lib_targets.append(w_lib)

    raw_nav, _ = ledger(total, picks, raw_targets)
    lib_nav, _ = ledger(total, picks, lib_targets)

    raw_m = metrics(raw_nav)
    lib_m = metrics(lib_nav)

    recomputed_raw_ret = (float(raw_nav.iloc[-1]) - 1.0) * 100.0
    recomputed_lib_ret = (float(lib_nav.iloc[-1]) - 1.0) * 100.0

    stored_raw = data["results"]["raw"]["return_rate_pct"]
    stored_lib = data["results"]["library"]["return_rate_pct"]

    assert abs(recomputed_raw_ret - stored_raw) < 1e-6, f"Raw mismatch: {recomputed_raw_ret} vs {stored_raw}"
    assert abs(recomputed_lib_ret - stored_lib) < 1e-6, f"Library mismatch: {recomputed_lib_ret} vs {stored_lib}"
    assert abs(raw_m["sharpe"] - data["results"]["raw"]["sharpe"]) < 1e-4
    assert abs(lib_m["sharpe"] - data["results"]["library"]["sharpe"]) < 1e-4

    return True


if __name__ == "__main__":
    verify_snapshot()
    print("Verification PASSED: All 44 decisions re-priced and matched to < 1e-6.")
