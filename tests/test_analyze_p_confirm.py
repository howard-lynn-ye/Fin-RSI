"""P-confirm analysis on synthetic runs: rates, intervals, primary test, routing."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fin_skills.rsi import pconfirm
from fin_skills.tasks.confirm import CONFIRM_SEEDS, confirm_tasks

spec = importlib.util.spec_from_file_location(
    "analyze_p_confirm", Path(__file__).resolve().parents[1] / "scripts" / "analyze_p_confirm.py")
ap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap)
TASKS = confirm_tasks()


def row(arm, task, seed, valid, routed="unset", code="x = 1", checks=None, accepted=None):
    step = dict(accepted=valid if accepted is None else accepted, checks=checks or {}, code=code,
                error="", executed=True, feedback="", library_calls=0)
    r = dict(method=f"confirm_{arm}_cause", task_algorithm=task.algorithm, task_id=task.id,
             seed=seed, valid=valid, accepted=valid if accepted is None else accepted, steps=[step])
    if routed != "unset":
        r["extra"] = dict(routed=routed)
    return r


def write_model(root, model, rate, route="right", job=7):
    """rate: arm -> probability-like pattern (valid on the first k of every 4 catalogued tasks)."""
    units = {"c0": ["C0"], "route": ["C2"], "lib": ["C1", "C3", "C4"]}
    for unit, arms in units.items():
        rows = []
        for arm in arms:
            for k, task in enumerate(TASKS):
                has = pconfirm.catalogued(task)
                for seed in CONFIRM_SEEDS:
                    valid = (k % 4) < rate[arm] if has else (k % 4) < rate.get("other", 1)
                    routed = "unset"
                    if arm == "C2":
                        routed = (task.algorithm if has else None) if route == "right" else "macd"
                    elif arm == "C3":
                        routed = task.algorithm if has else None
                    elif arm == "C4":
                        routed = pconfirm.retrieved_id(task, seed=seed)
                    rows.append(row(arm, task, seed, bool(valid), routed))
        d = root / model / unit / f"job-{job}"
        d.mkdir(parents=True)
        (d / f"{unit}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        (d / f"{unit}.jsonl.terminal.json").write_text(json.dumps(dict(status="complete")))


def test_rates_contrasts_primary_and_routing(tmp_path):
    big = dict(C0=1, C1=2, C2=3, C3=4, C4=2)
    for m in pconfirm.PRIMARY_MODELS[:3]:
        write_model(tmp_path, m, big)
    write_model(tmp_path, "small-7b", dict(C0=0, C1=0, C2=0, C3=1, C4=0, other=0), route="wrong")
    (tmp_path / "half-7b" / "c0" / "job-1").mkdir(parents=True)      # no complete unit
    out = ap.analyze(tmp_path, draws=300)
    assert out["missing"] == ["half-7b"] and out["feedback"] == "cause" and out["seeds_per_task"] == 2
    assert len(out["primary_models"]) == 3 and len(out["tasks"]) == 24
    r = out["rates"]["primary|catalogued"]
    assert r["C0"]["n"] == 3 * 16 * 2 and r["C0"]["valid_pct"] == 25.0 and r["C2"]["valid_pct"] == 75.0
    assert r["C3"]["valid_pct"] == 100.0 and r["C3"]["ci95"] == [100.0, 100.0]
    c = {x["contrast"]: x for x in out["contrasts"]["primary|catalogued"]}
    assert c["C2-C0"]["points"] == 50.0 and c["C2-C0"]["models_better"] == 3
    lo, hi = c["C2-C0"]["ci95"]
    assert 0 < lo <= 50.0 <= hi <= 100
    p = out["primary"]
    assert p["supported"] and p["sign_flip"]["rules"] == 11
    assert p["sign_flip"]["p_one_sided"] <= p["sign_flip"]["p_two_sided"] <= 1
    assert out["no_harm"]["points"] == 0.0 and out["no_harm"]["within_margin"]
    assert out["routing"]["primary"]["all"]["right_pct"] == 100.0
    # "macd" for every task: right only for the macd task's two episodes
    assert out["routing"]["small"]["counts"] == dict(wrong=30, right=2, false_route=16)
    assert out["routing"]["small"]["all"]["right"] == 2
    assert out["retrieval"]["catalogued"] == 16 and 0 <= out["retrieval"]["right_catalogued"] <= 16
    assert out["per_model"]["small-7b"]["C3"]["valid"] == 8      # k % 4 == 0 among 16 x 2
    assert out["diagnostics"]["C0"]["episodes"] == 4 * 48


def test_sign_flip_is_exact_and_incomplete_cells_are_refused(tmp_path):
    s = ap.sign_flip([0.5] * 5)
    assert s["p_one_sided"] == 1 / 32 and s["p_two_sided"] == 2 / 32
    assert ap.sign_flip([0.5, -0.5, 0.25, -0.25])["p_two_sided"] == 1.0
    write_model(tmp_path, pconfirm.PRIMARY_MODELS[0], dict(C0=1, C1=1, C2=1, C3=1, C4=1))
    path = tmp_path / pconfirm.PRIMARY_MODELS[0] / "c0" / "job-7" / "c0.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]))                         # one episode missing
    with pytest.raises(ValueError, match="episodes per model and task"):
        ap.analyze(tmp_path, draws=50)
    assert ap.arm_of("confirm_C3") == "C3" and ap.arm_of("confirm_C3_cause") == "C3"
    with pytest.raises(ValueError):
        ap.arm_of("check_F2")


def test_benchmark_runs_the_confirmatory_tasks_end_to_end():
    """One catalogued and one other task through the real judge with a scripted model."""
    from fin_skills import rsi
    from fin_skills.algorithms import routing
    from fin_skills.rsi.meter import Budget
    from fin_skills.tasks.confirm import confirm_suite
    suite = confirm_suite()
    has = next(t for t in suite.heldout if t.id == "c_donchian_breakout_1")
    other = next(t for t in suite.heldout if t.id == "c_threshold_momentum_1")

    def generate(messages):
        if messages[0]["content"] == routing.ROUTE_SYSTEM:
            return "donchian_breakout" if "highest and the lowest" in messages[1]["content"] else "NONE"
        hinted = 'positions("donchian_breakout"' in messages[1]["content"]
        call = "positions('donchian_breakout', prices, split_factors, lookback=15)"
        zero = "prices * 0.0"
        return ("```python\nfrom fin_skills.algorithms import positions\n"
                f"def solve(prices, split_factors):\n    return {call if hinted else zero}\n```")
    report = rsi.benchmark((pconfirm.C2Cause,), generate, suite, seeds=CONFIRM_SEEDS[:1],
                           library=(True,), budget=Budget(attempts=1), heldout=[has, other])
    rows = {r["task_id"]: r for r in report.rows}
    assert rows[has.id]["valid"] and rows[has.id]["extra"]["routed"] == "donchian_breakout"
    assert rows[other.id]["extra"]["routed"] is None and not rows[other.id]["valid"]
    assert rows[other.id]["steps"][0]["checks"]["takes_positions"] is False
