"""P-fix analysis on synthetic runs: first complete run per unit, denominators, funnel, scan."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import analyze_p_fix as ap  # noqa: E402

TASKS = [("macd", "macd_a"), ("macd", "macd_b"), ("rsi_reversion", "rsi_reversion_a"),
         ("rsi_reversion", "rsi_reversion_b"), ("bollinger", "bollinger_a"),
         ("bollinger", "bollinger_b")]


def row(method, fam, task, seed, valid, calls=0, executed=True, code="x = 1", error="", accepted=None):
    step = dict(accepted=valid if accepted is None else accepted, attempt_index=0, checks={}, code=code, error=error,
                executed=executed, feedback="", library_calls=calls, number=1,
                reply_format="fenced")
    return dict(method=method, task_algorithm=fam, task_id=task, seed=seed, valid=valid,
                accepted=valid if accepted is None else accepted,
                first_valid=valid, library_used=calls > 0, steps=[step])


def write_unit(root, model, unit, job, rows, status="complete"):
    d = root / model / unit / f"job-{job}"
    d.mkdir(parents=True)
    (d / f"{unit}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (d / f"{unit}.jsonl.terminal.json").write_text(json.dumps(dict(status=status)))


def episodes(method, valid_every=2, **kw):
    return [row(method, f, t, s, i % valid_every == 0, **kw)
            for i, ((f, t), s) in enumerate((ft, s) for ft in TASKS for s in (101, 211, 307))]


def test_first_complete_run_contrasts_funnel_and_scan(tmp_path):
    write_unit(tmp_path, "m1", "check_F0", 5, episodes("check_F0", valid_every=3))
    write_unit(tmp_path, "m1", "check_F0", 3, episodes("check_F0", valid_every=1),
               status="interrupted")
    fix = episodes("check_F1", calls=1) + episodes("check_F2", valid_every=1, calls=1)
    fix[0]["steps"][0].update(executed=False, error="ValueError: unknown algorithm 'foo'")
    fix[1]["steps"][0]["code"] = "import subprocess\nopen('/etc/passwd')"
    write_unit(tmp_path, "m1", "check_fix", 7, fix)
    write_unit(tmp_path, "m1", "check_fix", 9, [])  # later complete run is ignored
    write_unit(tmp_path, "m2", "check_F0", 4, episodes("check_F0"))  # no check_fix: missing
    out = ap.analyze(tmp_path)
    assert out["missing"] == ["m2"] and out["runs"]["m1"] == {"check_F0": "job-5",
                                                               "check_fix": "job-7"}
    c = out["per_model"]["m1"]
    assert (c["F0"]["valid"], c["F1"]["valid"], c["F2"]["valid"]) == (6, 9, 18)
    contrasts = {x["contrast"]: x for x in out["contrasts"]}
    assert contrasts["F2-F0"]["pooled_pct_points"] == round(100 * 12 / 18, 1)
    assert contrasts["F1-F0"]["better"] == 1
    assert out["funnel"]["F1"]["last_calls_library"] == 18
    assert out["funnel"]["F1"]["last_calls_library_ran"] == 17
    assert out["library_errors"]["F1"] == {"algorithm id not in the catalogue": 1}
    assert out["infrastructure_suspect"] == []
    assert out["pooled"]["F0"]["accepted_wrong"] == 0 and out["pooled"]["F2"]["models_none_valid"] == 0
    assert out["programs"]["F2"].get("uses_imports_fin_skills", 0) == 0
    flagged = out["scan"]["flagged"]
    assert len(flagged) == 1 and set(flagged[0]["matched"]) == {"file", "process"}


def test_wrong_denominator_fails(tmp_path):
    write_unit(tmp_path, "m1", "check_F0", 5, episodes("check_F0")[:-1])
    write_unit(tmp_path, "m1", "check_fix", 7, episodes("check_F1") + episodes("check_F2"))
    with pytest.raises(ValueError, match="episodes per arm"):
        ap.analyze(tmp_path)


def test_runtime_file_errors_are_listed(tmp_path):
    f0 = episodes("check_F0")
    f0[3]["steps"][0].update(executed=False, error="FileNotFoundError: [Errno 2] No such file or "
                             "directory: '/scratch/zt1/x/venv/lib/python3.12/site-packages/jinja2/a.py'")
    write_unit(tmp_path, "m1", "check_F0", 5, f0)
    write_unit(tmp_path, "m1", "check_fix", 7, episodes("check_F1") + episodes("check_F2"))
    out = ap.analyze(tmp_path)
    assert [(x["task"], x["seed"]) for x in out["infrastructure_suspect"]] == [("macd_b", 101)]


def test_library_use_read_from_program_text(tmp_path):
    code = ("from fin_skills.algorithms import positions, signal_positions, adjust_prices\n"
            "def solve(prices, split_factors):\n"
            "    p = adjust_prices(prices, split_factors)\n"
            "    return signal_positions(p > p.rolling(5).mean())\n")
    write_unit(tmp_path, "m1", "check_F0", 5, episodes("check_F0"))
    write_unit(tmp_path, "m1", "check_fix", 7,
               episodes("check_F1", code=code) + episodes("check_F2", code="signal = 1"))
    out = ap.analyze(tmp_path)
    f1 = out["programs"]["F1"]
    assert f1["uses_imports_fin_skills"] == f1["uses_signal_positions"] == f1["uses_adjust_prices"] == 18
    assert f1["uses_positions"] == 0 and f1["last_uses_signal_positions_valid"] == 9
    assert out["scan"]["counts"] == {}


def test_check_fix_from_a_later_version(tmp_path):
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    write_unit(v1, "m1", "check_F0", 5, episodes("check_F0", valid_every=3))
    write_unit(v1, "m1", "check_fix", 7, episodes("check_F1") + episodes("check_F2"))
    write_unit(v2, "m1", "check_fix", 9, episodes("check_F1", valid_every=1) +
               episodes("check_F2", valid_every=1))
    out = ap.analyze(v1, fix_root=v2)
    assert out["runs"]["m1"] == {"check_F0": "job-5", "check_fix": "job-9"}
    assert out["per_model"]["m1"]["F1"]["valid"] == 18 and out["per_model"]["m1"]["F0"]["valid"] == 6


@pytest.mark.parametrize("error, group", [
    ("TypeError: positions() takes the id of a catalogued algorithm, not a function. ...",
     "own function passed as the algorithm"),
    ("TypeError: positions() has no states= argument: it computes x from prices.",
     "states passed to positions()"),
    ("TypeError: window=15 applies to portfolio algorithms only and would be ignored",
     "argument that does not apply to the algorithm"),
    ("ValueError: split_factors look like event ratios (1.0 except on split days)",
     "split factors in the other convention"),
    ("ValueError: bollinger_reversion requires >= 22 observations", "too few rows (called on a slice)"),
])
def test_error_groups(error, group):
    assert ap.error_group(error) == group


def test_rescaled_library_output_is_counted(tmp_path):
    code = ("from fin_skills.algorithms import positions\n"
            "def solve(prices, split_factors):\n"
            "    r = positions('macd', prices, split_factors)\n"
            "    return r / len(r.columns)\n")
    write_unit(tmp_path, "m1", "check_F0", 5, episodes("check_F0"))
    write_unit(tmp_path, "m1", "check_fix", 7, episodes("check_F1") +
               episodes("check_F2", valid_every=1000, calls=1, code=code, accepted=True))
    f = ap.analyze(tmp_path)["funnel"]["F2"]
    assert f["last_calls_library_accepted_wrong"] == 17 and f["last_calls_library_accepted_wrong_rescaled"] == 17


def test_scan_ignores_comments_and_scipy_signal(tmp_path):
    quiet = "from scipy import signal\n# prices = pd.read_csv('prices.csv')\nx = 1\n"
    loud = "import subprocess\nopen('/etc/passwd')\n"
    write_unit(tmp_path, "m1", "check_F0", 5, episodes("check_F0", code=quiet))
    write_unit(tmp_path, "m1", "check_fix", 7, episodes("check_F1", code=quiet) +
               episodes("check_F2", code=loud))
    out = ap.analyze(tmp_path)
    assert {f["arm"] for f in out["scan"]["flagged"]} == {"F2"}
    assert out["scan"]["counts"] == {"file": 18, "process": 18}


def test_scale_argument_is_not_counted_as_rescaling(tmp_path):
    code = ("from fin_skills.algorithms import positions\n"
            "def solve(prices, split_factors):\n"
            "    return positions('macd', prices, split_factors, scale=1/len(prices.columns))\n")
    write_unit(tmp_path, "m1", "check_F0", 5, episodes("check_F0"))
    write_unit(tmp_path, "m1", "check_fix", 7, episodes("check_F1") +
               episodes("check_F2", valid_every=1000, calls=1, code=code, accepted=True))
    f = ap.analyze(tmp_path)["funnel"]["F2"]
    assert f["last_calls_library_accepted_wrong"] == 17 and f["last_calls_library_accepted_wrong_rescaled"] == 0


def test_documented_arm_is_exploratory_and_optional(tmp_path):
    root, fix_root, doc_root = tmp_path / "v1", tmp_path / "v2", tmp_path / "v3"
    for m in ("qwen-a-32b", "small-7b"):
        write_unit(root, m, "check_F0", 5, episodes("check_F0", valid_every=3))
        write_unit(fix_root, m, "check_fix", 7,
                   episodes("check_F1", calls=1) + episodes("check_F2", valid_every=1, calls=1))
    write_unit(doc_root, "qwen-a-32b", "check_doc", 8, episodes("check_F3", valid_every=1, calls=1))
    write_unit(doc_root, "small-7b", "check_doc", 9, episodes("check_F3")[:5],
               status="interrupted")  # not complete: keeps F0-F2, no F3
    without = ap.analyze(root, fix_root=fix_root)
    out = ap.analyze(root, fix_root=fix_root, doc_root=doc_root)
    assert out["missing"] == [] and out["missing_doc"] == ["small-7b"]
    assert out["contrasts"] == without["contrasts"]  # declared contrasts do not move
    assert {a: out["pooled"][a] for a in without["pooled"]} == without["pooled"]
    assert out["pooled"]["F3"] == dict(n=18, valid=18, valid_pct=100.0, models=1, first_valid=18,
                                       accepted_wrong=0, models_none_valid=0)
    e = {x["contrast"]: x for x in out["exploratory_contrasts"]}
    assert e["F3-F1"]["models"] == 1 and e["F3-F1"]["per_model"] == {"qwen-a-32b": 9}
    assert e["F3-F2"]["same"] == 1 and e["F3-F0"]["pooled_pct_points"] == round(100 * 12 / 18, 1)
    assert out["funnel"]["F3"]["last_calls_library"] == 18
    assert "F3" not in without["pooled"] and without["exploratory_contrasts"] == []


def test_size_split(tmp_path):
    for m in ("qwen3-30b-a3b", "phi4-mini", "phi4-14b", "granite-2b", "odd"):
        write_unit(tmp_path, m, "check_F0", 5, episodes("check_F0", valid_every=3))
        write_unit(tmp_path, m, "check_fix", 7,
                   episodes("check_F1") + episodes("check_F2", valid_every=1))
    out = ap.analyze(tmp_path)
    assert out["sizes_b"] == {"granite-2b": 2.0, "odd": None, "phi4-14b": 14.0,
                              "phi4-mini": 3.8, "qwen3-30b-a3b": 30.0}
    assert out["unsized"] == ["odd"]
    big, small = out["by_size"]["ge14b"], out["by_size"]["lt14b"]
    assert big["models"] == ["phi4-14b", "qwen3-30b-a3b"]
    assert small["models"] == ["granite-2b", "phi4-mini"]
    assert big["pooled"]["F2"]["valid"] == 36 and small["pooled"]["F0"]["valid"] == 12
    assert {c["contrast"]: c["models"] for c in big["contrasts"]} == {
        "F1-F0": 2, "F2-F0": 2, "F2-F1": 2}


def test_routed_arm_and_failed_checks(tmp_path):
    """v4: every unit in one root; F4 rows carry what the router named."""
    for m in ("qwen-a-32b", "small-7b"):
        write_unit(tmp_path, m, "check_F0", 5, episodes("check_F0", valid_every=3))
        write_unit(tmp_path, m, "check_fix", 5,
                   episodes("check_F1", calls=1) + episodes("check_F2", valid_every=1, calls=1))
        write_unit(tmp_path, m, "check_doc", 5, episodes("check_F3", calls=1))
        routed = episodes("check_F4", valid_every=2, calls=1)
        for i, r in enumerate(routed):
            r["extra"] = dict(routed=[r["task_algorithm"], None, "macd"][i % 3], route_reply="x")
            r["steps"][0]["checks"] = dict(finite_output=True, position_scale=i % 2 == 0)
        write_unit(tmp_path, m, "check_route", 5, routed)
    out = ap.analyze(tmp_path, doc_root=tmp_path, route_root=tmp_path)
    assert out["missing_route"] == [] and out["pooled"]["F4"]["n"] == 36
    assert out["routing"]["right"] + out["routing"]["none"] + out["routing"]["wrong"] == 36
    assert out["routing"]["wrong"] == 8          # "macd" named for a macd task counts as right
    assert out["routing"]["right_valid"] <= out["routing"]["right"]
    e = {x["contrast"]: x for x in out["exploratory_contrasts"]}
    assert set(e) == {"F3-F1", "F3-F2", "F3-F0", "F4-F1", "F4-F2", "F4-F0"}
    assert e["F4-F0"]["models"] == 2
    assert out["failed_checks"]["F4"] == dict(programs_ran=36, finite_output=0, position_scale=18)
    assert out["by_size"]["ge14b"]["routing"]["right"] == out["routing"]["right"] // 2
    assert out["funnel"]["F4"]["episodes"] == 36


def test_scan_skips_text_that_is_not_a_program(tmp_path):
    """A reply with no code (a model's reasoning) cannot run; its words are not a finding."""
    rows = episodes("check_F0", valid_every=3)
    rows[0]["steps"][0].update(code="<think>The function should not make network requests or "
                                    "open a socket.\nSo, I'll loop over rows.", executed=False)
    rows[1]["steps"][0].update(code="import socket\ndef solve(prices, split_factors):\n    return prices * 0\n")
    write_unit(tmp_path, "m-7b", "check_F0", 5, rows)
    write_unit(tmp_path, "m-7b", "check_fix", 5, episodes("check_F1") + episodes("check_F2"))
    out = ap.analyze(tmp_path)
    assert out["scan"]["not_python"] == 1 and out["scan"]["counts"] == {"network": 1}
    assert [f["task"] for f in out["scan"]["flagged"]] == [rows[1]["task_id"]]


def test_told_the_rule_and_repair_after_a_property_check(tmp_path):
    call = "from fin_skills.algorithms import positions\ndef solve(p, f):\n    return positions('macd', p, f)\n"
    own = "def solve(p, f):\n    return p * 0\n"
    f0 = episodes("check_F0", valid_every=3)
    f0[0]["steps"][0]["checks"] = dict(finite_output=True, takes_positions=False)   # flagged, no attempt left
    second = dict(f0[1]["steps"][0], checks=dict(takes_positions=True))
    f0[1]["steps"] = [dict(f0[1]["steps"][0], checks=dict(position_scale=False)), second]
    f0[1]["valid"] = True                                                            # flagged, then valid
    f0[2]["steps"] = [dict(f0[2]["steps"][0], checks=dict(warmup_rows_zero=False)), dict(second)]
    f0[2]["valid"] = False                                                           # flagged, not repaired
    f2 = episodes("check_F2", valid_every=2, calls=1, code=call)
    f4 = episodes("check_F4", valid_every=2, code=own)
    for i, r in enumerate(f4):
        r["extra"] = dict(routed=r["task_algorithm"] if i < 12 else None)
        if i < 6:
            r["steps"][0]["code"] = call
    for m in ("big-32b",):
        write_unit(tmp_path, m, "check_F0", 5, f0)
        write_unit(tmp_path, m, "check_fix", 5, episodes("check_F1") + f2)
        write_unit(tmp_path, m, "check_route", 5, f4)
    out = ap.analyze(tmp_path, route_root=tmp_path)
    assert out["repair"]["F0"] == dict(episodes_flagged=3, with_another_attempt=2, then_valid=1)
    assert out["told"]["F2"]["episodes"] == 18 and out["told"]["F2"]["last_calls_positions"] == 18
    assert out["told"]["F2"]["last_calls_positions_valid"] == 9 == out["told"]["F2"]["valid"]
    t4 = out["told"]["F4"]                       # 12 right routes; 6 of those call positions()
    assert (t4["episodes"], t4["last_calls_positions"], t4["last_own_rule"]) == (12, 6, 6)
    assert t4["last_calls_positions_valid"] == 3 and t4["last_own_rule_valid"] == 3
    assert out["by_size"]["ge14b"]["told"]["F4"] == t4 and out["by_size"]["lt14b"]["told"]["F4"] == {}
    assert out["told_by_model"]["F4"]["big-32b"]["no_program_ran"] == 0
