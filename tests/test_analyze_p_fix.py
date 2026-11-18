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
    step = dict(accepted=valid, attempt_index=0, checks={}, code=code, error=error,
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
