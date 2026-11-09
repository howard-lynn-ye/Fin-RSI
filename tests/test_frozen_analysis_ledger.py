import json
from types import SimpleNamespace

import pytest

from scripts.analyze_frozen_study import attach_event_costs, numeric_pattern, relative_tolerance_pass


def test_numeric_patterns_are_diagnostics_and_never_regrade():
    row = dict(valid=False, steps=[dict(executed=True)])
    assert numeric_pattern(row, .25, "25") == "power_of_ten_-2"
    assert row["valid"] is False
    assert numeric_pattern(row, -25, "25") == "opposite_sign_exact"
    assert numeric_pattern(row, 26, "25") == "other_numeric_mismatch"
    assert numeric_pattern(dict(valid=False, steps=[]), None, "25") == "no_program"
    with pytest.raises(ValueError, match="original recorded value"):
        numeric_pattern(row, None, "25")


def test_derived_ledger_requires_every_charged_call_and_execution(tmp_path):
    path = tmp_path / "chunk.jsonl"
    row = dict(method="external_code", library=True, task_id="t", seed=101, valid=False,
               usage=dict(calls=1, generated_tokens=7), steps=[dict(executed=True)])
    events = [dict(kind="episode_started", **{k:row[k] for k in ("method", "library", "task_id", "seed")}),
              dict(kind="response_received", response=dict(usage=dict(prompt_tokens=100, completion_tokens=7))),
              dict(kind="program_completed", result=dict(value=.25))]
    event_path = tmp_path / "chunk.jsonl.events.jsonl"
    event_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    report = SimpleNamespace(rows=[row])
    attach_event_costs(report, path, {"t":"25"})
    assert row["numeric_error_pattern"] == "power_of_ten_-2"
    assert row["derived_prompt_tokens"] == 100
    assert not row["valid"]
    events[1]["response"]["usage"]["completion_tokens"] = 6
    event_path.write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="recorded episode costs"):
        attach_event_costs(report, path, {"t":"25"})


@pytest.mark.parametrize("value,reference,expected", [
    (101, "100", True), (101.00001, "100", False), (-101, "-100", True),
    (-98.999, "-100", False), (0, "0", True), (1e-300, "0", False),
    (.101, ".1", True), (.10100001, ".1", False),
    (10**80+10**78, str(10**80), True), (10**80+10**78+1, str(10**80), False),
    (float("inf"), "100", False), (float("nan"), "100", False),
])
def test_relative_sensitivity_exact_boundary_and_zero(value, reference, expected):
    row = dict(valid=False, steps=[dict(executed=True)])
    assert relative_tolerance_pass(row, value, reference) is expected
    assert row["valid"] is False


def test_relative_sensitivity_does_not_hide_missing_output_or_change_primary():
    row = dict(valid=True, steps=[dict(executed=True)])
    with pytest.raises(ValueError, match="exact primary"):
        relative_tolerance_pass(row, 150, "100")
    with pytest.raises(ValueError, match="original recorded value"):
        relative_tolerance_pass(row, None, "100")
    assert relative_tolerance_pass(dict(valid=False, steps=[]), None, "100") is False
