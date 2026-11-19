"""P-confirm arms: what each arm adds, the routing call, retrieval, feedback and the runner."""
import pytest

from fin_skills.algorithms import routing
from fin_skills.rsi import pconfirm, pfix
from fin_skills.rsi.run_record import digest
from fin_skills.tasks import frozen_suite

TASKS = frozen_suite()
HELDOUT = TASKS.heldout
TASK = next(t for t in HELDOUT if t.id == "rsi_reversion_b")


def _program(task, tail="", parameters=True):
    args = "".join(f", {k}={v!r}" for k, v in task.parameters.items()) if parameters else ""
    return ("from fin_skills.algorithms import positions\n"
            "def solve(prices, split_factors):\n"
            f"    return positions('{task.algorithm}', prices, split_factors{args}){tail}\n")


def test_arms_differ_only_by_the_hint():
    f0 = pfix.pfix_messages(TASK, seed=101, arm="F0")
    f1 = pfix.pfix_messages(TASK, seed=101, arm="F1")
    hint = "\n\n" + routing.hint(TASK.algorithm)
    assert pconfirm.confirm_messages(TASK, seed=101, arm="C0") == f0
    assert pconfirm.confirm_messages(TASK, seed=101, arm="C1") == f1
    for arm in ("C2", "C3", "C4"):
        m = pconfirm.confirm_messages(TASK, seed=101, arm=arm, named=TASK.algorithm)
        assert m[0] == f1[0] and m[1]["content"] == f1[1]["content"] + hint
        assert pconfirm.confirm_messages(TASK, seed=101, arm=arm) == f1    # no hint: C1's prompt
    assert "lookback=..., lower=..., upper=..." in hint and str(TASK.parameters["lookback"]) not in hint
    for bad in (dict(arm="C1", named="macd"), dict(arm="C2", named="rsi"), dict(arm="C9")):
        with pytest.raises(ValueError):
            pconfirm.confirm_messages(TASK, seed=101, **bad)


def test_library_condition_is_fixed_per_arm_and_judged_with_six_checks():
    for cls in pconfirm.METHODS["check"] + pconfirm.METHODS["cause"]:
        library = cls.confirm_arm != "C0"
        arm = cls(library=library)
        assert len(arm.make_judge(TASK, seed=101, meter=None).check(
            "def solve(p, f):\n    return p * 0.0\n").checks) == 6
        with pytest.raises(ValueError):
            cls(library=not library)
        with pytest.raises(ValueError):
            cls(library=library, cards=True)
    assert [c.name for c in pconfirm.METHODS["cause"]][2] == "confirm_C2_cause"


def test_upper_bound_and_retrieval_name_an_id_without_the_model():
    c3, c4 = pconfirm.C3(library=True), pconfirm.C4(library=True)
    assert c3.named(TASK, 101) == TASK.algorithm
    assert c4.named(TASK, 101) == pconfirm.retrieved_id(TASK, seed=101) in routing.RULES
    right = sum(pconfirm.retrieved_id(t, seed=101) == t.algorithm for t in TASKS.tasks)
    assert right == 16                      # on the earlier tasks; the documents were chosen there

    class Other:                            # a rule positions() does not have
        algorithm, id = "wilder_rsi", "wilder_rsi_a"
    assert not pconfirm.catalogued(Other) and c3.named(Other, 101) is None


def _run(method, route_reply, program, attempts=3):
    from fin_skills import rsi
    from fin_skills.rsi.meter import Budget
    seen = []

    def generate(messages):
        seen.append(messages)
        if messages[0]["content"] == routing.ROUTE_SYSTEM:
            return route_reply
        return f"```python\n{program}```"
    report = rsi.benchmark((method,), generate, TASKS, seeds=(101,), library=(True,),
                           budget=Budget(attempts=attempts), heldout=[TASK])
    return report.rows[0], seen


def test_c2_routes_then_hints_and_records_the_route():
    row, seen = _run(pconfirm.C2, TASK.algorithm, _program(TASK))
    assert seen[0] == pconfirm.route_messages(TASK, seed=101)
    assert seen[1] == pconfirm.confirm_messages(TASK, seed=101, arm="C3", named=TASK.algorithm)
    assert row["method"] == "confirm_C2" and row["extra"]["routed"] == TASK.algorithm
    assert row["valid"] and row["usage"]["roles"] == ["router", "writer"]
    row, seen = _run(pconfirm.C2, "NONE", _program(TASK))
    assert seen[1] == pconfirm.confirm_messages(TASK, seed=101, arm="C1")
    assert row["extra"]["routed"] is None and row["valid"]
    row, seen = _run(pconfirm.C3, "unused", _program(TASK))
    assert row["extra"] == dict(routed=TASK.algorithm) and row["usage"]["roles"] == ["writer"]
    # window= is read as the rule's lookback, so this common call is now right as well.
    by_window = _program(TASK, parameters=False).replace(
        "split_factors)", "split_factors, window=%d, lower=%r, upper=%r)" % (
            TASK.parameters["lookback"], TASK.parameters["lower"], TASK.parameters["upper"]))
    assert _run(pconfirm.C3, "unused", by_window)[0]["valid"]


def test_cause_feedback_adds_the_cause_of_each_failed_check():
    twice = _program(TASK, " / prices.shape[1]")
    plain, _ = _run(pconfirm.C3, "unused", twice, attempts=2)
    cause, _ = _run(pconfirm.C3Cause, "unused", twice, attempts=2)
    assert plain["steps"][0]["feedback"].startswith("Failed checks: position_scale.\nReturn")
    text = cause["steps"][0]["feedback"]
    assert text.startswith("Failed checks: position_scale.\nCheck position_scale failed: likely")
    assert "divided by the number of assets twice" in text and text.endswith(
        "Return the complete corrected program in one python fence.")
    assert cause["method"] == "confirm_C3_cause" and not cause["valid"]


def test_manifest_and_runner_units():
    from scripts.run_p_confirm import DEV_UNITS, check_first_prompts, stage_design
    methods = DEV_UNITS["dev_hint"][0]
    m = pconfirm.manifest(HELDOUT, methods, seeds=(101, 211, 307), stage="dev")
    assert m["version"] == "p-confirm-v1" and m["exploratory"] and len(m["packets"]) == 36
    assert {p["messages_sha256"] for p in m["packets"]} == {
        digest(pconfirm.route_messages(t, seed=s)) for t in HELDOUT for s in (101, 211, 307)}
    assert m["hint"] == routing.HINT and len(m["visible_checks"]) == 6
    assert check_first_prompts(m, methods, [p["messages_sha256"] for p in m["packets"]]) == 36
    with pytest.raises(ValueError, match="never reached the model"):
        check_first_prompts(m, methods, [])
    tasks, heldout, seeds, _, units = stage_design("dev", False)
    assert [t.id for t in heldout] == [t.id for t in HELDOUT] and seeds == (101, 211, 307)
    assert len(stage_design("dev", True)[1]) == 1
    with pytest.raises(NotImplementedError):
        stage_design("main", False)
    main = pconfirm.manifest(HELDOUT, pconfirm.METHODS["check"], seeds=(1, 2))
    assert not main["exploratory"] and len(main["packets"]) == 5 * 6 * 2
