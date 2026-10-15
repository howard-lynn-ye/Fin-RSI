"""RSI contract/regression tests. Synthetic evaluators, no model or financial claims."""
from dataclasses import replace
import hashlib
import json

import pytest

from fin_skills.rsi import (
    CaseResult, Entry, Evaluation, JSONProposer, Playbook, Policy, RSILoop, Trace,
)

PAST = "2025-01-01T00:00:00Z"
NOW = "2025-02-01T00:00:00Z"
FUTURE = "2025-03-01T00:00:00Z"
EVIDENCE = hashlib.sha256(b"synthetic execution evidence; not a return result").hexdigest()
CHANGE = dict(key="data-shape", surface="tool_usage", text="Check the table shape before covariance.",
              rationale="The development trace failed because a Series was used as a matrix.")


def trace(**kwargs):
    return replace(Trace("t1", "dev1", PAST, "tool_contract", "Series.cov missing other",
                         "fixture-v1", "series.cov()", "TypeError"), **kwargs)


def evaluator(book, split, cutoff):
    # Deliberately simple deterministic independent test scorer. Real callers must
    # execute their own tasks, retain raw logs and provide the corresponding digest.
    return Evaluation(book.version, split, "fixture-v1", (
        CaseResult("dev1" if split == "development" else "hidden1",
                   1.0 if book.entries else 0.0, PAST, EVIDENCE, 0, True),))


def make(tmp_path, **kwargs):
    defaults = dict(model_id="test-generator", protocol_id="fixture-v1", available_at=PAST,
                    policy=Policy(enabled=True), proposer=lambda ctx: json.dumps(CHANGE),
                    evaluator=evaluator)
    defaults.update(kwargs)
    return RSILoop(tmp_path / "archive", **defaults)


def observed(tmp_path, **kwargs):
    loop = make(tmp_path, **kwargs)
    loop.observe(trace())
    return loop


def test_disabled_is_inert(tmp_path):
    def forbidden(*args):
        pytest.fail("disabled module called a backend")
    loop = make(tmp_path, policy=Policy(), proposer=forbidden, evaluator=forbidden)
    loop.observe(trace())
    assert loop.improve(as_of=NOW).status == "disabled"
    assert loop.instructions(as_of=NOW) == ""
    assert loop.audit() == ()
    assert not (tmp_path / "archive").exists()


def test_accept_resume_rollback_freeze_and_audit(tmp_path):
    loop = observed(tmp_path)
    decision = loop.improve(as_of=NOW)
    assert decision.status == "accepted"
    assert CHANGE["text"] in make(tmp_path).instructions(as_of=NOW)
    events = loop.audit()
    assert sum(e["kind"] == "evaluation" for e in events) == 4
    assert next(e for e in events if e["kind"] == "proposal_response")["data"]["raw"]
    loop.rollback(Playbook().version, reason="operator reverts accepted context", as_of=NOW)
    assert loop.instructions(as_of=NOW) == ""
    loop.freeze(as_of=NOW)
    assert make(tmp_path).improve(as_of=FUTURE).status == "frozen"
    with pytest.raises(ValueError, match="frozen"):
        loop.observe(trace(trace_id="t2"))
    with pytest.raises(ValueError, match="unfrozen"):
        loop.rollback(decision.active_version, reason="cannot change frozen", as_of=NOW)


def test_only_development_available_information_reaches_generator(tmp_path):
    prompts = []
    def generate(prompt):
        prompts.append(prompt)
        return "```json\n" + json.dumps(CHANGE) + "\n```"
    loop = observed(tmp_path, proposer=JSONProposer(generate))
    loop.observe(trace(trace_id="later", available_at=FUTURE, summary="FUTURE_SECRET"))
    loop.observe(trace(trace_id="passed", kind="success", summary="PASSING_BEHAVIOR"))
    assert loop.improve(as_of=NOW).status == "accepted"
    assert "PASSING_BEHAVIOR" in prompts[0]
    assert "FUTURE_SECRET" not in prompts[0] and "hidden1" not in prompts[0]
    with pytest.raises(ValueError, match="development"):
        trace(split="validation")
    with pytest.raises(ValueError, match="cutoff"):
        loop.instructions(as_of=PAST)


@pytest.mark.parametrize("kind", ["environment", "financial_outcome", "success"])
def test_low_return_or_transport_failure_is_not_a_code_bug(tmp_path, kind):
    loop = make(tmp_path, proposer=lambda ctx: pytest.fail("no addressable failure"))
    loop.observe(trace(kind=kind))
    assert loop.improve(as_of=NOW).status == "no_evidence"


@pytest.mark.parametrize("change", [dict(CHANGE, surface="accounting"),
                                     dict(CHANGE, key="../guards.py"),
                                     dict(CHANGE, permissions="unlimited"),
                                     dict(CHANGE, text="x" * 1601)])
def test_reject_forbidden_or_oversized_edits_before_evaluation(tmp_path, change):
    loop = observed(tmp_path, proposer=lambda ctx: json.dumps(change),
                    evaluator=lambda *args: pytest.fail("forbidden edit evaluated"))
    with pytest.raises(ValueError):
        loop.improve(as_of=NOW)
    assert loop.instructions(as_of=NOW) == ""
    assert loop.audit()[-1]["kind"] == "attempt_error"


@pytest.mark.parametrize("failure", ["regression", "guards", "execution", "coverage", "overlap",
                                      "observed_validation", "flat"])
def test_gate_rejects_bad_candidates(tmp_path, failure):
    def measure(book, split, cutoff):
        report = evaluator(book, split, cutoff)
        case = report.cases[0]
        if failure == "flat":
            case = replace(case, score=0.)
        if failure == "overlap":
            case = replace(case, case_id="dev1")
        if failure == "observed_validation" and split == "validation":
            case = replace(case, case_id="dev-observed")
        if book.entries and split == "validation":
            if failure == "regression":
                case = replace(case, score=-1.)
            elif failure == "guards":
                case = replace(case, guards_passed=False)
            elif failure == "execution":
                case = replace(case, execution_errors=1)
            elif failure == "coverage":
                case = replace(case, case_id="changed-case")
        return replace(report, cases=(case,))
    loop = observed(tmp_path, evaluator=measure)
    loop.observe(trace(trace_id="extra", task_id="dev-observed"))
    decision = loop.improve(as_of=NOW)
    assert decision.status == "rejected"
    assert loop.instructions(as_of=NOW) == ""
    with pytest.raises(ValueError, match="accepted ancestor"):
        loop.rollback(decision.candidate_version, reason="cannot promote rejected", as_of=NOW)


@pytest.mark.parametrize("failure", ["future", "wrong_version", "wrong_protocol", "wrong_split"])
def test_unbound_evaluation_receipt_fails_closed(tmp_path, failure):
    def measure(book, split, cutoff):
        report = evaluator(book, split, cutoff)
        if failure == "future":
            return replace(report, cases=(replace(report.cases[0], available_at=FUTURE),))
        return replace(report, **{{"wrong_version": "version", "wrong_protocol": "protocol_id",
                                   "wrong_split": "split"}[failure]: "invalid"})
    loop = observed(tmp_path, evaluator=measure)
    with pytest.raises(ValueError, match="receipt"):
        loop.improve(as_of=NOW)
    assert loop.instructions(as_of=NOW) == ""
    assert sum(e["kind"] == "evaluation_started" for e in loop.audit()) == 1


def test_failed_calls_consume_persisted_budget_and_raw_response_is_retained(tmp_path):
    policy = Policy(enabled=True, max_proposals=1)
    loop = observed(tmp_path, policy=policy, proposer=lambda ctx: "not JSON")
    with pytest.raises(ValueError):
        loop.improve(as_of=NOW)
    assert make(tmp_path, policy=policy).improve(as_of=NOW).status == "budget_exhausted"
    assert any(e["data"].get("raw") == "not JSON" for e in loop.audit())


def test_evaluation_budget_checked_before_proposer(tmp_path):
    loop = observed(tmp_path, policy=Policy(enabled=True, max_evaluations=3),
                    proposer=lambda ctx: pytest.fail("budget already insufficient"))
    assert loop.improve(as_of=NOW).status == "budget_exhausted"


def test_noop_and_duplicate_candidate_skip_evaluation(tmp_path):
    loop = observed(tmp_path)
    assert loop.improve(as_of=NOW).status == "accepted"
    assert loop.improve(as_of=NOW).status == "rejected"
    assert sum(e["kind"] == "evaluation_started" for e in loop.audit()) == 4


def test_failed_future_attempt_cannot_influence_past_proposal(tmp_path):
    loop = observed(tmp_path, proposer=lambda ctx: "bad JSON")
    with pytest.raises(ValueError):
        loop.improve(as_of=FUTURE)
    with pytest.raises(ValueError, match="cutoff"):
        loop.improve(as_of=NOW)


@pytest.mark.parametrize("kwargs", [dict(model_id="different"), dict(protocol_id="new"),
                                    dict(policy=Policy(enabled=True, max_proposals=99))])
def test_resume_cannot_reset_model_protocol_or_budget(tmp_path, kwargs):
    observed(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        make(tmp_path, **kwargs).audit()


def test_journal_tampering_and_partial_writes_are_detected(tmp_path):
    loop = observed(tmp_path)
    path = tmp_path / "archive" / "events.jsonl"
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace("Series.cov", "forged.cov"), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        loop.audit()
    path.write_text(original + '{"partial":', encoding="utf-8")
    with pytest.raises(ValueError):
        loop.audit()


def test_second_writer_fails_without_changing_archive(tmp_path):
    loop = observed(tmp_path)
    lock = tmp_path / "archive" / ".writer.lock"
    lock.write_text("another writer", encoding="utf-8")
    with pytest.raises(FileExistsError):
        loop.improve(as_of=NOW)
    assert lock.read_text(encoding="utf-8") == "another writer"


def test_observe_is_idempotent_but_cannot_rewrite_evidence(tmp_path):
    loop = observed(tmp_path)
    loop.observe(trace())
    assert sum(e["kind"] == "trace" for e in loop.audit()) == 1
    with pytest.raises(ValueError, match="different evidence"):
        loop.observe(trace(summary="rewrite failure as success"))


@pytest.mark.parametrize("kwargs", [dict(score=float("nan")), dict(score=True),
                                    dict(guards_passed="yes"), dict(execution_errors=-1),
                                    dict(available_at="2025-01-01"), dict(evidence_sha256="")])
def test_invalid_case_evidence_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(CaseResult("case", 1., PAST, EVIDENCE, 0, True), **kwargs)


def test_update_preserves_other_entries_and_previous_attempts_are_visible(tmp_path):
    contexts = []
    seed = Playbook((Entry("existing", "retrieval", "Read the tool documentation."),))
    def propose(ctx):
        contexts.append(ctx)
        return json.dumps(CHANGE)
    def measure(book, split, cutoff):
        report = evaluator(book, split, cutoff)
        return replace(report, cases=(replace(report.cases[0], score=float(len(book.entries))),))
    loop = observed(tmp_path, seed=seed, proposer=propose, evaluator=measure)
    assert loop.improve(as_of=NOW).status == "accepted"
    assert seed.entries[0] in contexts[0].playbook.entries
    loop.improve(as_of=NOW)
    assert "Read the tool documentation." in loop.instructions(as_of=NOW)
    assert CHANGE["text"] in contexts[1].previous_attempts[0]


def test_missing_guard_measurement_is_not_assumed_to_pass():
    with pytest.raises(TypeError):
        CaseResult("case", 1., PAST, EVIDENCE)


def test_evaluator_exception_consumes_budget_after_restart(tmp_path):
    policy = Policy(enabled=True, max_evaluations=4)
    def broken(*args):
        raise RuntimeError("execution failed")
    loop = observed(tmp_path, policy=policy, evaluator=broken)
    with pytest.raises(RuntimeError):
        loop.improve(as_of=NOW)
    resumed = make(tmp_path, policy=policy, evaluator=broken)
    assert resumed.improve(as_of=NOW).status == "budget_exhausted"
    assert sum(e["kind"] == "proposal_started" for e in resumed.audit()) == 1


def test_two_successive_improvements_use_current_playbook_and_cached_baseline(tmp_path):
    seen = []
    def propose(ctx):
        seen.append(len(ctx.playbook.entries))
        return json.dumps(dict(CHANGE, key=f"lesson-{len(ctx.playbook.entries)}",
                               text=f"Distinct tool lesson {len(ctx.playbook.entries)}"))
    def measure(book, split, cutoff):
        report = evaluator(book, split, cutoff)
        return replace(report, cases=(replace(report.cases[0], score=float(len(book.entries))),))
    loop = observed(tmp_path, policy=Policy(enabled=True, max_evaluations=6),
                    proposer=propose, evaluator=measure)
    assert loop.improve(as_of=NOW).status == "accepted"
    assert loop.improve(as_of=NOW).status == "accepted"
    assert seen == [0, 1]
    assert sum(e["kind"] == "evaluation_started" for e in loop.audit()) == 6
    assert loop.improve(as_of=NOW).status == "budget_exhausted"


def test_duplicate_lesson_under_new_key_is_not_progress(tmp_path):
    def propose(ctx):
        return json.dumps(dict(CHANGE, key="same" if not ctx.playbook.entries else "new-key"))
    loop = observed(tmp_path, proposer=propose)
    assert loop.improve(as_of=NOW).status == "accepted"
    assert loop.improve(as_of=NOW).status == "rejected"
    assert sum(e["kind"] == "evaluation_started" for e in loop.audit()) == 4
