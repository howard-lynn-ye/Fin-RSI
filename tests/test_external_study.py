"""Synthetic external task integration: a wrong finite answer is accepted but not correct."""
import json

import pytest

from benchmarks.agent_study import external_study as study
from benchmarks.agent_study.external_packets import ScalarPolicy, packets


def tasks():
    row = dict(task="CodeFinQA", question="What percentage of 80 is 20?",
               context="part=20 total=80", context_type="text", options=[], answer="25")
    policy = ScalarPolicy("Use the percentage number.", 0, 0, "synthetic exact convention")
    public, private = packets(row, revision="a"*40, split="train", row_index=0,
                              source_cluster="document-A", policy=policy)
    return [study.ScalarTask(public, private)]


def test_external_wrong_numeric_answer_never_provides_feedback_or_extra_attempt(tmp_path, monkeypatch):
    requests = []
    def backend(messages):
        requests.append(messages)
        return "```python\n20 / 80\n```"
    monkeypatch.setattr(study, "evaluate_scalar", lambda *a, **k: dict(executed=True, value=.25))
    report = study.run(backend, tasks(), out=tmp_path / "external.jsonl", model_id="fixture",
                       seeds=(11,), execution_mode="trusted-fixtures")
    assert len(requests) == 2 and len(report.rows) == 2
    assert all(r["accepted"] and not r["valid"] and r["usage"]["attempts"] == 1 for r in report.rows)
    assert all("25" not in json.dumps(packet) for packet in requests)
    assert report.protocol["study_manifest"]["role"] == "development"
    assert report.protocol["study_manifest"]["primary"] is False
    from scripts.summarize_rsi_qualification import checked_report, describe
    verified = checked_report(tmp_path / "external.jsonl", expected=2)
    counts = describe(verified)
    assert counts["episodes"] == 2 and counts["calls"] == 2
    assert all(arm["accepted_but_wrong"] == 1 and arm["first_valid"] == 0
               for arm in counts["arms"].values())
    with pytest.raises(ValueError, match="disagree"):
        checked_report(tmp_path / "external.jsonl", expected=3)
    with (tmp_path / "external.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hashes differ"):
        checked_report(tmp_path / "external.jsonl", expected=2)


def test_external_reference_is_bound_and_primary_rejects_relaxed_tolerance(tmp_path):
    task = tasks()[0]
    task.scoring["public_sha256"] = "b"*64
    with pytest.raises(ValueError, match="different public"):
        study.ScalarTask(task.public, task.scoring)
    task = tasks()[0]
    task.scoring["scoring_policy"]["relative_tolerance"] = .1
    with pytest.raises(ValueError, match="one frozen policy"):
        study.run(lambda _: "unused", [task], out=tmp_path / "never.jsonl", model_id="fixture", seeds=(11,))
    assert not (tmp_path / "never.jsonl").exists()
    exact, bizbench = tasks()[0], tasks()[0]
    bizbench.scoring["scoring_policy"]["relative_tolerance"] = .01
    with pytest.raises(ValueError, match="one frozen policy"):     # one policy per run
        study.run(lambda _: "unused", [exact, bizbench], out=tmp_path / "mixed.jsonl",
                  model_id="fixture", seeds=(11,))
    assert not (tmp_path / "mixed.jsonl").exists()


def test_external_bizbench_one_percent_policy_grades_near_answers(tmp_path, monkeypatch):
    """BizBench's own criterion (within 1%) accepts 24.9 for a reference of 25; exact does not."""
    monkeypatch.setattr(study, "evaluate_scalar", lambda *a, **k: dict(executed=True, value=24.9))
    for tolerance, valid in ((0, False), (.01, True)):
        task = tasks()[0]
        task.scoring["scoring_policy"]["relative_tolerance"] = tolerance
        task = study.ScalarTask(task.public, task.scoring)
        report = study.run(lambda _: "```python\n24.9\n```", [task],
                           out=tmp_path / f"policy-{tolerance}.jsonl", model_id="fixture",
                           seeds=(11,), execution_mode="trusted-fixtures")
        assert all(row["valid"] is valid for row in report.rows)
        endpoint = report.protocol["study_manifest"]["endpoint"]
        assert ("within 1%" in endpoint) is (tolerance == .01)


def test_external_real_scalar_transport_and_post_selection_grade(tmp_path, monkeypatch):
    from benchmarks.agent_study.linux_sandbox import confinement_available
    if not confinement_available():
        pytest.skip("production-style scalar process integration needs Linux confinement")
    monkeypatch.setenv("FIN_TASKS_REQUIRE_CONFINEMENT", "1")
    monkeypatch.setenv("FIN_TASKS_CONFINE", "benchmarks.agent_study.linux_sandbox:confine")
    report = study.run(lambda _: "```python\n100 * 20 / 80\n```", tasks(),
                       out=tmp_path / "external-real.jsonl", model_id="synthetic-backend",
                       seeds=(11,), execution_mode="trusted-fixtures")
    assert report.complete and len(report.rows) == 2
    assert all(row["valid"] and row["first_valid"] for row in report.rows)
    assert report.contrasts(draws=40)[0]["effect"] == 0


def test_interface_lists_only_algorithms_that_run_here():
    from fin_skills import algorithms
    status = {c["id"]: c["status"] for c in algorithms.catalog()}
    listed = {c["id"] for c in study.runnable()}
    assert listed and all(status[i] == "ready" for i in listed)
    text = study.interface_text()
    for entry in algorithms.catalog():
        if entry["status"] != "ready":
            assert f'"id": "{entry["id"]}"' not in text


def test_reply_format_is_recorded_per_step(tmp_path, monkeypatch):
    replies = iter(["The answer is 25 percent.", "```Python\n20 / 80\n```"])
    monkeypatch.setattr(study, "evaluate_scalar", lambda *a, **k: dict(executed=True, value=.25))
    report = study.run(lambda messages: next(replies), tasks(), out=tmp_path / "e.jsonl",
                       model_id="fixture", seeds=(11,), execution_mode="trusted-fixtures")
    formats = sorted(step["reply_format"] for row in report.rows for step in row["steps"])
    assert formats == ["fenced", "no_fence"]
