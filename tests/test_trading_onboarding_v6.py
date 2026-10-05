import hashlib
import json
import re

import pytest

import fin_skills
from benchmarks.agent_study import trading_onboarding_v6 as onboarding
from benchmarks.agent_study.trading_runtime_v6 import Controller, decide
from benchmarks.agent_study.trading_tools_v6 import Tools
from benchmarks.agent_study.trading_worker_v6 import run_python


def response(tool, arguments):
    return {"choices": [{"message": {"content": json.dumps(
        {"tool": tool, "arguments": arguments})}, "finish_reason": "stop"}]}


@pytest.mark.parametrize("arm", ["raw", "library"])
def test_actual_first_backend_request_contains_the_guide_only_in_library(arm, tmp_path):
    controller = Controller(tmp_path, tmp_path, arm)
    requests = []

    def backend(messages):
        requests.append([dict(m) for m in messages])
        return response("read_file", {"path": "README.md"}) if len(requests) == 1 else response(
            "submit", {"weights": {"SPY": 1}})

    (tmp_path / "README.md").write_text("Only visible input files.")
    result = decide(backend, controller, "Choose your own portfolio.")
    guide, manifest = onboarding.orientation(arm)
    assert result["initial_request"] == requests[0]
    assert result["library_orientation"] == manifest
    assert result["backend_responses"] == 2
    assert result["submitted"] and result["target"]["SPY"] == 1
    assert all(weight == 0 for ticker, weight in result["target"].items() if ticker != "SPY")
    assert all("Current call" in r[0]["content"] for r in requests)
    if arm == "library":
        assert all(r[0]["content"].count(guide) == 1 for r in requests)
        assert manifest["sha256"] == hashlib.sha256(guide.encode()).hexdigest()
    else:
        assert "# fin-skills: read before" not in requests[0][0]["content"]
    assert not any(c["tool"] == "read_skill" for c in result["tool_calls"])


def test_backend_failure_does_not_become_evidence_of_model_reading(tmp_path):
    def backend(messages):
        raise ValueError("context rejected")
    result = decide(backend, Controller(tmp_path, tmp_path, "library"), "task")
    assert result["initial_request"] and result["library_orientation"]
    assert result["backend_responses"] == 0 and not result["submitted"]
    assert result["tool_calls"] == []


def test_guide_references_real_skills_and_distinguishes_capability_from_input():
    guide, manifest = onboarding.orientation("library")
    rows = [line.split("|")[2] for line in guide.splitlines()
            if line.startswith("| ") and "`" in line]
    names = re.findall(r"`([^`]+)`", " ".join(rows))
    assert names and set(names) <= set(fin_skills.names())
    assert "network is disabled" in guide and "NOT supplied" in guide
    assert "synthetic" in guide and "public" in guide
    assert manifest["chars"] == len(guide)


@pytest.mark.parametrize("content", [None, "", "x" * 6501])
def test_missing_or_oversized_guide_stops_before_inference(tmp_path, monkeypatch, content):
    path = tmp_path / "guide.md"
    if content is not None:
        path.write_text(content)
    monkeypatch.setattr(onboarding, "GUIDE", path)
    def backend(messages):
        pytest.fail("inference must not start without the complete guide")
    with pytest.raises((OSError, ValueError)):
        decide(backend, Controller(tmp_path, tmp_path, "library"), "task")
    assert onboarding.orientation("raw") == ("", None)


def test_document_receipts_identify_pages_in_json_and_python_without_claiming_use(tmp_path):
    controller = Controller(tmp_path, tmp_path, "library")
    args = {"name": "congressional-trading-disclosures", "limit": 500}
    first = controller.call("read_skill", args)
    assert first["ok"] and first["next_offset"]
    receipt = controller.calls[-1]
    assert receipt["skill_name"] == args["name"] and receipt["offset"] == 0
    assert receipt["page_sha256"] == hashlib.sha256(first["text"].encode()).hexdigest()
    second = controller.call("read_skill", {**args, "offset": first["next_offset"]})
    assert second["text"] != first["text"]
    assert controller.calls[-1]["offset"] == first["next_offset"]
    tools = Tools(tmp_path, "library")
    result = run_python(f"page=read_skill(**{args!r})\nprint('retrieved')", tools)
    assert result["ok"] and result["output"].strip() == "retrieved"
    assert result["tool_calls"][-1]["evidence"].startswith("retrieval only")
    assert result["tool_calls"][-1]["returned_chars"] == len(first["text"])
    failed = controller.call("read_skill", {"name": "does-not-exist"})
    assert not failed["ok"] and "page_sha256" not in controller.calls[-1]
