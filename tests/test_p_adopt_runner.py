"""P-adopt runner pieces that need no GPU: designs, resumption and prompt checks."""
import importlib.util
import json
from pathlib import Path

import pytest

from fin_skills.rsi import adopt
from fin_skills.rsi.meter import BudgetExceeded

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_p_adopt", ROOT / "scripts/run_p_adopt.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_units_partition_the_six_arms_once():
    names = [m.name for u in runner.UNITS.values() for m in u[0]]
    assert sorted(names) == sorted(m.name for m in adopt.SCRATCH + adopt.LIBRARY)
    parts = [u for k, units in runner.PARTS.items() if k != "all" for u in units]
    assert sorted(parts) == sorted(runner.UNITS) == sorted(runner.PARTS["all"])
    for methods, library in runner.UNITS.values():
        assert all((m.arm != "L0") is library for m in methods)


def test_main_design_is_the_declared_one_and_dev_never_touches_heldout():
    tasks, seeds, fit = runner.design(False)
    assert seeds == (101, 211, 307) and len(tasks.heldout) == 6 and len(tasks.dev) == 12
    assert vars(fit) == dict(episodes=48, proposals=6, calls=600, generated_tokens=1_200_000)
    dev, dev_seeds, _ = runner.design(True)
    heldout = {t.id for t in tasks.heldout}
    assert not heldout & {t.id for t in (*dev.dev, *dev.heldout)}
    assert not {t.algorithm for t in dev.dev} & {t.algorithm for t in dev.heldout}


def test_only_complete_terminal_receipts_count_as_done(tmp_path):
    unit = tmp_path / "check_L0"
    (unit / "job-1").mkdir(parents=True)
    (unit / "job-1" / "check_L0.jsonl.terminal.json").write_text(json.dumps({"status": "invalid"}))
    (unit / "job-2").mkdir()
    (unit / "job-2" / "check_L0.jsonl").write_text("")          # interrupted, no receipt
    assert runner.complete(unit) == []
    (unit / "job-3").mkdir()
    (unit / "job-3" / "check_L0.jsonl.terminal.json").write_text(json.dumps({"status": "complete"}))
    assert len(runner.complete(unit)) == 1


def test_first_prompt_check_names_missing_packets():
    tasks, seeds, _ = runner.design(False)
    manifest = adopt.manifest(tasks.heldout, seeds=seeds)
    methods, library = runner.UNITS["check_lib"]
    sent = [p["messages_sha256"] for p in manifest["packets"] if p["library"]]
    assert runner.check_first_prompts(manifest, methods, library, sent + ["x"]) == 54
    with pytest.raises(ValueError, match="1 declared first prompts"):
        runner.check_first_prompts(manifest, methods, library, sent[1:])
    assert runner.check_first_prompts(manifest, *runner.UNITS["playbook_L1"], []) == 0


def test_context_overflow_becomes_an_episode_budget_stop():
    from benchmarks.agent_study.adopt_backend import OVERFLOW, ContextAsBudget

    class Chat:
        def describe(self):
            return {}

        def generate_bounded(self, messages, *, max_new_tokens, seed):
            if messages[0]["content"] == "long":
                raise ValueError(OVERFLOW)
            if messages[0]["content"] == "other":
                raise ValueError("something else")
            return "ok"

    backend = ContextAsBudget(Chat())
    assert backend.generate_bounded([{"role": "user", "content": "x"}],
                                    max_new_tokens=5, seed=1) == "ok"
    with pytest.raises(BudgetExceeded) as exc:
        backend.generate_bounded([{"role": "user", "content": "long"}], max_new_tokens=5, seed=1)
    assert exc.value.limit == "context" and backend.overflows == 1
    with pytest.raises(ValueError, match="something else"):
        backend.generate_bounded([{"role": "user", "content": "other"}], max_new_tokens=5, seed=1)
    assert len(backend.forwarded) == 1
