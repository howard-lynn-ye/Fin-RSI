"""Declared no-Landlock deviation (P-fix on Zaratan): opt-in only, recorded, system calls still denied."""
import json
import subprocess
import sys
import textwrap

import pytest

from benchmarks.agent_study import linux_sandbox
from fin_skills import rsi
from fin_skills.rsi.bench import NO_LANDLOCK_HOOK, QUALIFIED_HOOK
from fin_skills.tasks import frozen_suite


def unused(messages):
    raise AssertionError("no model dispatch permitted")


def gate(tmp_path):
    rsi.benchmark([rsi.NoImprove], unused, frozen_suite(), out=tmp_path / "run.jsonl",
                  model_id="fixture", execution_mode="model")


def test_no_landlock_hook_needs_the_declared_deviation(tmp_path, monkeypatch):
    monkeypatch.setenv("FIN_TASKS_REQUIRE_CONFINEMENT", "1")
    monkeypatch.setenv("FIN_TASKS_CONFINE", NO_LANDLOCK_HOOK)
    monkeypatch.delenv("FIN_TASKS_DECLARED_DEVIATION", raising=False)
    with pytest.raises(ValueError, match="qualified Linux confinement hook"):
        gate(tmp_path)
    monkeypatch.setenv("FIN_TASKS_DECLARED_DEVIATION", "something-else")
    with pytest.raises(ValueError, match="qualified Linux confinement hook"):
        gate(tmp_path)
    assert not list(tmp_path.iterdir())


def test_declared_deviation_still_requires_seccomp(tmp_path, monkeypatch):
    monkeypatch.setenv("FIN_TASKS_REQUIRE_CONFINEMENT", "1")
    monkeypatch.setenv("FIN_TASKS_CONFINE", NO_LANDLOCK_HOOK)
    monkeypatch.setenv("FIN_TASKS_DECLARED_DEVIATION", "no-landlock")
    monkeypatch.setattr(linux_sandbox, "seccomp_available", lambda: False)
    with pytest.raises(ValueError, match="confinement is unavailable"):
        gate(tmp_path)
    monkeypatch.setenv("FIN_TASKS_REQUIRE_CONFINEMENT", "0")
    with pytest.raises(ValueError, match="qualified Linux confinement hook"):
        gate(tmp_path)


def test_qualified_hook_is_unchanged_by_the_deviation(tmp_path, monkeypatch):
    monkeypatch.setenv("FIN_TASKS_REQUIRE_CONFINEMENT", "1")
    monkeypatch.setenv("FIN_TASKS_CONFINE", QUALIFIED_HOOK)
    monkeypatch.setenv("FIN_TASKS_DECLARED_DEVIATION", "no-landlock")
    monkeypatch.setattr(linux_sandbox, "confinement_available", lambda: False)
    monkeypatch.setattr(linux_sandbox, "seccomp_available", lambda: True)
    with pytest.raises(ValueError, match="confinement is unavailable"):
        gate(tmp_path)


@pytest.mark.skipif(not linux_sandbox.seccomp_available(), reason="needs Linux x86_64 libseccomp")
def test_confine_without_landlock_denies_sockets_and_subprocesses(tmp_path):
    child = textwrap.dedent(f"""
        import json, os, socket, sys
        sys.path.insert(0, {str(__import__('pathlib').Path(__file__).resolve().parents[1])!r})
        from benchmarks.agent_study.linux_sandbox import confine_without_landlock
        evidence = confine_without_landlock({str(tmp_path)!r}, [])
        out = dict(evidence=evidence)
        try:
            socket.socket()
            out["socket"] = "allowed"
        except OSError as exc:
            out["socket"] = exc.errno
        try:
            os.fork()
            out["fork"] = "allowed"
        except OSError as exc:
            out["fork"] = exc.errno
        print(json.dumps(out))
    """)
    done = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["socket"] == 1 and out["fork"] == 1
    assert out["evidence"]["network"] == "socket syscalls denied"
    assert out["evidence"]["subprocesses"] == "denied"
    assert "not restricted" in out["evidence"]["writes"]
