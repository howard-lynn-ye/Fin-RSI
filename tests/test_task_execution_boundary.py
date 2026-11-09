"""Adversarial synthetic fixtures; never run historical model output as trusted code."""
import os
from pathlib import Path

import pytest

from fin_skills.tasks import Judge, suite
from benchmarks.agent_study.linux_sandbox import confinement_available


@pytest.fixture
def confined(monkeypatch):
    if not confinement_available():
        pytest.skip("requires Linux x86_64 Landlock ABI >= 3 and libseccomp")
    monkeypatch.setenv("FIN_TASKS_CONFINE", "benchmarks.agent_study.linux_sandbox:confine")
    monkeypatch.setenv("FIN_TASKS_REQUIRE_CONFINEMENT", "1")


@pytest.mark.parametrize("library", [False, True])
def test_candidate_has_no_grader_modules_or_files(confined, library):
    import fin_skills.tasks.judge as judge_module
    private = str(Path(judge_module.__file__).with_name("references.py"))
    code = f'''
import sys, pathlib, importlib
assert not any(n.startswith(("fin_skills.tasks", "fin_skills.rsi")) for n in sys.modules)
try:
    pathlib.Path({private!r}).read_text()
except PermissionError:
    pass
else:
    raise AssertionError("private reference file readable")
try:
    importlib.import_module("fin_skills.tasks.references")
except (ImportError, PermissionError):
    pass
else:
    raise AssertionError("private reference module importable")
def solve(prices, split_factors):
    return prices * 0
'''
    seen = Judge(suite().heldout[0], seed=11, library=library).check(code)
    assert seen.accepted, seen.error


def test_raw_arm_blocks_importlib_and_direct_source_loading(confined):
    import fin_skills.algorithms as algorithms
    private = str(Path(algorithms.__file__).resolve())
    code = f'''
import importlib, importlib.util
try:
    importlib.import_module("fin_skills.algorithms")
except (ImportError, PermissionError):
    pass
else:
    raise AssertionError("raw arm imported algorithms")
try:
    spec = importlib.util.spec_from_file_location("stolen_library", {private!r})
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
except PermissionError:
    pass
else:
    raise AssertionError("raw arm read library source")
def solve(prices, split_factors):
    return prices * 0
'''
    seen = Judge(suite().heldout[0], seed=11, library=False).check(code)
    assert seen.accepted, seen.error


def test_candidate_monkeypatch_cannot_change_parent_checks_or_grade(confined):
    code = '''
import numpy as np
np.allclose = lambda *a, **k: True
def solve(prices, split_factors):
    return prices * 2
'''
    judge = Judge(suite().heldout[0], seed=11, library=False)
    seen = judge.check(code)
    assert seen.executed and not seen.accepted
    assert not seen.checks["split_representation"]
    assert not seen.checks["execution_lag"]
    assert not judge.grade(code)


def test_candidate_cannot_network_or_spawn(confined):
    code = '''
import socket, subprocess
for operation in (lambda: socket.socket(), lambda: subprocess.run(["true"])):
    try:
        operation()
    except PermissionError:
        pass
    else:
        raise AssertionError("forbidden capability available")
def solve(prices, split_factors):
    return prices * 0
'''
    seen = Judge(suite().heldout[0], seed=11, library=False).check(code)
    assert seen.accepted, seen.error


@pytest.mark.parametrize("library", [False, True])
def test_parent_environment_is_not_inherited(confined, monkeypatch, library):
    monkeypatch.setenv("FIN_REVIEW_NONSECRET_CANARY", "synthetic-canary")
    code = '''
import os
assert "FIN_REVIEW_NONSECRET_CANARY" not in os.environ
def solve(prices, split_factors):
    return prices * 0
'''
    seen = Judge(suite().heldout[0], seed=11, library=library).check(code)
    assert seen.accepted, seen.error
    assert "synthetic-canary" not in seen.error


@pytest.mark.parametrize("payload", [
    'import os; os._exit(42)',
    'import os, sys, pathlib; '
    'pathlib.Path(sys.argv[1], "result.json").write_text("not-json"); os._exit(0)',
    'import os, sys, pathlib; pathlib.Path(sys.argv[1], "result.json").write_text('
    '\'{"executed": false, "infrastructure_error": true}\'); os._exit(0)',
    'import os, sys, pathlib; pathlib.Path(sys.argv[1], "result.json").write_text('
    '"x" * ((1 << 20) + 1)); os._exit(0)',
])
def test_post_start_exit_and_bad_payload_are_candidate_failures(confined, payload):
    judge = Judge(suite().heldout[0], seed=11, library=False)
    seen = judge.check(payload)
    assert not seen.executed
    assert not judge.records[-1]["infrastructure_error"]


def test_flooded_stdout_hits_candidate_timeout_without_capture(confined):
    code = 'import os\nwhile True: os.write(1, b"x" * 4096)\n'
    judge = Judge(suite().heldout[0], seed=11, library=False, timeout=5)
    seen = judge.check(code)
    assert not seen.executed and "TimeoutExpired" in seen.error
    assert not judge.records[-1]["infrastructure_error"]


def test_trace_limit_is_a_candidate_rejection(confined):
    code = '''
from fin_skills.algorithms import positions
def solve(prices, split_factors):
    for _ in range(65):
        result = positions("ma_crossover", prices, split_factors)
    return result
'''
    judge = Judge(suite().heldout[0], seed=11, library=True)
    seen = judge.check(code)
    assert not seen.executed and "trace limit" in seen.error
    assert not judge.records[-1]["infrastructure_error"]


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_result_link_or_pipe_cannot_make_parent_read_or_block(confined, kind):
    code = f'''
import os, sys, pathlib
target = pathlib.Path(sys.argv[1], "result.json")
if {kind!r} == "symlink":
    target.symlink_to("/etc/passwd")
else:
    os.mkfifo(target)
os._exit(0)
'''
    judge = Judge(suite().heldout[0], seed=11, library=False)
    seen = judge.check(code)
    assert not seen.executed and "regular file" in seen.error
    assert not judge.records[-1]["infrastructure_error"]
