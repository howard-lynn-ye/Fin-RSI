"""Qualification cannot pass skipped tests or silently changed execution sources."""
import hashlib
import json

import pytest

from benchmarks.agent_study.rsi_qualify import (
    collected_nodes, development_tasks, junit_counts, verify_source, write_new,
)


def test_qualification_material_remains_original_development():
    from fin_skills.tasks import suite
    tasks = development_tasks()
    assert not tasks.dev
    assert all(t in suite().dev for t in tasks.heldout)


@pytest.mark.parametrize("attributes", ["tests='0'", "tests='1' skipped='1'",
                                       "tests='1' errors='1'", "tests='1' failures='1'"])
def test_zero_or_skipped_or_failed_cpu_qualification_is_rejected(tmp_path, attributes):
    path = tmp_path / "tests.xml"
    path.write_text(f"<testsuites><testsuite {attributes}/></testsuites>")
    with pytest.raises(RuntimeError, match="unskipped"):
        junit_counts(path)


def test_source_mutation_and_duplicate_receipt_are_rejected(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    manifest = tmp_path / "source-manifest.json"
    write_new(manifest, {source.name: hashlib.sha256(source.read_bytes()).hexdigest()})
    assert verify_source(tmp_path)
    source.write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        verify_source(tmp_path)
    with pytest.raises(FileExistsError):
        write_new(manifest, {})


def test_clean_junit_counts(tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text("<testsuites><testsuite tests='7' errors='0'/></testsuites>")
    assert junit_counts(path) == dict(tests=7, errors=0, failures=0, skipped=0)


def test_collected_plan_keeps_parameterized_nodes_and_rejects_empty_or_duplicates():
    node = "tests/test_contract.py::test_case[one]"
    assert collected_nodes(node + "\n" + node.replace("[one]", "[two]") +
                           "\n2 tests collected") == [node, node.replace("[one]", "[two]")]
    for invalid in ("no tests ran", node + "\n" + node):
        with pytest.raises(ValueError, match="unique"):
            collected_nodes(invalid)
