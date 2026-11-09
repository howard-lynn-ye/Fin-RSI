from copy import deepcopy

import pytest

from benchmarks.agent_study.frozen_analysis import external, match_complete, source_mean, blocked_interval, tolerance_sensitivity


def fixture():
    metadata, rows, cells = [], [], []
    for i, source in enumerate(("CodeFinQA", "CodeFinQA", "CodeTAT-QA")):
        task = str(i)
        metadata.append(dict(task_id=task, source_task=source, dependency_block="issuer-"+task,
            template_sha256="template-"+task, units=dict(tags=["unspecified"]),
            development_exposure=dict(same_context=False, same_template=False, same_verified_issuer=False)))
        for model in ("m1", "m2"):
            for lib in (False, True):
                for seed in (101, 211, 307):
                    row = dict(model_id=model, method="external_code", library=lib, task_id=task,
                        task_sha256=task+"-hash", seed=seed, valid=lib and i < 2, accepted=True,
                        usage=dict(calls=1, generated_tokens=10))
                    rows.append(row)
                    cells.append(dict(row, model=model))
    return metadata, rows, cells


def test_external_weights_sources_equally_and_never_counts_seeds_as_clusters():
    selected, rows, cells = fixture()
    result = external(rows, cells, selected, draws=60)
    assert result["source_balanced_effect"] == .5  # pooled episode mean would be 2/3
    assert result["questions"] == 3 and result["episodes"] == 36
    assert result["dependency_bootstrap"]["blocks"] == 3
    assert result["effective_groups"] == pytest.approx(1/(.25**2+.25**2+.5**2))
    assert all(value["effect"] == .5 for value in result["model_effects"].values())
    assert result["unit_strata"]["question-explicit-count"]["questions"] == 0
    assert result["unit_strata"]["question-explicit-count"]["source_balanced_effect"] is None
    assert not result["independent_population_inference"]


def test_all_unresolved_is_one_block_not_three_documents():
    selected, rows, cells = fixture()
    for item in selected:
        item["dependency_block"] = "unresolved"
    result = external(rows, cells, selected, draws=60)
    assert result["effective_groups"] == 1
    assert result["unresolved_weight"] == 1
    assert result["dependency_bootstrap"]["interval"] is None
    assert result["dependency_bootstrap"]["blocks"] == 1


def test_missing_failed_or_changed_cell_is_not_a_zero_score():
    _, rows, cells = fixture()
    with pytest.raises(ValueError, match="complete fixed"):
        match_complete(rows[:-1], cells)
    bad = deepcopy(rows)
    bad[0]["valid"] = None
    with pytest.raises(ValueError, match="invalid grade"):
        match_complete(bad, cells)
    bad[0]["valid"] = False
    bad[0]["task_sha256"] = "different"
    with pytest.raises(ValueError, match="task identity"):
        match_complete(bad, cells)


def test_shared_block_draw_keeps_two_sources_together():
    metadata = {str(i):dict(source_task=source, dependency_block=block)
                for i, (source, block) in enumerate((
                    ("CodeFinQA", "A"), ("CodeTAT-QA", "A"),
                    ("CodeFinQA", "B"), ("CodeTAT-QA", "B")))}
    values = dict(zip(metadata, (1, -1, .5, -.5)))
    result = blocked_interval(values, metadata, "dependency_block", draws=100)
    assert result["interval"] == [0, 0]
    assert result["undefined_draws"] == 0
    assert source_mean({"0":1}, metadata) is None


def test_sensitivity_retains_primary_grades_complete_denominator_and_source_weights():
    metadata, rows, cells = fixture()
    for row in rows:
        row["relative_tolerance_1pct"] = bool(row["valid"] or not row["library"])
    original = deepcopy(rows)
    result = tolerance_sensitivity(rows, cells, metadata)
    assert rows == original
    assert result["source_balanced_effects"] == dict(exact=.5, relative_1pct=-.5)
    assert result["effect_direction_changed"]
    assert result["transitions"]["False"] == dict(episodes=18, exact_pass=0, exact_fail_relative_pass=18, both_fail=0)
    assert not result["confirmatory_tests"]
    with pytest.raises(ValueError, match="complete fixed"):
        tolerance_sensitivity(rows[:-1], cells, metadata)
    rows[0]["relative_tolerance_1pct"] = None
    with pytest.raises(ValueError, match="every original"):
        tolerance_sensitivity(rows, cells, metadata)
