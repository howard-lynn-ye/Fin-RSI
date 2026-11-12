import json

import pytest

from benchmarks.agent_study.external_enrollment import (
    context_key, map_origin, origin_index, select, template_key, unit_tags,
)


def test_selection_is_per_source_group_and_order_invariant():
    rows = [dict(row_index=i, source_task=source, sampling_group=f"g{i//2}",
                 stable_key=f"revision/test/{i}")
            for source, indices in (("CodeFinQA", range(8)), ("CodeTAT-QA", range(8, 12)))
            for i in indices]
    chosen = select(rows, cap=2)
    assert chosen == select(list(reversed(rows)), cap=2)
    assert len(chosen) == 4
    assert len({(r["source_task"], r["sampling_group"]) for r in chosen}) == 4
    with pytest.raises(ValueError, match="public enrollment"):
        select([dict(rows[0], answer="better outcome")])
    with pytest.raises(ValueError, match="duplicate"):
        select([rows[0], rows[0]])


def test_context_normalization_preserves_cells_and_list_order():
    assert context_key('{"b":"  value ","a":[1,2]}', "json") == context_key(
        '{"a":[1,2],"b":"value"}', "json")
    assert context_key('{"a":[1,2]}', "json") != context_key('{"a":[2,1]}', "json")
    assert context_key('{"a":"1"}', "json") != context_key('{"a":1}', "json")
    with pytest.raises(ValueError, match="merges"):
        context_key('{"a":1," a ":2}', "json")


def test_units_use_only_public_lexical_evidence_and_keep_ambiguity():
    tags = unit_tags("What percentage of USD revenue?", "units in millions")
    assert "question-explicit-percent-or-bp" in tags["tags"]
    assert "question-explicit-currency-or-scale" in tags["tags"]
    assert "multiple-or-ambiguous" in tags["tags"]
    assert not tags["semantic_unit_truth"]
    assert unit_tags("What was revenue?", "USD millions")["tags"] == ["context-only-unit"]
    assert template_key(" Revenue in 2018? ") == template_key("Revenue in 2019?")


def test_origin_requires_question_and_complete_supported_context():
    record = dict(filename="ACME/2020/page_1.pdf", pre_text=["An original paragraph."],
                  post_text=["The closing paragraph."], table=[["item", "year"], ["sales", "4"]],
                  qa=dict(question="What were sales?", answer="ignored", program="ignored"))
    index = origin_index([("CodeFinQA", "source-v1", [record])])
    context = "An original paragraph.\nitem | year\nsales | 4\nThe closing paragraph."
    actual = map_origin("CodeFinQA", "What were sales?", context_key(context, "string"), index)
    assert actual["status"] == "verified_source_document"
    assert actual["sampling_group"] == "filing:ACME/2020"
    assert actual["dependency_block"] == "issuer:ACME"
    unrelated = map_origin("CodeFinQA", "What were sales?", context_key("other", "string"), index)
    assert unrelated["status"] == "unresolved_context_group"
    assert unrelated["dependency_block"] == "unresolved"
    assert "answer" not in json.dumps(actual) and "program" not in json.dumps(actual)


def test_table_match_never_promotes_a_table_uid_to_independent_document():
    record = dict(table=dict(uid="table-A", table=[["", "2020"], ["sales", "4"]]),
                  questions=[dict(question="What were sales?")])
    index = origin_index([("CodeTAT-QA", "source-v1", [record])])
    key = context_key('{"sales":{"2020":"4"}}', "json")
    actual = map_origin("CodeTAT-QA", "What were sales?", key, index)
    assert actual["status"] == "unresolved_context_group"
    assert actual["dependency_block"] == "unresolved"
    assert actual["evidence"][0]["table_uid"] == "table-A"


def test_normalized_upstream_label_collision_is_unresolved_not_an_intake_crash():
    from benchmarks.agent_study.external_enrollment import source_variants
    record = dict(table=dict(uid="table-A", table=[["", "2020", " 2020 "], ["sales", "4", "5"]]),
                  questions=[dict(question="What were sales?")])
    assert source_variants(record, "CodeTAT-QA") == set()
    index = origin_index([("CodeTAT-QA", "source-v1", [record])])
    result = map_origin("CodeTAT-QA", "What were sales?", context_key('{"sales":{"2020":"4"}}', "json"), index)
    assert result["status"] == "unresolved_context_group"
    assert result["evidence"] == []


def test_full_episode_plan_keeps_core_and_extension_and_all_cells():
    from scripts.enroll_bizbench import episode_plan
    selected = [dict(task_id="t1", source_task="CodeFinQA", row_index=7,
                     public_sha256="public", scoring_policy_sha256="policy")]
    protocol = dict(external=dict(experiment=dict(conditions=[False, True], seeds=[101, 211, 307])),
                    models=dict(core_ids=["small"], twenty_hf_models=[
                        dict(id="small", model="org/small", revision="r1"),
                        dict(id="large", model="org/large", revision="r2")]))
    cells = episode_plan(selected, protocol)
    assert len(cells) == 12
    assert sum(c["stage"] == "core" for c in cells) == 6
    assert len({(c["model_id"], c["task_id"], c["library"], c["seed"]) for c in cells}) == 12
    assert all(c["public_sha256"] == "public" for c in cells)
