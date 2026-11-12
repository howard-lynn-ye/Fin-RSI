"""fin_skills.api.rules - rule cards delivered to the model, tied to their skill passages."""
from __future__ import annotations

import pytest

import fin_skills
from fin_skills.api.rules import CARDS, CAUSES, TRAITS, card, explain, prompt_block, rules_for


@pytest.mark.parametrize("c", CARDS, ids=lambda c: c.id)
def test_every_card_cites_a_passage_that_is_still_in_its_skill(c):
    skill, doc, passage = c.source
    text = fin_skills.load(skill) if doc == "SKILL.md" else fin_skills.references(skill)[doc]
    assert passage in text, f"{c.id}: '{passage}' no longer in {skill}/{doc}"


@pytest.mark.parametrize("c", CARDS, ids=lambda c: c.id)
def test_cards_are_short_ascii_and_name_their_checks(c):
    assert c.text.isascii() and c.render().isascii() and c.tool.isascii()
    assert 40 <= len(c.text) <= 420
    assert "fin_skills" not in c.text and "fin_skills" not in c.render(tools=False)
    assert c.checks and all(name in CAUSES for name in c.checks)
    assert c.applies and c.applies <= set(TRAITS)


def test_rules_for_selects_by_declared_traits_in_a_fixed_order():
    ids = [c.id for c in rules_for({"positions", "split_factors"})]
    assert ids == ["execution_lag", "feature_causality", "split_once", "same_convention", "warmup"]
    assert [c.id for c in rules_for({"returns_from_prices"})] == ["first_return"]
    assert rules_for(set()) == []
    with pytest.raises(ValueError, match="unknown traits"):
        rules_for({"alpha"})


def test_prompt_block_lists_each_rule_with_its_source():
    block = prompt_block(rules_for({"split_factors"}))
    assert block.startswith("Rules that apply to this task:")
    assert "research-integrity-guards/adjustment-conventions.md" in block
    assert "cumprod()" in block
    assert prompt_block([]) == ""


def test_explain_gives_cause_and_rule_without_expected_output():
    note = explain(["execution_lag", "split_representation", "execution_lag"])
    assert note.count("Check execution_lag failed") == 1
    assert "decided with the close it trades over" in note
    assert "[execution_lag]" in note and "[split_once]" in note and "[same_convention]" in note
    assert "weight" not in note.lower()          # no reference weights or expected values
    with pytest.raises(KeyError):
        explain(["no_such_check"])


def test_card_lookup():
    assert card("split_once").family == "corporate_actions"
    with pytest.raises(KeyError):
        card("nope")


def test_tools_false_leaves_out_library_calls():
    cards = rules_for({"positions", "split_factors"})
    assert "fin_skills.algorithms.positions()" in prompt_block(cards)
    assert "fin_skills" not in prompt_block(cards, tools=False)
    assert "fin_skills" not in explain(["execution_lag", "split_representation"], tools=False)
    assert "adjust_prices" in explain(["split_representation"])


def test_split_card_steers_away_from_double_adjustment():
    """The v2 card ended by pointing at adjust_prices(); 14B then composed adjust_prices() with
    positions(..., split_factors). The card must name the single-call path and forbid that."""
    text = card("split_once").render()
    assert "positions(algorithm_id, raw, split_factors)" in text
    assert "split_factors=None" in text
    assert text.isascii()


def test_split_card_distinguishes_cumulative_factors_from_event_ratios():
    """The same card reaches both task conventions, including no-library feedback."""
    rule = card("split_once")
    for text in (rule.render(tools=False), explain(["split_representation"], tools=False)):
        assert "For cumulative factors" in text
        assert "For event ratios" in text
        assert "split_factors.cumprod()" in text
        assert "fin_skills" not in text
    assert "factors='ratio'" in rule.render()
