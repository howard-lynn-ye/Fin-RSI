"""Rule cards: the few rules a position-producing program must follow, delivered to the model.

Agents in our audits did not open optional skill documents (no successful read in 2,112 trading
decisions; no reference to the library in 24 code responses with its documentation attached).
Rules are therefore delivered, not offered:

    from fin_skills.api.rules import rules_for, prompt_block, explain
    cards = rules_for({"positions", "split_factors"})      # what the task declares
    text = prompt_block(cards)                              # put this in the prompt
    note = explain(["execution_lag"])                       # after a check fails

Each card condenses one passage of a skill and names it; the tests check that the passage is
still in the skill, so a card cannot silently drift from its source. Cards state a rule and how
it is verified. They never contain a task's expected output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

TRAITS = ("positions", "features", "split_factors", "returns_from_prices", "rolling_window")


@dataclass(frozen=True)
class RuleCard:
    id: str
    family: str                       # timing | corporate_actions | data_hygiene
    text: str                         # the rule, plain ASCII, a few sentences
    source: tuple[str, str, str]      # (skill, "SKILL.md" or reference file, passage)
    checks: tuple[str, ...]           # checks that verify the rule
    applies: frozenset[str]           # task traits that select the card
    tool: str = ""                    # the library call that follows the rule, if any

    def render(self, tools: bool = True) -> str:
        skill, doc, _ = self.source
        where = skill if doc == "SKILL.md" else f"{skill}/{doc}"
        text = f"{self.text} {self.tool}" if tools and self.tool else self.text
        return f"[{self.id}] {text} (source: {where})"


CARDS: tuple[RuleCard, ...] = (
    RuleCard(
        "execution_lag", "timing",
        "A position on row t earns the return from row t-1 to row t. Decide it from closes "
        "through row t-1 only: compute the signal at each close, then shift the positions down "
        "one row. A feature may use row t; a position may not.",
        ("signal-construction", "SKILL.md", "It tests features, not positions"),
        ("execution_lag",), frozenset({"positions"}),
        "fin_skills.algorithms.positions() applies this lag for you."),
    RuleCard(
        "feature_causality", "timing",
        "No value before row k may depend on rows at or after k: no centred windows, no "
        "full-sample means or z-scores, no negative shift(). Use trailing rolling() or "
        "expanding() windows.",
        ("signal-construction", "SKILL.md", "Indicator classes that inherently leak"),
        ("assert_causal",), frozenset({"features", "positions"})),
    RuleCard(
        "split_once", "corporate_actions",
        "Cumulative split factors already contain every earlier split. Apply them once: "
        "adjusted = raw * split_factors, then compute returns from the adjusted closes. Never "
        "take cumprod() of factors that are already cumulative.",
        ("research-integrity-guards", "adjustment-conventions.md", "`adjusted = raw \u00d7 factor`"),
        ("split_representation", "adjustment_check"), frozenset({"split_factors"}),
        "fin_skills.algorithms.positions(algorithm_id, raw, split_factors) does this once for "
        "you. The output of adjust_prices(raw, split_factors) is already adjusted: pass it with "
        "split_factors=None, never together with the factors again."),
    RuleCard(
        "same_convention", "corporate_actions",
        "Use one adjustment convention for the signal and for the returns the position earns. "
        "Do not mix raw and adjusted prices in one calculation.",
        ("research-integrity-guards", "adjustment-conventions.md",
         "Use the same convention for the signal and for the execution price"),
        ("split_representation",), frozenset({"split_factors"})),
    RuleCard(
        "warmup", "data_hygiene",
        "Rows before a rule has its full window are warm-up rows. Do not compute the rule on a "
        "partial window there; set those positions to 0 so the output has no missing values.",
        ("signal-construction", "SKILL.md", "discard warm-up generously"),
        ("finite_output",), frozenset({"positions", "rolling_window"})),
    RuleCard(
        "first_return", "data_hygiene",
        "pct_change() leaves the first row missing. Drop it (returns.iloc[1:]) before passing "
        "returns on, and keep each return on the date of the close that ends it.",
        ("portfolio-and-risk", "SKILL.md", "losing the first observation"),
        ("finite_output",), frozenset({"returns_from_prices"})),
)

CAUSES = {
    "execution_lag": "a position moved when only prices on or after its own row changed, so "
                     "it was decided with the close it trades over",
    "assert_causal": "an earlier value moved when only later rows changed, so the calculation "
                     "reads data from the future",
    "split_representation": "the output changed when the same economic prices were given as "
                            "adjusted closes with unit factors, so the split factors are applied "
                            "more than once or raw and adjusted prices are mixed",
    "adjustment_check": "the price series still jumps at a split date, so it is not adjusted",
    "finite_output": "the output has missing or infinite values, often on warm-up rows",
}


def card(card_id: str) -> RuleCard:
    for c in CARDS:
        if c.id == card_id:
            return c
    raise KeyError(f"unknown rule card {card_id!r}; known: {[c.id for c in CARDS]}")


def rules_for(traits: Iterable[str]) -> list[RuleCard]:
    """Cards whose traits intersect the task's declared traits, in a fixed order."""
    wanted = set(traits)
    unknown = wanted - set(TRAITS)
    if unknown:
        raise ValueError(f"unknown traits {sorted(unknown)}; known: {list(TRAITS)}")
    return [c for c in CARDS if c.applies & wanted]


def prompt_block(cards: Iterable[RuleCard], *, tools: bool = True) -> str:
    """The cards as a prompt section. tools=False leaves out the library calls, for a
    program that must not import fin_skills."""
    cards = list(cards)
    if not cards:
        return ""
    return "Rules that apply to this task:\n" + "\n".join(f"- {c.render(tools)}" for c in cards)


def explain(failed: Iterable[str], *, tools: bool = True) -> str:
    """Cause and rule for each failed check, for feedback to the model. No expected output."""
    lines = []
    for name in dict.fromkeys(failed):
        if name not in CAUSES:
            raise KeyError(f"no explanation for check {name!r}; known: {sorted(CAUSES)}")
        rules = [c for c in CARDS if name in c.checks]
        lines.append(f"Check {name} failed: likely {CAUSES[name]}.")
        lines.extend(f"  Rule: {c.render(tools)}" for c in rules)
    return "\n".join(lines)
