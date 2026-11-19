"""Which catalogue algorithm computes a rule that is described in words.

positions() runs an algorithm by id. A task usually describes its rule without naming it, and in
our study models then wrote the rule themselves even when the catalogue had it. route() asks the
connected model that one question before any program is written:

    algorithm_id = route(description, generate)     # a positions() id, or None
    hint = POINTER.format(algorithm=algorithm_id)   # one sentence to add to the task

generate is the library's usual callable, generate(messages) -> str. Nothing here knows which
rule is right: the answer is the model's, and a wrong or missing one is returned as it comes.
"""
from __future__ import annotations

import re

# What each position algorithm computes at a close, before positions() applies lag, warm-up and
# scale; written from fin_skills/algorithms (technical.py, runtime.py, catalog.py). Shown by the
# documented interface (library_text(documented=True)) and by route().
RULES = {
    "equal_weight": "1 / N on every asset.",
    "inverse_volatility": "weights proportional to 1 / volatility of each asset's returns over "
                          "the window, summing to 1.",
    "min_variance": "long-only weights that minimise the variance of portfolio returns over the "
                    "window.",
    "hrp": "hierarchical risk parity weights from the window's returns.",
    "cross_sectional_momentum": "1 / top_k on each of the top_k assets by summed log return over "
                                "the last lookback rows, only where that sum is positive; the "
                                "rest is cash.",
    "ma_crossover": "sign of (mean of the last fast closes - mean of the last slow closes).",
    "momentum": "sign of the summed log return over the last lookback returns.",
    "vol_target_momentum": "sign of the change over the last lookback closes, times "
                           "min(target_vol / (sample std of the last lookback simple returns "
                           "* sqrt(periods_per_year)), max_exposure), 0 when that std is 0.",
    "macd": "sign of (line - EMA(signal_span) of line), with line = EMA(fast) - EMA(slow) of "
            "closes; every EMA is pandas ewm(span, adjust=False).",
    "bollinger_reversion": "z = (close - mean) / sample std (ddof=1) over the last lookback "
                           "closes including the current one, z = 0 when std is 0. Walking "
                           "forward from flat: +1 when z < -entry_z, -1 when z > entry_z, back "
                           "to 0 when long and z >= 0 or short and z <= 0, otherwise unchanged.",
    "rsi_reversion": "RSI = 100 * gain / (gain + loss), with gain and loss the simple means of "
                     "the positive and negative close-to-close changes over the last lookback "
                     "changes (not Wilder smoothing), RSI = 50 when both are 0. Walking forward "
                     "from flat: +1 when RSI < lower, -1 when RSI > upper, back to 0 when long "
                     "and RSI >= 50 or short and RSI <= 50, otherwise unchanged.",
    "donchian_breakout": "upper and lower = max and min of the lookback closes before the "
                         "current one. From flat once defined: +1 on a close above upper, -1 on "
                         "a close below lower, otherwise unchanged.",
}

POINTER = 'The fin_skills library implements this rule as positions("{algorithm}", ...).'
ROUTE_SYSTEM = ("You match a rule described in words to a library catalogue. Reply with one line: "
                "the id of the catalogue rule that computes it, or NONE.")
ROUTE_QUESTION = ("Which catalogue id computes the rule above, apart from parameter values? Reply "
                  "with that id alone, or NONE if no catalogue rule computes it.")


def route_messages(description: str) -> list[dict]:
    """The one question route() asks: the description, the catalogue's rules, id or NONE."""
    if not isinstance(description, str) or not description.strip():
        raise TypeError("description must be a non-empty string")
    listing = "\n".join(f"{name}: {rule}" for name, rule in RULES.items())
    user = ("Rule to match:\n" + description.strip() + "\n\nCatalogue (id: the rule it computes "
            "at a close; lookbacks, thresholds and other parameters can be set to any value, and "
            "lag, warm-up and scaling are applied separately):\n" + listing + "\n\n"
            + ROUTE_QUESTION)
    return [dict(role="system", content=ROUTE_SYSTEM), dict(role="user", content=user)]


def parse_route(reply: str) -> str | None:
    """The catalogue id a reply names, or None. The last line decides; reasoning before it
    (including a <think> block) is ignored unless the last line names nothing at all."""
    text = re.sub(r"<think>.*?</think>", " ", reply or "", flags=re.S).split("</think>")[-1]
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None

    def named(part):
        return {w for w in re.findall(r"[a-z_]+", part.lower()) if w in RULES}

    last = named(lines[-1])
    if len(last) == 1:
        return last.pop()
    if last or re.search(r"\bnone\b", lines[-1], re.I):
        return None
    everything = named(text)
    return everything.pop() if len(everything) == 1 else None


def route(description: str, generate) -> str | None:
    """Ask the model which catalogue algorithm computes the described rule: its id, or None."""
    return parse_route(generate(route_messages(description)))
