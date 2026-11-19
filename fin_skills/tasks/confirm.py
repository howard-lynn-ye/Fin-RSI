"""The 24 tasks of the confirmatory study (docs/P_CONFIRM_STUDY_DESIGN_20261119.md).

Written on 2026-11-19 after the library's pre-freeze fixes and before any model saw them.
Sixteen describe a rule positions() has, with parameter values and wording no earlier task
used; eight describe a rule it does not have, each a near miss of a catalogued one, so that
the right answer to "which catalogued rule is this?" is none. Every rule is stated in full and
never named. References are in tasks/references.py and do not call the library.

Which rules get a second task was drawn once: numpy default_rng(20261119), five of the eleven
catalogued rules with a task (equal_weight has none: its output does not depend on prices).
"""
from __future__ import annotations

from .spec import Task, TaskSet

CONFIRM_SEEDS = (409, 523)
DRAW_SEED = 20261119
SECOND = ("bollinger_reversion", "cross_sectional_momentum", "donchian_breakout", "hrp",
          "inverse_volatility")
WEIGHT = ("The weight on row t is that value divided by 4, the number of assets; on rows "
          "before the rule can be evaluated the weight is 0.")
RETURNS = frozenset({"positions", "split_factors", "returns_from_prices", "rolling_window"})
PRICES = frozenset({"positions", "split_factors", "rolling_window"})
ORDERS = (("A", "B", "C", "D"), ("B", "D", "A", "C"), ("D", "C", "B", "A"), ("C", "A", "D", "B"))


def _state_machine(low, high, centre, what):
    return (f"Keep a state per asset that starts at 0 and is updated at each of those closes in "
            f"order: {what} under {low} sets it to +1; {what} over {high} sets it to -1; while "
            f"it is +1, {what} of {centre} or more resets it to 0; while it is -1, {what} of "
            f"{centre} or less resets it to 0; in every other case it stays as it was. The "
            f"value at close t-1 is the state after that close's update. ")


def _ma(fast, slow):
    return (f"Per asset, compare two trailing arithmetic averages of the adjusted close, one "
            f"over {fast} closes and one over {slow} closes, each ending with close t-1. The "
            f"value is +1 when the shorter average is above the longer one, -1 when it is "
            f"below, and 0 when they are equal. " + WEIGHT)


def _momentum(lookback):
    return (f"Per asset, look at the adjusted close {lookback} sessions apart: from close "
            f"t-1-{lookback} to close t-1. The value is +1 if the close rose over that span, -1 "
            f"if it fell, and 0 if it is unchanged. " + WEIGHT)


def _vol_target(lookback, target):
    return (f"Per asset, the direction is the sign of the change in the adjusted close from "
            f"close t-1-{lookback} to close t-1. The size is the smaller of 1 and {target} "
            f"divided by the annualised volatility, where the annualised volatility is the "
            f"sample standard deviation (ddof=1) of the {lookback} simple returns ending with "
            f"the return into close t-1, multiplied by the square root of 252; when that "
            f"standard deviation is 0 the size is 0. The value is direction times size. "
            + WEIGHT)


def _macd(fast, slow, span):
    return (f"Per asset, smooth the adjusted close exponentially. For a span s the smoothing "
            f"constant is 2/(s+1); the smoothed series equals the close on row 0 and afterwards "
            f"is constant times the close plus (1 - constant) times its own previous value; "
            f"count a span-s series as available from row s-1. Let D be the span-{fast} "
            f"smoothed close minus the span-{slow} smoothed close, available from row "
            f"{slow - 1}. Smooth D in the same way with span {span}, starting at D's first "
            f"available value; that smoothed D is available once {span} values of D exist. At "
            f"close t-1 the value is the sign of D minus smoothed D. " + WEIGHT)


def _bollinger(lookback, z):
    return (f"Per asset and at every close from row {lookback - 1} on, measure how far the "
            f"adjusted close is from the average of the last {lookback} adjusted closes (the "
            f"current one included), in units of their sample standard deviation (ddof=1). "
            f"Call it the score, and use 0 when that standard deviation is 0. "
            + _state_machine(-z, z, 0, "a score") + WEIGHT)


def _rsi(lookback, lower, upper):
    return (f"Per asset, let d be the day-to-day change of the adjusted close. At each close "
            f"from row {lookback} on, take the last {lookback} changes and form U, the plain "
            f"average of max(d, 0), and D, the plain average of max(-d, 0); these are ordinary "
            f"averages over the window, with no recursive smoothing. The index is "
            f"100 * U / (U + D), or 50 when U + D is 0. "
            + _state_machine(lower, upper, 50, "an index") + WEIGHT)


def _donchian(lookback):
    return (f"Per asset, at each close t from row {lookback} on, compare the adjusted close "
            f"with the highest and the lowest of the {lookback} adjusted closes just before it "
            f"(rows t-{lookback} to t-1; the current close is not among them). A state starts "
            f"at 0 on row {lookback}. It becomes +1 when the close is above that highest value "
            f"and -1 when it is below that lowest value; otherwise it keeps its value. The "
            f"value at close t-1 is the state after that close. " + WEIGHT)


def _inverse_volatility(window):
    return (f"From session {window + 1} on, each asset's weight on row t is proportional to the "
            f"reciprocal of the sample standard deviation (ddof=1) of its last {window} simple "
            f"returns of the adjusted close, the latest being the return into close t-1. Scale "
            f"the weights so that they add up to 1. Every earlier row is 0.")


def _xsmom(lookback, top_k, min_history=60):
    first = max(lookback, min_history) + 1
    return (f"No position is held until {min_history} returns of history exist. For every "
            f"session t >= {first}, score each asset by its compounded simple return of the "
            f"adjusted close over the {lookback} returns ending with the return into close t-1. "
            f"Sort the assets by score, best first, keeping the column order among equal "
            f"scores. Of the first {top_k} in that order, each asset whose score is above 0 "
            f"gets a weight of 1/{top_k}; nothing else is held and the remainder stays in "
            f"cash. Every earlier row is 0.")


def _min_variance(window, min_history=20):
    first = max(window, min_history) + 1
    return (f"For every session t >= {first}, estimate the sample covariance matrix S (ddof=1) "
            f"of the assets from the {window} simple returns of the adjusted closes ending "
            f"with the return into close t-1. Hold the fully invested, long-only portfolio of "
            f"smallest variance: weights w >= 0 that add up to 1 and minimise w' S w, accurate "
            f"to 1e-5. Every earlier row is 0.")


def _hrp(window, linkage, min_history=20):
    first = max(window, min_history) + 1
    return (f"For every session t >= {first}, use the "
            f"{window} simple returns of the adjusted closes ending with the return into close "
            f"t-1, columns in the order given. From them take the correlation matrix C and the "
            f"sample covariance matrix S (ddof=1). Turn C into distances sqrt((1 - C) / 2), "
            f"cluster the assets hierarchically on those distances with scipy's '{linkage}' "
            f"linkage, and list the assets in the order of the dendrogram's leaves (scipy "
            f"leaves_list). Give every asset a weight of 1 and treat the whole list as one "
            f"group. Repeatedly cut each group of two or more assets into its first "
            f"len // 2 members and the remaining ones. For the two parts of a cut, let V1 and "
            f"V2 be the variance under S of each part's inverse-variance portfolio (weights "
            f"proportional to 1 / own variance within the part); multiply the weights of the "
            f"first part by 1 - V1/(V1+V2) and those of the second by V1/(V1+V2). Every earlier "
            f"row is 0.")


# Rules positions() does not have ------------------------------------------------------------

def _wilder_rsi(lookback, lower, upper):
    return (f"Per asset, let d be the day-to-day change of the adjusted close, up = max(d, 0) "
            f"and down = max(-d, 0). On row {lookback}, U and D are the plain averages of up "
            f"and down over the first {lookback} changes. On every later row they are updated "
            f"recursively with the newest change: U = (U * {lookback - 1} + up) / {lookback} "
            f"and D = (D * {lookback - 1} + down) / {lookback}. The index is 100 * U / (U + D), "
            f"or 50 when U + D is 0, at each close from row {lookback} on. "
            + _state_machine(lower, upper, 50, "an index") + WEIGHT)


def _band_breakout(lookback, entry, exit_):
    return (f"Per asset and at every close from row {lookback - 1} on, measure how far the "
            f"adjusted close is from the average of the last {lookback} adjusted closes (the "
            f"current one included), in units of their sample standard deviation (ddof=1). "
            f"Call it the score, and use 0 when that standard deviation is 0. Keep a state per "
            f"asset that starts at 0 and is updated at each of those closes in order: a score "
            f"over {entry} sets it to +1; a score under -{entry} sets it to -1; while it is +1, "
            f"a score under {exit_} resets it to 0; while it is -1, a score over -{exit_} "
            f"resets it to 0; in every other case it stays as it was. The value at close t-1 "
            f"is the state after that close's update. " + WEIGHT)


def _ema_crossover(fast, slow):
    return (f"Per asset, smooth the adjusted close exponentially. For a span s the smoothing "
            f"constant is 2/(s+1); the smoothed series equals the close on row 0 and afterwards "
            f"is constant times the close plus (1 - constant) times its own previous value; "
            f"count a span-s series as available from row s-1. At close t-1 the value is +1 "
            f"when the span-{fast} smoothed close is above the span-{slow} smoothed close, -1 "
            f"when it is below, and 0 when they are equal. " + WEIGHT)


def _threshold_momentum(lookback, threshold):
    return (f"Per asset, let m be the simple return of the adjusted close from close "
            f"t-1-{lookback} to close t-1. The value is +1 when m is above {threshold}, -1 when "
            f"m is below -{threshold}, and 0 in between (bounds included). " + WEIGHT)


def _range_position(lookback, lower, upper):
    return (f"Per asset and at every close from row {lookback - 1} on, locate the adjusted "
            f"close within the range of the last {lookback} adjusted closes (the current one "
            f"included): k = (close - lowest) / (highest - lowest), or 0.5 when highest equals "
            f"lowest. " + _state_machine(lower, upper, 0.5, "a k") + WEIGHT)


def _ma_envelope(lookback, band):
    return (f"Per asset, let M be the arithmetic average of the {lookback} adjusted closes "
            f"ending with close t-1. The value is +1 when close t-1 is above M * (1 + {band}), "
            f"-1 when it is below M * (1 - {band}), and 0 otherwise. " + WEIGHT)


def _long_short(lookback):
    return (f"For every session t >= {lookback + 1}, score each asset by its compounded simple "
            f"return of the adjusted close over the {lookback} returns ending with the return "
            f"into close t-1. Sort the assets by score, best first, keeping the column order "
            f"among equal scores. The first asset in that order gets the value +1 and the last "
            f"one -1, unless their scores are equal, in which case every value is 0; all other "
            f"assets get 0. " + WEIGHT)


def _inverse_variance(window):
    return (f"From session {window + 1} on, each asset's weight on row t is proportional to the "
            f"reciprocal of the sample variance (ddof=1) of its last {window} simple returns of "
            f"the adjusted close, the latest being the return into close t-1. Scale the weights "
            f"so that they add up to 1. Every earlier row is 0.")


def _task(name, algorithm, family, text, parameters, index, *, options=None, rows=128,
          traits=PRICES, tolerance=1e-8):
    """Variants alternate the factor convention and rotate the column order."""
    return Task(name, algorithm, family, text, dict(parameters), dict(options or {}),
                traits=traits, rows=rows, factors=("cumulative", "ratio")[index % 2],
                order=ORDERS[index % 4], tolerance=tolerance, origin="confirm")


def confirm_tasks() -> list[Task]:
    catalogued = [
        ("ma_crossover", "signal", _ma(7, 25), dict(fast=7, slow=25), {}, PRICES, 128, 1e-8),
        ("momentum", "signal", _momentum(30), dict(lookback=30), {}, RETURNS, 120, 1e-8),
        ("vol_target_momentum", "signal", _vol_target(25, 0.12),
         dict(lookback=25, target_vol=0.12), {}, RETURNS, 144, 1e-8),
        ("macd", "signal", _macd(10, 30, 7), dict(fast=10, slow=30, signal_span=7), {}, PRICES,
         144, 1e-8),
        ("bollinger_reversion", "signal", _bollinger(25, 1.75), dict(lookback=25, entry_z=1.75),
         {}, PRICES, 128, 1e-8),
        ("rsi_reversion", "signal", _rsi(9, 20.0, 80.0), dict(lookback=9, lower=20.0, upper=80.0),
         {}, PRICES, 120, 1e-8),
        ("donchian_breakout", "signal", _donchian(15), dict(lookback=15), {}, PRICES, 128, 1e-8),
        ("inverse_volatility", "portfolio", _inverse_volatility(30), dict(window=30),
         dict(window=30), RETURNS, 120, 1e-8),
        ("cross_sectional_momentum", "portfolio", _xsmom(50, 3), dict(lookback=50, top_k=3), {},
         RETURNS, 144, 1e-8),
        ("min_variance", "portfolio", _min_variance(30), dict(window=30), dict(window=30),
         RETURNS, 128, 1e-5),
        ("hrp", "portfolio", _hrp(45, "average"), dict(window=45, linkage="average"),
         dict(window=45), RETURNS, 144, 1e-8),
        # second tasks of the five drawn rules
        ("bollinger_reversion", "signal", _bollinger(12, 1.25), dict(lookback=12, entry_z=1.25),
         {}, PRICES, 144, 1e-8),
        ("cross_sectional_momentum", "portfolio", _xsmom(20, 1), dict(lookback=20, top_k=1), {},
         RETURNS, 128, 1e-8),
        ("donchian_breakout", "signal", _donchian(40), dict(lookback=40), {}, PRICES, 144, 1e-8),
        ("hrp", "portfolio", _hrp(35, "single"), dict(window=35, linkage="single"),
         dict(window=35), RETURNS, 120, 1e-8),
        ("inverse_volatility", "portfolio", _inverse_volatility(50), dict(window=50),
         dict(window=50), RETURNS, 144, 1e-8),
    ]
    other = [
        ("wilder_rsi_reversion", "signal", _wilder_rsi(12, 30.0, 70.0),
         dict(lookback=12, lower=30.0, upper=70.0), PRICES, 128),
        ("band_breakout", "signal", _band_breakout(20, 1.5, 0.5),
         dict(lookback=20, entry_z=1.5, exit_z=0.5), PRICES, 144),
        ("ema_crossover", "signal", _ema_crossover(6, 24), dict(fast=6, slow=24), PRICES, 120),
        ("threshold_momentum", "signal", _threshold_momentum(15, 0.03),
         dict(lookback=15, threshold=0.03), RETURNS, 128),
        ("range_position_reversion", "signal", _range_position(14, 0.2, 0.8),
         dict(lookback=14, lower=0.2, upper=0.8), PRICES, 144),
        ("ma_envelope", "signal", _ma_envelope(20, 0.02), dict(lookback=20, band=0.02), PRICES,
         128),
        ("long_short_momentum", "signal", _long_short(25), dict(lookback=25), RETURNS, 120),
        ("inverse_variance", "portfolio", _inverse_variance(35), dict(window=35), RETURNS, 144),
    ]
    tasks, seen = [], {}
    for i, (algorithm, family, text, parameters, options, traits, rows, tol) in enumerate(
            catalogued):
        seen[algorithm] = seen.get(algorithm, 0) + 1
        tasks.append(_task(f"c_{algorithm}_{seen[algorithm]}", algorithm, family, text,
                           parameters, i, options=options, rows=rows, traits=traits,
                           tolerance=tol))
    for i, (algorithm, family, text, parameters, traits, rows) in enumerate(other):
        tasks.append(_task(f"c_{algorithm}_1", algorithm, family, text, parameters,
                           len(catalogued) + i, rows=rows, traits=traits))
    return tasks


def confirm_suite() -> TaskSet:
    """All 24 tasks are test tasks; the confirmatory arms learn nothing, so there is no
    development split here (the earlier suite is the development material)."""
    tasks = tuple(confirm_tasks())
    return TaskSet(tasks, {t.algorithm: "heldout" for t in tasks})
