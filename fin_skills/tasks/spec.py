"""Graded financial programming tasks: what the model is asked, and how it is graded.

Every task asks for solve(prices, split_factors) -> date x asset positions. The definition is
the complete rule in plain text. The reference (fin_skills.tasks.references) is written
independently of the library; the library call that computes the same rule is recorded only
so the qualification tests can compare the two.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import random

from .data import ASSETS, FACTOR_KINDS

SPLITS = ("dev", "heldout")
TABLE_FAMILIES = ("volatility", "risk", "forecast")
# E4 tasks were used to design the checks and to repair the library (v1 and v2), so their
# algorithms can never be held out.
E4_ALGORITHMS = ("inverse_volatility", "cross_sectional_momentum", "ma_crossover",
                 "vol_target_momentum")
# Any algorithm a model has been run on before the protocol is frozen is development data too.
# Beacon job 3443316 (model qualification) runs e4_ma_crossover and momentum_a.
QUALIFICATION_ALGORITHMS = ("momentum",)
FORCED_DEV = E4_ALGORITHMS + QUALIFICATION_ALGORITHMS
SPLIT_SEED = 20261103


@dataclass(frozen=True)
class Task:
    id: str
    algorithm: str                       # catalog id; the dev/held-out split is by algorithm
    family: str                          # portfolio | signal | volatility | risk | forecast
    definition: str
    parameters: dict = field(default_factory=dict)      # reference and library parameters
    library_options: dict = field(default_factory=dict)  # positions() / rolling() keywords
    traits: frozenset = frozenset({"positions", "split_factors", "rolling_window"})
    rows: int = 128
    factors: str = "cumulative"
    order: tuple = ASSETS
    tolerance: float = 1e-8
    origin: str = "new"                  # "e4" for the four original tasks

    def __post_init__(self):
        if self.factors not in FACTOR_KINDS:
            raise ValueError(f"{self.id}: factors must be one of {FACTOR_KINDS}")

    def manifest(self) -> dict:
        """The complete task contract; no behavioural fields are omitted from its hash."""
        body = asdict(self)
        body["traits"] = sorted(self.traits)
        body["order"] = list(self.order)
        return json.loads(json.dumps(body, sort_keys=True, allow_nan=False))

    @classmethod
    def from_manifest(cls, body: dict) -> "Task":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(body, dict) or set(body) != fields:
            raise ValueError("task manifest must contain exactly the complete Task fields")
        values = json.loads(json.dumps(body, allow_nan=False))
        values["traits"] = frozenset(values["traits"])
        values["order"] = tuple(values["order"])
        return cls(**values)

    @property
    def kind(self) -> str:
        """"positions" (weights held over the next return) or "table" (a value known at close
        t-1, such as a volatility, VaR or forecast)."""
        return "table" if self.family in TABLE_FAMILIES else "positions"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.manifest(), sort_keys=True).encode()).hexdigest()


FACTOR_TEXT = {
    "cumulative": ("split_factors are CUMULATIVE: split_factors[t] is the product of every split "
                   "ratio effective on or before row t. The continuous adjusted close is "
                   "prices * split_factors."),
    "ratio": ("split_factors are EVENT RATIOS: split_factors[t] is the ratio of a split "
              "effective on row t and 1.0 on every other row. The cumulative factor is their "
              "running product, and the continuous adjusted close is prices times that "
              "cumulative factor."),
}

SIGNAL_TAIL = ("The weight on row t is that value divided by the number of assets (4). Rows "
               "where it is not yet defined are 0.")


def _e4() -> list[Task]:
    # Definitions copied from benchmarks/agent_study/codegen_utility.DEFINITIONS (E4).
    return [
        Task("e4_diagonal_risk_parity", "inverse_volatility", "portfolio",
             "For every session t >= 61, use the 60 simple returns ending at t-1. Normalize "
             "inverse sample standard deviations (ddof=1) to sum to 1. This is diagonal risk "
             "parity (inverse-volatility), not full-covariance equal risk contribution. "
             "Earlier rows are zero.",
             dict(window=60), dict(window=60), rows=96, origin="e4",
             traits=frozenset({"positions", "split_factors", "returns_from_prices",
                               "rolling_window"})),
        Task("e4_cross_sectional_momentum", "cross_sectional_momentum", "portfolio",
             "For every t >= 61, compound the 60 simple returns ending at t-1. Rank assets "
             "descending; stable ties follow column order. Among the top two assets, give each "
             "with strictly positive momentum weight 0.5; leave the remainder in cash. Earlier "
             "rows are zero.",
             dict(lookback=60, top_k=2), rows=96, origin="e4",
             traits=frozenset({"positions", "split_factors", "returns_from_prices",
                               "rolling_window"})),
        Task("e4_ma_crossover", "ma_crossover", "signal",
             "For each asset, compute sign(mean(last 5 adjusted closes)-mean(last 20 adjusted "
             "closes)) at close t-1, and divide by 4 for the weight at t. Earliest "
             "nonzero-capable row is t=20. Earlier rows are zero. Use simple rolling means, not "
             "exponential means.",
             dict(fast=5, slow=20), rows=96, origin="e4"),
        Task("e4_vol_target_momentum", "vol_target_momentum", "signal",
             "For each asset at close t-1, compute sign(P[t-1]/P[t-21]-1), multiply by "
             "min(1, 0.10/(sample_std(last 20 simple returns,ddof=1)*sqrt(252))), then divide "
             "by 4. Put that weight on row t. Earliest nonzero-capable row is t=21; earlier rows "
             "are zero.",
             dict(lookback=20, target_vol=0.10), rows=96, origin="e4",
             traits=frozenset({"positions", "split_factors", "returns_from_prices",
                               "rolling_window"})),
    ]


def _inverse_volatility(window):
    return (f"For every session t >= {window + 1}, use the {window} simple returns of the "
            f"adjusted closes ending at close t-1. Weight each asset by the inverse of its sample "
            f"standard deviation (ddof=1) over those returns, normalized to sum to 1. Earlier "
            f"rows are 0.")


def _xsmom(lookback, top_k, min_history=60):
    first = max(lookback, min_history) + 1
    return (f"Decisions start once {min_history} returns of history exist: for every session "
            f"t >= {first}, compound the {lookback} simple returns of "
            f"the adjusted closes ending at close t-1 into each asset's momentum. Rank assets by "
            f"momentum, highest first; ties keep column order. Each of the top {top_k} assets "
            f"whose momentum is strictly positive gets weight 1/{top_k}; the rest is cash. "
            f"Earlier rows are 0.")


def _ma(fast, slow):
    return (f"For each asset at close t-1, take sign(simple mean of the last {fast} adjusted "
            f"closes - simple mean of the last {slow} adjusted closes), both windows ending at "
            f"close t-1. " + SIGNAL_TAIL)


def _vt(lookback, target):
    return (f"For each asset at close t-1, take sign(P[t-1]/P[t-1-{lookback}] - 1) times "
            f"min(1, {target} / (sample standard deviation (ddof=1) of the last {lookback} "
            f"simple returns ending at close t-1, times sqrt(252))), where P is the adjusted "
            f"close; use 0 when that standard deviation is 0. " + SIGNAL_TAIL)


def _momentum(lookback):
    return (f"For each asset at close t-1, take sign(P[t-1]/P[t-1-{lookback}] - 1), where P is "
            f"the adjusted close (0 when unchanged). " + SIGNAL_TAIL)


def _macd(fast, slow, span):
    return (f"For each asset, compute exponential moving averages of the adjusted close with "
            f"alpha = 2/(span+1), started at the first close (EMA[0] = P[0], EMA[i] = alpha*P[i] "
            f"+ (1-alpha)*EMA[i-1]); an EMA with span s counts as defined from row s-1 on. The "
            f"MACD line is EMA(span {fast}) - EMA(span {slow}), defined from row {slow - 1}. The "
            f"signal line is the same kind of EMA with span {span} applied to the MACD line, "
            f"started at its first defined value and defined once {span} MACD values exist. At "
            f"close t-1 take sign(MACD line - signal line). " + SIGNAL_TAIL)


def _walk_rule(lower, upper, center):
    return (f"Starting flat (0) and walking forward through the closes where the indicator is "
            f"defined: become long (+1) when it is below {lower}, short (-1) when it is above "
            f"{upper}, flat when long and it is at or above {center} or short and it is at or "
            f"below {center}, and otherwise keep the previous state. Take the state at close "
            f"t-1. ")


def _bollinger(lookback, z):
    return (f"For each asset, at every close compute z = (P - mean) / std over the last "
            f"{lookback} adjusted closes including the current one (sample std, ddof=1; z = 0 "
            f"when std is 0); z is defined from row {lookback - 1}. "
            + _walk_rule(-z, z, 0) + SIGNAL_TAIL)


def _rsi(lookback, lower, upper):
    return (f"For each asset let delta[t] = P[t] - P[t-1] on adjusted closes. At close t, gain "
            f"is the simple mean of max(delta, 0) over the last {lookback} deltas and loss the "
            f"simple mean of max(-delta, 0) over the same deltas (not Wilder smoothing). RSI = "
            f"100 * gain / (gain + loss), or 50 when gain + loss is 0; RSI is defined from row "
            f"{lookback}. " + _walk_rule(lower, upper, 50) + SIGNAL_TAIL)


def _donchian(lookback):
    return (f"For each asset, at close t (t >= {lookback}) let upper be the maximum and lower the "
            f"minimum of the {lookback} adjusted closes on rows t-{lookback} to t-1. Starting "
            f"flat at row {lookback}, the state becomes +1 on a close above upper, -1 on a close "
            f"below lower, and otherwise keeps its last value. Take the state at close t-1. "
            + SIGNAL_TAIL)


def _min_variance(window, min_history=20):
    first = max(window, min_history) + 1
    return (f"For every session t >= {first}, take the {window} simple returns of the adjusted "
            f"closes ending at close t-1 and their sample covariance matrix S (ddof=1). The "
            f"weights minimize w' S w subject to sum(w) = 1 and w >= 0 (long-only, fully "
            f"invested minimum variance); weights must be accurate to 1e-5. Earlier rows "
            f"are 0.")


def _hrp(window, linkage, min_history=20):
    first = max(window, min_history) + 1
    return (f"Hierarchical risk parity. For every session t >= {first}, take the {window} simple "
            f"returns of the adjusted closes ending at close t-1, with the columns in the given "
            f"order. Compute their correlation C and sample covariance S (ddof=1), the distance "
            f"d = sqrt((1 - C) / 2), and a hierarchical clustering of d with scipy's "
            f"'{linkage}' linkage. Order the assets by the dendrogram leaves (scipy leaves_list). "
            f"Then bisect recursively: start every weight at 1 and the list as one cluster; "
            f"split each cluster of two or more assets into its first half (length // 2) and "
            f"the rest; for each pair, with V the variance under S of the inverse-variance "
            f"portfolio of each half, multiply the first half's weights by 1 - V1/(V1+V2) and "
            f"the second half's by V1/(V1+V2). Earlier rows are 0.")


RETURNS = frozenset({"positions", "split_factors", "returns_from_prices", "rolling_window"})
SHUFFLED = ("C", "A", "D", "B")


def _new() -> list[Task]:
    t = []
    # Variant "b" of each algorithm: other parameters, event-ratio factors, shuffled columns.
    t += [Task("inverse_volatility_b", "inverse_volatility", "portfolio", _inverse_volatility(40),
               dict(window=40), dict(window=40), factors="ratio", order=SHUFFLED,
               traits=RETURNS),
          Task("cross_sectional_momentum_b", "cross_sectional_momentum", "portfolio",
               _xsmom(40, 1), dict(lookback=40, top_k=1), factors="ratio", order=SHUFFLED,
               traits=RETURNS),
          Task("ma_crossover_b", "ma_crossover", "signal", _ma(10, 30), dict(fast=10, slow=30),
               factors="ratio", order=SHUFFLED),
          Task("vol_target_momentum_b", "vol_target_momentum", "signal", _vt(30, 0.15),
               dict(lookback=30, target_vol=0.15), factors="ratio", order=SHUFFLED,
               traits=RETURNS)]
    # Portfolio optimizers. min_variance is solved by SLSQP in the library and exactly (by subset
    # enumeration) in the reference, so it is graded at 1e-5. equal_weight is left out: its
    # output does not depend on prices, so none of the traps can be exercised.
    t += [Task("min_variance_a", "min_variance", "portfolio", _min_variance(60),
               dict(window=60), dict(window=60), traits=RETURNS, tolerance=1e-5),
          Task("min_variance_b", "min_variance", "portfolio", _min_variance(40),
               dict(window=40), dict(window=40), factors="ratio", order=SHUFFLED,
               traits=RETURNS, tolerance=1e-5),
          Task("hrp_a", "hrp", "portfolio", _hrp(60, "single"),
               dict(window=60, linkage="single"), dict(window=60), traits=RETURNS),
          Task("hrp_b", "hrp", "portfolio", _hrp(40, "complete"),
               dict(window=40, linkage="complete"), dict(window=40), factors="ratio",
               order=SHUFFLED, traits=RETURNS)]
    for algorithm, a, b, text in (
            ("momentum", dict(lookback=20), dict(lookback=40), _momentum),
            ("macd", dict(fast=12, slow=26, signal_span=9), dict(fast=8, slow=21, signal_span=5),
             _macd),
            ("bollinger_reversion", dict(lookback=20, entry_z=2.0),
             dict(lookback=15, entry_z=1.5), _bollinger),
            ("rsi_reversion", dict(lookback=14, lower=30.0, upper=70.0),
             dict(lookback=10, lower=25.0, upper=75.0), _rsi),
            ("donchian_breakout", dict(lookback=20), dict(lookback=30), _donchian)):
        traits = RETURNS if algorithm == "momentum" else frozenset(
            {"positions", "split_factors", "rolling_window"})
        t.append(Task(f"{algorithm}_a", algorithm, "signal", text(*a.values()), a,
                      traits=traits))
        t.append(Task(f"{algorithm}_b", algorithm, "signal", text(*b.values()), b,
                      factors="ratio", order=SHUFFLED, traits=traits))
    return t


TABLE_TAIL = ("Put that value on row t; rows without a full window are 0. Do not lag it a "
              "second time: the value on row t already uses closes through row t-1 only.")


def _returns_window(window):
    return (f"For every session t >= {window + 1}, take the {window} simple returns of the "
            f"adjusted closes ending at close t-1. ")


def _hist_vol(window):
    return (_returns_window(window) + "For each asset compute their sample standard deviation "
            "(ddof=1) times sqrt(252). " + TABLE_TAIL)


def _ewma_vol(window, decay):
    return (_returns_window(window) + f"For each asset run v = r[0]^2 on the first of those "
            f"returns, then v = {decay} * v + (1 - {decay}) * r^2 for each later return in order "
            f"(zero mean); the value is sqrt(v * 252). " + TABLE_TAIL)


def _hist_risk(window, confidence, output):
    what = ("the historical value at risk: the {c} quantile of the losses (loss = -return; "
            "numpy's default linear interpolation)" if output == "var" else
            "the historical expected shortfall: the mean of the losses (loss = -return) that are "
            "at or above their {c} quantile (numpy's default linear interpolation)")
    return (_returns_window(window) + "For each asset compute " + what.format(c=confidence)
            + ", as a positive decimal loss. " + TABLE_TAIL)


def _normal_risk(window, confidence, output):
    what = ("the normal value at risk mu + sd * z" if output == "var" else
            "the normal expected shortfall mu + sd * phi(z) / (1 - c)")
    return (_returns_window(window) + f"With loss = -return, mu the mean loss, sd the sample "
            f"standard deviation (ddof=1), c = {confidence}, z the standard normal quantile at c "
            f"and phi the standard normal density, compute {what} for each asset, as a positive "
            f"decimal loss. " + TABLE_TAIL)


def _drift(window):
    return (f"One-step forecast of each asset's adjusted close by the drift method: for every "
            f"session t >= {window}, take the {window} adjusted closes on rows t-{window} to t-1, "
            f"and extend the line through the first and last of them by one step: last + (last - "
            f"first) / {window - 1}. " + TABLE_TAIL)


def _mean(window):
    return (f"One-step forecast of each asset's adjusted close by the mean method: for every "
            f"session t >= {window}, the simple mean of the {window} adjusted closes on rows "
            f"t-{window} to t-1. " + TABLE_TAIL)


def _tables() -> list[Task]:
    level = frozenset({"positions", "split_factors", "rolling_window"})
    t = []
    for algorithm, family, (a, a_text), (b, b_text), options in (
            ("historical_volatility", "volatility",
             (dict(window=20, periods_per_year=252), _hist_vol(20)),
             (dict(window=40, periods_per_year=252), _hist_vol(40)), ("window",)),
            ("ewma_volatility", "volatility",
             (dict(window=60, decay=0.94, periods_per_year=252), _ewma_vol(60, 0.94)),
             (dict(window=40, decay=0.97, periods_per_year=252), _ewma_vol(40, 0.97)), ("window",)),
            ("historical_var_es", "risk",
             (dict(window=60, confidence=0.95, output="var"), _hist_risk(60, 0.95, "var")),
             (dict(window=40, confidence=0.975, output="expected_shortfall"),
              _hist_risk(40, 0.975, "es")), ("window", "output")),
            ("normal_var_es", "risk",
             (dict(window=60, confidence=0.99, output="var"), _normal_risk(60, 0.99, "var")),
             (dict(window=40, confidence=0.95, output="expected_shortfall"),
              _normal_risk(40, 0.95, "es")), ("window", "output")),
            ("drift", "forecast", (dict(window=30), _drift(30)), (dict(window=20), _drift(20)),
             ("window",)),
            ("mean", "forecast", (dict(window=20), _mean(20)), (dict(window=40), _mean(40)),
             ("window",))):
        traits = RETURNS if family != "forecast" else level
        for name, params, text, variant in (("a", a, a_text, {}),
                                            ("b", b, b_text, dict(factors="ratio",
                                                                  order=SHUFFLED))):
            t.append(Task(f"{algorithm}_{name}", algorithm, family, text, params,
                          {k: params[k] for k in options}, traits=traits, **variant))
    return t


def assign_splits(algorithms, *, seed: int = SPLIT_SEED, dev_share: float = 1 / 3) -> dict:
    """Algorithm -> split. E4 and qualification algorithms are dev; of the rest, a seeded
    third are dev."""
    algorithms = set(algorithms)
    rest = sorted(a for a in algorithms if a not in FORCED_DEV)
    rng = random.Random(seed)
    rng.shuffle(rest)
    n_dev = round(len(rest) * dev_share)
    split = {a: "dev" for a in algorithms if a in FORCED_DEV}
    split.update({a: ("dev" if i < n_dev else "heldout") for i, a in enumerate(rest)})
    return split


@dataclass(frozen=True)
class TaskSet:
    tasks: tuple
    splits: dict                        # algorithm -> "dev" | "heldout"

    @property
    def dev(self) -> tuple:
        return tuple(t for t in self.tasks if self.splits[t.algorithm] == "dev")

    @property
    def heldout(self) -> tuple:
        return tuple(t for t in self.tasks if self.splits[t.algorithm] == "heldout")

    def get(self, task_id: str) -> Task:
        for t in self.tasks:
            if t.id == task_id:
                return t
        raise KeyError(f"no task {task_id!r}")

    @property
    def sha256(self) -> str:
        body = dict(tasks=[t.sha256 for t in self.tasks], splits=self.splits)
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def suite() -> TaskSet:
    """The current task suite. PROVISIONAL until the benchmark protocol freezes it."""
    tasks = tuple(_e4() + _new() + _tables())
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate task ids")
    return TaskSet(tasks, assign_splits(t.algorithm for t in tasks))


# The suite the frozen RSI study ran on Beacon (2026-11-08 on) and its follow-up P-adopt: the 18
# position tasks that existed before the suite grew to 34, with that study's split. The tasks
# are the same objects as in suite() (same manifests and hashes); only the selection and the
# split differ (suite() now puts bollinger_reversion in dev).
FROZEN_STUDY_SPLITS = {
    "inverse_volatility": "dev", "cross_sectional_momentum": "dev", "ma_crossover": "dev",
    "vol_target_momentum": "dev", "momentum": "dev", "macd": "dev",
    "bollinger_reversion": "heldout", "rsi_reversion": "heldout", "donchian_breakout": "heldout"}
FROZEN_STUDY_SHA256 = "a000f661cadc1b27035dbf8f8cc3ce327f9e5b601ed774e7d958e2bcf644a44e"


def frozen_suite() -> TaskSet:
    """The frozen study's 18-task suite (12 dev, 6 held-out); checked against its hash."""
    tasks = tuple(t for t in suite().tasks if t.algorithm in FROZEN_STUDY_SPLITS)
    frozen = TaskSet(tasks, dict(FROZEN_STUDY_SPLITS))
    if len(tasks) != 18 or frozen.sha256 != FROZEN_STUDY_SHA256:
        raise RuntimeError("the frozen study's tasks changed; its suite is not reproducible")
    return frozen

