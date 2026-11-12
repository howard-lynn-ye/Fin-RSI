"""fin_skills.tasks.real - the real-price panel (AMZN, SHOP, GOOGL, TSLA through their 2022
splits). The prices are not shipped, so tests that need them skip when the file is absent;
the rest run on toy frames."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from fin_skills import rsi
from fin_skills.tasks import Judge, library_positions, panel, reference, suite
from fin_skills.tasks import real
from fin_skills.tasks.data import ASSETS, REAL, as_seed, sampling_seed

from test_tasks_suite import BUGS, TABLE_BUGS, VOL, _library_program

TASKS = suite()
have_panel = pytest.mark.skipif(not real.default_path().exists(),
                                reason="real-price panel not fetched (python -m fin_skills.tasks "
                                       "real-fetch)")


def test_letters_are_the_task_assets():
    assert real.LETTERS == ASSETS


def test_seeds_are_integers_or_real():
    assert as_seed(11) == 11 and as_seed(REAL) == REAL == "real"
    assert sampling_seed(REAL) == 2022 and sampling_seed(23) == 23
    for bad in (True, "11", 1.5, "synthetic"):
        with pytest.raises((ValueError, TypeError)):
            as_seed(bad)


def test_as_traded_multiplies_back_every_later_split():
    idx = pd.to_datetime(["2022-06-02", "2022-06-03", "2022-06-06", "2022-06-07"])
    served = pd.DataFrame({"AMZN": [123.15, 122.35, 124.79, 123.00]}, index=idx)
    splits = pd.DataFrame(dict(ticker=["AMZN", "AMZN"], ratio=[2.0, 20.0],
                               date=pd.to_datetime(["1999-09-02", "2022-06-06"])))
    got = real.as_traded(served, splits)["AMZN"].tolist()
    assert got == [2463.00, 2447.00, 124.79, 123.00]


def test_as_traded_refuses_a_series_that_is_not_in_cents():
    idx = pd.to_datetime(["2022-06-03", "2022-06-06"])
    served = pd.DataFrame({"AMZN": [122.3537, 124.79]}, index=idx)
    splits = pd.DataFrame(dict(ticker=["AMZN"], ratio=[20.0], date=[pd.Timestamp("2022-06-06")]))
    with pytest.raises(ValueError, match="not whole cents"):
        real.as_traded(served, splits)


def test_vendor_splits_must_match_the_verified_table():
    first, last = pd.Timestamp(real.FIRST), pd.Timestamp(real.LAST)
    rows = [dict(ticker=t, date=pd.Timestamp(d), ratio=r) for t, d, r, _ in real.SPLITS]
    real._check_vendor_splits(pd.DataFrame(rows), first, last)
    rows[0]["ratio"] = 10.0
    with pytest.raises(ValueError, match="differ"):
        real._check_vendor_splits(pd.DataFrame(rows), first, last)


def test_cumulative_factors_start_on_the_split_sessions():
    idx = pd.bdate_range(real.FIRST, real.LAST)
    f = real.cumulative(idx)
    for ticker, date, ratio, _ in real.SPLITS:
        on = idx.get_loc(pd.Timestamp(date))
        assert f[ticker].iloc[on - 1] == 1.0 and f[ticker].iloc[on] == ratio


def test_a_missing_or_altered_file_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "panel.csv"
    monkeypatch.setenv(real.ENV, str(path))
    with pytest.raises(FileNotFoundError, match="real-fetch"):
        real.load()
    path.write_text("date,AMZN,SHOP,GOOGL,TSLA\n2022-03-31,1.00,1.00,1.00,1.00\n")
    with pytest.raises(ValueError, match="SHA-256"):
        real.load()


def test_report_keeps_real_and_synthetic_apart():
    row = dict(method="NoImprove", library=True, cards=True, arm="cards+library", task_id="t",
               task_algorithm="a", valid=True, accepted=True, library_used=True,
               usage=dict(calls=1, generated_tokens=10))
    # A summary needs a complete run whose rows match its plan.
    plan = [["NoImprove", "cards+library", "t", seed] for seed in (11, 23, REAL)]
    report = rsi.Report(rows=[dict(row, seed=11), dict(row, seed=23),
                              dict(row, seed=REAL, valid=False)],
                        complete=True, protocol=dict(conditions="arms", plan=plan))
    by_data = {s["data"]: s for s in report.summary(draws=50)}
    assert by_data["synthetic"]["episodes"] == 2 and by_data["synthetic"]["valid"] == 1.0
    assert by_data["real"]["episodes"] == 1 and by_data["real"]["valid"] == 0.0
    assert " real " in report.table()


@have_panel
def test_the_file_is_the_verified_file():
    body = real.default_path().read_bytes()
    assert hashlib.sha256(body).hexdigest() == real.SHA256


@have_panel
def test_rebuilt_quotes_match_published_pre_split_closes():
    raw = real.load()
    for ticker, (date, close, _) in real.PRE_SPLIT_CLOSES.items():
        assert raw.loc[date, ticker] == close
        split = next(d for t, d, _, _ in real.SPLITS if t == ticker)
        assert raw.index[raw.index.get_loc(pd.Timestamp(date)) + 1] == pd.Timestamp(split)


@have_panel
def test_layout_matches_the_synthetic_panels():
    raw, factors, adjusted = panel(REAL, factors="ratio", order=("C", "A", "D", "B"))
    syn = panel(11, factors="ratio", order=("C", "A", "D", "B"))
    assert raw.shape == syn[0].shape and list(raw.columns) == ["GOOGL", "AMZN", "TSLA", "SHOP"]
    assert (factors.to_numpy() != 1).sum() == 4                  # four split events
    assert raw.index.get_loc(pd.Timestamp("2022-06-06")) == 45   # Amazon's, like asset A's
    _, cum, adj = panel(REAL)
    np.testing.assert_allclose(factors.cumprod().to_numpy(), cum[list(raw.columns)].to_numpy(),
                               rtol=0, atol=0)
    assert adjusted.pct_change().abs().max().max() < 0.2          # no split left in adjusted
    assert raw.pct_change().abs().max().min() > 0.6               # every raw series jumps
    assert panel(REAL, rows=96)[0].equals(panel(REAL)[0].iloc[:96])


@have_panel
@pytest.mark.parametrize("task", TASKS.tasks, ids=lambda t: t.id)
def test_reference_matches_library_on_the_real_panel(task):
    raw, factors, _ = panel(REAL, rows=task.rows, factors=task.factors, order=task.order)
    lib = library_positions(task, raw, factors).to_numpy()
    atol = 1e-12 if task.tolerance <= 1e-8 else task.tolerance / 5
    np.testing.assert_allclose(lib, reference(task, REAL), atol=atol, rtol=0)


@have_panel
@pytest.mark.parametrize("task", TASKS.heldout, ids=lambda t: t.id)
def test_a_correct_program_passes_every_check_on_the_real_panel(task):
    code = _library_program(task)
    judge = Judge(task, seed=REAL, library=True)
    seen = judge.check(code)
    assert seen.accepted, (seen.error, seen.checks)
    assert judge.grade(code)


@have_panel
@pytest.mark.parametrize("bug", sorted(BUGS))
def test_planted_defects_fail_their_check_on_the_real_panel(bug):
    code, check = BUGS[bug]
    judge = Judge(TASKS.get("macd_b"), seed=REAL, library=False)
    seen = judge.check(code)
    assert seen.executed and seen.checks[check] is False and not judge.grade(code)


@have_panel
@pytest.mark.parametrize("bug", sorted(TABLE_BUGS))
def test_table_defects_fail_their_check_on_the_real_panel(bug):
    code, check = TABLE_BUGS[bug]
    judge = Judge(TASKS.get("historical_volatility_a"), seed=REAL, library=False)
    assert judge.check(VOL).accepted and judge.grade(VOL)
    seen = judge.check(code)
    assert seen.executed and seen.checks[check] is False and not judge.grade(code)
