"""Protocol tests use invented fixture records, never real study decisions."""
import pytest

pytest.importorskip("bs4")
from benchmarks.agent_study.manual_multisource import (
    ManualStudy, asof_records, cot_eligible, validate_weights, write_new,
)


def record(**updates):
    r = dict(id="fixture", category="news", series="test", event_date="2025-01-01",
             eligible_date="2025-01-03", vintage="current_archive_historical_version_unverified",
             availability_basis="dated_official_page_plus_one_day")
    return dict(r, **updates)


def test_future_release_cannot_enter_earlier_packet():
    assert asof_records([record()], "2025-01-02") == []
    assert len(asof_records([record()], "2025-01-03")) == 1


def test_strict_mode_does_not_relabel_backfill_as_historical_observation():
    assert asof_records([record()], "2025-01-03", strict=True) == []
    verified = record(vintage="verified_historical_version")
    assert len(asof_records([verified], "2025-01-03", strict=True)) == 1


def test_shutdown_data_not_available_by_report_date_plus_three():
    assert cot_eligible("2025-09-30") == "2025-11-20"
    assert cot_eligible("2025-11-10") == "2025-12-11"
    assert cot_eligible("2025-12-23") == "2026-01-06"


def test_revised_record_enters_only_after_its_own_eligibility():
    revised = record(id="revision", eligible_date="2025-01-10")
    assert asof_records([record(), revised], "2025-01-09")[0]["id"] == "fixture"
    assert asof_records([record(), revised], "2025-01-10")[0]["id"] == "revision"


def test_stale_is_explicit_not_silently_fresh():
    assert asof_records([record()], "2025-05-01")[0]["stale"]


@pytest.mark.parametrize("bad", [{"SPY": float("nan")}, {"SPY": -0.1},
    {"SPY": 0.6, "QQQ": 0.5}, {"SPY": True}, {"NOT_REAL": 1}, {}])
def test_invalid_weights_rejected(bad):
    with pytest.raises(ValueError):
        validate_weights(bad)


def test_hold_and_cash_are_distinct():
    assert validate_weights(None) is None
    assert sum(validate_weights({"SPY": 0}).values()) == 0


def test_append_only_and_chronological_arm_order(tmp_path):
    path = tmp_path / "00-raw.json"
    write_new(path, {"fixture": True})
    with pytest.raises(FileExistsError):
        write_new(path, {"fixture": False})
    write_new(tmp_path / "00-library.json", {})
    study = object.__new__(ManualStudy)
    study.decisions = tmp_path
    assert [p.name for p in study.locked()] == ["00-raw.json", "00-library.json"]


def test_partial_score_is_rejected_before_reading_hidden_prices(tmp_path):
    study = object.__new__(ManualStudy)
    study.decisions, study.dates = tmp_path, ["2025-01-02"]
    with pytest.raises(ValueError, match="partial-path"):
        study.score()


def test_changed_calendar_rejected_on_restore(tmp_path):
    import json
    study = object.__new__(ManualStudy)
    study.root, study.dates, study.sessions = tmp_path, ["2025-01-02"], ["2025-01-06"]
    (tmp_path / "protocol.json").write_text(json.dumps({"dates": ["2025-01-03"],
                                                       "valuation_end": "2025-01-06"}))
    with pytest.raises(ValueError, match="calendar changed"):
        study.verify_protocol()


def test_actual_holdings_use_same_prices_as_score(tmp_path):
    import json
    import pandas as pd
    from benchmarks.agent_study.manual_multisource import TICKERS
    dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
    total = pd.DataFrame(100., index=dates, columns=TICKERS)
    total.loc[dates[-1], "SPY"] = 110.
    total.to_csv(tmp_path / "total_return_close.csv")
    decisions = tmp_path / "decisions"
    decisions.mkdir()
    weights = validate_weights({"SPY": .5})
    (decisions / "00-raw.json").write_text(json.dumps(dict(
        index=0, arm="raw", weights=weights)))
    study = object.__new__(ManualStudy)
    study.data_dir, study.decisions = tmp_path, decisions
    study.sessions, study.dates, study.picks = dates, dates[:1], [0]
    # No visible quotes file is supplied: this must come from the accounting prices.
    assert study.holdings("raw", dates[-1])["SPY"] == pytest.approx(.55/1.05)
