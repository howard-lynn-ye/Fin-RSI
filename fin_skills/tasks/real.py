"""A real-price panel: four large caps through their 2022 stock splits.

AMZN, SHOP, GOOGL and TSLA each split within twelve weeks in mid-2022. The panel holds their
as-traded daily closes over 128 sessions, so the four splits fall inside it the way the
synthetic splits do (Amazon's on row 45, like asset A's). Columns are the tickers; asset
letters A-D map to them in that order, so a task's column order applies unchanged.

Prices come from Yahoo Finance through yfinance. Yahoo's terms forbid redistribution, so the
repository ships the split table, the window, this fetch code and the SHA-256 of the file it
writes, never the prices themselves:

    python -m fin_skills.tasks real-fetch [PATH]    # writes the panel, prints its SHA-256
    python -m fin_skills.tasks real-check [PATH]    # verifies an existing file
    python real.py fetch [PATH]                     # the same, with this file on its own

Set FIN_TASKS_REAL_PANEL to the file, or keep it at the default path below.

Yahoo's "Close" with auto_adjust=False is split-adjusted, not as traded. The fetch rebuilds
the as-traded close by multiplying back every later split Yahoo reports, requires Yahoo's
splits inside the window to equal SPLITS below, and rounds to cents only when every value is
already within half a tenth of a cent of one.

This module imports only the standard library, numpy and pandas (yfinance inside fetch), so
the file can run on its own on a machine that has no fin_skills installed.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

NAME = "real"                                  # the "seed" that selects this panel
TICKERS = ("AMZN", "SHOP", "GOOGL", "TSLA")
LETTERS = ("A", "B", "C", "D")                 # fin_skills.tasks.data.ASSETS
ROWS = 128
ANCHOR = ("AMZN", 45)                          # Amazon's split lands on row 45

# (ticker, first session traded on a split-adjusted basis, ratio, source). Checked 2026-11-05.
SPLITS = (
    ("AMZN", "2022-06-06", 20.0,
     "Amazon 20-for-1, record date 2022-05-27, split-adjusted trading from 2022-06-06 "
     "(SEC filing as reported by Nasdaq, 2022-03-10; Morningstar, 2022-06-07)"),
    ("SHOP", "2022-06-29", 10.0,
     "Shopify 10-for-1, record date 2022-06-22, split-adjusted trading from 2022-06-29 "
     "(Benzinga, 2022-06-29; The Motley Fool, 2022-06-16)"),
    ("GOOGL", "2022-07-18", 20.0,
     "Alphabet 20-for-1, record date 2022-07-01, shares distributed after the close on "
     "2022-07-15, split-adjusted trading from 2022-07-18 (Investing.com, 2022-06-06)"),
    ("TSLA", "2022-08-25", 3.0,
     "Tesla 3-for-1, record date 2022-08-17, split-adjusted trading from 2022-08-25 "
     "(Tesla press release, SEC Form 8-K Exhibit 99.1, 2022-08-05)"),
)

FIRST, LAST = "2022-03-31", "2022-10-03"      # from the fetch of 2026-11-05 (yfinance 1.7.0)
SHA256 = "16f3f83be8bd7b7baf3af63093f90ffd4e76b1125405f05a7eb38bc8d521260a"

# The as-traded close on the last session before each split, as published elsewhere, so the
# rebuilt quotes can be checked against something other than the vendor that produced them.
PRE_SPLIT_CLOSES = {
    "AMZN": ("2022-06-03", 2447.00, "AlphaSpread split history"),
    "SHOP": ("2022-06-28", 350.26, "FXStreet, 2022-06-29; AlphaSpread split history"),
    "GOOGL": ("2022-07-15", 2235.55, "TickerLeague split history ($2,236); Capway"),
    "TSLA": ("2022-08-24", 891.29, "Forex.com, 2022; AlphaSpread split history"),
}
VENDOR = "Yahoo Finance via yfinance, auto_adjust=False, Close"
TERMS = "https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html"
FETCH_RANGE = ("2022-01-03", "2022-12-31")     # wide enough to place the 128-row window
ENV = "FIN_TASKS_REAL_PANEL"
DEFAULT_PATH = Path.home() / ".cache" / "fin-skills" / "real_panel_2022.csv"


def default_path() -> Path:
    return Path(os.environ.get(ENV) or DEFAULT_PATH)


def _canonical(raw: pd.DataFrame) -> bytes:
    """The exact bytes the hash is taken over: date and one close in cents per ticker."""
    out = io.StringIO()
    out.write("date," + ",".join(TICKERS) + "\n")
    for date, row in raw[list(TICKERS)].iterrows():
        out.write(pd.Timestamp(date).strftime("%Y-%m-%d") + ","
                  + ",".join(f"{v:.2f}" for v in row.to_numpy()) + "\n")
    return out.getvalue().encode()


def as_traded(close: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Split-adjusted closes -> as-traded closes, rounded to cents.

    close: date x ticker, adjusted for every split up to today (Yahoo's Close).
    splits: rows of (ticker, date, ratio), every split the vendor reports, any date.
    The close on date t is multiplied by each split of that ticker effective after t.
    """
    out = close.astype(float).copy()
    for ticker in out.columns:
        for _, s in splits[splits["ticker"] == ticker].iterrows():
            later = out.index < pd.Timestamp(s["date"])
            out.loc[later, ticker] *= float(s["ratio"])
    cents = out.round(2)
    off = float(np.nanmax(np.abs(out.to_numpy() - cents.to_numpy())))
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("a close is missing or not finite")
    if off > 0.0005:
        raise ValueError(f"rebuilt closes are not whole cents (off by up to {off:.4f}); "
                         f"the vendor series is not the split-adjusted close it is said to be")
    return cents


def window(raw: pd.DataFrame) -> pd.DataFrame:
    """The ROWS sessions that put ANCHOR's split on its row."""
    ticker, row = ANCHOR
    date = pd.Timestamp(next(d for t, d, _, _ in SPLITS if t == ticker))
    at = raw.index.get_loc(date)
    if at < row or at - row + ROWS > len(raw):
        raise ValueError("the fetched range does not cover the window")
    return raw.iloc[at - row: at - row + ROWS]


def _check_vendor_splits(splits: pd.DataFrame, first, last) -> None:
    inside = splits[(splits["date"] >= first) & (splits["date"] <= last)]
    got = sorted((t, pd.Timestamp(d).strftime("%Y-%m-%d"), float(r))
                 for t, d, r in inside[["ticker", "date", "ratio"]].itertuples(index=False))
    want = sorted((t, d, r) for t, d, r, _ in SPLITS)
    if got != want:
        raise ValueError(f"vendor splits inside the window {got} differ from SPLITS {want}")


def _check_jumps(raw: pd.DataFrame) -> None:
    """Each split shows as a jump in the as-traded closes and vanishes once adjusted."""
    for ticker, date, ratio, _ in SPLITS:
        at = raw.index.get_loc(pd.Timestamp(date))
        move = raw[ticker].iloc[at] / raw[ticker].iloc[at - 1]
        if not (abs(move * ratio - 1) < 0.25 and abs(move - 1) > 0.5):
            raise ValueError(f"{ticker} on {date}: as-traded close moved by x{move:.4f}; a "
                             f"{ratio:g}-for-1 split should move it by about x{1 / ratio:.4f}")


def fetch(path: str | Path | None = None) -> str:
    """Download, rebuild as-traded closes, write the panel file, return its SHA-256."""
    import yfinance as yf
    start, end = FETCH_RANGE
    frame = yf.download(list(TICKERS), start=start, end=end, interval="1d", auto_adjust=False,
                        actions=False, progress=False, threads=False)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    close = frame["Close"][list(TICKERS)]
    rows = []
    for t in TICKERS:
        s = yf.Ticker(t).splits                   # full history, so later splits are undone
        for d, r in s.items():
            rows.append(dict(ticker=t, date=pd.Timestamp(d).tz_localize(None).normalize(),
                             ratio=float(r)))
    splits = pd.DataFrame(rows, columns=["ticker", "date", "ratio"])
    raw = window(as_traded(close, splits))
    _check_vendor_splits(splits, raw.index[0], raw.index[-1])
    _check_jumps(raw)
    body = _canonical(raw)
    path = Path(path) if path is not None else default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    sidecar = dict(vendor=VENDOR, terms=TERMS, yfinance=getattr(yf, "__version__", ""),
                   fetched_range=FETCH_RANGE, first=str(raw.index[0].date()),
                   last=str(raw.index[-1].date()), rows=len(raw),
                   vendor_splits=[dict(ticker=t, date=str(pd.Timestamp(d).date()), ratio=r)
                                  for t, d, r in splits.itertuples(index=False)],
                   sha256=hashlib.sha256(body).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    served = frame[["Close", "Adj Close"]].loc[raw.index[0]:raw.index[-1]]
    served.to_csv(path.with_name(path.stem + "_vendor.csv"))     # as served, for audit only
    return sidecar["sha256"]


def load(path: str | Path | None = None, *, verify: bool = True) -> pd.DataFrame:
    """The as-traded closes, date x ticker. Raises with the fetch command when absent."""
    path = Path(path) if path is not None else default_path()
    if not path.exists():
        raise FileNotFoundError(
            f"the real-price panel is not at {path}. Fetch it with "
            f"`python -m fin_skills.tasks real-fetch {path}` (Yahoo's terms forbid "
            f"redistributing the prices, so they are not shipped) or set {ENV}.")
    body = path.read_bytes()
    if verify and SHA256 and hashlib.sha256(body).hexdigest() != SHA256:
        raise ValueError(f"{path} has SHA-256 {hashlib.sha256(body).hexdigest()}, expected "
                         f"{SHA256}; the vendor may have revised its history")
    raw = pd.read_csv(io.BytesIO(body), index_col="date", parse_dates=True)
    if list(raw.columns) != list(TICKERS) or len(raw) != ROWS:
        raise ValueError(f"{path} must hold {ROWS} rows of {list(TICKERS)}")
    if verify and FIRST and (str(raw.index[0].date()), str(raw.index[-1].date())) != (FIRST, LAST):
        raise ValueError(f"{path} covers {raw.index[0].date()}..{raw.index[-1].date()}, "
                         f"expected {FIRST}..{LAST}")
    return raw.astype(float)


def cumulative(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Product of every SPLITS ratio effective on or before each date."""
    f = pd.DataFrame(1.0, index=index, columns=list(TICKERS))
    for ticker, date, ratio, _ in SPLITS:
        f.loc[f.index >= pd.Timestamp(date), ticker] *= ratio
    return f


def panel(*, rows: int = ROWS, factors: str = "cumulative", order=LETTERS, path=None
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(raw prices, split factors as given to the program, adjusted closes), like
    fin_skills.tasks.data.panel. Adjusted closes are anchored at the first row:
    raw * cumulative factor, so the program's adjustment reproduces them exactly."""
    if sorted(order) != sorted(LETTERS):
        raise ValueError(f"order must be a permutation of {LETTERS}")
    if not 60 <= rows <= ROWS:
        raise ValueError(f"rows must be between 60 and {ROWS}")
    raw = load(path).iloc[:rows]
    cum = cumulative(raw.index)
    given = cum.copy()
    if factors == "ratio":
        given.iloc[1:] = cum.iloc[1:].to_numpy() / cum.iloc[:-1].to_numpy()
    elif factors != "cumulative":
        raise ValueError("factors must be 'cumulative' or 'ratio'")
    columns = [TICKERS[LETTERS.index(a)] for a in order]
    raw.index.name = given.index.name = None
    return raw[columns], given[columns], (raw * cum)[columns]


def main(argv) -> int:
    command = argv[0] if argv else "check"
    target = argv[1] if len(argv) > 1 else None
    if command == "fetch":
        print(fetch(target))
        return 0
    if command == "check":
        raw = load(target)
        print(f"ok: {len(raw)} rows {raw.index[0].date()}..{raw.index[-1].date()}, sha256 "
              f"{hashlib.sha256((Path(target) if target else default_path()).read_bytes()).hexdigest()}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
