"""Real US daily market data for the trading study, rebuilt deterministically from a raw download.

Source: Yahoo Finance chart API (`v8/finance/chart`, interval 1d, events div and split),
downloaded 2026-09-29 for 12 instruments from 2022-10-03 to 2026-09-25 and saved verbatim
as `yahoo_raw_20260929.json`. Yahoo's `close` is split-adjusted; `adjclose` is split- and
dividend-adjusted. The agent-visible files undo the split adjustment so that quotes look
like a raw vendor feed, and list splits and cash dividends as dated corporate actions.
The hidden ledger uses `adjclose` (total return). `adjust()` rebuilds a total-return series
from the visible files only; a test checks it against `adjclose`.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE/'data'/'market-us-20260929'
RAW = DATA_DIR/'yahoo_raw_20260929.json'
TICKERS = ('SPY', 'QQQ', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'DBC', 'NVDA', 'AVGO', 'ORLY',
           'FAST')
VISIBLE = ('quotes.csv', 'corporate_actions.csv', 'README.md')
HIDDEN = 'total_return_close.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _date(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()


def build(raw_path=RAW):
    """Return (quotes long frame, actions frame, hidden total-return wide frame)."""
    raw = json.loads(Path(raw_path).read_text(encoding='utf-8'))
    quotes, actions, adjusted = [], [], {}
    for ticker in TICKERS:
        result = raw[ticker]['chart']['result'][0]
        dates = [_date(t) for t in result['timestamp']]
        quote = result['indicators']['quote'][0]
        adj = result['indicators']['adjclose'][0]['adjclose']
        events = result.get('events', {})
        splits = {}
        for item in events.get('splits', {}).values():
            num, den = item['splitRatio'].split(':')
            splits[_date(item['date'])] = float(num)/float(den)
        dividends = {_date(item['date']): float(item['amount'])
                     for item in events.get('dividends', {}).values()}
        # Yahoo divides pre-split prices by the ratio; multiply back for dates before it.
        factor = pd.Series(1.0, index=pd.Index(dates))
        for day, ratio in splits.items():
            factor[factor.index < day] *= ratio
        for i, day in enumerate(dates):
            f = factor.iloc[i]
            quotes.append(dict(date=day, ticker=ticker, open=round(quote['open'][i]*f, 6),
                               high=round(quote['high'][i]*f, 6), low=round(quote['low'][i]*f, 6),
                               close=round(quote['close'][i]*f, 6),
                               volume=int(round(quote['volume'][i]/f))))
            adjusted[(day, ticker)] = adj[i]
        for day, ratio in splits.items():
            actions.append(dict(date=day, ticker=ticker, kind='split', value=ratio,
                                note=f'{ratio:g} new shares per old share; quotes before this '
                                     'date are on the old share basis'))
        for day, amount in dividends.items():
            # Dividend amounts are reported by Yahoo on the split-adjusted basis.
            f = float(factor.get(day, 1.0))
            actions.append(dict(date=day, ticker=ticker, kind='dividend',
                                value=round(amount*f, 6), note='cash per share, ex-date'))
    quotes = pd.DataFrame(quotes).sort_values(['date', 'ticker']).reset_index(drop=True)
    actions = pd.DataFrame(actions).sort_values(['date', 'ticker', 'kind']).reset_index(drop=True)
    total = pd.Series(adjusted).unstack()
    total.index.name = 'date'
    total = total[list(TICKERS)]
    return quotes, actions, total


README = """# Market data (raw vendor-style feed)

`quotes.csv`: daily open, high, low, close and volume for 12 US-listed instruments, as
quoted on each day. Quotes are NOT adjusted for splits or dividends.
`corporate_actions.csv`: dated splits (value = new shares per old share) and cash dividends
(value = cash per share on the ex-date). A split changes the quote basis on its date, so
returns computed across a split date from raw quotes are wrong unless adjusted.
Dates are exchange sessions (US). All files end at the current decision date.
"""


def write_dataset(out_dir, raw_path=RAW):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quotes, actions, total = build(raw_path)
    quotes.to_csv(out_dir/'quotes.csv', index=False, lineterminator='\n')
    actions.to_csv(out_dir/'corporate_actions.csv', index=False, lineterminator='\n')
    total.to_csv(out_dir/HIDDEN, float_format='%.6f', lineterminator='\n')
    (out_dir/'README.md').write_text(README, encoding='utf-8', newline='\n')
    return {name: sha(out_dir/name) for name in (*VISIBLE, HIDDEN)}


def load_visible(data_dir):
    data_dir = Path(data_dir)
    quotes = pd.read_csv(data_dir/'quotes.csv')
    actions = pd.read_csv(data_dir/'corporate_actions.csv')
    return quotes, actions


def adjust(quotes, actions, through=None):
    """Total-return-adjusted close (wide frame) from the visible files only.

    Splits: divide quotes before the split date by the ratio. Dividends: standard
    back-adjustment, multiplying prices before the ex-date by (1 - D / P_prev).
    """
    close = quotes.pivot(index='date', columns='ticker', values='close').sort_index()
    if through is not None:
        close = close[close.index <= through]
        actions = actions[actions['date'] <= through]
    out = close.copy()
    for _, a in actions.sort_values('date').iterrows():
        before = out.index < a['date']
        if a['ticker'] not in out.columns or not before.any():
            continue
        if a['kind'] == 'split':
            out.loc[before, a['ticker']] /= a['value']
        elif a['kind'] == 'dividend':
            prev = out.loc[before, a['ticker']].iloc[-1]
            out.loc[before, a['ticker']] *= 1 - a['value']/prev
    return out


def truncate(data_dir, out_dir, through):
    """Copy the visible files with every row after `through` removed (no lookahead)."""
    quotes, actions = load_visible(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quotes[quotes['date'] <= through].to_csv(out_dir/'quotes.csv', index=False,
                                             lineterminator='\n')
    actions[actions['date'] <= through].to_csv(out_dir/'corporate_actions.csv', index=False,
                                               lineterminator='\n')
    (out_dir/'README.md').write_text(README, encoding='utf-8', newline='\n')


if __name__ == '__main__':
    print(json.dumps(write_dataset(DATA_DIR), indent=1))
