"""Point-in-time adjusted input shared by all cells; no model sees the oracle."""
from pathlib import Path
import json
import pandas as pd
from benchmarks.agent_study import market_data as md
from benchmarks.agent_study import trading_study as old
from benchmarks.agent_study import trading_tools_v4 as t4
TRANSFER=('XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY','SHY')
ORIGINAL=tuple(md.TICKERS)
README='''prices.csv contains date,ticker,close. close is a normalized total-return price index built only from quotes, splits and cash dividends available through as_of. It is not a tradable dollar quote. Returns, momentum and portfolio covariance can be estimated directly; do not adjust this series again. All conditions receive identical data. No future rows or hidden ledger are visible. Execution occurs at next-session close in the external ledger, at 5 bps per traded side.\n'''

def configure(tickers):
    md.TICKERS=tuple(tickers);old.TICKERS=tuple(tickers);t4.TICKERS=tuple(tickers)

def visible(data,workspace,date):
    data,workspace=Path(data),Path(workspace);workspace.mkdir(parents=True,exist_ok=False)
    q=pd.read_csv(data/'quotes.csv');a=pd.read_csv(data/'corporate_actions.csv')
    adjusted=md.adjust(q[q.date<=date],a[a.date<=date])
    adjusted=adjusted/adjusted.iloc[0]
    adjusted.index.name='date';adjusted.columns.name='ticker'
    adjusted.stack().rename('close').reset_index().to_csv(workspace/'prices.csv',index=False)
    (workspace/'README.md').write_text(README)
    assert str(adjusted.index.max())==date
    return adjusted

def write_data(raw_path,out,tickers):
    configure(tickers);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    quotes,actions,total=md.build(Path(raw_path))
    quotes.to_csv(out/'quotes.csv',index=False);actions.to_csv(out/'corporate_actions.csv',index=False)
    total.to_csv(out/md.HIDDEN)
    assert total.notna().all().all() and (total>0).all().all()
    (out/'tickers.json').write_text(json.dumps(list(tickers)))
    return total
