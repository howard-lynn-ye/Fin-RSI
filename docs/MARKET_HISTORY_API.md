# Explicit market history

`fin_skills.data.market_snapshot` returns one session of price levels. It explicitly
reports zero return samples and is unsuitable for return estimation.
`prepare_history` returns complete, labelled pandas matrices for calculation:

```python
import pandas as pd
from fin_skills.data import prepare_history

quotes = pd.read_csv("quotes.csv")
actions = pd.read_csv("corporate_actions.csv")
h = prepare_history(
    quotes, actions, as_of="2025-01-02", lookback=252,
    input_adjustment="raw", actions_complete=True,
)
returns = h["returns"]
prices = h["prices"]
print(h["metadata"])
```

The caller must establish the raw-price convention and complete visible action
coverage before declaring them. An empty event table is not evidence of no actions.
For already total-return-adjusted closes, use `input_adjustment="total_return"`
without actions. Neither convention is guessed. Date filtering does not establish
historical publication availability or protect against revised vendor data.

`lookback=252` means 252 returns and requires 253 common sessions ending at `as_of`.
Missing, nonfinite, nonpositive or duplicate prices, unknown tickers and insufficient
history fail explicitly. No padding, forward filling or silent shorter window occurs.
The API cannot detect a session missing from every asset without an exchange calendar.

For raw closes, gross daily returns are `(close * split_ratio + cash_dividend) /
previous_close`. Cash dividends are reinvested at the ex-date close. The returned price
matrix is a total-return index anchored to the last raw close. Same-day mixed action
types, duplicate actions and unsupported events fail rather than guessing per-share
conventions. This method differs from v5's backward dividend-factor approximation;
existing v5 returns and frozen source must not be replaced or relabelled.

## Agent adapter

The opt-in `trading_runtime_v6.Controller` supplies:

- `read_snapshot`: raw current levels, explicitly not history.
- `read_market`: raw inspection pages with page-session counts.
- Library `read_history`: bounded JSON pages of validated returns for inspection.
- Library Python `load_history`: complete matrices in memory for calculations.
- Library `run_algorithm`: the same validated input pipeline and labelled output.

Tool replies are dictionaries. In one Python call:

```python
h = load_history(lookback=252)
print(h["metadata"])
# Choose an algorithm yourself; no method is selected by the history reader.
reply = run_algorithm(algorithm_id=chosen_method, lookback=252)
assert reply["ok"], reply
submit(reply["result"]["weights"])
```

`load_history` is Python-only so full numeric arrays do not pass through a text
truncation budget. Its successful and failed uses are recorded in tool receipts.
Misplaced `read_market(lookback=...)` and attribute-style `reply.weights` return
specific corrections. JSON history pages never stand in for the full matrices.

The new history helpers are library capabilities. The raw condition still receives
raw files and common snapshot tools. Any comparison with equal adjusted inputs must
declare that input treatment separately. Use a new frozen experimental protocol for
v6; no v5 or factorial run is resumed with this code. CPU tests establish interface
and arithmetic properties, not model uptake, improved decisions, or higher returns.
