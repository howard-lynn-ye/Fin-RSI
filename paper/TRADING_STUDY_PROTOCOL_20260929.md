# Out-of-sample trading study on real data — protocol (frozen before inference, 2026-09-29)

Question: does an agent that can use fin-skills earn higher net out-of-sample returns than
the same agent without it? This is the paper's main line; the code-audit studies (E4) become
supporting evidence about why returns computed by unassisted code cannot be trusted.

Data: Yahoo Finance daily bars for 12 US-listed instruments (SPY, QQQ, EFA, EEM, TLT, IEF,
GLD, DBC, NVDA, AVGO, ORLY, FAST), 2022-10-03 to 2026-09-25, downloaded 2026-09-29 and stored
verbatim (`yahoo_raw_20260929.json`, SHA-256 recorded in `protocol.json`). The agent sees a
vendor-style feed: raw un-adjusted OHLCV quotes plus a dated table of splits and cash
dividends (`market_data.py`; a test shows the visible files reproduce Yahoo's total-return
series to 5e-6). Four splits fall inside the history, two of them (FAST 2:1, ORLY 15:1)
inside the test window. The hidden ledger uses total-return closes.

Window: decisions every 10 sessions from the first session of 2025 (44 decisions); a target
submitted at decision session t executes at the close of t+1 and is held to the next trade.
Qwen2.5-Coder models were released in 2024, so 2025-2026 prices postdate their training
data by public statements; contamination cannot be excluded with certainty.

Arms (identical prompt, data, turn budget of 8, 1,024 output tokens per turn, temperature
0.1, top-p 1, seeds 11/23/37 varying model sampling only):
- `raw`: read_file, run_python (numpy/pandas/scipy, sandboxed, 60 s, no fin_skills), submit.
- `library`: the same plus fin_skills importable in run_python, list/describe_algorithm,
  run_algorithm (the library builds split- and dividend-adjusted inputs from the visible
  files and returns weights or lagged signals), run_guard (adjustment_check, data_quality),
  read_skill, and a short catalog description in the prompt.
Long-only, weights >= 0 summing to at most 1, remainder cash at zero yield; 5 bps per side on
traded notional; an invalid or missing submission leaves holdings unchanged (recorded as a
non-submission). Models: Qwen2.5-Coder 7B, 14B and 32B, one Beacon job each.

Primary endpoint: net Sharpe ratio and net cumulative return per path over the test window,
paired library minus raw by (model, seed), with a block bootstrap over holding-period
returns. Secondary: submission rate, library calls per decision, turnover, max drawdown,
tokens, seconds, parse failures. Baselines: cash, equal-weight buy-and-hold, equal-weight
rebalanced every decision, 60/40 SPY/IEF rebalanced.

Qualification before inference (no model): sandbox denies network and writes, the raw arm
cannot import fin_skills, library tools return adjusted weights and signals, the adjustment
guard flags a raw split series, the ledger prices baselines; a scripted agent reproduces
the equal-weight baseline exactly through controller, ledger and scorer (slow test).

Limits: one real market path, so seeds are not independent market samples; ~21 months of
data; daily closes without intraday execution, slippage or capacity; the library arm reads
more documentation, so tools and text vary together; three model sizes from one family; an
unchanged holding after a failed decision is a design choice. Results are reported for the
complete hash-verified batch of each model; an interrupted job is recorded, never rerun over
existing decisions. A negative or null result is reported as such.
