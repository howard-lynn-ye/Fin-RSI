# Financial knowledge and numerical tools: bounded factorial study

Created 2026-10-03 before this batch's inference and before downloading transfer assets.
Owner chat: 01a0fd91-3d08-7a53-b0a5-dd10235a6f56.
Source base: published ff7605e / merge bcf42c1, extracted from the existing publication archive.
This is a new protocol, not a relabeling of v5. Preserve all prior source, traces, scores and failures.

Question: under identical point-in-time total-return price inputs, what changes when an agent has financial skill documents, numerical algorithm tools, or both?
Four conditions (knowledge K x numerical tool T): base=(0,0), knowledge=(1,0), tools=(0,1), full=(1,1).
All conditions receive identical adjusted price CSVs, NumPy/pandas/SciPy execution, data readers, submission format, budget reminders, 8 turns and 1024 output tokens per turn. The readout names available interfaces but does not prescribe a strategy.
K exposes a frozen financial skill corpus with paginated discovery/read access. T exposes the same algorithm menu, required calling contracts and outputs. No guards are added to one factorial cell. No condition may read other conditions' data, future prices or the scoring ledger.
Minimal algorithm API documentation is part of T; K is the additional domain skill documents. Their interaction is reported. This does not isolate prose length from knowledge content.
All arms may decline tools or choose their own strategy. Natural use is an outcome, not an inclusion criterion. Scripted reference calls test technical availability separately and cannot establish model uptake.

Models: the same immutable Qwen2.5-Coder 7B/14B revisions used in v5. Seeds: 11,23,37. Long-only; cash yield zero; 5 bps per side; next-session-close execution; ten-session decisions; continuous holdings preserved after missing submission.
Primary endpoint: cumulative net return, with paired K and T effects and K-by-T interaction per model/seed. Secondary: Sharpe, drawdown, exposure, turnover, submission, discovery, actual successful calls, adoption of tool outputs, errors, tokens and time. Report all cells, not just successful submissions.
Development basket: the original twelve instruments, 2025-01-03 through 2026-09-25. Already exposed to development; not unseen.
Transfer basket selected by named taxonomy, before outcomes: the eleven Select Sector SPDR tickers XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY plus SHY. Same fixed calendar and decision schedule; no asset substitutions after seeing returns. Warmup history starts 2022-10-03. This is held-out asset transfer, NOT an independent market or an unseen time period; overlap with SPY economic exposures is explicit.
Historical skill documents are the current frozen library, not claimed to be historically available documents. Inspect the chosen corpus for asset-specific outcomes before exposing it. New periods after protocol freeze require future data and are not represented by slicing previously observed history.

Gate: CPU tests must establish cell access restrictions, identical price history, future-read denial, raw-to-adjusted parity, scripted equal-weight accounting, and successful supported algorithm calls with valid submissions. A single development date per model/cell tests model transport, with no return selection or minimum profit gate. Preserve poor adoption and all failures. After gate verification, freeze all batch receipts before full inference. Do not tune after observing this batch.
Bounded scope: two baskets x two models x three seeds x four conditions, at most three concurrent L40S jobs. No 32B/new-family or broad budget sweep in this batch. If transfer download/qualification fails, do not replace its assets or count it completed.
Baselines: cash, equal-weight hold/rebalance, 60/40 for the original basket only, inverse-volatility, and supported portfolio algorithms under fixed defaults. Baselines share costs and timing.
Uncertainty: sampling seeds are not independent markets. Do not pool 44 decisions x seeds as independent observations. Report per-seed effects and dependence-aware paired time-block intervals as descriptive sensitivity, not a broad population significance claim.
