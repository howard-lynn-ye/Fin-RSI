# 32B full-cycle comparison, October 17, 2026

The completed paired cycle produced **38.00% with fin-skills** and
**18.98% without fin-skills**, a difference of
**19.03 percentage points** after modeled trading costs.

| Arm | Initial capital (USD) | Ending capital (USD) | Return Rate | Decisions |
| --- | ---: | ---: | ---: | ---: |
| Without library | 100000.00 | 118976.93 | 18.976928% | 44/44 |
| With library | 100000.00 | 138003.85 | 38.003847% | 44/44 |

`Return Rate = 100 * (ending capital / initial capital - 1)`.
Both arms have zero failed final decisions. Raw submitted 43 allocations and one hold;
library submitted 44 allocations. The library arm executed library capabilities in all
44 periods. This is execution evidence, not proof of understanding or use of every skill.

## Frozen experiment and completion

- Qwen/Qwen2.5-Coder-32B-Instruct, revision
  `381fc969f78efac66bc87ff7ddeadb7e73c218a7`, BF16, seed 11.
- Execution window: 2025-01-03 through 2026-09-25; first observation: 2025-01-02.
- Twelve allowed instruments, decisions every ten sessions, next-session-close execution,
  5 bps per traded side, zero cash interest. Both arms receive the same dated market data
  and evidence packet, with an equal six-research plus two-final-response budget.
- Frozen arm order: library, then raw. RSI disabled. No paid model calls.
- Job 2317005 completed at 2026-10-17 08:02 UTC (04:02 America/New_York), exit 0.
  Actual runtime: 05:46:15 on two L40S GPUs, four CPUs, 64 GiB host RAM.
  This window has no remaining GPU allocation. Other windows' jobs were untouched.
- Source lineage: frozen source `f36c3f63e317e6990b6692f6771bdb333b72b181`
  with only the benchmark loader replaced by the tested PR #57 version. The complete
  source and input hashes are bound in protocol.json. The active experiment was not
  changed after it started.

## Full accounting records

- `ledgers.json`: all 88 dated decisions, target allocations, holdings before each
  decision, source-record hashes and library-use receipts; 434 NAV rows per arm
  (initial capital plus 433 valuations); all 44 execution/hold rows per arm.
- NAV is normalized to initial capital. Multiply it by USD 100,000 for dollar equity.
  Trade `cost` is the fee fraction of pre-trade equity, not a dollar amount;
  `turnover` is the absolute risky-asset weight change. These are simulator records,
  not broker fills or actual stock-share statements.
- `scores.json`, `summary.json`: unrounded returns, final capital, usage and failure counts.
- `independent-model-audit.json`: separate units/cash replay agreed with the original
  daily NAV within 1.1e-14 and independently reproduced both returns.
- `inference-receipt.json`, `completed.json`, `protocol.json`: hashes, frozen settings
  and completion bindings. The final decision files themselves stay in the private
  run because their conversations embed market/news source data. Their full derived
  decisions and accounting are included in ledgers.json.
- `publication-checks.json`: fresh read-only replay, original-export equality, full
  ledger coverage, return identity, privacy checks and file hashes.
- `build-validation.txt`: build_index, build_package, validate passed on Slurm.

## Remaining limitations

This is one model, one seed and one development market path. It is not a new
20-model campaign, an unseen holdout, or evidence that every model benefits.
Historical news/disclosure availability uses stated archive-lag assumptions, not
verified point-in-time vintages; installed skill knowledge is current. The treatment
combines library knowledge, retrieval and execution interfaces, so this result does
not isolate the contribution of any one feature.

Research still had 21 unsuccessful turns with the library and 33 without it.
Five library responses were truncated. The remaining failed research turns have
parse label `json`: their JSON was recognized, but the research operation failed.
That label is not evidence of a JSON syntax error or its underlying cause.
One raw period had no successful research response but still produced a valid final
decision. No failed research turns or unfavorable periods were discarded.

Allocator OOM warnings occurred during the original job. The full job completed,
all 88 final decisions were accepted, and independent accounting passed; this does
not establish the cause or harmlessness of every research failure. Runtime and
interface issues remain distinct from the observed return difference.

## Recheck without rerunning model inference

On Slurm, with access to the unchanged frozen source and private run inputs:

```sh
python publish_results.py SOURCE PAIRS ORIGINAL_PUBLIC NEW_OUTPUT
```

Place the successful `build-validation.txt` beside the script. The audit checks
existing receipts read-only, compares its export with the original, and refuses to
overwrite an output directory. It never generates new model decisions.

Follow-up: [return attribution, market baselines and research-call audit](../20261017-research-reliability/README.md). The original results and ledgers above are unchanged.
