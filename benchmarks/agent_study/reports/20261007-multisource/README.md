# Multisource trading results: October 7, 2026 snapshot

Eleven models completed all three paired seeds by the verification cutoff,
October 7 at 16:42:10 UTC (12:42:10 New York). Slurm job `1961154` checked 33
pairs with no verification failures. This extends the [October 6 snapshot](../20261006/README.md)
with complete results for Mistral Nemo, Qwen Coder 32B, OLMo and Falcon.
The requested twenty-model comparison remains incomplete.

These are newly completed outputs from the previously frozen **v7** experiment.
They retain its original risk-adjusted objective and interface, while reporting
net cumulative Return Rate as requested. They do not measure the subsequent v8
terminal-return prompt, v9 direct retrieval interface, or parser repairs in
[PR 22](https://github.com/howard-lynn-ye/fin-skills/pull/22) and
[PR 26](https://github.com/howard-lynn-ye/fin-skills/pull/26).
The [v9 single-date pilot](../20261006-rag-pilot/README.md) is a separate experiment
without full-period Return Rates. No completed run was rewritten or rescored with
the newer parser for this publication.

## Return Rate

Each row is the arithmetic mean of three net cumulative Return Rates, seeds
11, 23 and 37, over January 3, 2025 through September 25, 2026. Return Rate is
`(ending equity / initial equity - 1) * 100%`; these figures are not annualized.
Each arm has 44 decision opportunities per seed. Orders execute at the next
session close, trading costs are 5 basis points per side on traded notional,
and cash earns zero interest. An invalid or missing submission preserves the
existing holdings and remains recorded as a failed decision.

| Model | Without library | With library | Library minus without, percentage points |
|---|---:|---:|---:|
| Qwen2.5 Coder 7B | 37.239007% | 56.649502% | +19.410495 |
| Qwen2.5 Coder 14B | 32.313750% | 27.834846% | -4.478904 |
| Qwen2.5 Coder 32B | 34.486601% | 29.279650% | -5.206950 |
| Qwen2.5 7B Instruct | 31.993609% | 33.396372% | +1.402763 |
| Qwen3 4B Instruct 2507 | 12.232357% | 0.000000% | -12.232357 |
| Granite 3.3 8B | 25.486892% | 38.106928% | +12.620035 |
| Phi-4 Mini | 37.147272% | 23.426641% | -13.720631 |
| SmolLM3 3B | 22.309886% | 11.347148% | -10.962738 |
| Mistral Nemo 12B | 25.476791% | 29.519211% | +4.042420 |
| OLMo 3 7B | 32.292949% | 31.213293% | -1.079655 |
| Falcon3 10B | 61.942589% | 27.941113% | -34.001475 |

Four models have a positive mean difference and seven a negative difference.
This is one development market path; three random seeds are not three independent
markets. These results do not establish a universal library advantage or isolate
the causal contribution of a tool, prompt, error, allocation, or transaction cost.

All per-seed returns, successful submissions, response counts, error classes,
model revisions and result fingerprints are in [snapshot.json](snapshot.json).
Completion here means all planned decision records were produced and scored;
it does not mean every model successfully submitted a portfolio. In particular,
Qwen3's library arm submitted none of its 132 decision opportunities. It remained
in cash, explaining its measured zero return. This is distinct from an unfinished
experiment, for which no full-period Return Rate is published.

## Outstanding runs and later status

At the fixed verification cutoff, DeepSeek R1 Distill Qwen 14B had three library
decision records for seed 11 and none for the other seeds. The original run log
shows a context-budget exception followed by an infrastructure-qualification
abort. It is an incomplete run, not a zero-return model result.

Mistral Small 24B had 44 library and 39 raw decisions in seed 11 at that cutoff.
The later read-only status check at 17:05:28 UTC (13:05:28 New York) showed job
`1908420` still running, now at seed 23 library decision 5/44. Its log also contains
GPU allocation warnings; their effect has not been audited here. These later
outputs are outside this fixed snapshot and do not add to its 33 verified pairs.
The DeepSeek job had already left the active queue with the recorded exception.
No job was restarted or cancelled for this publication.

Native Claude/Gemini/GPT trading runs and other model coverage remain outstanding;
code reviews do not count as model trading experiments. The separate
[fixed-strategy report](../20261007-expert-agent-trading/README.md) also does not
add an autonomous model to this count: its runner supplies programmed allocation
rules for both arms.

## Verification and reproduction

The unchanged [verifier](../20261006/verify_snapshot.py) ran as job `1961154`,
using each run's archived source and checking source/input bindings, decision
receipts, completion and independent accounting-audit hashes. It wrote
`verified-status-1961154.json` on Beacon. There were no verification failures.
This establishes record and accounting consistency, not correct execution of
every library function or completeness of historical evidence.

[response_diagnostics.py](response_diagnostics.py) is the October 6 diagnostic
script with its input changed to that new verification file. The unchanged
[publisher](../20261006/publish_snapshot.py) checks the means and seed inventory,
rechecks decision/result fingerprints, and exports public aggregates. The exact
launcher is [export.sbatch](export.sbatch), submitted as job `1961764`. Its
diagnostic input is `response-diagnostics-1961764.json`; public and private outputs
are written exclusively under `publication-20261007-v1` in the existing Beacon
study directory. Output files cannot silently overwrite an earlier export.

The public snapshot contains the hashes of the verification and diagnostic files
and of each paired run's protocol, inference receipt, score, accounting audit,
source manifest and evidence inputs. Original responses, evidence text and
private error examples remain on Beacon. A later verification may find additional
completed runs; publish that as another dated snapshot, preserving this one.

The pre-commit launcher [validate.sbatch](validate.sbatch) regenerates the index
and package, runs repository validation and the existing publication regression
tests, then runs [validate_publication.py](validate_publication.py). That final
check reproduces the exact public export from the original Beacon records,
compares every previously published paired result, checks the table against the
full-precision data, and verifies that regeneration leaves generated content
unchanged. It does not perform new inference or change trading decisions.

Evidence coverage remains limited to selected monetary news, surveys, futures
positioning and House disclosures. Historical archive vintages and some public
availability dates remain unverified. Neither this snapshot nor the earlier
pilots demonstrate that every library capability or required information source
was successfully exercised.
