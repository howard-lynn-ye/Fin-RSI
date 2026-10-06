# Multisource trading results: October 6, 2026 snapshot

At 16:25:56 UTC, seven models had completed all three paired seeds. This snapshot
contains 24 verified pairs: 21 from those seven models, two from Mistral Nemo and
one from OLMo. The requested twenty-model comparison is **not complete**.

## Return Rate

Each number below is the arithmetic mean of three **net cumulative Return Rates**
over January 3, 2025 through September 25, 2026. It is not an annualized return.
Trading costs are 5 basis points per side on traded notional. Decisions execute
at the next session close. Each arm has 44 decision dates per seed; unallocated
capital earns zero interest. No submission preserves the existing holdings.

| Model | Without library | With library | Library minus without, percentage points |
|---|---:|---:|---:|
| Qwen2.5 Coder 7B | 37.239007% | 56.649502% | +19.410495 |
| Qwen2.5 Coder 14B | 32.313750% | 27.834846% | -4.478904 |
| Qwen2.5 7B Instruct | 31.993609% | 33.396372% | +1.402763 |
| Granite 3.3 8B | 25.486892% | 38.106928% | +12.620035 |
| Phi-4 Mini | 37.147272% | 23.426641% | -13.720631 |
| SmolLM3 3B | 22.309886% | 11.347148% | -10.962738 |
| Qwen3 4B Instruct 2507 | 12.232357% | 0.000000% | -12.232357 |

Three models have a positive mean difference and four a negative one. Seed
variation is substantial: Qwen Coder 7B differences are +13.343093, -8.635358 and
+53.523750 points. These are three repeats on one development market path, not
three independent markets or evidence of a universal library advantage.

Partial results are excluded from that table and from the seven-model count:

| Model | Completed seeds | Without library | With library | Difference, points |
|---|---|---:|---:|---:|
| Mistral Nemo 12B | 11, 23 only | 22.537184% | 24.518925% | +1.981741 |
| OLMo 3 7B | 11 only | 36.974433% | 31.616444% | -5.357989 |

The exact per-seed values, successful submission counts, error classes, model
revisions and receipt fingerprints are in [snapshot.json](snapshot.json).
This publication preserves losses and execution failures. The earlier personal
case and legacy v6 results are separate protocols and are not pooled here.

## Why some library results are lower

The independent cash-and-units accounting checks passed for all 24 pairs. The
snapshot verification also checked source/input bindings, every decision receipt,
completion hashes and audit hashes. That establishes accounting consistency; it
does **not** certify every library path, historical evidence vintage or model call.

| Observation | What the records establish | What remains unproven |
|---|---|---|
| Qwen3 library submits 0 of 132 decisions | Its 1,056 turns contain 1,044 successful reads, ten generation truncations and two unparsed responses. With no trades from an all-cash start, Return Rate is zero. | A numerical portfolio defect is not established. The contribution of prompt length, evidence pagination and tool choice requires a separate controlled comparison. |
| Coder 14B submits 111 raw versus 91 library decisions | Library calls include 111 `NameError: evidence` failures, 58 generation truncations and 46 unparsed responses. | These counts do not quantify how many return points each failure caused, or prove all failures originate in library implementation. |
| SmolLM3 library frequently fails at API discovery | There are 186 rejected unknown algorithm IDs, 96 unsupported `query` arguments to `list_algorithms`, and 33 unsupported `offset` arguments to `describe_algorithm`. | Replacing guessed calls with valid calls has not yet been tested in a separately frozen experiment. |
| Phi Mini library submits 45 versus 42 raw decisions but earns less | It has 296 unparsed library responses and 33 rejected unknown algorithm IDs. More submissions alone do not guarantee better returns. | Asset choices, timing, exposure, costs and execution failures have not been causally separated. |

Qwen3's result exposes an interaction failure: the model keeps reading until its
budget expires. The runner already states the remaining calls and warns on the
last call; a proposed repair cannot honestly be described as merely adding that
missing warning. The current protocol has no separate, guaranteed final decision
stage. A future comparison should test a reserved final decision stage in **both**
arms, with allocations still chosen by the model and invalid submissions retained.

The API failures also expose an integration problem. Package functions and the
prepared-history study adapters have different signatures; some functions are
Python-only. Documentation is present, but these smaller models repeatedly mix
the interfaces. Improving discoverability and executable examples is justified.
The recorded data do not justify claiming that all return declines are a library
bug, or that the existing library is bug-free.

## Provenance and executable analysis

- Verification: Slurm **1908418**, completed with exit 0; 24 pairs, no validation
  failures. [verify_snapshot.py](verify_snapshot.py) is the executed verifier.
- Response counting: Slurm **1908424**, completed with exit 0.
  [response_diagnostics.py](response_diagnostics.py) counts recorded response and
  tool events, rather than assigning failure causes from the return alone.
- Publication export: Slurm **1908428**, completed with exit 0.
  [publish_snapshot.py](publish_snapshot.py) rechecks result/receipt bindings and
  exports summary data. Detailed error text stays in a private Beacon file.
- Public snapshot SHA256:
  `a9674e332b85d93b283c68add40fac47d8769d89a195a68b3b4031459a5e2e89`.
- Pre-commit validation: Slurm **1908433** regenerated the index and package,
  printed `OK` from repository validation, passed seven publication regression
  tests, and verified that generated content and the public snapshot were unchanged.
- Source/library repairs were already merged through PRs
  [19](https://github.com/howard-lynn-ye/fin-skills/pull/19) and
  [20](https://github.com/howard-lynn-ye/fin-skills/pull/20).

The first two scripts preserve the exact site paths and input filenames used for
this dated snapshot. Execute them on a Slurm compute node with the archived study
environment, never on a login node. A later verifier run may see more completed
pairs and creates a new report; it does not replace this published snapshot.
The publisher takes explicit paths and refuses to overwrite either output:

```bash
python publish_snapshot.py --base "$B" \
  --verification "$B/verified-status-1908418.json" \
  --diagnostics "$B/response-diagnostics-1908424.json" \
  --output "$B/new-publication.json" --private-errors "$B/new-private-errors.json"
```

Here `$B` is the existing `fin-multisource-models-20261005` directory on Beacon.
The Python executable must be its qualified `v7-models-v1/env/bin/python`.
Raw market data, source documents, model responses and private error examples
are not published in this directory.

The requested additional Opus 5.5 review was attempted on October 6 but returned
an account-limit error before a verified model response. It is **not** counted as
a completed independent review of this publication. Prior completed reviews of
the library/runtime repairs are documented with PRs 19 and 20.

## Remaining work and limits

Mistral Nemo seed 37 hit its 12-hour job limit. Resume job **1908419** checks and
preserves the 75 existing decisions. Job **1908420** runs the previously unsubmitted
Qwen Coder 32B/Mistral Small 24B batch. Job **1907749** continues the remaining small
models. These are launch/status facts, not completed results in this snapshot.
Native Claude/Gemini/GPT trading comparisons remain outstanding; reviews do not
count as those experiments. Llama access remains gated.

Evidence covers selected monetary news, surveys, futures positioning and House
disclosures. Historical archive vintages and some availability dates are still
unverified. The study does not cover all library capabilities or all information
sources. Any interface/budget repair needs a new frozen protocol; changing a
finished run to improve its return would invalidate this record.
