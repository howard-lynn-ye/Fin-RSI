# Dataframe error recovery and per-model continuation

The completed source29 pilot (Slurm 2180196) produced all six final decisions for each model.
Both Qwen2.5-7B-Instruct and Qwen2.5-Coder-32B-Instruct executed library tools on all three
library dates. Coder32B passed research readiness. General7B did not: every research attempt
failed on the middle no-library date. These independent all-cash scenarios are interface
checks, not a cumulative Return Rate result.

The 7B code called Series.cov() without another Series, then treated Series.mean() as an
indexed vector. Generic TypeError feedback incorrectly directed it to bound-tool contracts.
The patch explains the data shape and supplies an optional raw-matrix inspection example
equally to both arms. No suggested inspection executes automatically, no return calculation
or portfolio is supplied, and the original failure remains a failure. Model-generated analysis
mistakes remain recorded; this does not establish that the revised prompt eliminates them.

CPU job 2210699 reproduced seven regression cases on the preceding source, then passed 101
focused tests after the patch (295.82 seconds). Index/package generation and validation passed.
The source fingerprints in verification.json match the tested copy. Existing discovery-budget
warnings are unchanged. The revised interaction is 20261015-dataframe-recovery-v10.

The user directed that the qualified 32B model proceed independently of 7B. Job 2232777 runs
the unchanged source29 32B pair at seed 11: 44 scheduled decisions per arm, initial capital
USD 100,000, window 2025-01-03 to 2026-09-25. It revalidates existing receipts without new
interface-model calls, executes the full cycle, scores it and independently exports every
decision, daily NAV and trade. At preparation of this report, that job was queued for Priority;
no full-cycle result is claimed. The new 7B-oriented diagnostic patch is not inserted into
the frozen 32B experiment. Results from different interaction revisions must remain labeled.

prior-model-decisions.json includes all 12 pilot decisions, generated requests, accepted targets,
errors and execution receipts. Its disclosure lists omitted source prose, prompts, feedback and
stdout. Original hash-bound records remain on Beacon; no private scoring labels are published.
Positive and negative outcomes are retained. No paid model or RSI loop is enabled.

The optional RSI integration design is recorded in docs/RSI_INTEGRATION_DESIGN_20261015.md.
It is a proposal, not a shipped capability or an intervention in this return experiment.
