# Flexible 32B scheduling and dual-device qualification

This change adds opt-in placement for the pinned Qwen2.5-Coder-32B-Instruct BF16
model on two GPUs. The default backend remains single-device. These are execution
and validation records, not a completed financial result.

The user's final scope is at most one RUNNING allocation per owning chat/window;
serial dependent pending stages are allowed. Other windows may run concurrently
within actual Slurm quotas. The matching instruction was synchronized to shared
Codex/Claude instructions, project instructions and Beacon's shared files.

## Why the job waited

The previous loader placed the complete model on one GPU. Its 48 GB alternatives
could not hold the unchanged weights on that device. Earlier comparisons also
relied too heavily on hypothetical `sbatch --test-only` start times. Those estimates
were not promises and did not justify treating October 20 as the earliest possible
start. Idle GPU counts did not describe free CPUs: several inspected L40S nodes had
all 32 CPUs allocated while two GPUs remained unallocated.

Beacon was already using automatic Backfill. An untyped two-GPU request with four
CPUs and 64 GiB RAM admitted other eligible nodes without pinning or excluding any.
The first two replacements started 10 seconds after submission. The current repair
started after 27 seconds. Availability changed between inspections; this is evidence
that flexible requests can start promptly, not a controlled estimate of speedup.

Scavenger was inspected. It is a lower-tier, preemptible opportunity partition;
its test-only estimates were not earlier than the actual ordinary-partition starts.
The new dual-device placement also has no verified preempted-job recovery yet.
The current run therefore uses the ordinary partition and explicitly disables
automatic requeue. No job from another window was changed or cancelled.

## Retained attempts

| Job | Requested GPUs | Allocated GPUs | Outcome |
| --- | --- | --- | --- |
| 2232777 | one H200 | none | Pending predecessor cancelled at zero runtime, with no cycle decisions. |
| 2316162 | two untyped | two L40S | Started in 10 seconds; exited after 46 seconds because staged tests could not import their sibling helper. No model inference. |
| 2316360 | two untyped | two RTX 6000 Ada | Started in 10 seconds; 66 tests and build/spec checks passed. Exact BF16 peer-copy validation failed before weights were loaded. |
| 2317005 | two untyped | two L40S | Started in 27 seconds; 36 exact transfers and 32,768-token generation passed; real tool/decision preflight running. |

The test import failure was fixed by running tests inside their complete source
`tests/` directory and including that directory on the import path. All 66 checks
then passed. The peer-transfer failure is preserved as infrastructure evidence;
no corrupted outputs were accepted as investment decisions.

## Transfer repair and acceptance

The explicit map assigns 32 transformer layers to each GPU and keeps the language
head on the input GPU. The opt-in `dual_qwen32_bf16_host` mode transfers activations
through CPU memory using blocking copies; all model parameters remain CUDA BF16.
The serial benchmark scopes the Accelerate transfer hook to each generation and
restores it even after an exception. It does not change library tools or strategy
choices. Regression cases cover nested tensors, skipped cache keys and restoration.

Acceptance requires exact copies in both directions for BF16, float32 and integer
tensors, including transposed tensors; a full configured context generation with
finite scores; both devices holding BF16 weights; the original confinement and RAG
checks; and the original real-tool and decision preflight. Any failure stops before
publishing a financial result. Allocation or model loading alone is not acceptance.

The current request is two GPUs, four CPUs, 64 GiB host RAM, and 11 hours 50 minutes.
Requested allocation caps are not measured usage or billing credits. The failed
attempts consumed allocations too; their evidence is retained.

## Experiment preserved

- Model: Qwen/Qwen2.5-Coder-32B-Instruct, revision
  `381fc969f78efac66bc87ff7ddeadb7e73c218a7`.
- BF16 weights, original context/output budgets, seed 11, original inputs and order.
- 44 decisions per arm, USD 100,000 initial capital, 2025-01-03 through 2026-09-25.
- Primary outcome: `100 * (ending capital / initial capital - 1)`, after modeled costs.
- RSI remains disabled; no paid model call, speculative queue copy or heartbeat.
- A new protocol binds the loader change while asserting every other experiment
  field and every input hash remains unchanged; the old frozen source is preserved.

Complete Return Rates and decision/NAV/trade ledgers are still unavailable at this
record's initial writing. They must be independently audited and published when the
paired cycle completes; absent returns must not be reported as zero.

References: [Slurm backfill](https://slurm.schedmd.com/sched_config.html),
[Slurm preemption](https://slurm.schedmd.com/preempt.html),
[Accelerate device dispatch](https://huggingface.co/docs/accelerate/package_reference/big_modeling).

## Observed hardware acceptance

At 2026-10-17T02:25:11Z, job 2317005 had loaded the complete BF16 model on two L40S
GPUs and passed all 36 exact host-staged transfer checks plus 32,768-token generation.
The full receipt is in hardware-qualification.json. This validates the observed L40S
allocation; other allocations still must pass the same gates. Real tool/decision
preflight and full-cycle financial outcomes remain separate acceptance stages.

GitHub CI was not readable through the dedicated connector because it required an
account connection. CI success is not asserted; Beacon test/build/hardware checks
above are the verification evidence. No CI CLI fallback was used.
