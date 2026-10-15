# 32B full-cycle queue resource correction

At 2026-10-15 18:32:50 UTC, the sole fin-skills job 2232777 was pending Priority with
one H100, eight CPUs, 64G host memory and a 24-hour limit. Its estimated start was
2026-10-17 15:04:18 America/New_York. There was no dependency, account-limit reason,
invalid-resource reason, requested node or excluded node.

## Observed bottleneck and applied change

The beacon partition exposes eight H100 GPUs and 96 H200 GPUs. All GPUs of both types
were allocated at inspection; MIXED node state did not mean there was a free GPU.
The H100-only request unnecessarily restricted eligible capacity. A 24-hour continuous
allocation also restricted backfill opportunities. Neither observation implies model
or library failure: this full cycle had not started.

The existing pending job was held briefly, changed with:

```sh
scontrol update JobId=2232777 Gres=gpu:nvidia_h200:1 TimeLimit=08:00:00
```

and released. No job was cancelled or submitted. Job ID and SubmitTime remained unchanged;
Slurm reset EligibleTime/AccrueTime on release, which is visible in the receipt. The displayed
Priority returned to 1; all configured priority weights were zero at inspection. Other
projects' jobs were untouched. No persistent SSH master was created or closed.

At 18:34:00 UTC, Slurm confirmed one H200, eight CPUs, 64G and eight hours, and estimated
start at **2026-10-16 14:02:43 America/New_York**. This is a changing scheduler estimate,
not an execution or completion promise. The job remained pending and the start/completion/
summary files were absent. No new full-cycle Return Rate or ledger exists in this report.

## Runtime basis and experimental isolation

The preserved 32B pilot finished qualification at 17:56:06.571013 UTC and model preflight
at 18:10:14.969000 UTC on October 14: 848.397987 seconds for loading, access checks and
six decisions. Scaling that whole interval to 88 decisions gives about 3.46 hours; this
is a rough planning estimate, not a measured cycle runtime. The eight-hour request retains
margin for varying analysis turns and longer history. No model-token budget, date, trade,
validation criterion or decision was removed to make it fit. An incomplete run must still
be reported as incomplete; it cannot be scored as a full cycle.

The pilot's H100 run logged a transient allocator OOM warning but subsequently completed
32B preflight successfully. H200 also offers more memory headroom. The frozen backend
loads bf16 on one GPU, without quantization or CPU offload. A generic GPU request could
place the model on an undersized card, so it was not used. H200 hardware may change
floating-point execution and generated samples relative to the H100 pilot; both full-cycle
arms run on the same allocated GPU, and hardware receipts must accompany their results.

The frozen source29, data, Qwen/Qwen2.5-Coder-32B-Instruct revision, seed 11, prompts,
44 dates per arm and starting capital USD 100,000 remain unchanged. RSI stays disabled.
The existing script will execute, score and independently export all decision, daily NAV
and trade ledgers. The metric remains 100 * (ending capital / initial capital - 1), after
modeled costs. No paid model calls, new pilot, 7B run or automation was added.

## Evidence and reproduction

- `slurm-evidence.txt`: actual update receipt, after-update scheduling, partition/node resources,
  priority settings, pilot timestamps and absent output markers. Other projects' queue rows
  are excluded and trailing spaces are trimmed. This is operational, not financial evidence.
- `resource-change.json`: machine-readable change and pending-result status.
- `cycle32_33_h200.sbatch`: original published launcher with only GPU and walltime directives
  changed. It describes the effective allocation but **was not submitted**. Do not submit it
  while job 2232777 exists or after its start markers appear.
- The original historical H100 launcher remains in `../20261015-dataframe-recovery/`.
  Slurm uses its stored batch payload plus the controller overrides recorded here; changing
  a launcher on disk does not modify an already submitted job.

No Python runtime, package content, test or generated catalog changed. The required index
and package generation commands were rerun, followed by repository validation: OK, 131 skills,
with the existing fin-libraries discovery-budget warning. See `validation.log`. Generation
produced no tracked changes. This publication adds queue evidence and a launcher-only resource
diff; GitHub CI validates the publication head. Full-cycle results and complete ledgers must
be published separately after the existing job completes its independent audit.

Slurm documents job resource and time-limit updates in its
[scontrol reference](https://slurm.schedmd.com/scontrol.html). The scheduler's accepted
before/after state, rather than documentation alone, is the verification of this change.
