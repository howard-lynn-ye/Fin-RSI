# Review under the October 15 earliest-compatible-capacity policy

The updated shared connection guide was reread at the user's instruction. This review did
not submit, cancel or add a job, change frozen source, or expand the one-GPU allocation.
The original resource correction passed 21/21 GitHub CI checks at
`e87d191ea11df45a28bc4f1a8ba63001d3e6851b` before this documentation-only supplement.

## Compatibility and scheduler capabilities

The exact source29 loader specifies bf16 and `device_map={"": 0}`. The 32B weights alone
exceed a 48GB card's capacity before context/cache overhead. Beacon's L40S and RTX 6000
Ada single-card options therefore cannot run the unchanged task. A generic `gpu:1` request
would admit those cards. CPU offload or quantization would change the authorized workload.

Both H100 80GB and H200 are compatible single-card choices. All nodes have null feature
labels. The only partitions shown are beacon and scavenger, covering the same GPU types;
there is no compatible-GPU-only partition or configured feature expression for an H100/H200
OR request. A comma-separated GRES list requests resources together, not alternatives.
Do not invent `nvidia_h*` types, treat host RAM as GPU VRAM, pin nodes, or exclude arbitrary
nodes to emulate a feature the cluster has not configured.

A two-card small-GPU layout would require replacing the frozen single-card loader and
validating the actual topology and full acceptance flow. The source also records the previous
reproduced RTX peer-copy corruption. Merely allocating two cards would still load the whole
model on GPU 0. That is not a validated compatible alternative for this unchanged run.
No such source change, additional pilot or larger allocation was made here.

The account association query succeeded and showed QOS default/high/medium/normal/scavenger,
with one association row limited to 6 running jobs and 11 submitted jobs. These cluster
limits do not replace this project's stricter one-job limit. The existing job has no account
or QOS-limit pending reason. Scavenger uses the same hardware with lower partition priority
and preemptible requeue behavior; the frozen launcher has exclusive start markers and is not
an automatic resume path, so it is not a complete-task scheduling improvement.

## Same-resource scheduling comparison, without submission

At 2026-10-15 18:51:07 UTC, two `sbatch --test-only` calls used the existing full-cycle
script, one GPU, eight CPUs, 64G and eight hours:

| Hypothetical new request | Scheduler estimate (America/New_York) |
| --- | --- |
| H100 80GB | 2026-10-22 08:55:18 |
| H200 | 2026-10-20 15:48:58 |

These are hypothetical **new-job** estimates; their displayed IDs are not queued jobs.
They include the current queue, including the older pending job, so they are not equivalent
to an in-place rescheduling estimate for that job. They support H200 as the earlier observed
compatible new-request option, not a universal H200 preference or an exact start guarantee.

The actual existing job 2232777 remained the sole fin-skills job. At 18:50:27 UTC its estimate
had moved to **2026-10-16 21:56:16 America/New_York**, from 14:02:43 at the earlier observation.
Queue forecasts change; retain that distinction instead of promising the first forecast.
No completed Return Rate or ledger was produced by this scheduling review.

Current decision: retain the older pending H200 job as the earliest supported compatible
choice observed. If later evidence favors H100 or the cluster exposes a suitable capability
constraint, reassess the same pending job. Preserve a healthy running job. The policy does
not authorize duplicate fallback submissions, another pilot, paid calls or a heartbeat.

Evidence: `compatible-pools.txt` contains the actual partition/features, frozen loader and
existing job snapshot; `compatible-estimates.txt` contains association limits, both test-only
estimates and the final one-job queue. These supplement the earlier `slurm-evidence.txt`.
The deployment record on disk is historical: only scheduler state controls the live job.

References: Slurm's [GRES documentation](https://slurm.schedmd.com/gres.html) describes typed
and untyped requests; [sbatch](https://slurm.schedmd.com/sbatch.html) documents constraints,
GRES lists and `--test-only`. The actual configured features and accepted request estimates
are the evidence for this cluster, not generic assumptions from those documents.
