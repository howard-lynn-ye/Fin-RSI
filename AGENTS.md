<!-- fin-skills-parallel-execution:2026-11-08:start -->
## Current fin-skills experiment: parallel execution authorized

On 2026-11-08 (America/New_York), the user explicitly instructed: “换成那种 scavenger 的模式，
然后哪张卡空就直接哪张卡上，对卡没有任何要求吧……分小批，然后在不同的上面跑嘛，
然后尽量并行，全部一口气跑完。” The original statement is preserved in
`.progress-audit/parallel-scavenger-user-authorization-20261108.json`
(SHA256 `64073af709c6170d8b8348a31f31b63743edc911e8f5eff78637544ee49aa9eb`).

- For this project's already authorized original experiment, multiple independently owned
  RUNNING jobs in this same chat are allowed. This supersedes the older one-job and
  one-RUNNING-per-window limits below for this task only. Do not edit cross-project policy.
- Prefer eligible scavenger opportunity capacity and every GPU type that can run the complete
  task. Use untyped GPU requests when compatible. Check actual account/association/QoS limits,
  full weights, precision, context/output allowance, RAM and verified device placement.
- Partition disjoint cells/units without duplication. Preserve original20 models, complete
  inputs, seeds, comparisons, all completed evidence and total authorized budgets. Splitting
  questions does not reduce model weight memory. No implicit retry or cap expansion.
- Remove unnecessary cross-track dependencies. A model's completed qualification permits its
  independent P/E/S work; other models' pending qualifications must not block it solely due to
  an agent-created execution order. Record prospective execution-order amendments; the full
  analysis still requires the original complete population.
- Preserve healthy running jobs. Use verified persistence/resumption for preemptible work;
  distinguish completed-unit durability from resumability inside an interrupted unit. Record
  concrete reasons when ordinary capacity is the usable path while recovery is incomplete.
- Remote owner remains chat `01a1bc9a-3383-7b41-800d-b1cb5bfefff9`; helpers share that ownership.
  The shared six-SSH limit, privacy, scientific scope and no-heartbeat/paid-call rules remain.
<!-- fin-skills-parallel-execution:2026-11-08:end -->

<!-- beacon-gpu-no-type-pinning:2026-10-28:start -->
## Beacon GPU type: never pin by preference; retarget long-pending typed jobs

User instruction, October 28, 2026, for all projects, Codex and Claude:
"我没有设置GPU限制，看到空就直接用就行". The user has never set a GPU-model
restriction. A GPU type chosen by an earlier agent, plan, tie-break or sbatch
header is not a user requirement. This block overrides any project file, plan,
script header or earlier decision that names a preferred GPU model.

Why: CAPRISA job 2371177 was pinned by an agent tie-break to
`gpu:nvidia_h100_80gb_hbm3:1` (only 8 such GPUs on Beacon) and waited 8 days
although H200 also fit. Retargeted in place to H200 on 2026-10-28, its Slurm
start estimate moved from about 31 hours away to about 1 hour away.

- Default request: `--gres=gpu:N` with no type. Add a type only when other
  types physically cannot run the complete task (for example single-card
  VRAM). Never pin a type for a tie-break, prior evidence on that type, billing,
  habit or a copied script.
- When a type is required, the allowed set is EVERY type that can run the
  complete task. Beacon nodes have no features (`Features=(null)` on
  2026-10-28), so Slurm cannot express "H100 or H200" in one job. Compare all
  allowed types at submission and submit once, to the earliest.
- `sbatch --test-only` for a new job ignores an existing job's older submit
  time and queue position, so it is not a valid comparison for a job that is
  already pending. Compare by retargeting the owned pending job in place:
  `scontrol hold` -> `scontrol update jobid=<id> TresPerNode=gres/gpu:<type>:<N>`
  -> verify walltime, memory, CPUs, partition, QoS, requeue and dependency are
  unchanged -> `scontrol release` -> read its new `StartTime`; revert if later.
  This is the same job: no new submission, retry, allocation or budget.
- Any owned typed job still PENDING for Resources/Priority after 2 hours, or
  with an estimate more than 2 hours away, must be re-compared at the next check.
  "Waiting for Resources" is not an acceptable final status without that check.
- Do not hardcode `#SBATCH --gres=gpu:<type>` in new sbatch files; pass the
  GPU request on the sbatch command line (it overrides headers in frozen
  scripts). Hardware-bound acceptance code must accept every allowed type; if a
  qualification run binds later stages to its hardware, record that consequence
  instead of using it as a reason to pin.
- Beacon snapshot 2026-10-28 (verify live; not a guarantee): 80 GB-class
  single cards are H100 80GB HBM3 (2 nodes x 4 = 8) and H200 (12 nodes x 8 = 96).
  48 GB cards are L40S and RTX 6000 Ada. A single-card task that needs more than
  48 GB uses H100 or H200; a task that fits on 48 GB stays untyped.
- Unchanged: one RUNNING job per owning window, budgets, release gates, no
  duplicate jobs, and never modify another window's jobs. Only the owning window
  retargets its own pending job.

Existing agents must reread this block; file synchronization is not a broadcast.
<!-- beacon-gpu-no-type-pinning:2026-10-28:end -->

<!-- beacon-resource-limits-policy:2026-10-16:start -->
# Mandatory Beacon resource and execution policy

User instruction, 2026-10-16: apply these requirements to all local projects,
Codex and Claude. These are shared operating instructions plus a dated controller
snapshot, not a claim of new administrator legislation or execution permission.
They supersede older conflicting GPU preferences and limit interpretations.
These requirements apply when using Beacon; they do not move unrelated local,
HULK or other-server work to Beacon or change those projects' execution scope.

## Verified controller snapshot: 2026-10-16 17:51:23 UTC

| Scope | GPU/job ceiling | CPU/job ceiling | RAM/job ceiling | Max walltime |
|---|---:|---:|---:|---:|
| default QoS | 1 | 4 | 32 GiB | 72 h |
| medium QoS | 2 | 8 | 64 GiB | 72 h |
| high QoS | 4 | 16 | 512 GiB | 96 h |

The beacon partition itself has MaxTime=72 h and DefaultTime=1 h. Its effective
limit remains 72 h even with high QoS; request walltime explicitly. The observed
user/partition association has MaxJobs=6 and MaxSubmitJobs=11. These are job-count
limits, not a six-GPU quota. Each listed QoS had Priority=0; selecting high does
not establish a priority advantage. Other partitions may have different rules.
Do not extrapolate this snapshot into current eligibility or remaining credit.

## Required before any submission

1. Read the current connection guide, inspect new local/remote shared pauses,
   and inspect live partition, QoS, association limits, configured GRES and
   resource availability. Inspect prior jobs and output markers before acting.
2. Establish the COMPLETE task's VRAM, host RAM, CPU, GPU count, topology,
   runtime, storage and walltime needs. A free card or successful model load
   does not prove full-task eligibility or acceptance.
3. Use earliest compatible capacity. Prefer --gres=gpu:N without a model type
   when the eligible pool can run the complete task; otherwise constrain only
   demonstrated minimum capabilities. Do not pin nodes or arbitrarily exclude
   them, and do not wait for H200/L40S or another type without technical need.
   Use exact configured GRES names when a type is necessary.
4. Preserve authorized weights, precision, context/output limits, inputs,
   stages and ALL acceptance tests. No shrinking or quantization to fit a card.
   Multiple small cards do not combine memory unless the executor supports and
   verifies that placement; four GPUs is not a universal default.
5. Count requested GPU hours as GPU count times allocation walltime. Include
   startup, data checks, loading, training, evaluation, checkpointing and saving;
   set soft/process/Slurm limits with finishing margin. Queue time is separate.
   Submitted caps, measured runtime, GPU-hour credit and billing TRES are distinct;
   differing GPU billing weights mean equal GPU hours need not mean equal cost.
6. Preserve healthy jobs and immutable evidence. Correct only authorized pending
   work, without duplicate runs or speculative fan-out. Never cancel or alter
   another project's jobs to make capacity. Recheck eligibility before submission
   and record requested versus allocated resources plus complete acceptance.

## Authorization and shared connection boundaries

Respect each project's budget, pause, release gate and maximum active job count.
This document grants no experiment, new budget, retry, timer, restart or migration.
The user clarified that one Beacon job means at most one RUNNING job per
owning chat/window, with serial dependent pending jobs allowed. The account
may have multiple independently authorized window tasks. Consult each task's
latest release record for its budget, pause and scientific release gates.

The shared SSH cap is six simultaneous connections across projects, separate
from GPU quotas. Prefer the existing shared connection and preserve its owner.
Use one coordinated owner for recovery. After a failed or hung attempt, record
the shared pause and stop; retry only after a concrete fix, recovery evidence or
explicit user instruction. Do not switch clients or use a timer to bypass it.
Keep command stderr privately so a remote CLI error is not mislabeled as an
authentication failure. For sinfo, do not combine -a with -p; for job guards,
use squeue --all --states=all so nonterminal states are not silently omitted.

Run computation, numerical tests, data construction, analysis, inference,
training and large-weight hashing through Slurm. New remote storage belongs on
the approved radfm project storage. Do not send support email without explicit
authorization. Never copy credentials, private infrastructure or raw research
data into a repository. Existing sessions must reread updated instructions;
file synchronization is not a live broadcast or scheduler-level enforcement.

Canonical local policy: `D:/ct_agent_vqa/BEACON_RESOURCE_POLICY.md`.
Before connecting, read `D:/ct_agent_vqa/BEACON_CONNECTION_GUIDE.md`.
Claude and Codex must apply this block and the project-specific AGENTS.md;
do not treat an older CLAUDE.md quota as overriding this dated policy.
<!-- beacon-resource-limits-policy:2026-10-16:end -->

<!-- beacon-flexible-gpu-policy:2026-10-15:start -->
## Mandatory Beacon GPU scheduling: earliest compatible capacity

User instruction dated 2026-10-15, for all projects and all agents, including
Codex and Claude: use available eligible GPUs that can run the COMPLETE task;
do not wait specifically for H200 or another GPU model without a demonstrated
technical requirement. This supersedes older default GPU-model preferences.

- Before submission, inspect current Slurm resources, configured GRES names,
  account/QOS limits, and the task's actual VRAM, RAM, CPU, GPU-count, topology,
  runtime and walltime needs. An idle GPU alone does not establish eligibility.
- Prefer an untyped GPU request such as `--gres=gpu:N` when the eligible pool
  supports the full workload. Otherwise use only the minimum supported
  capability/type constraints needed for compatibility, with the reason recorded.
  Let Slurm select nodes; no fixed hostnames or arbitrary node exclusions.
- Keep the exact authorized models/weights, precision, context/output budgets,
  inputs/cases, pipeline stages and acceptance tests. Do not shrink or quantize
  them merely to fit a smaller card or shorten the queue. Compatible multi-GPU
  placement is allowed within the existing resource budget; record and validate
  the actual device layout and runtime. Four GPUs is not a universal default.
- For a long-pending job, inspect unnecessary type/node restrictions and feasible
  alternative layouts; adjust authorized pending work rather than simply waiting
  for one GPU type. Preserve healthy running jobs and frozen experiment evidence.
  If an original fallback must remain, coordinate a single winner before any
  non-idempotent work. No speculative job fan-out or duplicate experiments.
- Respect each project's execution authorization, pause, budget and maximum job
  count, and the cluster's quotas. This scheduling rule does not resume stopped
  projects, grant new experiments, increase budgets or move work off its approved
  server. Never cancel another project's jobs to free capacity.
- Run compute through Slurm; record requested versus allocated resources and
  verify model loading plus the complete acceptance flow. Allocation/startup is
  not a passed test. Report queue estimates as estimates, never guarantees.

Read the current connection guide before Beacon access:
`D:\ct_agent_vqa\BEACON_CONNECTION_GUIDE.md` on Windows or
`/beacon-projects/radfm/wy891/BEACON_CONNECTION_GUIDE.md` on Beacon.
The cross-project SSH limit remains six; it is not a GPU quota. Existing storage,
connection recovery, privacy and project-specific restrictions remain in force.
Existing sessions must reread updated instructions; file edits are not a live
broadcast or scheduler-level enforcement.
<!-- beacon-flexible-gpu-policy:2026-10-15:end -->

<!-- beacon-window-ownership-policy:2026-10-16:start -->
## Beacon: one RUNNING job per window; earliest compatible capacity

User clarification, October 16, 2026, applies to Codex, Claude and all projects.
The user explicitly answered: "最多一个运行中作业，允许串行依赖排队".
Thus each owning chat/window may have **at most one RUNNING Beacon job**;
serial dependent PENDING jobs are allowed. This supersedes the earlier
"one nonterminal job" interpretation, including pending jobs in that cap.
The account may run multiple independently authorized window tasks concurrently,
subject to actual live account/association/partition/QoS limits and task budgets.
This is not an account-wide or repository-wide one-job limit.

- Record the owner chat/window identifier and its submitted Slurm IDs. Use
  `squeue --all --states=all` and filter by owned IDs for the window guard;
  inspect account totals separately for actual Slurm quotas. Other windows'
  jobs do not consume this window's one-running-job allowance. Never cancel
  or change another window's jobs or connections.
- Serialize the owned stages with dependencies (for example, `afterok`) so
  pending stages cannot run concurrently. Do not start a new allocation while
  the preceding owned job is configuring, running, suspended or completing.
  A single experiment can have several submitted stage IDs. Subagents/helpers
  share their parent window's allowance; do not create windows, arrays or
  duplicate experiments to evade this rule. Check old outputs before any retry.
- For ALL authorized Beacon GPU submissions, use the earliest capacity that
  supports the COMPLETE task. Prefer untyped `--gres=gpu:N` when the eligible
  pool fits; otherwise use only demonstrated minimum capability constraints.
  Let Slurm choose nodes. Preserve required weights, precision, context, inputs
  and every acceptance stage. Check GPU/CPU/RAM/topology/storage/walltime needs.
- Backfill is an automatic Slurm scheduler mechanism, not an `sbatch` switch
  that bypasses priority. Request realistic resources and walltime with finishing
  margin. An idle GPU alone is insufficient when memory, CPUs, reservations,
  task compatibility or eligibility block it. `sbatch --test-only` start times
  are estimates, not guarantees; do not wait for a model preference alone.
- Always consider eligible scavenger opportunity capacity alongside ordinary
  partitions. Use it when the complete task is eligible, preemption is acceptable
  under its authorization, and required checkpoint/resume behavior is verified.
  Record a concrete reason when a different compatible path is preferable.
  Scavenger is not guaranteed faster, immediate or free. Do not silently enable
  requeue, expand retry budgets, blanket-migrate projects or submit queue copies.
- Preserve healthy jobs, immutable evidence, budgets, pauses and experiment
  release gates. This scheduling clarification grants no new experiments,
  spending, retries, timers or migrations. The shared SSH cap remains six across
  projects, separate from job/GPU counts. Reread the connection guide and live
  resource policy before access/submission. Existing sessions must reread this
  update: file synchronization does not broadcast to running agents or modify
  Slurm jobs/submission-guard code.
<!-- beacon-window-ownership-policy:2026-10-16:end -->

# AGENTS.md

Instructions for coding agents working in this repository.

## What this repo is

A library of [Agent Skills](https://agentskills.io/specification) about the Python
quantitative-finance ecosystem. The content is not tutorials — it is the set of defaults, licence
changes and silent failure modes that a model's training prior gets wrong, each one dated and
marked with how it was verified.

Two tiers:

- **domain skills** (`fin-core`, `fin-china`, `fin-asia`, `fin-crypto`, `fin-futures-fx`,
  `fin-llm`) — one per task a person has. These answer "which library, and what will bite me".
- **`fin-libraries`** — one skill per library, opt-in because it costs ~5,300 tokens of listing
  budget on its own (`build_index.py` prints the current figure per plugin). These answer "how do I use this one correctly", and each links back to its
  domain skill.

## Before you commit

```bash
python scripts/build_index.py   # 1. regenerate the catalog and README table
python scripts/build_package.py # 2. regenerate fin_skills/ - AFTER build_index, it copies index.json
python scripts/validate.py      # 3. must print OK (it runs build_package.py --check)
```

`catalog/index.json`, the README skill table and its prose counts, and everything under `fin_skills/`
are **generated** - with the exceptions listed in `scripts/build_package.py` as `HAND_WRITTEN` and
`HAND_WRITTEN_DIRS`, currently `__init__.py` and the hand-written layers `api/`, `tools/`, `mcp/`,
`engine/`, `data/` and `bridges/`. Read that tuple rather than this sentence; it is the mechanism,
and `--check` enforces it. Never hand-edit anything else under `fin_skills/`: it is overwritten on
the next regeneration. A new hand-written layer must be added to that tuple in the same commit that
creates it.

## Rules that exist because they were broken

**Do not state a number you did not produce.** Numbers in a SKILL.md should come from a script in
that skill, or from a command whose output you saw. Several claims here were wrong on the first
pass and were caught only by running them.

**Do not reflow prose you are not editing.** Re-wrapping a file to a different column width makes
every line show as modified, which hides deletions. An agent doing this once flattened three
markdown tables into slash-separated prose and deleted a set of cross-references, and the diff
looked like pure whitespace churn. If you must check, compare with whitespace normalised:

```bash
python -c "
import re,subprocess,pathlib,sys
f=sys.argv[1]
old=subprocess.run(['git','show',f'HEAD:{f}'],capture_output=True,text=True).stdout
new=pathlib.Path(f).read_text(encoding='utf-8')
n=lambda s: re.sub(r'\s+',' ',s).strip()
print('content identical' if n(old)==n(new) else 'CONTENT CHANGED, not just wrapping')
" <path>
```

**Do not edit files another agent is writing.** Concurrent writes to the same SKILL.md have
silently reverted committed work here. If you are one of several agents, stay inside the files you
were assigned.

**The frontmatter spec is exactly six fields** — `name`, `description`, `license`, `compatibility`,
`metadata`, `allowed-tools`. Extra keys pass in Claude Code and hard-fail on claude.ai upload, so
`validate.py` rejects them. A mis-indented block scalar silently swallows the keys that follow it;
the validator checks for that specifically.

## Testing

`python -m pytest -q` runs the unit-test suite under `tests/` - one module per generated skill
script plus the `fin_skills.api` layer (over 1,600 tests; slow demo paths are marked `slow` and
deselected by default, `-m slow` runs them). Tests import the GENERATED package, so after editing
a plugin script run `build_package.py` before the tests can see the change. Library-specific
tests skip when the library is absent. Beyond the unit tests, verification is:

| Script | Checks |
|---|---|
| `scripts/validate.py` | spec compliance, name/directory match, description caps, dead reference links, empty `references/`, domain backlinks, README skill count |
| `scripts/build_index.py` | regenerates the catalog and README table |
| `scripts/eval_triggers.py` | lexical smoke test — reports strict top-1 and routed accuracy. Deliberately reproducible across `PYTHONHASHSEED` values |
| `scripts/eval_blind.py` | historical replay; does not enforce answer isolation; refuses incomplete batches |
| `scripts/eval_sealed.py` | separate public inputs/private labels, stateless tool-free API inference, frozen receipt before independent scoring |
| `scripts/check_drift.py` | re-checks version claims against PyPI (network) |
| `scripts/check_repo_stats.py` | re-checks GitHub stats. Note `open_issues_count` includes pull requests |
| `scripts/check_scripts.py` | runs every skill script in a subprocess with `PYTHONIOENCODING` stripped. A script that prints a non-ASCII character passes under UTF-8 and dies on a stock Windows console |
| `scripts/build_package.py` | regenerates `fin_skills/` - the skills as an importable package (`--check` is run by `validate.py`). Never edit `fin_skills/` by hand |

`eval_triggers.py` is a bag-of-words proxy and degrades once two skills cover the same package.
`eval_blind.py` cannot prove an agent did not read answer files. Use `eval_sealed.py` for
constrained API evaluation and report the declared dataset exposure and protocol limits.
Do not tune descriptions to the proxy. Never publish private holdout labels or treat public
regression fixtures as unseen questions. See `docs/EVALUATION_SECURITY.md`.

## Verifying a claim at a primary source

**A 403 is not evidence that the source is unreachable.** Several exchange and regulator sites
refuse programmatic requests while serving a browser normally. Measured on 2026-09-10:
`idx.co.id` returns 403 to WebFetch on every URL and full page text through the browser tools
(`navigate` then `get_page_text`, in a tab you created yourself); `jpx.co.jp` returns 403 to both
and its material had to be sourced from another exchange's equivalent notice, labelled
secondhand; `cmegroup.com` returns 403 to all programmatic access. Try the browser before
recording a fact as unverifiable, and say which route worked.

**An official English translation can be years stale.** Japan's `japaneselawtranslation.go.jp`
serves the 2011 text of the FIEA Enforcement Order and the 2008 text of the trading-restriction
ordinance - both predate the November 2013 short-selling reform, so neither contains the rule
that is actually in force. The current text is on `laws.e-gov.go.jp`. Prefer the national
in-force database in its own language over a translation with an older version stamp.

**Client-rendered pages need a wait.** Press-release pages on several exchange sites return an
empty shell to an immediate read; wait a few seconds after navigating before extracting text.

**"Everyone knows" is where the errors are.** Three market verifications on 2026-09-10 each
overturned facts that circulate widely: Indonesia restored symmetric auto-rejection in September
2023 and reverted to an asymmetric 15% floor in April 2025, so the commonly-cited "symmetry was
restored" is 17 months out of date; the Philippines' static threshold is +50%/-30%, not
symmetric, and its dynamic threshold is tiered 20/15/10 by trade frequency rather than a flat
figure; Japan's 10% short-sale trigger is measured against an exchange base price derived from
the previous close, not the close itself. Date every rule and name the decree.

## Conventions

- Lines wrap at about 98 columns
- Markers: ✅ verified at a primary source · ⚠️ secondhand · 🚨 silently wrong · 🔴 dead or broken
- `metadata.verified_on` is a date, and it should move when you re-check the file
- Scripts use only numpy / pandas / scipy, run with a fixed seed, and work whether or not the
  library they describe is installed

## Beacon experiment resource budget (user instruction, 2026-10-13)

Allow at most ONE fin-skills Slurm job in total, counting running and pending jobs.
Do not expand model experiments while source/interface defects are being repaired.
Use only necessary focused verification, preferably CPU-only; finish the repairs and
verification before resuming a model campaign. Preserve partial and completed results.
No paid model calls, heartbeat monitors, automatic full-campaign restarts or parallel
version sweeps. This instruction supersedes earlier multi-job campaign launch plans.
This is a project job limit; the shared six-connection SSH limit is a separate rule.


<!-- layout-explains-policy:2026-10-24:start -->
## Design: let the layout do the explaining

Standing user instruction, 2026-10-24, for all projects and agents:
**Avoid unnecessary secondary labels and let the layout do the explaining.**

- For user-facing websites, apps, games, dashboards and visual artifacts, use grouping, alignment, spacing, scale, imagery and reading order to make relationships clear.
- Omit decorative category labels, redundant subtitles, badges, step numbers and helper copy when they merely repeat what the title, content or layout already communicates. Before adding an explanatory label, first improve the layout.
- Keep concise labels that identify an action, input, meaningful status or unit; preserve accessible names and necessary concept/readiness disclosures. The goal is clarity, not unexplained controls or missing context.
- Apply this principle when creating or reviewing designs. It does not authorize deleting or hiding content the user asked to retain, or redesigning unrelated projects without a task.
- Treat this as a standing design and review requirement across projects, not a one-time prompt. When creating or adopting a new project, record this principle in its root `AGENTS.md` as well as applying the shared rule.
<!-- layout-explains-policy:2026-10-24:end -->


<!-- project-progress-contract:2026-10-31:start -->
## Local project progress contract
The user requested automatic desktop progress reporting on 2026-10-31.
After meaningful work within this project's existing authorization, update its
local status receipt using .project-progress/README.txt and connection.json.
Publish evidence for the exact acceptance item IDs and definition in status.json;
use measured counts for partial completion and an explicit complete-scope receipt
for a pass. Do not report arbitrary percentages, use a code commit as acceptance,
run extra experiments solely to update progress, or weaken existing gates.
The desktop collector owns status.json. The working agent owns immutable receipts
under .project-progress/events/. Preserve old receipts and report regressions.
Keep credentials and raw research/clinical data out of receipts. No new network,
compute, retries, spending, publishing or cross-window messages are authorized.
This metadata directory is local-only; it is not a public project deliverable.
<!-- project-progress-contract:2026-10-31:end -->
