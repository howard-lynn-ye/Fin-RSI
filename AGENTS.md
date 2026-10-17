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
