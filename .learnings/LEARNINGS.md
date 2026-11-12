# Learnings

## [LRN-20261016-WINDOW-OWNERSHIP] correction

**Logged**: 2026-10-17T02:15:56Z
**Priority**: high
**Status**: promoted by explicit user instruction
**Area**: infra

The final user clarification allows one RUNNING Beacon job per owning window and
serial dependent pending stages. Neither account totals nor a repository-name
prefix defines the user allowance. Inspect actual Slurm quotas separately, track
owned IDs, and prevent overlapping allocations during transitions. Subagents share
their parent's allowance. Prefer complete-task-compatible capacity and automatic
backfill; compare scavenger only with its preemption and recovery requirements.

- Source: user_feedback
- Pattern-Key: beacon.window_owned_running_limit
- Recurrence-Count: 1
- Related Files: AGENTS.md
- Promoted: shared/project AGENTS and Claude instructions, Beacon shared policies

### Context Maintenance

- Decay-Type: decision; Decay-Rate: durable until a new user instruction
- Authority: explicit user clarification; Owner: owning chat/window
- Revalidation: reread current policy and live quotas before submission
- Validation: verified 2026-10-16; local and remote instruction headers read back
- Disposition: retain; Application: applied


## [LRN-20261108-BBZ] correction

**Logged**: 2026-11-08
**Priority**: high
**Status**: verified
**Area**: docs

Verify a dataset's official numeric tolerance in the primary Methods section. Do not infer it
from a demo or from this project's adapted endpoint. BizBench ACL2024 Section3.1 states1%
reference tolerance, whereas this project's previously frozen endpoint is exact numerical
label equality. The documentation now distinguishes both; no score, protocol, model output
or old receipt was rewritten. A methods statement is not proof of an executable official scorer.

- Source: primary-paper recheck, independently confirmed
- Pattern-Key: research.verify-metric-in-primary-methods
- Recurrence-Count: 1
- Evidence: .progress-audit/external-publisher-facts-correction-20261108.json;
  https://aclanthology.org/2024.acl-long.452.pdf Section3.1.
- Related Files: docs/EXTERNAL_EVALUATION_20261108.md
- Decay-Type: dependency; Decay-Rate: slow; Authority: primary paper for this revision
- Owner: main project chat for shared learning files
- Revalidation: check the primary Methods and executable scorer when adopting a new revision;
  retain uncertainty when the implementation is unavailable.
- Validation: verified2026-11-08; Disposition: retain; Application: applied to documentation

## [LRN-20261109-DELEGATED-DESIGN] use explicit judgment delegation

The user resolved the native16K versus declared32K choice with “你自己做判断就行”.
Choose and record the defensible prospective amendment, preserve original evidence and
scientific population, then execute the already authorized work. Do not keep showing
awaiting-user or ask the same design question again. This does not authorize fabrication,
paid calls, new campaign scope or weakening fixed output/input budgets.

- Source: direct user instruction; Pattern-Key: autonomy.delegated_research_decision
- Status: applied; Recurrence-Count: 1; Owner: main project chat
- Evidence: .progress-audit/context-ceiling-amendment-20261109.json

## [LRN-20261109-PARALLEL] apply the current task-specific scheduling authorization

The user explicitly authorized compatible scavenger/idle capacity, disjoint batches and parallel completion for the existing fin-skills study. The old one-RUNNING and all-core-before-any-P/E ordering no longer applies to this task. Update the project override, bind the prospective execution amendment, and release already-qualified independent work. Preserve scientific population, completed evidence, actual quotas and other owners. Five existing S jobs started simultaneously after their dependencies were removed; no extra cells or fits were introduced.

- Pattern-Key: research.current_parallel_authority
- Status: applied; Recurrence-Count: 1; Owner: main project chat
- Evidence: .progress-audit/parallel-scavenger-user-authorization-20261108.json; execution-order-amendment-20261108.json; PE-v4-CPU-result-20261109.txt
