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
