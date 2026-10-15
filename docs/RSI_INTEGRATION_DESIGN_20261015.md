# Optional agent self-improvement with fin-skills

Status: proposed design, not an installed or enabled feature. The active 32B paired return
experiment uses frozen source29 and does not use this design. No additional Slurm job or paid
model call is authorized by this document. The project remains limited to one active/pending job.

## Role in the library

Keep fin-skills responsible for documented financial tools, data conventions, retrieval and
executable guards. An optional agent-framework adapter can use those capabilities to improve
the agent's tool selection, retrieval queries, API examples and short experience notes between
development episodes. A successful API call is evidence of delivery, not of financial benefit.

The repository already has experimental reflection/retrieval memory in
`benchmarks/agent_study/decision_memory.py`, framework experiments in `framework_tools.py`,
and an upstream Reflexion software qualification script. Those are research components, not a
production self-improvement engine. The earlier v2 Skill-RSI compiler was excluded from the
selective integration for documented validation and publication-order problems; see
`V2_INTEGRATION_20261012.md`. Do not enable it merely because its name contains RSI.

## First implementation boundary

1. Record actual requests, tool results, errors, data availability times and source versions.
2. Distinguish transport/environment faults, tool-contract errors, model analysis mistakes and
   financial outcomes. Low Return Rate alone is not a software-failure label.
3. Propose one bounded change to an agent playbook or tool-use instruction. For example, learn
   that a long-format close column must not be treated as a multi-asset price matrix. Preserve
   successful examples and failed proposals. Do not rewrite the entire accumulated playbook.
4. Evaluate on development tasks and a separate validation partition that the proposer cannot
   inspect. Require executable correctness, unchanged guards and a documented improvement;
   preserve uncertainty and reject regressions. The final test partition is used only after
   the candidate is frozen, not repeatedly for selecting candidates.
5. Promote a versioned candidate only after the checks pass; otherwise retain the prior version.
   A configured framework adapter controls promotion. Each candidate records its parent, diff,
   traces, evaluation receipts and decision. Empty edits do not count as improvement.

The agent's model weights need not change. The initial mechanism improves instructions and
experience retrieval. A system that also changes its own improvement procedure would add a
deeper recursive step; that is a separate, later experiment, not an existing capability claim.

## Locked components

The proposer cannot edit accounting, prices, fees, execution timing, guards, the evaluator,
logging, budgets, permissions, historical cutoffs or sealed labels. Enforce these boundaries
outside mutable prompts. Extra reflection calls count toward the declared compute budget.
Use existing permitted model backends; do not silently select a paid provider.

In a frozen return experiment, the model can reason and choose tools, but the adapter's code,
instructions and stored experience remain fixed. A later online-adaptation experiment must
declare its update policy in advance and use only feedback actually available before each
decision. It cannot reflect on a trade's future realized outcome before that outcome exists.

## Evaluation

First finish the requested 32B comparison: no library versus library, cumulative net Return
Rate computed from initial and ending capital, with every decision and the full daily ledger.
Do not add RSI mid-cycle or relabel this baseline as an RSI result.

Then test a separate 2-by-2 design: no library/no improvement; library/no improvement;
no library/improvement; library/improvement. Keep market inputs, transaction rules, model,
seed and deployment budgets matched, and disclose the separate cost of candidate search.
Freeze the selected candidate before evaluation on a later market window. Compare Return Rate,
execution errors, actual tool delivery and total inference cost; retain negative outcomes.
This separates the library contribution from the improvement-loop contribution.

## Algorithm starting point and evidence

GEPA uses execution traces and natural-language reflection to propose and evaluate prompt
updates. That is a plausible starting point for bounded tool-use instruction improvement:
[GEPA paper](https://arxiv.org/abs/2507.19457). Its reported results do not demonstrate financial
return improvement for fin-skills; that transfer is an untested proposal here.

Darwin Godel Machine studies broader agent-code self-modification and empirical validation:
[DGM paper](https://arxiv.org/abs/2505.22954). Its coding-task evidence does not justify enabling
unrestricted code rewriting in the financial runtime. Begin with the narrower editable surface.
