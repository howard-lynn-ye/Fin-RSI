# Factorial trading manuscript: evidence and release review

Status: working research draft. GitHub publication and merge were explicitly authorized
by the user on October 3, 2026. Journal submission, author order, final declarations and
human scientific approval are not represented as complete. The existing anonymous ACL
format is retained; current venue length and submission policies have not been certified.
No clinical, participant or private-review material is part of this experiment.

## Evidence map

| ID | Claim scope | Source and locator | Verification |
|---|---|---|---|
| E01 | Full design, interventions, timing and exposure | `20261003-factorial-trading/PROTOCOL.md`, `ANALYSIS_PLAN.md`, all `receipts/*/protocol.json` | Frozen before full inference; hash-checked on Slurm |
| E02 | All 2,112 decisions, 12 pairs, 48 paths | `verification.json`, `completed.json`, receipt maps | Source/data/decision hashes checked; independent dollar-holdings replay |
| E03 | Return means, four paired effects and seed signs | `analysis.json`: `results`, `groups` | Reconciled against complete per-pair scores; tables generated and regression-checked |
| E04 | Zero successful document reads and rare numerical calls | `analysis.json`: `results[*].paths[*]` | Coverage recomputed from frozen decision records; counts are calls/decisions, not inferred knowledge |
| E05 | Risk, exposure, turnover, tokens and errors | Same paths, including `metrics`, `exposure` and separate failure fields | Full denominators retained; seed-level values preserved |
| E06 | Fixed baselines, cost and uncertainty sensitivity | `baselines`, `fixed_decision_cost_sensitivity_pct`, `descriptive_block_sensitivity` | Frozen finalizer; no independent-market or significance claim |
| E07 | Earlier positive and negative bundled-access results | Separate `20261002-trading-v5` archive and posthoc directory | Original completion and 528 decision receipt hashes rechecked; not pooled |
| E08 | Supporting guard/code-generation/context/memory claims | Existing evidence and generated macros referenced by the preserved appendix sections | Existing published evidence retained; not re-run or relabeled as factorial findings |

Evidence paths in this table are relative to `benchmarks/agent_study/evidence/`.
Automated checks are recorded as automated checks, not fabricated human sign-off.
Independent human source review and manuscript approval remain pending.

## Interpretation and reporting coverage

The question, intervention differences, protocol date, fixed budget, model revisions,
seed set, calendars, asset selection, inclusion of failures, analysis population, primary
and secondary outcomes, effect units, dependence sensitivity, data access and limitations
are stated in the methods, result tables or evidence archive. Clinical reporting checklists
are not applicable to this simulated agent experiment. No checklist certification is claimed.

The original assets are exposed development data. Transfer assets share the historical
calendar and economic exposures. Current skill documents are not historically available
knowledge. Pretraining contamination is not ruled out. K and T represent optional access,
including prompt text; zero successful skill reads do not support knowledge-use causality.
All full-access means trail equal-weight rebalancing. All reported block intervals span zero.
No additional model inference, seed selection, tuned budget or discarded failure was used.

## Source and disclosure boundaries

Code, protocols, derived aggregate results and receipt hashes are in the authorized public
release. Datasets, weights, raw model outputs and decision contents remain on Beacon.
Receipt hashes permit identity checks but do not supply restricted artifacts or independent
custody. The paper distinguishes public table reproduction from full inference reproduction.

The four finance-agent reference abstracts and bibliographic entries were checked by the
assistant against their arXiv pages on October 5, 2026: 2402.18485, 2510.02209, 2508.00828,
2510.07920. Their short related-work descriptions are retained without borrowing their
performance claims. Other pre-existing references remain from the earlier manuscript;
this release does not claim a new human reference audit.

Authorship, contributions, funding and conflicts are unresolved for submission. AI assisted
execution, verification scripting and manuscript editing in this release; accountable human
authors must review the results and the venue-specific disclosure before submission.
