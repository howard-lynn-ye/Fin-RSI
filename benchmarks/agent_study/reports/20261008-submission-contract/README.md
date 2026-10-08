# Decision submission review — October 8, 2026

The first 18 processed Qwen2.5-Coder-7B library decisions (seed 11) in the
`20261007-final-contract` full-period run included seven failed decisions.
Inspection of their original final responses found:

| Final response | Count | Why it did not execute |
|---|---:|---|
| Complete submit JSON, weights total 1.10 | 4 | Exceeds available capital |
| Complete submit JSON, weights total 1.15 | 1 | Exceeds available capital |
| Complete submit JSON, weights total 1.13 | 1 | Exceeds available capital |
| Bare weights JSON, weights total 1.40 | 1 | Missing tool envelope; also exceeds capital |

All seven responses ended normally; none was cut off by the generation limit.
These are portfolio allocation instructions, rather than three-way buy/sell/hold
classifications. A trade also needs an instrument and an amount. Rejecting 110%
absolute exposure in a cash-only account is correct. Requiring the model to keep
repairing the sum, with no correction opportunity after its final response, is
an avoidable usability problem in the experimental execution interface.

This sample does not establish a numerical bug in a fin-skills portfolio
algorithm, a cause of every other failure, or the return a rejected trade would
have produced. The original decisions and running experiments remain unchanged.

## New submission contract

New freezes use interaction revision `20261008-explicit-allocation`:

- A model may submit `{"allocation":{"SPY":2,"CASH":1}}`. This is an illustration
  of the format, not a strategy. The model chooses the assets and relative parts,
  including an explicit cash part. The executor divides each part by the total.
  Missing assets have zero target weight. No default cash exposure is inferred.
- `HOLD` or `{"action":"hold"}` explicitly preserves existing units and cash.
  Allocating everything to cash instead sells the positions and incurs the
  normal trading costs. These operations are distinct.
- Bare `{"weights":{...}}` and the existing submit tool remain supported.
  Absolute weights still obey the budget constraint. Invalid weights are never
  silently converted into relative allocations.
- Both arms get six research responses and up to two decision responses, within
  the original eight-response limit. An invalid seventh response receives
  validation feedback before the eighth. A valid earlier submission ends the
  decision immediately.
- The receipt retains the original allocation, computed target weights and cash
  fraction. Conflicting instructions, unknown assets, invalid numeric values and
  missing cash allocations remain errors; none receives a default strategy.

Only new protocols use this change. The existing full-period batch keeps its
archived source and `20261007-final-contract` revision. Its results cannot be
relabeled as outcomes of this repair.

## Validation

Beacon diagnostic job: `1972511`. Original record hashes and failure metadata are
in `diagnosis.json`; prompts and source-document text are excluded.

Candidate validation job `1972522` passed: **114 tests**, required catalog and
package regeneration, repository validator, and generated-content comparison.
The tested patch SHA-256 is
`ef55bf7529fcafb6d3e01d7194a997cbdd6c359d9afb2a9ac487df04dc457c5b`.
A first-date paired Qwen7B probe, job `1972524`, completed after that validation.
Both arms submitted on response 7 using explicit allocations: SPY 0.4, QQQ 0.3,
EFA 0.2 and CASH 0.1. Both still had two research-call errors (`start_date` and
singular `ticker` were unsupported). Neither used a library-specific research
tool in this probe. This establishes two accepted decisions, not comprehensive
tool use, elimination of all failures, or a library performance advantage.

Audit job `1972571` verified the frozen source/qualification, both decision hashes,
the delivered prompt chains and allocation conversion. Its public output is
`pilot.json`. Full prompts and source documents remain on Beacon. The probe
measures interface usability and has no Return Rate score.
Neither test success nor this small probe proves all models will submit valid
orders or that the library increases investment returns.

## Full-period follow-up

Array job `1972585` was accepted for the separately frozen repaired batch on
October 8. It uses the same five models and three seeds as the existing wave,
44 dates per arm, and the same market/evidence inputs. `manifest.json`,
`run_pair.py` and `full_cycle.sbatch` record the plan and launcher; their hashes
were checked by the pilot audit before submission. No complete repaired-cycle
Return Rate was available at submission. Existing jobs `1972429_0` and
`1972430_1` retain their prior protocol and records.
