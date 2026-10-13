# Selective v2 integration, 2026-10-12

Base: `master` at `078b8f55497fe378afd3f83767d44e99ce4a6224`.
Reviewed source: Shwai He's `v2` at `2c2c665f19165e0d2d78d90d059340b525e46850`.
The user requested useful, verified additions while preserving the current master work.
This is a selective adaptation on top of master, not a merge of v2's research history.

## Selection

| Source | Included | Excluded and reason |
|---|---|---|
| `2461140` financial JEV | Financial question builders; timestamp check; strict answer guard using master's existing wire validator | Analytical predictions use hand-set coefficients and fabricated token usage. The neural adapter creates random weights by default; no trained/calibrated artifact was supplied. Both are excluded. |
| `a7dbf26` liquidity shield | Daily rolling liquidity proxy; built-in calculation audit; registered Python/JSON guard | Execution cost, board classification, fill probability and borrow-fee presets lack demonstrated calibration. They are not imported as market rules. |
| `ff42cde` six-gate report and Skill-RSI compiler | No stored pass verdict is imported as evidence for this integration | The report predates the JEV commit. Its blind scorer explicitly lacks answer isolation. The compiler synchronizes other directories before evaluating the aggregate pass flag and defaults to personal paths. Existing build/check commands remain authoritative. |
| `5248bea` v5/PR #11 integration | Existing master implementations retained | Its sys.path-based Linux read grants are not independently qualified here. Current trading, confinement and later v6-v9 changes remain untouched. |
| `2c2c665` hostname cleanup | Existing master hygiene retained | No need to import unrelated older files to gain this cleanup. |
| Earlier v2 paper, model-family, panel, macro and cross-board work | Preserved on v2 | Outside the selected October 2 additions; not represented as accepted experiments or validated functionality. |

The template text and liquidity weighting originate with Shwai He. The integration narrows
their public claims, fixes the checks below, and adds independent regression tests. It does
not claim new investment performance or independently reproduce the archived v2 experiments.

## Reproduced problems and repairs

Direct probes of the frozen v2 source produced:

- `stock_jev_gate` passed an answer with `type="invented"`.
- It also passed choice probabilities `{a: 2.0, b: -1.0}` because their sum was one.
- The timestamp audit accepted the invalid string `0000-bad` before a valid cutoff.
- The liquidity audit passed a supplied score with a component of the next observation's
  return added to it. Its correlation threshold did not establish equality or causality.

The integrated JEV guard requires the original questions and reuses master's existing
validation of IDs, types, finite probability ranges, declared options and ordinal means.
It returns `GuardResult`; a caller can optionally impose a disqualification-rate limit.
It does not reject a fully disqualified universe unless the caller requests that policy.
Timestamp checks parse UTC instants, reject missing/invalid/numeric dates, and describe the
limits of checking supplied dates rather than the feature-producing program.

The integrated liquidity proxy requires observed features, preserves warmup NaNs, parses
timestamps, rejects duplicate symbol/timestamp pairs, and preserves positional alignment
even with duplicate DataFrame index labels. Rankings depend only on the current timestamp.
The audit compares a supplied signal to the formula within numerical tolerance rather
than accepting a correlated signal. It perturbs all consumed features after a cutoff.
This probes the built-in calculation; it is not certification of arbitrary caller code.

`fin_skills/jev/` is explicitly preserved by `HAND_WRITTEN_DIRS`. The v2 change omitted
this entry, so simply copying its package would lose it during regeneration. Optional
PyTorch imports and additional model-catalog entries are not added.

## Verification

Validation is performed on this adapted implementation, not inferred from v2's report.
Tests include an independent numerical calculation, every chronological prefix, shuffled
rows, duplicate index labels, missing inputs, contaminated scores, invalid responses,
timezones, JSON access, optional policy limits and import without PyTorch/network access.

The repository validator is authoritative for this repository's six-field frontmatter.
The generic installed skill-creator validator rejects `compatibility`, despite the explicit
repository allowance; no repository requirement was removed to satisfy that older helper.

The lexical routing check gives the same result with and without the two new skills:
71/108 strict top-1, 99/108 routed, five thin margins. Its existing nonzero exit is retained;
descriptions are not tuned to this proxy, and no private holdout labels were read.

The PyPI drift check completed with zero wrong historical version claims; it reported
14 newer versions and one unmatched package name. These existing dated entries were left
unchanged.

- Repository validation: `OK`, with the pre-existing fin-libraries discovery warning.
- Focused new/API/JSON/JEV/Bundle tests: 283 passed on NumPy 1.24.0, pandas 2.2.2,
  SciPy 1.11.4. One pandas categorical conversion warning was emitted.
- Full regression, script execution and distribution checks are pending completion.

## Preservation

No v2 manuscript, benchmark receipt, dataset, private label, model outcome, global description
rewrite, trading worker or RAG change is copied over master. Generated catalog/package files
are rebuilt from the selected sources. Existing source history stays intact on both branches.
