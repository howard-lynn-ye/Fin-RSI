# Library reliability correction — 2026-09-28

The current manuscript centers on fin-skills, executable research guards, and numerical
reliability. HiSTrim and memory are auxiliary integrations. This supersedes the September 22
memory-centered writing direction without changing historical experimental records.

## What the evidence supports

| Claim | Evidence and interpretation |
| --- | --- |
| 129 skills | `scripts/build_index.py` and `scripts/validate.py`, executed for this revision. |
| 36/36 planted defects detected | `benchmarks/GUARD_ROBUSTNESS.json`: 12 defect families in three seeded worlds, with no false alarms on clean references. This is suite-level recall; it is neither 36 distinct leakage types nor proof of universal protection. |
| 8/8 numerical checks pass | `benchmarks/PARITY_AND_COST_RESULTS.json`, also summarized in `paper/evidence.json`. Maximum absolute error is 1.1827553714205408e-8. Each case retains its own tolerance. |
| Completed exploratory execution | `paper/AUTONOMOUS_LIBRARY_STUDY.md`: 64 valid decisions across four conditions, including 16 autonomous decisions and 43 successful requests. Requests include discovery, reading and advice; they are not 43 numerical algorithms. |
| Unguarded algorithm-access ablation | The historical dispatcher did not mandate research guards. The manuscript preserves the returns and execution traces, and no longer uses them to motivate a claim that tools are ineffective or that memory repairs the losses. No matched guarded return arm exists. |

The proposed exact “71.5 false Sharpe” example was not promoted to a verified result because
the inspected evidence did not establish its raw provenance. Short anonymous FX exposure,
trading costs and missing mandatory guards limit the earlier experiment's interpretation;
they do not establish the cause of its losses individually.

## Software corrections

Month-end, quarter-end and business-quarter-end code uses pandas offset objects instead of
aliases introduced in pandas 2.2. The universe interface accepts old and new alias spellings,
including multipliers and fiscal quarter anchors. Plugin sources were regenerated into the
package. The declared pandas floor and minimum-dependency CI job now use 2.1.4.

FinQA now has an explicit final ReAct/JSON example and a versioned parser. New runs accept
one complete JSON object, optionally inside one complete Markdown fence. They do not repair
programs or extract a favorable answer from prose. Old protocols keep strict legacy scoring.
Diagnostics retain generation failures, missing final answers, strict JSON compliance and
length-limited responses. New job bundles require an explicit qualification root for answer
runs and include the prospective protocol amendment.

`FINQA_INTERFACE_CORRECTION_20260928.md` describes the required next model qualification and
matched rerun. No new model inference, retrieval gain, numerical score or guarded trading
result is claimed by this correction. Historical FinQA scores remain unchanged.

## Manuscript delivery

Verification on Windows:

| pandas | Default full suite | Final regenerated PIT/API/FinQA regression |
| --- | --- | --- |
| 2.1.4 | 3,493 passed; 285 skipped; 53 deselected; 0 failed | Included in final full suite |
| 2.2.2 | 3,426 passed; 352 skipped; 53 deselected; 0 failed | 114 passed |
| 3.0.6 | 3,488 passed; 290 skipped; 53 deselected; 0 failed | 114 passed |

The 2.2.2 and 3.0.6 full runs preceded the final equivalent BQuarterEnd replacement in
the generated PIT module; its affected API/PIT tests were rerun afterward. Skips reflect
missing optional backends/platform requirements; slow tests were deselected by the repository
default. This is not an all-backends or all-pandas-versions claim. Ordered index/package
generation and `scripts/validate.py` passed (129 skills, one existing discovery-budget
warning). `git diff --check` and compilation of changed job/audit scripts passed.

Local logs are under `runs/pytest-pandas214-final-20260928.log`,
`runs/pytest-pandas222-20260928.log`, `runs/pytest-pandas3-20260928.log`,
`runs/targeted-pandas222-final-20260928.log` and `runs/targeted-pandas3-final-20260928.log`.

Local `paper/latex_naacl/main.pdf` compiles to 13 pages. The title, first page, framework table
and core evidence table were visually inspected; the final local log has no overfull boxes
or undefined references. Overleaf compiled the new manuscript with Errors 0 and Warnings 0;
45 underfull typesetting notices remain.

Project: <https://www.overleaf.com/project/6aad9a03f27c3c07a182965d>

History label: `Fin-skills library reliability - 2026-09-28`, on the September 28, 4:03 pm
upload shown by Overleaf. Before editing, the project's source ZIP was saved locally and the
affected files were compared with the previous synchronized sources. Six revised TeX files
were uploaded. Cloud-only outline content, coauthor sources, figures and auxiliary studies
were preserved. The cloud outline was not overwritten with the different local outline.

The final source-ZIP download did not complete, so all six updated files were instead read
back through the source editor. SHA-256 hashes below normalize CRLF to LF and match the local
files exactly:

| File | SHA-256 |
| --- | --- |
| `main.tex` | `b59b7dbac2823f9503f6b138b6cf17dc33bc66c25a3d8b25366c9a0eba32fe89` |
| `framework_overview.tex` | `1e8b4e1d15112beb762385312d747900be928cacbb76cbe8d1ea80dff263ab3b` |
| `capability_evaluation.tex` | `687771c7ab4e331bd4749a6a6c8e278297676827756cfbba56d85011ecbbec66` |
| `autonomous_study.tex` | `868bc4a7a6369b14963b4a610f5f59212747b2945fd8c7a36921904c2b4b9c34` |
| `supporting_evidence.tex` | `26a03b6304b3c7d634f932d70bb43e7fcc5af2ec01d1c616f2af6df086c37491` |
| `evidence_numbers.tex` | `561fa436e6b3364c3cfe3af350399faa5581b5cf636eaa8ddc5dd285515dee01` |
