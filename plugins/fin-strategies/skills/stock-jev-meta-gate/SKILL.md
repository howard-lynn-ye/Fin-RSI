---
name: stock-jev-meta-gate
description: >-
  Build financial JEV questions and validate returned answers against the original questions.
  TRIGGER - stock_jev_gate, financial JEV question templates, regime choice, noul
  disqualification, ordinal rating, malformed decision probabilities or timestamp cutoff.
  SKIP for fitting a predictive model (time-series-forecasting-models), portfolio optimization
  (portfolio-and-risk), or multi-agent orchestration (finance-agent-architectures).
license: MIT
compatibility: Python 3.10+ with pandas; templates and validation need no torch or credentials.
metadata:
  verified_on: "2026-10-12"
allowed-tools: "Bash(python:*)"
---

# Financial JEV questions and answer validation

The templates are adapted from Shwai He's `v2` financial JEV module. They define market
regime choices, per-ticker disqualification questions and ordinal quality ratings for a
caller-selected model. They do not generate forecasts, train a model or select positions.

✅ Code verification, 2026-10-12: `tests/test_stock_jev.py` exercises these helpers and
rejects malformed responses. Validation reuses the existing `fin_skills.model_zoo.jev`
wire validator; the hosted adapter's network and credential controls remain unchanged.

```python
from fin_skills.jev import build_macro_regime_question, build_disqualification_question
from fin_skills.api.guards.stock_jev_gate import check_stock_jev_gate

questions = build_macro_regime_question()
questions.update(build_disqualification_question("EXAMPLE"))
# decision_result is the actual response from the model selected by the caller.
# result = check_stock_jev_gate(decision_result, questions)
```

Always retain the original `questions`. The guard checks exact question IDs and types,
declared probability keys, finite probability ranges and sums, selected choices, ordinal
expected scores, model identity, and usage fields. It returns the standard `GuardResult`.
No usage figures are fabricated. Malformed answers fail; malformed question definitions
are rejected as bad inputs.

The optional `max_disqualification_rate` is a caller-selected policy limit. Without it,
a model may legitimately disqualify every candidate. Passing the answer contract does
not mean probabilities are calibrated or the recommended stocks are safe.

`audit_stock_jev_causality(features_df, asof_date)` checks the supplied `date` column
against an inclusive UTC cutoff. Pass source availability timestamps, not just fiscal
period dates. Empty, missing and invalid dates fail. A date-only cutoff means midnight;
use an explicit timestamp to include observations later that day.

🚨 This timestamp check does not establish how features were calculated. Use the actual
signal producer with `assert_causal` for numerical look-ahead tests. Neural weights,
hardcoded regime forecasts and automatic exposure mappings from `v2` are not included.

For hosted calls, retain the explicit network opt-in of `fin_skills.model_zoo.jev.JevModel`.
See `../../../fin-llm/skills/finance-agent-architectures/SKILL.md` for agent orchestration.
