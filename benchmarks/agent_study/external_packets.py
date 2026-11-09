"""Explicit public/reference separation for externally authored scalar code tasks.

This is an adapter contract, not the official BizBench evaluator. It never runs reference
programs, fetches data, chooses a favourable subset, or claims private/unseen task custody.
Numerical tolerance and units must be supplied by a separately frozen scoring policy.
"""
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ScalarPolicy:
    """Task-wide conventions fixed from public instructions before examining outcomes."""
    unit_instruction: str
    absolute_tolerance: float
    relative_tolerance: float
    provenance: str

    def __post_init__(self):
        if not self.unit_instruction.strip() or not self.provenance.strip():
            raise ValueError("public unit convention and scoring provenance are required")
        for value in (self.absolute_tolerance, self.relative_tolerance):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("tolerances must be explicit finite nonnegative numbers")

    def correct(self, candidate, reference):
        # A bool, container, NaN or string is not a successfully delivered numerical scalar.
        if (type(candidate) not in (int, float) or
                (type(candidate) is float and not math.isfinite(candidate))):
            return False
        expected = numeric_reference(reference)
        return abs(Decimal(str(candidate)) - expected) <= max(
            Decimal(str(self.absolute_tolerance)),
            Decimal(str(self.relative_tolerance)) * abs(expected))


def numeric_reference(text):
    """Only unambiguous finite numeric literals; never choose a percent scale from a score."""
    if not isinstance(text, str) or not re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text.strip()):
        raise ValueError("reference needs a reviewed source-specific numeric/unit parser")
    try:
        value = Decimal(text.strip())
    except InvalidOperation as exc:
        raise ValueError("invalid numerical reference") from exc
    if not value.is_finite():
        raise ValueError("reference must be finite")
    return value


def packets(row, *, revision, split, row_index, source_cluster, policy):
    """Project a source row into public inputs and a separate scoring record.

    A source document/template cluster must come from the frozen provenance mapping;
    numbering each question as its own cluster would inflate independent sample size.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("full dataset revision required")
    if split not in ("train", "test") or type(row_index) is not int or row_index < 0:
        raise ValueError("explicit source split and row coordinate required")
    if not isinstance(source_cluster, str) or not source_cluster.strip():
        raise ValueError("source document/template cluster must be assigned before evaluation")
    if row.get("task") not in ("CodeFinQA", "CodeTATQA", "CodeTAT-QA", "FinCode"):
        raise ValueError("unsupported source task; do not silently change its output type")
    fields = {}
    for key in ("question", "context", "context_type"):
        value = row.get(key)
        if value is None and key != "question":
            value = ""
        if not isinstance(value, str) or (key == "question" and not value.strip()):
            raise ValueError(f"invalid public {key}")
        fields[key] = value
    options = row.get("options")
    if hasattr(options, "tolist"):
        options = options.tolist()
    if options not in (None, []):
        raise ValueError("multiple-choice rows require a separate declared contract")
    reference = numeric_reference(row.get("answer"))
    source = dict(dataset="kensho/bizbench", revision=revision, split=split, row_index=row_index)
    public = dict(schema="external-scalar-v1", id=digest(source), source=source,
        source_task=row["task"], source_cluster=source_cluster, **fields,
        output_contract=dict(kind="python_last_expression_scalar",
                             unit_instruction=policy.unit_instruction),
        exposure="public; model training contamination unknown")
    private = dict(public_sha256=digest(public), source=source,
                   original_answer=row["answer"], reference_literal=str(reference),
                   scoring_policy=asdict(policy), scoring_policy_sha256=digest(asdict(policy)))
    return public, private


def prompt(public, *, interface_text=""):
    """Public-only rendering. No answer, reference program, tolerance or score is accepted."""
    required = {"schema", "id", "source", "source_task", "source_cluster", "question",
                "context", "context_type", "output_contract", "exposure"}
    if set(public) != required or public["schema"] != "external-scalar-v1":
        raise ValueError("unexpected public packet fields")
    if (set(public["output_contract"]) != {"kind", "unit_instruction"}
            or public["output_contract"]["kind"] != "python_last_expression_scalar"):
        raise ValueError("scalar output contract required")
    system = ("Write Python code in one python fence. The last statement must be an expression "
              "whose value is one finite numerical scalar. Preserve the quantities and units "
              "specified in the question. " + public["output_contract"]["unit_instruction"])
    if interface_text:
        system += "\nAvailable library interface:\n" + interface_text
    return [dict(role="system", content=system),
            dict(role="user", content=f"Context:\n{public['context']}\n\nQuestion: "
                                      + public["question"])]
