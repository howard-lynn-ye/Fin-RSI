"""Validate the whole declared population before producing P/E study summaries in Slurm."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_pattern(row, value, reference):
    """Post-selection diagnostics only; this never changes the frozen exact primary grade."""
    if row["valid"]:
        return "exact"
    if not row["steps"]:
        return "no_program"
    if not row["steps"][-1]["executed"]:
        return "format_or_execution_failure"
    if type(value) not in (float, int):
        raise ValueError("executed scalar row lacks its original recorded value")
    actual, expected = Decimal(str(value)), Decimal(reference)
    if expected and actual == -expected:
        return "opposite_sign_exact"
    if expected:
        for exponent in (-9, -6, -3, -2, 2, 3, 6, 9):
            if actual == expected * (Decimal(10) ** exponent):
                return "power_of_ten_" + str(exponent)
    return "other_numeric_mismatch"


def relative_tolerance_pass(row, value, reference):
    """Prespecified E sensitivity on the same final output; never edits primary grades."""
    if not row["steps"] or not row["steps"][-1]["executed"]:
        if row["valid"]:
            raise ValueError("exact pass lacks a delivered scalar")
        return False
    if type(value) not in (int, float):
        raise ValueError("executed scalar row lacks its original recorded value")
    actual, expected = Decimal(str(value)), Decimal(reference)
    if not expected.is_finite():
        raise ValueError("finite frozen reference required")
    # Fractions avoid Decimal context rounding at large integer / small decimal boundaries.
    passed = actual.is_finite() and abs(Fraction(actual)-Fraction(expected))*100 <= abs(Fraction(expected))
    if row["valid"] and not passed:
        raise ValueError("exact primary pass must also pass relative sensitivity")
    return passed


def attach_event_costs(report, path, references):
    active = None
    costs, values, counts = defaultdict(list), {}, Counter()
    for line in Path(str(path) + ".events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["kind"] == "episode_started":
            active = tuple(event[k] for k in ("method", "library", "task_id", "seed"))
        elif event["kind"] == "fit_started":
            active = None
        elif event["kind"] == "response_received" and active is not None:
            usage = event["response"].get("usage", {})
            if type(usage.get("prompt_tokens")) is not int or type(usage.get("completion_tokens")) is not int:
                raise ValueError("exact usage is missing from a completed bounded call")
            costs[active].append(usage)
        elif event["kind"] == "program_completed" and active is not None:
            counts[active] += 1
            values[active] = event["result"].get("value")
    for row in report.rows:
        key = tuple(row[k] for k in ("method", "library", "task_id", "seed"))
        if (len(costs[key]) != row["usage"]["calls"] or
                sum(c["completion_tokens"] for c in costs[key]) != row["usage"]["generated_tokens"]):
            raise ValueError("response ledger differs from recorded episode costs")
        row["derived_prompt_tokens"] = sum(c["prompt_tokens"] for c in costs[key])
        if references:
            if counts[key] != len(row["steps"]) or counts[key] > 1:
                raise ValueError("external one-shot execution ledger differs")
            row["numeric_error_pattern"] = numeric_pattern(row, values.get(key), references[row["task_id"]])
            row["relative_tolerance_1pct"] = relative_tolerance_pass(
                row, values.get(key), references[row["task_id"]])


def main(args):
    try:
        import frozen_analysis as analysis
        from run_frozen_study import (verify_plan, selected_chunks, allocation_qualification,
                                      accepted_chunk, allocation_context)
        from summarize_rsi_qualification import checked_report
    except ModuleNotFoundError as exc:
        if exc.name not in ("frozen_analysis", "run_frozen_study", "summarize_rsi_qualification"):
            raise
        from benchmarks.agent_study import frozen_analysis as analysis
        from scripts.run_frozen_study import (verify_plan, selected_chunks, allocation_qualification,
                                             accepted_chunk, allocation_context)
        from scripts.summarize_rsi_qualification import checked_report
    plan = verify_plan(args.plan)
    protocol = read(args.plan / "protocol.json")
    allowed_stages = {"core", "extension"} if args.stage == "all20" else {args.stage}
    cells = [c for c in read(args.plan / "cells.json") if c["track"] == args.track and c["stage"] in allowed_stages]
    chunks = {c["id"]:c for c in read(args.plan / "chunks.json")
              if c["track"] == args.track and c["cells"][0]["stage"] in allowed_stages}
    enrollment = Path(plan["enrollment_root"])
    references = {}
    if args.track == "E":
        public = [json.loads(line) for line in (enrollment / "public.jsonl").read_text().splitlines()]
        scoring = [json.loads(line) for line in (enrollment / "scoring.jsonl").read_text().splitlines()]
        references = {p["id"]:s["reference_literal"] for p,s in zip(public, scoring)}
    rows, seen, allocations = [], set(), []
    for root in args.runs:
        allocation = read(root / "plan.json")
        if allocation["release"]["plan_sha256"] != sha(args.plan / "plan.json"):
            raise ValueError("allocation belongs to another frozen plan")
        released = {c["id"]:c for c in selected_chunks(args.plan, allocation["release"], protocol)}
        model_id=next(iter(released.values()))['model_id']
        context_profile=allocation_context(args.plan,plan,protocol,allocation['release'],model_id)
        qualified = allocation_qualification(plan, protocol, allocation["release"],model_id,context_profile)
        if allocation["qualification"] != qualified or allocation.get('context_profile') != context_profile:
            raise ValueError("allocation qualification differs from admitted receipt")
        terminal = read(root / "terminal.json")
        if terminal["status"] not in ("complete_released_chunks", "invalid"):
            raise ValueError("allocation is not terminal")
        allocations.append(dict(root=str(root), status=terminal["status"], terminal_sha256=sha(root / "terminal.json"),
            job_id=allocation["slurm_job_id"], released_resources=allocation["release"].get("resources"),
            qualification=qualified,
            context_profile=context_profile,context_guard_accounting=terminal.get('context_guard_accounting'),
            partial_provider_costs=terminal["provider_event_accounting"]))
        for item in terminal["completed"]:
            chunk_id = item["chunk_id"]
            if chunk_id not in released:
                raise ValueError("unreleased completed chunk")
            if chunk_id not in chunks:
                continue
            if chunk_id in seen or chunk_id not in allocation["release"]["chunk_ids"]:
                raise ValueError("duplicate or unreleased chunk")
            seen.add(chunk_id)
            path = root / (chunk_id + ".jsonl")
            report, verified = accepted_chunk(root, released[chunk_id], qualified,context_profile)
            if item != verified:
                raise ValueError("allocation completed item differs from verified chunk")
            attach_event_costs(report, path, references)
            rows.extend(report.rows)
    if seen != set(chunks):
        raise ValueError(f"incomplete {args.track}/{args.stage} population: {len(seen)}/{len(chunks)} chunks; no quality summary")
    if args.track == "E":
        result = analysis.external(rows, cells, read(enrollment / "selected-metadata.json"))
        result["relative_tolerance_sensitivity"] = analysis.tolerance_sensitivity(
            rows, cells, read(enrollment / "selected-metadata.json"))
        result["analysis_amendment_sha256"] = sha(args.plan / "analysis-amendment.json")
    else:
        contrasts = read(args.plan / "position-prompts.json")["primary_contrasts"]
        result = analysis.positions(rows, cells, contrasts)
    result.update(track=args.track, stage=args.stage, plan_sha256=sha(args.plan / "plan.json"),
        context_amendment_sha256=plan['context_amendment_sha256'],
        coordinator_sha256=sha(__file__), analysis_sha256=sha(analysis.__file__), allocations=allocations,
        allocation_cost_limit="Recorded episode costs plus separately reported allocation resources; invalid partial calls are not silently zero",
        complete_chunks=len(seen), model_count=len({c["model_id"] for c in cells}),
        scope="Complete frozen track and stage only; other project acceptance remains separate")
    verify_plan(args.plan)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps(dict(track=args.track, stage=args.stage, episodes=len(rows), complete_chunks=len(seen))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--track", choices=("P", "E"), required=True)
    parser.add_argument("--stage", choices=("core", "extension", "all20"), default="core")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("research analysis requires Slurm")
    main(args)
