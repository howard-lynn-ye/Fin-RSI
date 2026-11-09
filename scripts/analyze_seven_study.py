"""Require the entire frozen S matrix and fit ledger before descriptive summaries."""
import argparse
import os
from pathlib import Path

try:
    import run_seven_study as runner
    import seven_study as seven
    from summarize_rsi_qualification import checked_report
except ModuleNotFoundError as exc:
    if exc.name not in ("run_seven_study", "seven_study", "summarize_rsi_qualification"):
        raise
    from scripts import run_seven_study as runner, seven_study as seven
    from scripts.summarize_rsi_qualification import checked_report


def analyze(args):
    runner.verify_plan(args.plan)
    units = runner.read(args.plan / "units.json")
    indexed = {unit["id"]:unit for unit in units}
    seen, rows, receipts, allocations = set(), [], [], []
    for root in args.runs:
        allocation = runner.read(root / "allocation.json")
        terminal = runner.read(root / "terminal.json")
        if (allocation["plan_sha256"] != runner.sha(args.plan / "plan.json")
                or terminal["status"] not in ("invalid", "complete_released_units")):
            raise ValueError("S allocation belongs to another plan or is not terminal")
        released = {u["id"] for u in runner.selected_units(args.plan, allocation["release"])}
        allocations.append(dict(path=str(root), status=terminal["status"],
            allocation=allocation["allocation"], terminal_sha256=runner.sha(root / "terminal.json"),
            partial_provider_costs=terminal["provider_event_accounting"]))
        for item in terminal["completed"]:
            key = item["unit_id"]
            if key not in indexed or key not in released or key in seen:
                raise ValueError("duplicate, unplanned or unreleased S unit")
            seen.add(key)
            report, cost = runner.accepted_unit(root, indexed[key])
            if item != cost:
                raise ValueError("S accepted costs differ from original provider ledger")
            receipts.append(cost)
            rows.extend(report.rows)
    result = seven.complete_summary(rows, units, receipts)
    result.update(plan_sha256=runner.sha(args.plan / "plan.json"), allocations=allocations,
        analyzer_sha256=runner.sha(__file__), accounting_sha256=runner.sha(seven.__file__))
    runner.verify_plan(args.plan)
    runner.write(args.output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal S analysis requires Slurm")
    analyze(args)
