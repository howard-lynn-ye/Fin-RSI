"""Audit actual repair coverage of one completed DEVELOPMENT qualification in Slurm."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_report(path, expected):
    from fin_skills.rsi.bench import Report
    from fin_skills.rsi.run_record import digest
    protocol = json.loads(Path(str(path) + ".protocol.json").read_text())
    terminal = json.loads(Path(str(path) + ".terminal.json").read_text())
    if (terminal["status"] != "complete" or terminal["protocol_sha256"] != digest(protocol)
            or terminal["episodes_sha256"] != sha(path)
            or terminal["events_sha256"] != sha(Path(str(path) + ".events.jsonl"))):
        raise ValueError("development receipt is incomplete or hashes differ")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    report = Report(rows=rows, complete=True, protocol=protocol)
    report._validate()
    if (len(rows) != expected or
            any(r["protocol_sha256"] != terminal["protocol_sha256"] for r in rows)):
        raise ValueError("qualification and development records disagree")
    return report


def describe(report):
    rows = report.rows
    by_arm = {}
    for arm, members in report.groups().items():
        by_arm[f"{arm[0]}_{'library' if arm[1] else 'scratch'}"] = dict(
            episodes=len(members), statuses=dict(Counter(r["status"] for r in members)),
            attempt_counts=dict(Counter(str(len(r["steps"])) for r in members)),
            accepted=sum(r["accepted"] for r in members),
            valid=sum(r["valid"] for r in members),
            first_valid=sum(r["first_valid"] for r in members),
            accepted_but_wrong=sum(r["accepted"] and not r["valid"] for r in members),
            final_library_used=sum(r["library_used"] for r in members),
            any_attempt_library_used=sum(any(s["library_calls"] for s in r["steps"])
                                         for r in members),
            execution_failed_attempts=sum(not s["executed"] for r in members for s in r["steps"]),
            failed_checks=dict(Counter(k for r in members for s in r["steps"]
                                       for k, ok in s["checks"].items() if not ok)))
    coverage = {str(k): sum(len(r["steps"]) >= k for r in rows) for k in (1, 2, 3)}
    return dict(episodes=len(rows), arms=by_arm,
        episodes_reaching_attempt=coverage, all_arms_reached_third_attempt=all(
            any(len(r["steps"]) >= 3 for r in members) for members in report.groups().values()),
        calls=sum(r["usage"]["calls"] for r in rows),
        generated_tokens=sum(r["usage"]["generated_tokens"] for r in rows))


def summarize(root):
    qualification = json.loads((root / "terminal.json").read_text())
    if qualification.get("status") != "passed" or qualification.get("phase") != "model":
        raise ValueError("complete successful model qualification required")
    path = root / "development.jsonl"
    report = checked_report(path, qualification["details"]["development_episodes"])
    result = dict(scope="public development diagnosis only", publication_ready=False,
        model_id=report.protocol["model_id"], positions=describe(report),
        qualification_sha256=sha(root / "terminal.json"),
        development_terminal_sha256=sha(Path(str(path) + ".terminal.json")),
        source_manifest_sha256=qualification["source_manifest_sha256"],
        limitations=["Natural episodes may not exercise every repair branch",
                     "No accuracy-based model exclusion or confirmatory inference",
                     "External scalar and positions results have different task contracts"])
    external_count = qualification["details"].get("external_development_episodes", 0)
    if external_count:
        external_path = root / "external-development.jsonl"
        external = checked_report(external_path, external_count)
        if external.protocol["study_manifest"]["role"] != "development":
            raise ValueError("qualification cannot include external test results")
        result["external_scalar"] = describe(external)
        result["external_terminal_sha256"] = sha(Path(str(external_path) + ".terminal.json"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("research result analysis requires Slurm")
    result = summarize(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True))
