"""Read complete development records in Slurm; export error classes, never programs."""
import argparse
import ast
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path


def main(root, verifier, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("diagnostic analysis requires Slurm")
    spec = importlib.util.spec_from_file_location("diagnostic_verifier", verifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qualification = json.loads((root / "terminal.json").read_text())
    if qualification["status"] != "passed":
        raise ValueError("complete qualification required")
    result = {}
    for name, count in (("development", qualification["details"]["development_episodes"]),
                        ("external-development", qualification["details"]["external_development_episodes"])):
        report = module.checked_report(root / (name + ".jsonl"), count)
        groups = {}
        for key, rows in report.groups().items():
            errors = Counter()
            for row in rows:
                for step in row["steps"]:
                    if not step["error"]:
                        continue
                    try:
                        error = ast.literal_eval(step["error"])
                    except (ValueError, SyntaxError):
                        error = dict(type="UnparsedError", message=step["error"])
                    errors[(error.get("type", "unknown"), str(error.get("message", ""))[:300])] += 1
            groups[f"{key[0]}_{'library' if key[1] else 'scratch'}"] = [
                dict(type=kind, message=message, count=n)
                for (kind, message), n in errors.most_common()]
        result[name] = groups
    with output.open("x") as stream:
        json.dump(dict(scope="complete public development errors; not efficacy evidence",
                       errors=result), stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("verifier", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.root, args.verifier, args.output)
