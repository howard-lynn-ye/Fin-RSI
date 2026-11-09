"""Slurm-only frozen public source intake; exports schema/identity metadata, no questions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def write(path, body):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, sort_keys=True, allow_nan=False)


def download(url, path, size, *, sha256=None, blob_sha=None):
    if size <= 0 or (sha256 is None) == (blob_sha is None):
        raise ValueError("one frozen source digest and size required")
    checksum = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    count = 0
    with urllib.request.urlopen(url, timeout=120) as response, path.open("xb") as stream:
        while block := response.read(1 << 20):
            count += len(block)
            if count > size:
                raise ValueError("source exceeds frozen size")
            checksum.update(block)
            blob.update(block)
            stream.write(block)
    if (count != size or (sha256 and checksum.hexdigest() != sha256)
            or (blob_sha and blob.hexdigest() != blob_sha)):
        raise ValueError("source differs from frozen digest/size")
    return dict(file=path.name, bytes=count, sha256=checksum.hexdigest(),
                git_blob_sha1=blob.hexdigest(), url=url)


def schema(value):
    if isinstance(value, dict):
        return {k: type(v).__name__ for k, v in value.items()}
    return type(value).__name__


def main(plan_path, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("public data intake requires Slurm")
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", dict(plan=plan, source_plan_sha256=hashlib.sha256(
        plan_path.read_bytes()).hexdigest(), started_utc=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ["SLURM_JOB_ID"]))
    records, metadata = [], {}
    try:
        item = plan["bizbench"]
        url = f"https://huggingface.co/datasets/{item['repository']}/resolve/{item['revision']}/{item['path']}"
        target = output / "bizbench-test.parquet"
        records.append(download(url, target, item["size"], sha256=item["sha256"]))
        import pandas as pd
        from benchmarks.agent_study.external_packets import numeric_reference
        frame = pd.read_parquet(target)
        groups = {}
        for name in ("CodeFinQA", "CodeTAT-QA", "CodeTATQA"):
            rows = frame[frame.task == name]
            if not len(rows):
                continue
            formats = Counter()
            for value in rows.answer:
                try:
                    numeric_reference(value)
                    formats["finite_numeric_literal"] += 1
                except ValueError:
                    formats["unscorable_literal"] += 1
            context_schema = []
            for value in rows.context.head(2):
                try:
                    parsed = json.loads(value)
                    # These keys are table labels, not stable schema field names.
                    # Keep research text on the compute host even in metadata exports.
                    context_schema.append(dict(type=type(parsed).__name__,
                        field_count=len(parsed) if isinstance(parsed, (dict, list)) else None,
                        value_types=sorted({type(v).__name__ for v in parsed.values()})
                            if isinstance(parsed, dict) else []))
                except (ValueError, TypeError):
                    context_schema.append(type(value).__name__)
            groups[name] = dict(rows=len(rows), reference_formats=dict(formats),
                context_types=dict(Counter(str(x) for x in rows.context_type)),
                context_schema_examples=context_schema,
                unique_exact_context_blocks=int(rows.context.nunique()))
        metadata["bizbench"] = dict(rows=len(frame), columns=list(frame.columns),
            task_counts=dict(Counter(str(x) for x in frame.task)), selected_source_groups=groups,
            grouping_limit="context blocks are not verified documents or independent clusters")
        for source in plan["upstream"]:
            repository, revision = source["repository"], source["revision"]
            files = []
            for entry in source["entries"]:
                filename = repository.replace("/", "--") + "--" + Path(entry["path"]).name
                target = output / filename
                url = f"https://raw.githubusercontent.com/{repository}/{revision}/{entry['path']}"
                records.append(download(url, target, entry["size"], blob_sha=entry["sha"]))
                rows = json.loads(target.read_text())
                if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
                    raise ValueError("upstream source must be a nonempty list of records")
                first = rows[0]
                detail = dict(file=filename, records=len(rows), record_schema=schema(first))
                for key in ("qa", "table", "questions", "paragraphs"):
                    value = first.get(key)
                    if isinstance(value, list):
                        value = value[0] if value else None
                    if value is not None:
                        detail[key + "_schema"] = schema(value)
                files.append(detail)
            metadata[repository] = files
        result = dict(status="verified_public_sources", files=records, metadata=metadata,
            selected_tasks=0, model_calls=0, reference_programs_executed=0,
            scope="Source intake and schema only; enrollment and source mapping remain pending",
            publication_ready=False)
    except BaseException as exc:
        write(output / "terminal.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}",
            completed_files=records, metadata=metadata, publication_ready=False))
        raise
    write(output / "terminal.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.plan, args.output)
