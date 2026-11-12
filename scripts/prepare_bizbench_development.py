"""Pin and inspect ONLY the public BizBench training file on a Slurm compute node.

No model calls or execution of upstream reference programs. Reports descriptive source
coverage, never independent task counts or experimental accuracy.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

REVISION = "0a793f2f886156902c72b4a22cab82bb9dceaecf"
FILE = "data/train-00000-of-00001-300a28d2e31daec3.parquet"
SHA256 = "d0af1868ac46b082f153981b813b97b78707153b7a5cd872b30490bec47b3cc1"
SIZE = 19140183
URL = f"https://huggingface.co/datasets/kensho/bizbench/resolve/{REVISION}/{FILE}"


def write(path, body):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, sort_keys=True, indent=2, allow_nan=False)


def main(root, existing=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("dataset preparation requires Slurm")
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(dataset="kensho/bizbench", revision=REVISION, source_path=FILE,
        sha256=SHA256, bytes=SIZE, source_url=URL, split="train",
        license_observed="Apache-2.0 in the pinned official dataset card",
        scope="development intake; no final test material or model calls",
        source_identity="official HF tree API LFS oid and size, read 2026-11-08",
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        owner_chat="01a1bc9a-3383-7b41-800d-b1cb5bfefff9",
        started_utc=datetime.now(timezone.utc).isoformat(), publication_ready=False)
    write(root / "plan.json", plan)
    try:
        destination = existing if existing is not None else root / "train.parquet"
        digest, total = hashlib.sha256(), 0
        if existing is None:
            with urllib.request.urlopen(URL, timeout=120) as response, destination.open("xb") as stream:
                while block := response.read(1 << 20):
                    total += len(block)
                    if total > SIZE:
                        raise ValueError("download exceeds frozen source size")
                    digest.update(block)
                    stream.write(block)
        else:
            with existing.open("rb") as response:
                while block := response.read(1 << 20):
                    total += len(block)
                    if total > SIZE:
                        raise ValueError("existing input exceeds frozen source size")
                    digest.update(block)
        if total != SIZE or digest.hexdigest() != SHA256:
            raise ValueError("download differs from frozen source identity")
        import pandas as pd
        import pyarrow
        data = pd.read_parquet(destination)
        required = {"question", "answer", "task", "context", "context_type", "options", "program"}
        if set(data.columns) != required:
            raise ValueError("unexpected source schema; inspect before adapting")
        counts, groups = Counter(data["task"]), {}
        for name in ("CodeFinQA", "CodeTATQA", "CodeTAT-QA", "FinCode"):
            rows = data[data.task == name]
            if not len(rows):
                continue
            contexts = [str(v) for v in rows.context]
            supported = sum(isinstance(v, str) and bool(re.fullmatch(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", v.strip()))
                for v in rows.answer)
            groups[name] = dict(rows=len(rows), plain_numeric_answers=int(supported),
                other_answer_formats=len(rows)-int(supported),
                unique_context_blocks=len(set(contexts)),
                context_types=dict(Counter(str(v) for v in rows.context_type)),
                max_question_characters=max(len(str(v)) for v in rows.question),
                max_context_characters=max(map(len, contexts)),
                cluster_warning="Exact context blocks are not verified independent source documents",
                source_rows=[int(v) for v in rows.index])
        result = dict(plan, status="verified_development_source", rows=len(data),
                      task_counts=dict(counts), code_tasks=groups,
                      input_path=str(destination.resolve()),
                      runtime=dict(pandas=pd.__version__, pyarrow=pyarrow.__version__),
                      execution="No program or model executed; no test split downloaded")
    except BaseException as exc:
        write(root / "terminal.json", dict(plan, status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise
    write(root / "terminal.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "code_tasks"}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--existing", type=Path)
    args = parser.parse_args()
    main(args.root, args.existing)
