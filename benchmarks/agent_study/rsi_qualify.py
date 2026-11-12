"""Production qualification on public development material, never a paper score.

Run once per fresh output directory inside Slurm. CPU qualification precedes any model
loading. Model failures are retained and are not retried by this program.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

TESTS = (
    "tests/test_task_execution_boundary.py", "tests/test_rsi_evaluation_contract.py",
    "tests/test_tasks_suite.py", "tests/test_rsi_methods.py",
    "tests/test_rsi_controlled_study.py", "tests/test_rsi_backend_adapter.py",
    "tests/test_rsi_qualification.py",
    "tests/test_scalar_execution.py", "tests/test_external_packets.py",
    "tests/test_external_study.py",
    "tests/test_rsi_fit_outcomes.py", "tests/test_learning_fit_qualification.py",
)
DEV_TASKS = ("e4_ma_crossover", "momentum_a")
OWNER = "01a1bc9a-3383-7b41-800d-b1cb5bfefff9"


def write_new(path, body):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(body, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def verify_source(source):
    manifest = json.loads((source / "source-manifest.json").read_text())
    for relative, expected in manifest.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("source manifest leaves its root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("source snapshot changed: " + relative)
    return hashlib.sha256((source / "source-manifest.json").read_bytes()).hexdigest()


def junit_counts(path):
    suites = ET.parse(path).getroot().findall("testsuite")
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if not counts["tests"] or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise RuntimeError(f"production tests require nonempty, unskipped success: {counts}")
    return counts


def collected_nodes(text):
    nodes = [line.strip() for line in text.splitlines()
             if line.startswith("tests/") and "::" in line]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("test collection must be nonempty with unique node IDs")
    return nodes


def cpu(source, output):
    from benchmarks.agent_study.linux_sandbox import confinement_available
    if not confinement_available():
        raise RuntimeError("production kernel lacks required confinement")
    base = [sys.executable, "-m", "pytest", "-q", f"--rootdir={source}",
            "-o", "cache_dir=" + str(output / "pytest-cache")]
    collection = subprocess.run([*base, "--collect-only", *TESTS], cwd=source,
                                capture_output=True, text=True, timeout=120)
    (output / "collection.log").write_text(collection.stdout + collection.stderr)
    collection.check_returncode()
    nodes = collected_nodes(collection.stdout)
    workers = min(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    if workers < 1:
        raise ValueError("positive allocated CPU count required")
    write_new(output / "test-plan.json", dict(nodes=nodes, workers=workers,
              scope="all originally declared test modules plus the new typed contract tests"))
    deadline = time.monotonic() + 3300

    def run_case(item):
        index, node = item
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("production CPU qualification reached its 55-minute process cap")
        case = output / f"case-{index:04d}"
        case.mkdir()
        command = [*base, "-o", f"cache_dir={case / 'cache'}",
                   f"--junitxml={case / 'tests.xml'}", node]
        write_new(case / "request.json", dict(node=node, command=command))
        with (case / "tests.log").open("x", encoding="utf-8") as stream:
            with subprocess.Popen(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True) as proc:
                try:
                    code = proc.wait(timeout=remaining)
                except BaseException:
                    # Only this newly created test process group; no other jobs or users.
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    raise
        if code:
            raise RuntimeError(f"production test {node} failed: exit {code}; inspect {case.name}")
        counts = junit_counts(case / "tests.xml")
        if counts["tests"] != 1:
            raise ValueError("one selected test node must produce exactly one test result")
        write_new(case / "terminal.json", dict(node=node, status="passed", counts=counts))
        return counts

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_case, enumerate(nodes)))
    counts = {key: sum(row[key] for row in results)
              for key in ("tests", "failures", "errors", "skipped")}
    if counts["tests"] != len(nodes):
        raise ValueError("production results do not cover the complete collected plan")
    return dict(tests=counts, confinement=True, workers=workers,
                complete_planned_test_nodes=len(nodes))


def development_tasks():
    from fin_skills.tasks import frozen_suite
    tasks = frozen_suite()
    selected = tuple(tasks.get(name) for name in DEV_TASKS)
    if any(tasks.splits[t.algorithm] != "dev" for t in selected):
        raise ValueError("qualification must use original development tasks only")
    # No fitting occurs. The benchmark's evaluation slot is used for DEVELOPMENT probes;
    # this does not reclassify the selected tasks as independently held out.
    return SimpleNamespace(dev=(), heldout=selected)


def full_context_probe(chat):
    import torch
    from benchmarks.agent_study.transformers_chat import FiniteScores, transfer_scope
    count = chat.context_limit - 1
    token = chat.tokenizer.encode("market", add_special_tokens=False)[-1]
    ids = torch.full((1, count), token, dtype=torch.long, device=chat.model.device)
    with torch.inference_mode(), transfer_scope(chat.hardware.get("placement_qualification"), torch):
        result = chat.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=1, do_sample=False, use_cache=True,
            logits_processor=[FiniteScores(count)], pad_token_id=chat.tokenizer.eos_token_id)
    if result.shape[-1] != count + 1:
        raise RuntimeError("full configured context did not complete")
    del result, ids
    torch.cuda.empty_cache()
    return dict(input_tokens=count, output_tokens=1,
                scope="GPU memory and generation path; synthetic token input, not task accuracy")


def external_development_tasks(external_train):
    from benchmarks.agent_study.external_packets import ScalarPolicy, packets
    from benchmarks.agent_study.external_study import ScalarTask
    from scripts.prepare_bizbench_development import SHA256, REVISION
    import pandas as pd
    if hashlib.sha256(external_train.read_bytes()).hexdigest() != SHA256:
        raise ValueError("external training source differs from frozen identity")
    frame = pd.read_parquet(external_train)
    records = frame[frame.task == "CodeFinQA"].head(2)
    if len(records) != 2:
        raise ValueError("two leading source CodeFinQA training rows required")
    policy = ScalarPolicy("Use the units and rounding explicitly requested by the question.",
                          0, 0, "adapted strict numeric match; public question controls units")
    return tuple(ScalarTask(*packets(source_row.to_dict(), revision=REVISION,
        split="train", row_index=int(index),
        source_cluster="training-context-block:" + hashlib.sha256(
            str(source_row.context).encode()).hexdigest(), policy=policy))
        for index, source_row in records.iterrows())


def model(source, output, *, cpu_receipt, inventory, model_id, external_train=None):
    from fin_skills.rsi import study
    from fin_skills.rsi.meter import Budget
    from benchmarks.agent_study.rsi_backend import BoundedTransformersChat
    from benchmarks.agent_study.transformers_chat import TransformersChat, tokenize_chat
    receipt = json.loads(cpu_receipt.read_text())
    if receipt.get("status") != "passed" or receipt.get("phase") != "cpu":
        raise ValueError("passed production CPU qualification required")
    if receipt["source_manifest_sha256"] != verify_source(source):
        raise ValueError("CPU qualification source differs from model source")
    rows = json.loads(inventory.read_text())
    rows = [row for row in rows if row["id"] == model_id and row.get("cached")]
    if len(rows) != 1:
        raise ValueError("one pinned cached inventory entry required")
    row = rows[0]
    if len(row["revision"]) != 40 or any(c not in "0123456789abcdef" for c in row["revision"]):
        raise ValueError("complete model revision required")
    os.environ["HF_HUB_CACHE"] = row["cache_check"]["cache"]
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("offline cache required; no model download is authorized here")
    tasks = development_tasks()
    external_tasks = external_development_tasks(external_train) if external_train else ()
    budget = Budget(attempts=3, calls=6, generated_tokens=12000)
    write_new(output / "model-plan.json", dict(model=row["model"], revision=row["revision"],
        model_id=model_id, cache=row["cache_check"]["cache"], per_call_output_cap=2048,
        seed=11, development_only=True, publication_ready=False,
        tasks=[t.manifest() for t in tasks.heldout],
        external_development_tasks=[t.manifest() for t in external_tasks],
        controlled_plan=study.manifest(tasks.heldout, seeds=(11,), budget=budget),
        probes=["full configured context", "identical-seed repeat", "two-turn cap/usage"],
        selection_rule="Engineering qualification only; no accuracy threshold or model ranking"))
    started = time.monotonic()
    chat = TransformersChat(row["model"], row["revision"], max_tokens=2048, seed=11)
    backend = BoundedTransformersChat(chat)
    write_new(output / "loaded.json", dict(seconds=time.monotonic() - started,
                                           backend=backend.describe()))
    context = full_context_probe(chat)
    write_new(output / "context.json", context)
    messages = [dict(role="system", content="Follow the requested output format."),
                dict(role="user", content="Write the Python assignment x = 1.")]
    probes = []
    for index, (packet, cap, seed) in enumerate(((messages, 32, 17), (messages, 32, 17),
            (messages + [dict(role="assistant", content="x = 1"),
                         dict(role="user", content="Now change x to 2." )], 7, 23))):
        write_new(output / f"probe-{index}-request.json", dict(messages=packet, cap=cap, seed=seed))
        response = backend.generate_bounded(packet, max_new_tokens=cap, seed=seed)
        write_new(output / f"probe-{index}-response.json", response)
        usage = response["usage"]
        count = len(tokenize_chat(chat.tokenizer, packet)["input_ids"])
        if not (usage["prompt_tokens"] == count and 0 < usage["completion_tokens"] <= cap
                and usage["total_tokens"] == count + usage["completion_tokens"]):
            raise RuntimeError("backend token/cap contract failed")
        probes.append(response)
    if (probes[0]["choices"] != probes[1]["choices"] or probes[0]["usage"] != probes[1]["usage"]):
        raise RuntimeError("identical seed did not reproduce the same qualification response")
    report = study.run(backend, tasks, seeds=(11,), budget=budget,
                       out=output / "development.jsonl", model_id=row["model"],
                       execution_mode="model")
    report._validate()
    external_report = None
    if external_tasks:
        from benchmarks.agent_study.external_study import run as external_run
        external_report = external_run(backend, external_tasks, seeds=(11,),
            out=output / "external-development.jsonl", model_id=row["model"])
        external_report._validate()
    import torch
    return dict(loaded=True, backend=backend.describe(), context=context,
                probes=3, development_episodes=len(report.rows),
                calls=sum(r["usage"]["calls"] for r in report.rows),
                generated_tokens=sum(r["usage"]["generated_tokens"] for r in report.rows),
                actual_attempt_counts=dict(Counter(str(len(r["steps"])) for r in report.rows)),
                actual_statuses=dict(Counter(r["status"] for r in report.rows)),
                external_development_episodes=len(external_report.rows) if external_report else 0,
                external_calls=sum(r["usage"]["calls"] for r in external_report.rows)
                               if external_report else 0,
                peak_gpu_bytes=[torch.cuda.max_memory_allocated(i)
                                for i in range(torch.cuda.device_count())],
                interpretation="Public development runtime qualification, not an efficacy result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cpu", "model"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--external-train", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("qualification requires a Slurm allocation")
    source = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(phase=args.phase, owner_chat=OWNER, slurm_job_id=os.environ["SLURM_JOB_ID"],
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=verify_source(source), publication_ready=False,
        scope="production qualification on public development fixtures", model_id=args.model_id)
    write_new(args.output / "plan.json", plan)
    started = time.monotonic()
    try:
        if args.phase == "cpu" and args.external_train is not None:
            write_new(args.output / "external-development-tasks.json", dict(
                tasks=[t.manifest() for t in external_development_tasks(args.external_train)],
                scope="two leading public training rows; no model called"))
        details = (cpu(source, args.output) if args.phase == "cpu" else
                   model(source, args.output, cpu_receipt=args.cpu_receipt,
                         inventory=args.inventory, model_id=args.model_id,
                         external_train=args.external_train))
        if verify_source(source) != plan["source_manifest_sha256"]:
            raise ValueError("source changed during qualification")
        result = dict(plan, status="passed", details=details, seconds=time.monotonic() - started)
    except BaseException as exc:
        write_new(args.output / "terminal.json", dict(plan, status="failed",
                  error=f"{type(exc).__name__}: {str(exc)[:2000]}", seconds=time.monotonic() - started))
        raise
    write_new(args.output / "terminal.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
