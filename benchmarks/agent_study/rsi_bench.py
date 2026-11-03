"""Beacon harness for the RSI benchmark (fin_skills.rsi, fin_skills.tasks).

Stage 1, model qualification (dev tasks only, no held-out task is touched):

    python -m benchmarks.agent_study.rsi_bench qualify ROOT --inventory INV --models a,b,c

For every model it starts one child process (each with its own HF_HUB_CACHE and a fresh GPU),
loads the pinned weights with TransformersChat, and runs 2 dev tasks x {no library, library} x
{NoImprove, CheckFeedback} on seed 11 under the standard Budget. ROOT/qualification/<id>.json
records load status and time, hardware, versions, peak GPU memory, every model response (text,
finish reason, tokens, seconds) and every episode with its dev grade. A model whose file
already exists is skipped, never overwritten; a load failure is recorded, not retried.

The task worker is confined with benchmarks.agent_study.linux_sandbox:confine unless
FIN_TASKS_CONFINE is already set. Run inside a Slurm GPU allocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

QUALIFY_TASKS = ("e4_ma_crossover", "momentum_a")
QUALIFY_METHODS = ("NoImprove", "CheckFeedback")
QUALIFY_SEED = 11
MAX_TOKENS = 2048


def inventory(path) -> dict:
    """Model id -> (repo, revision, cache) for entries the 10/17 inventory marks cached."""
    rows = json.loads(Path(path).read_text())
    return {r["id"]: (r["model"], r["revision"], r["cache_check"]["cache"])
            for r in rows if r.get("cached")}


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "numpy", "pandas", "scipy"):
        try:
            out[name] = __import__(name).__version__
        except Exception as exc:                 # recorded, not fatal
            out[name] = f"unavailable: {type(exc).__name__}"
    return out


class Recorder:
    """Passes calls to the backend and keeps each response for the record."""

    def __init__(self, backend):
        self.backend, self.responses = backend, []

    @property
    def seed(self):
        return self.backend.seed

    @seed.setter
    def seed(self, value):
        self.backend.seed = value

    def __call__(self, messages):
        response = self.backend(messages)
        choice = response["choices"][0]
        self.responses.append(dict(text=choice["message"]["content"],
                                   finish_reason=choice.get("finish_reason"),
                                   usage=response.get("usage"),
                                   seconds=response.get("generation_seconds")))
        return response


def qualify_one(root: Path, model_id: str, model: str, revision: str, *,
                backend_factory=None) -> dict:
    from fin_skills import rsi
    from fin_skills.tasks import suite

    out = root / "qualification" / f"{model_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return json.loads(out.read_text())
    tasks = suite()
    record = dict(model_id=model_id, model=model, revision=revision,
                  hf_hub_cache=os.environ.get("HF_HUB_CACHE"),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  confine=os.environ.get("FIN_TASKS_CONFINE"), versions=_versions(),
                  suite_sha256=tasks.sha256, tasks=list(QUALIFY_TASKS),
                  methods=list(QUALIFY_METHODS), seed=QUALIFY_SEED,
                  budget=rsi.Budget().as_dict(), max_tokens=MAX_TOKENS,
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    started = time.monotonic()
    try:
        if backend_factory is None:
            from benchmarks.agent_study.transformers_chat import TransformersChat
            backend = TransformersChat(model, revision, max_tokens=MAX_TOKENS, seed=QUALIFY_SEED)
        else:
            backend = backend_factory(model, revision)
    except Exception as exc:
        record.update(loaded=False, load_error=f"{type(exc).__name__}: {str(exc)[:2000]}",
                      load_seconds=time.monotonic() - started)
        out.write_text(json.dumps(record, indent=2))
        return record
    record.update(loaded=True, load_seconds=time.monotonic() - started,
                  hardware=getattr(backend, "hardware", None),
                  context_limit=getattr(backend, "context_limit", None))
    model_calls = Recorder(backend)
    episodes = []
    for task_id in QUALIFY_TASKS:
        task = tasks.get(task_id)
        assert tasks.splits[task.algorithm] == "dev", "qualification uses dev tasks only"
        for library in (False, True):
            for name in QUALIFY_METHODS:
                model_calls.seed = QUALIFY_SEED
                first = len(model_calls.responses)
                t0 = time.monotonic()
                episode = getattr(rsi, name)(library=library).solve(
                    model_calls, task, seed=QUALIFY_SEED, budget=rsi.Budget())
                judge = episode.__dict__.pop("_judge")
                episode.valid = bool(episode.steps) and judge.grade(episode.code)
                episodes.append(dict(
                    episode.as_dict(), wall_seconds=time.monotonic() - t0,
                    responses=model_calls.responses[first:],
                    infrastructure_errors=[r for r in judge.records
                                           if r.get("infrastructure_error")]))
    record["episodes"] = episodes
    try:
        import torch
        record["peak_gpu_bytes"] = [torch.cuda.max_memory_allocated(i)
                                    for i in range(torch.cuda.device_count())]
    except Exception:
        record["peak_gpu_bytes"] = None
    responses = [r for e in episodes for r in e["responses"]]
    record["summary"] = dict(
        episodes=len(episodes), responses=len(responses),
        fenced=sum("```" in r["text"] for r in responses),
        truncated=sum(r["finish_reason"] == "length" for r in responses),
        executed_final=sum(bool(e["steps"]) and e["steps"][-1]["executed"] for e in episodes),
        accepted=sum(e["accepted"] for e in episodes), valid=sum(e["valid"] for e in episodes),
        infrastructure_errors=sum(len(e["infrastructure_errors"]) for e in episodes),
        mean_episode_seconds=sum(e["wall_seconds"] for e in episodes) / max(1, len(episodes)))
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with out.open("x") as fh:                  # never overwrite an earlier record
        json.dump(record, fh, indent=2)
    return record


def qualify(root: Path, inventory_path: Path, models: list[str]) -> None:
    inv = inventory(inventory_path)
    missing = [m for m in models if m not in inv]
    if missing:
        raise SystemExit(f"not cached in the inventory: {missing}")
    os.environ.setdefault("FIN_TASKS_CONFINE", "benchmarks.agent_study.linux_sandbox:confine")
    for model_id in models:
        model, revision, cache = inv[model_id]
        env = dict(os.environ, HF_HUB_CACHE=cache, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        print(json.dumps(dict(phase="qualify", model=model_id, started=time.time())), flush=True)
        proc = subprocess.run([sys.executable, "-B", "-m", "benchmarks.agent_study.rsi_bench",
                               "qualify-one", str(root), model_id, model, revision], env=env)
        print(json.dumps(dict(phase="qualify", model=model_id, returncode=proc.returncode)),
              flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(prog="rsi_bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("qualify")
    q.add_argument("root", type=Path)
    q.add_argument("--inventory", type=Path, required=True)
    q.add_argument("--models", required=True)
    one = sub.add_parser("qualify-one")
    one.add_argument("root", type=Path)
    for name in ("model_id", "model", "revision"):
        one.add_argument(name)
    a = p.parse_args(argv)
    if a.cmd == "qualify":
        qualify(a.root, a.inventory, [m for m in a.models.split(",") if m])
    else:
        record = qualify_one(a.root, a.model_id, a.model, a.revision)
        print(json.dumps(dict(model=a.model_id, loaded=record.get("loaded"),
                              summary=record.get("summary"))), flush=True)


if __name__ == "__main__":
    main()
