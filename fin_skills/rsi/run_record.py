"""Exclusive, pre-inference records for the public regression benchmark.

This is an audit trail, not independent custody or an authorization to run a campaign.
An interrupted output cannot be appended to or silently retried by this runner.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from contextvars import ContextVar
import time

EXECUTION_BUNDLE = ContextVar("fin_task_execution_bundle", default=None)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    out = {"fin_skills/" + p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob("*")) if p.is_file()
           and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
    hook = root.parent / "benchmarks/agent_study/linux_sandbox.py"
    out["benchmarks/agent_study/linux_sandbox.py"] = hashlib.sha256(hook.read_bytes()).hexdigest()
    for name in ("external_packets.py", "external_study.py"):
        adapter = hook.with_name(name)
        if adapter.is_file():
            out[f"benchmarks/agent_study/{name}"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    return out


class RunRecord:
    def __init__(self, path: Path | None, protocol: dict):
        self.path, self.protocol = path, protocol
        self.protocol_sha256 = digest(protocol)
        self.rows = 0
        self.terminal = False
        self.started = time.monotonic()
        self.bundle = None
        if path is None:
            return
        self.protocol_path = Path(str(path) + ".protocol.json")
        self.state_path = Path(str(path) + ".state.json")
        self.events_path = Path(str(path) + ".events.jsonl")
        self.terminal_path = Path(str(path) + ".terminal.json")
        self.bundle = Path(str(path) + ".sources")
        if any(p.exists() for p in (path, self.protocol_path, self.state_path, self.events_path,
                                   self.terminal_path, self.bundle)):
            raise FileExistsError("benchmark output already exists; resume/retry is not implicit")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass
        with self.events_path.open("x", encoding="utf-8"):
            pass
        with self.protocol_path.open("x", encoding="utf-8") as fh:
            json.dump(protocol, fh, sort_keys=True, indent=2, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        if protocol.get("source_hashes"):
            self.bundle.mkdir()
            root = Path(__file__).resolve().parents[2]
            for name, expected in protocol["source_hashes"].items():
                content = (root / name).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("source changed while freezing execution bundle")
                target = self.bundle / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        self.state("running")

    def event(self, entry):
        if self.terminal:
            raise ValueError("terminal event log is immutable")
        if self.path is not None:
            with self.events_path.open("a", encoding="utf-8") as fh:
                body = dict(entry, recorded_utc=datetime.now(timezone.utc).isoformat(),
                            elapsed_seconds=time.monotonic() - self.started)
                fh.write(json.dumps(body, sort_keys=True, allow_nan=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def append(self, row: dict) -> None:
        if self.terminal:
            raise ValueError("a terminal run cannot receive more results")
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        self.rows += 1

    def state(self, status: str, **details) -> None:
        if self.terminal:
            raise ValueError("terminal receipt is immutable")
        if self.path is None:
            self.terminal = status in ("complete", "invalid")
            return
        body = dict(status=status, protocol_sha256=self.protocol_sha256,
                    completed_episodes=self.rows, expected_episodes=len(self.protocol["plan"]),
                    **details)
        self.event(dict(kind="run_state", **body))
        if status in ("complete", "invalid"):
            body["episodes_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
            body["events_sha256"] = hashlib.sha256(self.events_path.read_bytes()).hexdigest()
            with self.terminal_path.open("x", encoding="utf-8") as fh:
                json.dump(body, fh, sort_keys=True, indent=2, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            self.terminal = True
        temp = Path(str(self.state_path) + ".tmp")
        with temp.open("w", encoding="utf-8") as fh:
            json.dump(body, fh, sort_keys=True, indent=2, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, self.state_path)
