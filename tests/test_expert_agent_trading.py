"""Test verification and reproducibility of the expert agent trading study."""
import importlib.util
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "benchmarks/agent_study/reports/20261007-expert-agent-trading/verify_study.py"
SPEC = importlib.util.spec_from_file_location("verifier", FILE)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_expert_agent_snapshot_verification():
    assert verifier.verify_snapshot() is True
