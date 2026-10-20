"""Confined worker with direct, evidence-bounded RAG access."""
from benchmarks.agent_study.trading_runtime_v9 import Tools
from benchmarks.agent_study.trading_worker_v6 import main


if __name__ == '__main__':
    main(Tools)
