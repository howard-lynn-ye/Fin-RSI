"""Same OS confinement as v7, with on-demand study contracts in bound Python."""
from benchmarks.agent_study.trading_runtime_v8 import Tools
from benchmarks.agent_study.trading_worker_v6 import main


if __name__ == '__main__':
    main(Tools)
