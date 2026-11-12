"""Check that Figure 2 (b) shows what the judge really returns.

The figure (figures/episode.tex) quotes run1.py and run2.py and the judge's verdicts for them on
task momentum_a. This runs both programs through the judge and the hidden grader on every seed
and on the real panel, and fails if any verdict differs from what the figure shows.

    python3 check.py        (needs FIN_TASKS_REAL_PANEL or the cached real panel)
"""
from pathlib import Path

from fin_skills.tasks import suite
from fin_skills.tasks.judge import Judge

HERE = Path(__file__).resolve().parent
SHOWN = {
    "run1.py": ({"finite_output": True, "split_representation": False, "execution_lag": True}, False, False),
    "run2.py": ({"finite_output": True, "split_representation": True, "execution_lag": True}, True, True),
}


def main() -> None:
    task = next(t for t in suite().tasks if t.id == "momentum_a")
    for name, expected in SHOWN.items():
        code = (HERE / name).read_text()
        for seed in (11, 23, 37, "real"):
            judge = Judge(task, seed=seed, library=False)
            seen = judge.check(code)
            got = (dict(seen.checks), seen.accepted, judge.grade(code))
            assert got == expected, f"{name} seed {seed}: {got} != {expected}"
        print(f"{name}: as shown, on seeds 11, 23, 37 and the real panel")


if __name__ == "__main__":
    main()
