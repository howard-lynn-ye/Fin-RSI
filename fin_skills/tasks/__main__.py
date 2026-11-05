"""python -m fin_skills.tasks real-fetch [PATH] | real-check [PATH]

real-fetch downloads the real-price panel (fin_skills.tasks.real) and prints its SHA-256;
real-check verifies a file that is already there.
"""
from __future__ import annotations

import sys

from . import real


def main(argv) -> int:
    commands = {"real-fetch": "fetch", "real-check": "check"}
    if not argv or argv[0] not in commands:
        print(__doc__)
        return 2
    return real.main([commands[argv[0]], *argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
