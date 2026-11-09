"""Count how often a model answered without a python fence, from recorded run events.

A diagnostic over finished runs; it never changes a grade. The frozen study ran a reply with no
fence as code, so an answer written in prose was recorded as a program that failed with a
SyntaxError. This separates those replies from programs that ran or failed on their own.

    python scripts/reply_formats.py RUN_DIR_OR_EVENTS_FILE ... > formats.json

Every writer call is classified with fin_skills.rsi.fences.reply_format (fenced,
unclosed_fence, no_fence), grouped by the chunk or unit named in the events file and by the
library arm. "final" counts only each episode's last writer reply.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fin_skills.rsi.fences import reply_format  # noqa: E402

SUFFIX = ".jsonl.events.jsonl"
CHUNK = re.compile(r"^(?P<track>[PE])-(?P<model>.+)-\d{3}$")


def label(path: Path) -> str:
    """P-<model> / E-<model> for frozen chunks; the unit file name otherwise."""
    stem = path.name[:-len(SUFFIX)]
    match = CHUNK.match(stem)
    return f"{match['track']}-{match['model']}" if match else stem


def tally(paths):
    calls, final = defaultdict(Counter), defaultdict(Counter)
    for path in paths:
        group, active, last = label(path), None, {}
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["kind"] == "episode_started":
                active = (event["library"], tuple(event[k] for k in ("method", "task_id", "seed")))
            elif event["kind"] == "fit_started":
                active = None
            elif (event["kind"] == "call_completed" and active is not None
                  and event.get("role", "writer") == "writer"):
                kind = reply_format(event["content"])
                calls[(group, active[0])][kind] += 1
                last[active] = kind
        for (library, _), kind in last.items():
            final[(group, library)][kind] += 1
    return calls, final


def files(args):
    for item in args:
        item = Path(item)
        if item.is_dir():
            yield from sorted(item.rglob("*" + SUFFIX))
        else:
            yield item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    calls, final = tally(list(files(args.paths)))
    rows = [dict(group=group, library=library, calls=dict(calls[(group, library)]),
                 final=dict(final[(group, library)]))
            for group, library in sorted(calls, key=lambda k: (k[0], str(k[1])))]
    json.dump(rows, sys.stdout, indent=1, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
