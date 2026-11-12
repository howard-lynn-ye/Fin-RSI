"""Find the program in a model reply.

A reply is expected to hold the program in a python fence. Three things went wrong with a
single search for a lowercase-tagged fence, all seen in the frozen study:

* a fence tagged ```Python or ```python3 was not found, so the whole reply, prose included,
  ran as the program and failed with a SyntaxError on its first line;
* a reply cut off at the token limit has an opening fence and no closing one, and was also
  run whole;
* the closing fence of an earlier block (```bash, ```text) could be read as the opening of a
  python block, so the prose after it became the program.

blocks() pairs every opening fence with the next closing fence, in order. extract() takes
the first block tagged as python (or untagged); a block left open runs to the end of the
reply. A reply with no such block is still returned whole, so the judge reports the
SyntaxError as before; reply_format() says which case a reply was, so that answers in prose
can be counted apart from programs that do not run.
"""
from __future__ import annotations

import re

OPEN = re.compile(r"```[ \t]*([^\s`]*)[^\n`]*\n")
CLOSE = "```"
PYTHON_TAGS = frozenset({"", "python", "python3", "py", "py3"})


def blocks(text: str) -> list[tuple[str, str, bool]]:
    """Fenced blocks as (tag, body, closed), in the order they appear. Tags are lowercased."""
    found, pos = [], 0
    while True:
        start = text.find(CLOSE, pos)
        if start < 0:
            return found
        opening = OPEN.match(text, start)
        tag = opening.group(1).strip("{}.").lower() if opening else None
        line_start = text[text.rfind("\n", 0, start) + 1:start].strip() == ""
        # An opening fence starts a line, or names python mid-line ("Here: ```python").
        if opening is None or not (line_start or (tag and tag in PYTHON_TAGS)):
            pos = start + len(CLOSE)           # inline ``` or a stray closing fence
            continue
        end = text.find(CLOSE, opening.end())
        if end < 0:
            found.append((tag, text[opening.end():], False))
            return found
        found.append((tag, text[opening.end():end], True))
        pos = end + len(CLOSE)


def _python(text: str):
    return next((b for b in blocks(text) if b[0] in PYTHON_TAGS), None)


def extract(text: str) -> str:
    """The first python (or untagged) fence, closed or not; the whole reply when there is none."""
    block = _python(text)
    return (block[1] if block else text).strip() + "\n"


def reply_format(text: str) -> str:
    """fenced, unclosed_fence (cut off before the closing fence), or no_fence."""
    block = _python(text)
    if block is None:
        return "no_fence"
    return "fenced" if block[2] else "unclosed_fence"
