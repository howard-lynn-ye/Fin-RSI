"""Finding the program in a model reply (fin_skills.rsi.fences)."""
import pytest

from fin_skills import rsi
from fin_skills.agents import extract as team_extract
from fin_skills.rsi.fences import blocks, extract, reply_format

CASES = [
    # (reply, program, format)
    ("a\n```python\nx = 1\n```\n```python\ny\n```", "x = 1\n", "fenced"),
    ("```\nx = 1\n```", "x = 1\n", "fenced"),
    ("Here it is:\n```Python\nx = 1\n```\n", "x = 1\n", "fenced"),
    ("```python3\nx = 1\n```", "x = 1\n", "fenced"),
    ("```python\r\nx = 1\r\n```\r\n", "x = 1\n", "fenced"),          # Windows line endings
    ("```py title=solve.py\nx = 1\n```", "x = 1\n", "fenced"),
    # cut off at the token limit: the open block runs to the end of the reply
    ("Sure.\n```python\ndef solve(p, f):\n    return p", "def solve(p, f):\n    return p\n",
     "unclosed_fence"),
    # the closing fence of a non-python block is not the opening of a python one
    ("```bash\npip install x\n```\nThen run:\n```python\nx = 3\n```", "x = 3\n", "fenced"),
    ("```text\nnotes\n```\nThe answer is 3.", "```text\nnotes\n```\nThe answer is 3.\n",
     "no_fence"),
    ("The answer is 42.", "The answer is 42.\n", "no_fence"),
    ("", "\n", "no_fence"),
]


@pytest.mark.parametrize("reply, program, kind", CASES)
def test_extract_and_reply_format(reply, program, kind):
    assert extract(reply) == program
    assert reply_format(reply) == kind


def test_inline_backticks_are_not_fences():
    reply = "Use ```positions``` here.\n```python\nx = 1\n```"
    assert blocks(reply) == [("python", "x = 1\n", True)]
    assert extract(reply) == "x = 1\n"
    assert extract("Here is the program: ```python\nx = 1\n```") == "x = 1\n"


def test_one_parser_for_methods_and_team():
    assert rsi.extract is extract
    assert team_extract is extract
