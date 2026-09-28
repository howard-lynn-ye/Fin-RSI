import subprocess
import sys
from pathlib import Path

from benchmarks.agent_study.codegen_fence_reanalysis import first_fence
from benchmarks.agent_study.codegen_utility import parse_code

ROOT = Path(__file__).resolve().parents[1]


def test_trailing_prose_is_dropped_but_code_is_untouched():
    text = '```python\ndef solve(p, f):\n    return p * 0.0\n```\n\nThis uses the 61st row.'
    code, status = first_fence(text)
    assert status == 'first_fence'
    assert code == 'def solve(p, f):\n    return p * 0.0'
    # The pre-registered parser keeps the prose, which then fails to compile.
    assert parse_code(text) == text.strip()


def test_single_fence_and_unfenced_code_match_the_official_parser():
    for text in ('```python\nx = 1\n```', 'x = 1\n', '```\nx = 1\n```'):
        assert first_fence(text)[0] == parse_code(text)


def test_only_the_first_complete_fence_is_used():
    text = 'Intro\n```python\na = 1\n```\nthen\n```python\nb = 2\n```'
    assert first_fence(text) == ('a = 1', 'first_fence')


def test_unclosed_fence_and_empty_text_are_not_repaired():
    assert first_fence('```python\nx = 1')[1] == 'no_fence'
    assert first_fence('   ') == ('', 'empty')
    assert first_fence(None) == ('', 'empty')


def test_manuscript_macros_match_hash_checked_evidence():
    result = subprocess.run([sys.executable, 'scripts/build_codegen_evidence.py', '--check'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
