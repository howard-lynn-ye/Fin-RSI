"""Fin-RSI interface: tolerant action parser and the unchanged strict path (CPU only)."""
import pytest

from benchmarks.agent_study import fin_rsi_interface as fi
from benchmarks.agent_study import trading_runtime_v6 as runtime

PY1 = '```python\nprint(1)\n```'
PY2 = '```python\nprint(2)\n```'
READ = '{"tool":"read_file","arguments":{"path":"README.md"}}'
SUBMIT = '{"tool":"submit","arguments":{"weights":{"SPY":1}}}'

STRICT_CORPUS = [PY1, READ, PY1 + '\n' + PY2, 'hello', '```json\n' + SUBMIT + '\n```',
                 '<think>x</think>' + PY1, 'Here is my plan.\n' + PY1, READ + '\n' + SUBMIT, '']


@pytest.mark.parametrize('text', STRICT_CORPUS)
def test_strict_is_the_frozen_v6_rule(text):
    call, status = runtime.extract_call(text)
    assert fi.extract_call('strict', text, False) == (call, status, int(call is not None))


def test_strict_rejects_truncated_output():
    assert fi.extract_call('strict', PY1, True) == (None, 'generation-truncated', 0)


@pytest.mark.parametrize('text,tool,status,n', [
    (PY1, 'run_python', 'python-fence', 1),
    (PY1 + '\nthen\n' + PY2, 'run_python', 'python-last-of-2', 2),
    (PY1 + '\n' + READ, 'run_python', 'python-fence', 1),          # Python wins over a JSON echo
    (READ, 'read_file', 'json', 1),
    (READ + '\n' + SUBMIT, 'submit', 'json-last-of-2', 2),
    ('<think>' + READ + '</think>' + PY2, 'run_python', 'python-fence', 1),
    ('```\nprint(1)\n```', 'run_python', 'python-fence', 1),         # unlabelled non-JSON fence
])
def test_tolerant_runs_the_last_complete_action(text, tool, status, n):
    call, got, count = fi.extract_call('tolerant', text, False)
    assert (call['tool'], got, count) == (tool, status, n)


def test_tolerant_last_python_body_is_executed():
    call, _, _ = fi.extract_call('tolerant', PY1 + '\n' + PY2, False)
    assert call['arguments']['code'].strip() == 'print(2)'


def test_tolerant_truncation():
    call, status, _ = fi.extract_call('tolerant', PY1 + '\n```python\nprint(', True)
    assert call['tool'] == 'run_python' and status == 'python-fence-before-truncation'
    assert fi.extract_call('tolerant', READ, True) == (None, 'generation-truncated', 0)
    assert fi.extract_call('tolerant', 'no action', False) == (None, 'no-action', 0)


@pytest.mark.parametrize('text,status', [('<think>a', 'reasoning-incomplete'),
                                         ('<think><think>a</think>', 'reasoning-malformed')])
def test_tolerant_reasoning_errors(text, status):
    assert fi.extract_call('tolerant', text, False) == (None, status, 0)


def test_non_dict_arguments_become_empty():
    call, _, _ = fi.extract_call('tolerant', '{"tool":"list_skills","arguments":"x"}', False)
    assert call == {'tool': 'list_skills', 'arguments': {}}


def test_notes():
    assert 'only the last one was executed' in fi.tolerant_note('python-last-of-2', 2)
    assert 'output-token limit' in fi.tolerant_note('python-fence-before-truncation', 1)
    assert fi.tolerant_note('json', 1) is None


def test_agent_import_resolves_bound_tools_only():
    bound = {'run_algorithm': lambda **k: 'bound'}
    agent_import = fi._agent_import(bound)
    module = agent_import('fin_skills', fromlist=('run_algorithm',))
    assert module.run_algorithm() == 'bound'
    assert agent_import('json').dumps([1]) == '[1]'                  # other imports untouched
    real = agent_import('fin_skills', fromlist=('names',))           # unbound name: real module
    assert callable(real.names)


def test_exec_line_is_pinned():
    """Agent tracebacks show this line number; moving it changes model-visible feedback."""
    import inspect
    lines = inspect.getsource(fi).splitlines()
    assert lines[138].strip() == "exec(compile(code, 'agent_code.py', 'exec'), namespace)"
