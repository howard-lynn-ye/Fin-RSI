"""Learning traces follow the library condition (frozen-study leak, fixed on master in 78f2fed)."""
from fin_skills.rsi.wiring import Attempt, traces


def attempt():
    return Attempt(task_id="t", code="def solve(p, f):\n    return p\n", executed=True, error="",
                   checks={"execution_lag": False, "finite_output": True}, valid=False)


def test_no_library_traces_carry_no_library_calls():
    without = traces(attempt(), available_at="2026-11-03T00:00:00+00:00", source_version="v",
                     tools=False)[0].summary
    with_library = traces(attempt(), available_at="2026-11-03T00:00:00+00:00", source_version="v",
                          tools=True)[0].summary
    assert "execution_lag" in without and "fin_skills" not in without
    assert "fin_skills" in with_library


def test_playbook_fit_passes_its_library_condition_to_observe():
    import inspect
    from fin_skills.rsi import methods
    source = inspect.getsource(methods.PlaybookLearning.fit)
    assert 'source_version="fin-skills", tools=self.library)' in source
