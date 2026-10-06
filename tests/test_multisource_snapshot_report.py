"""Publication must not turn partial seeds or tampered means into complete results."""
import importlib.util
from pathlib import Path

import pytest

FILE = Path(__file__).resolve().parents[1] / 'benchmarks/agent_study/reports/20261006/publish_snapshot.py'
SPEC = importlib.util.spec_from_file_location('snapshot_publisher', FILE)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def fixture():
    return dict(failures=[], completed_models=0, verified_pairs=1,
        rows=[dict(family='sample', seed=11, paths={
            'raw': dict(return_rate_pct=10.), 'library': dict(return_rate_pct=12.)})],
        summary=dict(sample=dict(seeds=[11], complete=False,
            return_rate_pct=dict(raw=10., library=12.), difference_pp=2.)))


def test_partial_seed_remains_partial():
    publisher.validate_summary(fixture())


@pytest.mark.parametrize('change', ['complete', 'mean', 'duplicate', 'nonfinite', 'failed_audit'])
def test_rejects_misleading_publication(change):
    report = fixture()
    if change == 'complete':
        report['summary']['sample']['complete'] = True
    elif change == 'mean':
        report['summary']['sample']['return_rate_pct']['library'] = 120.
    elif change == 'duplicate':
        report['rows'] *= 2
        report['verified_pairs'] = 2
    elif change == 'nonfinite':
        report['rows'][0]['paths']['raw']['return_rate_pct'] = float('nan')
    else:
        report['failures'] = ['audit failed']
    with pytest.raises(ValueError):
        publisher.validate_summary(report)


def test_error_class_keeps_truncation_separate_from_python_error():
    assert publisher.error_class(dict(parse='generation-truncated', result=dict(ok=False))) == 'generation-truncated'
    assert publisher.error_class(dict(parse='python-fence', result=dict(
        ok=False, error="KeyError: 'weights'"))) == 'KeyError'
    assert publisher.error_class(dict(result=dict(ok=True))) is None
