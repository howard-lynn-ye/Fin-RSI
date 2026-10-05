"""The manuscript must retain the full paired design and receipt-bound evidence."""
import copy
import hashlib
import json
import pytest
from scripts import build_factorial_evidence as evidence


@pytest.fixture(scope='module')
def verified():
    return evidence.read_verified()


def test_published_tables_match_complete_evidence(verified):
    for name, text in evidence.render(verified).items():
        assert (evidence.PAPER / name).read_text(encoding='utf-8') == text


def test_rejects_missing_or_duplicate_pair(verified):
    missing = copy.deepcopy(verified)
    missing['results'].pop()
    with pytest.raises(ValueError, match='pairs'):
        evidence.validate_batch(missing)
    duplicate = copy.deepcopy(verified)
    duplicate['results'][-1] = duplicate['results'][0]
    with pytest.raises(ValueError, match='pairs'):
        evidence.validate_batch(duplicate)


def test_rejects_dropped_failure_denominator(verified):
    data = copy.deepcopy(verified)
    data['results'][0]['paths']['base']['decisions'] -= 1
    with pytest.raises(ValueError, match='denominator'):
        evidence.validate_batch(data)


def test_rejects_relabelled_effect(verified):
    data = copy.deepcopy(verified)
    data['results'][0]['effects_pp']['interaction'] += 1
    with pytest.raises(ValueError, match='effect'):
        evidence.validate_batch(data)


def test_rejects_changed_group_mean(verified):
    data = copy.deepcopy(verified)
    data['groups'][0]['mean_return_rate_pct']['base'] += 1
    with pytest.raises(ValueError, match='mean'):
        evidence.validate_batch(data)


def test_receipt_hash_is_checked_before_using_numbers(tmp_path):
    path = tmp_path / 'analysis.json'
    path.write_text('{}', encoding='utf-8')
    manifest = {'analysis.json': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / 'SHA256SUMS.json').write_text(json.dumps(manifest), encoding='utf-8')
    path.write_text('{"changed": true}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        evidence.read_verified(tmp_path)


def test_manifest_cannot_escape_evidence_directory(tmp_path):
    (tmp_path / 'SHA256SUMS.json').write_text(json.dumps({'../private': 'invalid'}), encoding='utf-8')
    with pytest.raises(ValueError, match='escapes'):
        evidence.read_verified(tmp_path)


def test_prior_study_is_generated_from_its_separate_evidence():
    from scripts import build_prior_trading_evidence as prior
    path = prior.REPO / 'paper/latex_naacl/prior_trading_table.tex'
    assert path.read_text(encoding='utf-8') == prior.render()
