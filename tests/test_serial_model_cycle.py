import json
import pytest
from benchmarks.agent_study.serial_model_cycle import check_cached_snapshot, selection, run, completion_summary


@pytest.mark.parametrize('seeds', [[], [11], [11,23], [11,11,37]])
def test_partial_or_duplicate_repeats_never_count_as_complete(seeds):
    rows = [dict(seed=s) for s in seeds]
    assert completion_summary(('7b','model','revision'), rows) is None


def test_complete_summary_preserves_bad_returns_and_requires_model_identity():
    rows = [dict(seed=s, family='7b', paths={a:dict(return_rate_pct=r) for a,r in
            [('raw', 5.0), ('library', -10.0)]}) for s in (11,23,37)]
    result = completion_summary(('7b','model','revision'), rows)
    assert result['counts_toward_twenty_models']
    assert result['return_rate_pct_mean'] == {'raw':5., 'library':-10.}
    rows[-1]['family'] = 'another-model'
    with pytest.raises(ValueError, match='another model'):
        completion_summary(('7b','model','revision'), rows)


def test_public_open_weight_selection_keeps_exact_identity():
    batch, spec, row = selection('granite-8b')
    assert batch == 'matrix_hf' and row['transport'] == 'hf'
    assert spec[1] == 'ibm-granite/granite-3.3-8b-instruct'
    assert len(spec[2]) == 40 and row['execution_enabled']


@pytest.mark.parametrize('family', ['claude-opus-5.5', 'nonexistent', 'llama31-8b'])
def test_unenabled_or_native_families_cannot_enter_free_runner(family):
    with pytest.raises((ValueError, KeyError, StopIteration)):
        selection(family)


def test_cache_gate_rejects_missing_weight_shards(tmp_path):
    s = tmp_path/'models--org--name'/'snapshots'/('a'*40)
    s.mkdir(parents=True)
    (s/'config.json').write_text('{}')
    (s/'tokenizer_config.json').write_text('{}')
    (s/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w':'shard.safetensors'}}))
    with pytest.raises(ValueError, match='shard missing'):
        check_cached_snapshot(tmp_path, 'org/name', 'a'*40)
    (s/'shard.safetensors').write_bytes(b'test')
    assert check_cached_snapshot(tmp_path, 'org/name', 'a'*40)['weight_shards'] == 1


def test_compute_refuses_local_invocation(tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        run('7b', tmp_path, tmp_path/'evidence', tmp_path/'cache')
