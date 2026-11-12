"""Frozen populations accept later qualifications without replacing earlier evidence."""
import copy
import json
from types import SimpleNamespace

import pytest

from scripts import run_frozen_study as runner


@pytest.fixture
def admission(tmp_path):
    models = [dict(id=f"m{i}", model=f"org/m{i}", revision=str(i)*40) for i in range(6)]
    protocol = dict(models=dict(core_ids=[m['id'] for m in models[:5]], twenty_hf_models=models,
        common_generation=dict(context_tokens=32768,per_call_output_cap=2048,temperature=.1,top_p=1.0)))
    paths = []
    for model in models:
        path = tmp_path / (model['id']+'.json')
        runner.write(path, dict(model_id=model['id'], phase='model', status='passed',
            source_manifest_sha256='source', details=dict(backend=dict(
                model_id=model['model'], model_revision=model['revision'], context_tokens=32768,
                precision='torch.bfloat16',max_new_tokens=2048,temperature=.1,top_p=1.0))))
        paths.append(path)
    plan = dict(source_manifest_sha256='source',
                qualifications=runner.checked_qualifications(paths[:5], protocol, 'source'))
    release = dict(status='released', qualification=dict(path=str(paths[5]), sha256=runner.sha(paths[5])))
    return SimpleNamespace(plan=plan, protocol=protocol, release=release, paths=paths, root=tmp_path)


def test_extension_admission_keeps_frozen_plan_and_all_core_receipts(admission):
    before = json.dumps(admission.plan, sort_keys=True)
    qualified = runner.allocation_qualification(admission.plan, admission.protocol, admission.release, 'm5')
    assert qualified['backend']['model_id'] == 'org/m5'
    for model_id, expected in admission.plan['qualifications'].items():
        assert runner.allocation_qualification(admission.plan, admission.protocol, {}, model_id) == expected
    assert json.dumps(admission.plan, sort_keys=True) == before
    with pytest.raises(ValueError, match='override'):
        runner.allocation_qualification(admission.plan, admission.protocol, admission.release, 'm0')
    with pytest.raises(ValueError, match='explicit'):
        runner.allocation_qualification(admission.plan, admission.protocol, {}, 'm5')


@pytest.mark.parametrize('field,value', [('status','failed'), ('phase','cpu'),
    ('source_manifest_sha256','other'), ('model_id','m4'), ('model_revision','wrong'),
    ('backend_model','org/other')])
def test_later_receipt_requires_same_source_revision_and_model(admission, field, value):
    receipt = runner.read(admission.paths[5])
    if field in ('model_revision', 'backend_model'):
        receipt['details']['backend']['model_id' if field == 'backend_model' else field] = value
    else:
        receipt[field] = value
    admission.paths[5].write_text(json.dumps(receipt), encoding='utf-8')
    admission.release['qualification']['sha256'] = runner.sha(admission.paths[5])
    with pytest.raises(ValueError):
        runner.allocation_qualification(admission.plan, admission.protocol, admission.release, 'm5')


def test_admission_rejects_changed_file_or_frozen_noncore_override(admission):
    old = copy.deepcopy(admission.release)
    admission.paths[5].write_text(admission.paths[5].read_text()+'\n')
    with pytest.raises(ValueError, match='changed'):
        runner.allocation_qualification(admission.plan, admission.protocol, old, 'm5')
    admission.plan['qualifications']['m5'] = runner.checked_model_qualification(
        admission.paths[5], admission.protocol, 'source', 'm5')
    with pytest.raises(ValueError, match='override'):
        runner.allocation_qualification(admission.plan, admission.protocol, old, 'm5')


@pytest.mark.parametrize('field,value', [('context_tokens',16384),('precision','torch.float16'),
    ('max_new_tokens',1024),('temperature',.7),('top_p',.9)])
def test_native_context_or_generation_cannot_silently_weaken_frozen_contract(admission, field, value):
    receipt = runner.read(admission.paths[5])
    receipt['details']['backend'][field] = value
    admission.paths[5].write_text(json.dumps(receipt))
    admission.release['qualification']['sha256'] = runner.sha(admission.paths[5])
    with pytest.raises(ValueError, match='generation/context contract'):
        runner.allocation_qualification(admission.plan,admission.protocol,admission.release,'m5')


def planned_chunks(admission):
    result = []
    for model in admission.protocol['models']['twenty_hf_models']:
        cell = dict(track='E', model_id=model['id'], model=model['model'], model_revision=model['revision'],
            stage='core' if model['id'] != 'm5' else 'extension', method='external_code', library=False,
            task_id='t', task_sha256='taskhash', seed=101)
        result.append(dict(id='E-'+model['id'], model_id=model['id'], cells=[cell]))
    runner.write(admission.root/'chunks.json', result)
    runner.write(admission.root/'plan.json', admission.plan)
    return result


def test_selector_uses_fixed_population_and_consistent_stage(admission):
    chunks = planned_chunks(admission)
    release = dict(status='released', plan_sha256=runner.sha(admission.root/'plan.json'), chunk_ids=['E-m5'])
    assert runner.selected_chunks(admission.root, release, admission.protocol) == [chunks[5]]
    for ids in (['E-m5','E-m5'], ['E-m0','E-m5'], ['not-planned']):
        with pytest.raises(ValueError):
            runner.selected_chunks(admission.root, dict(release, chunk_ids=ids), admission.protocol)
    chunks[5]['cells'][0]['stage'] = 'core'
    (admission.root/'chunks.json').write_text(json.dumps(chunks))
    with pytest.raises(ValueError, match='stage'):
        runner.selected_chunks(admission.root, release, admission.protocol)


def test_parent_rejects_missing_extension_admission_before_allocation(admission, monkeypatch):
    planned_chunks(admission)
    runner.write(admission.root/'protocol.json', admission.protocol)
    release = admission.root/'release.json'
    runner.write(release, dict(status='released', plan_sha256=runner.sha(admission.root/'plan.json'),
                              chunk_ids=['E-m5'], process_seconds=10))
    monkeypatch.setattr(runner, 'verify_plan', lambda _:admission.plan)
    touched = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs:touched.append(True))
    output = admission.root/'output'
    with pytest.raises(ValueError, match='explicit'):
        runner.run(SimpleNamespace(plan=admission.root, release=release,
                                   inventory=admission.root/'unused', output=output))
    assert not touched and not output.exists()


@pytest.mark.parametrize('broken', ['worker', 'marker', 'missing_report'])
def test_interrupted_PE_still_writes_invalid_parent_and_unknown_cost(admission, monkeypatch, broken):
    chunks = planned_chunks(admission)
    runner.write(admission.root/'protocol.json', admission.protocol)
    release, inventory = admission.root/'release.json', admission.root/'inventory.json'
    runner.write(inventory, [])
    runner.write(release, dict(status='released', plan_sha256=runner.sha(admission.root/'plan.json'),
                              chunk_ids=['E-m0'], process_seconds=10))
    monkeypatch.setattr(runner, 'verify_plan', lambda _:admission.plan)
    monkeypatch.setattr(runner, 'validate_allocation', lambda *args:{})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs:SimpleNamespace(stdout=''))
    monkeypatch.setenv('SLURM_JOB_ID','synthetic')
    def interrupt(command, *, seconds, output):
        (output/'worker-terminal.json').write_text('{"status":' if broken == 'worker' else
            json.dumps(dict(status='complete_released_chunks', completed=[])))
        if broken != 'worker':
            (output/'E-m0.accepted.json').write_text('{"chunk_id":' if broken == 'marker' else
                json.dumps(dict(chunk_id='E-m0',episodes=1,terminal_sha256='fictional')))
        (output/'E-m0.jsonl.events.jsonl').write_text(
            json.dumps(dict(kind='call_started',backend_mode='bounded'))+'\n')
        return dict(timed_out=True,returncode=-15)
    monkeypatch.setattr(runner,'supervised_command',interrupt)
    output = admission.root/'output'
    with pytest.raises(RuntimeError):
        runner.run(SimpleNamespace(plan=admission.root,release=release,inventory=inventory,output=output))
    terminal = runner.read(output/'terminal.json')
    assert terminal['status'] == 'invalid' and terminal['completed'] == []
    assert terminal['read_errors']
    assert terminal['provider_event_accounting']['E-m0']['unknown_cost_requests'] == 1
    assert terminal['provider_event_accounting']['E-m0']['unfinished_requests'] == 1


def test_report_backend_must_match_admission_not_just_model_name(admission, monkeypatch):
    from fin_skills.rsi.run_record import source_hashes
    chunk = planned_chunks(admission)[5]
    qualified = runner.allocation_qualification(admission.plan, admission.protocol, admission.release, 'm5')
    row = dict(chunk['cells'][0], model_id='org/m5', usage=dict(calls=1,generated_tokens=7))
    config = copy.deepcopy(qualified['backend'])
    report = SimpleNamespace(rows=[row], _validate=lambda:None,
        protocol=dict(source_hashes=source_hashes(),backend=dict(configuration=config)))
    monkeypatch.setattr(runner.report_module(),'checked_report',lambda *args:report)
    terminal = admission.root/'E-m5.jsonl.terminal.json'
    runner.write(terminal, {})
    marker = dict(chunk_id='E-m5',episodes=1,terminal_sha256=runner.sha(terminal))
    runner.write(admission.root/'E-m5.accepted.json',marker)
    events = [dict(kind='call_started',backend_mode='bounded'),
              dict(kind='call_completed',backend_mode='bounded',tokens=7)]
    (admission.root/'E-m5.jsonl.events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    assert runner.accepted_chunk(admission.root,chunk,qualified)[1] == marker
    config['model_revision'] = 'changed'
    with pytest.raises(ValueError, match='backend'):
        runner.accepted_chunk(admission.root,chunk,qualified)


def test_extension_worker_admits_once_and_rejects_existing_claim(admission, monkeypatch):
    from benchmarks.agent_study import transformers_chat
    planned_chunks(admission)
    runner.write(admission.root/'protocol.json', admission.protocol)
    release, inventory = admission.root/'release.json', admission.root/'inventory.json'
    body = dict(admission.release, plan_sha256=runner.sha(admission.root/'plan.json'),
                chunk_ids=['E-m5'], process_seconds=10)
    runner.write(release, body)
    model = admission.protocol['models']['twenty_hf_models'][5]
    runner.write(inventory, [dict(model,cached=True,cache_check=dict(cache='unused-cache'))])
    output = admission.root/'output'; output.mkdir()
    qualified = runner.allocation_qualification(admission.plan,admission.protocol,body,'m5')
    runner.write(output/'plan.json', dict(release_sha256=runner.sha(release),
        inventory_sha256=runner.sha(inventory),qualification=qualified,
        supervisor='owned-process-group-wallclock'))
    monkeypatch.setattr(runner,'verify_plan',lambda _:admission.plan)
    for key,value in (('SLURM_JOB_ID','synthetic'),('HF_HUB_OFFLINE','1'),('TRANSFORMERS_OFFLINE','1')):
        monkeypatch.setenv(key,value)
    loaded = []
    def stop_at_model(model, revision, **kwargs):
        loaded.append((model, revision))
        raise RuntimeError('synthetic boundary: no GPU called')
    monkeypatch.setattr(transformers_chat,'TransformersChat',stop_at_model)
    args = SimpleNamespace(plan=admission.root,release=release,inventory=inventory,output=output)
    with pytest.raises(RuntimeError, match='synthetic boundary'):
        runner.run_worker(args)
    assert loaded == [('org/m5','5'*40)]
    claim = admission.root/'claims/E-m5.json'
    before = claim.read_bytes()
    # A new allocation directory cannot consume an old claim or repeat its model load.
    again = admission.root/'again'; again.mkdir()
    runner.write(again/'plan.json',runner.read(output/'plan.json'))
    args.output = again
    with pytest.raises(FileExistsError):
        runner.run_worker(args)
    assert len(loaded) == 1 and claim.read_bytes() == before
