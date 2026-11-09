"""Native ceilings preserve inputs and keep context incompatibility out of quality scores."""
import copy
import json
from types import SimpleNamespace

import pytest

from scripts import study_context as context
from scripts import run_frozen_study as runner


class Provider:
    def __init__(self):
        self.calls=[]

    def describe(self):
        return dict(max_new_tokens=2048)

    def generate_bounded(self,messages,**kwargs):
        self.calls.append((messages,kwargs))
        return dict(choices=[dict(message=dict(content='answer'))],usage=dict(completion_tokens=2))


def guarded(count, cap=16384):
    provider,events=Provider(),[]
    backend=context.ContextGuardedBackend(provider,dict(native=dict(effective_tokens=cap)),
        count_tokens=lambda messages:count,emit=lambda kind,**fields:events.append(dict(kind=kind,**fields)))
    return provider,backend,events


def test_exact_boundary_forwards_identical_full_messages_and_arguments():
    provider,backend,events=guarded(16384-2048)
    messages=[dict(role='user',content='完整输入')]
    backend.generate_bounded(messages,max_new_tokens=12000,seed=307)
    assert provider.calls==[(messages,dict(max_new_tokens=12000,seed=307))]
    assert provider.calls[0][0] is messages
    assert events[0]['output_cap']==2048 and events[0]['forwarded'] is True


def test_third_turn_overflow_preserves_history_and_never_calls_provider():
    provider,backend,events=guarded(16384-2048+1)
    messages=[dict(role='user',content='task')]
    for _ in range(2):
        messages += [dict(role='assistant',content='previous full answer'),
                     dict(role='user',content='failure tail and fixed explanation')]
    before=copy.deepcopy(messages)
    with pytest.raises(context.ContextIncompatibility):
        backend.generate_bounded(messages,max_new_tokens=12000,seed=211)
    assert not provider.calls and messages==before
    assert events[0]['known_generated_tokens']==0 and events[0]['output_cap']==2048


def recorded_call(tmp_path, *, overflow=False, fail_after_forward=False):
    from fin_skills.rsi.meter import Meter,Budget,EVENT_SINK
    provider,backend,_=guarded(14337 if overflow else 14336)
    events=[]
    backend.emit=lambda kind,**fields:events.append(dict(kind=kind,**fields))
    if fail_after_forward:
        def failed(*args,**kwargs):raise TimeoutError('no response')
        provider.generate_bounded=failed
    token=EVENT_SINK.set(events.append)
    try:
        meter=Meter(backend,Budget(attempts=3,calls=6,generated_tokens=12000))
        if overflow or fail_after_forward:
            with pytest.raises((context.ContextIncompatibility,TimeoutError)):
                meter([dict(role='user',content='full')],seed=101)
        else:
            meter([dict(role='user',content='full')],seed=101)
    finally:
        EVENT_SINK.reset(token)
    path=tmp_path/'events.jsonl'
    path.write_text(''.join(json.dumps(e)+'\n' for e in events),encoding='utf-8')
    return path,events


def test_real_meter_unknown_is_preserved_with_separate_preprovider_zero_proof(tmp_path):
    path,_=recorded_call(tmp_path,overflow=True)
    original=runner.accounting_module().partial_accounting(path)
    proof=context.guard_accounting(path,effective_tokens=16384)
    assert original['unknown_cost_requests']==1 and original['known_generated_tokens']==0
    assert proof['proven_zero_generation_requests']==1
    assert not proof['unfinished_request'] and not proof['incomplete_final_event']


def test_failure_after_forward_cannot_be_inferred_free(tmp_path):
    path,_=recorded_call(tmp_path,fail_after_forward=True)
    assert context.guard_accounting(path)['proven_zero_generation_requests']==0
    assert runner.accounting_module().partial_accounting(path)['unknown_cost_requests']==1


@pytest.mark.parametrize('missing',['check','outcome','truncated'])
def test_missing_or_truncated_proof_never_becomes_zero(tmp_path,missing):
    path,events=recorded_call(tmp_path,overflow=True)
    if missing=='check':events=[e for e in events if e['kind']!='context_checked']
    else:events=events[:-1]
    path.write_text(''.join(json.dumps(e)+'\n' for e in events)+('{"kind":' if missing=='truncated' else ''))
    proof=context.guard_accounting(path)
    assert proof['proven_zero_generation_requests']==0
    assert proof['incomplete_final_event']==(missing=='truncated')


@pytest.mark.parametrize('field,value',[('messages_sha256','wrong'),('seed',307),
    ('output_cap',1024),('effective_context_tokens',32768),('forwarded',0)])
def test_mismatched_guard_is_rejected(tmp_path,field,value):
    path,events=recorded_call(tmp_path,overflow=True)
    next(e for e in events if e['kind']=='context_checked')[field]=value
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError):context.guard_accounting(path,effective_tokens=16384)


@pytest.fixture
def profile_fixture(tmp_path):
    cells=[dict(track='P',method='check',library=True,task_id='p',seed=101,
                task_sha256='task',messages_sha256='initial'),
           dict(track='E',method='external_code',library=False,task_id='e',seed=101,
                public_sha256='public')]
    rows=[dict(c,prompt_tokens=100,output_cap=2048) for c in cells]
    rows[0]['frozen_messages_sha256']=rows[0].pop('messages_sha256')
    evidence=tmp_path/'rows.jsonl'
    evidence.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    native=dict(native_model_tokens=16384,tokenizer_tokens=16384,effective_tokens=16384)
    expected=dict(model_id='phi4',model=dict(model='org/phi4',revision='fixed'),
        protocol_sha='protocol',enrollment_sha='enrollment',amendment_sha='amendment',
        source_sha='source',planned_cells=cells,helper_hashes={'guard':'hash'})
    profile=dict(status='initial_inputs_checked',model_id='phi4',model='org/phi4',revision='fixed',
        protocol_sha256='protocol',enrollment_terminal_sha256='enrollment',amendment_sha256='amendment',
        source_manifest_sha256='source',helper_hashes=expected['helper_hashes'],
        initial_cell_counts=dict(P=1,E=1),overflow_initial_cells=0,native=native,
        initial_rows_file=evidence.name,initial_rows_sha256=context.sha(evidence))
    path=tmp_path/'profile.json'; path.write_text(json.dumps(profile))
    return SimpleNamespace(path=path,profile=profile,rows=rows,evidence=evidence,expected=expected)


def validate(f):
    return context.validate_profile(dict(path=str(f.path),sha256=context.sha(f.path)),**f.expected)


def test_profile_binds_complete_cell_keys_and_native_limit(profile_fixture):
    f=profile_fixture
    assert validate(f)['native']['effective_tokens']==16384
    f.rows[1]['task_id']='substituted'
    f.evidence.write_text(''.join(json.dumps(r)+'\n' for r in f.rows))
    f.profile['initial_rows_sha256']=context.sha(f.evidence); f.path.write_text(json.dumps(f.profile))
    with pytest.raises(ValueError,match='cell identities'):validate(f)


@pytest.mark.parametrize('field,value',[('helper_hashes',{}),('revision','other'),
    ('amendment_sha256','other'),('overflow_initial_cells',1),('initial_cell_counts',{'P':1,'E':0})])
def test_profile_rejects_changed_binding(profile_fixture,field,value):
    f=profile_fixture; f.profile[field]=value; f.path.write_text(json.dumps(f.profile))
    with pytest.raises(ValueError):validate(f)


def test_native_qualification_needs_amended_profile_and_keeps_other_limits(tmp_path):
    protocol=dict(models=dict(twenty_hf_models=[dict(id='phi4',model='org/phi4',revision='fixed')],
        common_generation=dict(context_tokens=32768,per_call_output_cap=2048,temperature=.1,top_p=1.0)))
    config=dict(model_id='org/phi4',model_revision='fixed',precision='torch.bfloat16',
                context_tokens=16384,max_new_tokens=2048,temperature=.1,top_p=1.0)
    receipt=dict(model_id='phi4',status='passed',phase='model',source_manifest_sha256='source',
                 details=dict(backend=config))
    path=tmp_path/'qualification.json'; path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):runner.checked_model_qualification(path,protocol,'source','phi4')
    profile=dict(native=dict(native_model_tokens=16384,tokenizer_tokens=16384,effective_tokens=16384))
    assert runner.checked_model_qualification(path,protocol,'source','phi4',profile)['backend']==config
    config['max_new_tokens']=1024; path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):runner.checked_model_qualification(path,protocol,'source','phi4',profile)
