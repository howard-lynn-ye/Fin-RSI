import pytest
from benchmarks.agent_study.transformers_chat import qwen32_dual_map, allocated_layout

MODEL = 'Qwen/Qwen2.5-Coder-32B-Instruct'
REVISION = '381fc969f78efac66bc87ff7ddeadb7e73c218a7'
GIB = 1024**3


def test_layout_covers_all_layers_and_keeps_generation_on_input_device():
    layout = qwen32_dual_map(MODEL, REVISION, [48*GIB, 48*GIB])
    assert [layout[f'model.layers.{i}'] for i in range(64)] == [0]*32 + [1]*32
    assert layout['model.embed_tokens'] == layout['lm_head'] == 0
    assert set(layout.values()) == {0, 1}


@pytest.mark.parametrize('sizes', [[48*GIB], [48*GIB, 24*GIB], [48*GIB]*3])
def test_reject_incompatible_allocations(sizes):
    with pytest.raises(ValueError):
        qwen32_dual_map(MODEL, REVISION, sizes)


@pytest.mark.parametrize('model,revision', [('other', REVISION), (MODEL, 'other')])
def test_does_not_silently_apply_to_other_weights(model, revision):
    with pytest.raises(ValueError):
        qwen32_dual_map(model, revision, [48*GIB]*2)


def test_default_keeps_original_single_device(monkeypatch):
    monkeypatch.delenv('FIN_STUDY_GPU_LAYOUT', raising=False)
    assert allocated_layout(MODEL, REVISION, None) == ({'': 0}, None)


def test_unknown_layout_fails_closed(monkeypatch):
    monkeypatch.setenv('FIN_STUDY_GPU_LAYOUT', 'arbitrary')
    with pytest.raises(ValueError):
        allocated_layout(MODEL, REVISION, None)


def test_l40s_reported_usable_memory_is_not_rounded_up_to_45_gib():
    assert qwen32_dual_map(MODEL, REVISION, [46068 * 1024**2] * 2)['lm_head'] == 0


def test_dual_requires_slurm_before_touching_cuda(monkeypatch):
    monkeypatch.setenv('FIN_STUDY_GPU_LAYOUT', 'dual_qwen32_bf16')
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        allocated_layout(MODEL, REVISION, None)


def test_single_card_allocation_cannot_silently_run_dual(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('FIN_STUDY_GPU_LAYOUT', 'dual_qwen32_bf16')
    monkeypatch.setenv('SLURM_JOB_ID', 'test')
    cuda = SimpleNamespace(device_count=lambda: 1,
        get_device_properties=lambda i: SimpleNamespace(total_memory=48*GIB))
    with pytest.raises(ValueError, match='exactly two'):
        allocated_layout(MODEL, REVISION, SimpleNamespace(cuda=cuda))


def test_host_staging_preserves_nested_values_and_skip_keys_without_peer_copy():
    from types import SimpleNamespace
    from benchmarks.agent_study.transformers_chat import host_staged_sender
    transfers = []
    def device(value):
        if not isinstance(value, str):
            return value
        parts = value.split(':')
        return SimpleNamespace(type=parts[0], index=int(parts[1]) if len(parts) > 1 else None)
    class Tensor:
        def __init__(self, location, payload):
            self.device, self.payload = device(location), payload
        def to(self, target, non_blocking=False):
            target = device(target)
            assert not non_blocking
            assert not (self.device.type == target.type == 'cuda'
                        and self.device.index != target.index), 'unsafe peer copy'
            transfers.append((self.device.type, target.type))
            return Tensor(target, self.payload)
    torch = SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor), device=device)
    fallback = lambda value, target, **kwargs: value
    send = host_staged_sender(torch, fallback, lambda source, values: type(source)(values))
    source = Tensor('cuda:0', (1, -2, 3))
    value = send({'layers': [source, (source,)], 'cache': source, 'label': 'market'},
                 1, non_blocking=True, skip_keys='cache')
    assert value['layers'][0].device.index == 1
    assert value['layers'][0].payload == source.payload
    assert value['layers'][1][0].device.index == 1
    assert value['cache'] is source and value['label'] == 'market'
    assert transfers == [('cuda', 'cpu'), ('cpu', 'cuda')] * 2
    assert send(source, 'cuda:0') is source


def test_transfer_hook_restored_when_generation_fails(monkeypatch):
    import sys
    import types
    from benchmarks.agent_study.transformers_chat import transfer_scope
    accelerator = types.ModuleType('accelerate')
    hooks = types.ModuleType('accelerate.hooks')
    utils = types.ModuleType('accelerate.utils')
    operations = types.ModuleType('accelerate.utils.operations')
    original = lambda *args, **kwargs: None
    hooks.send_to_device = original
    operations.honor_type = lambda source, values: type(source)(values)
    accelerator.hooks, accelerator.utils, utils.operations = hooks, utils, operations
    for name, module in [('accelerate', accelerator), ('accelerate.hooks', hooks),
                         ('accelerate.utils', utils), ('accelerate.utils.operations', operations)]:
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(RuntimeError, match='generation failed'):
        with transfer_scope({'transfer_route': 'cpu_staging'}, None):
            assert hooks.send_to_device is not original
            raise RuntimeError('generation failed')
    assert hooks.send_to_device is original
