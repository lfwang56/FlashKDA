"""Irregular H6 tails, long affine carry, workspace bounds, and replay guards."""
import pytest
import torch
import flash_kda
import flash_kda_C
from test_ppu import recurrent_reference, requires_ppu

D, H = 128, 6


def _data(tokens, dtype):
    torch.manual_seed(2026102917 + tokens)
    projection = torch.randn(1, tokens, 3 * H * D, device='cuda', dtype=torch.bfloat16) * .2
    q, k, v = [x.unflatten(-1, (H, D)) for x in projection.split(H * D, -1)]
    gate = torch.randn_like(q) * .25 - 2
    beta = torch.randn(1, tokens, H, device='cuda', dtype=torch.bfloat16)
    initial = (torch.randn(1, H, D, D, device='cuda') * .01).to(dtype)
    return q, k, v, gate, beta, initial


def _native_call(data, *, packed=True, cu=None):
    q, k, v, gate, beta, initial = data
    if not packed:
        q, k, v = [x.contiguous() for x in (q, k, v)]
    protected_out = torch.full((256 + q.numel(),), 13, device=q.device, dtype=q.dtype)
    out = protected_out[128:-128].view_as(q)
    out.fill_(float('nan'))
    protected_state = torch.full((256 + initial.numel(),), 13, device=q.device, dtype=initial.dtype)
    final = protected_state[128:-128].view_as(initial)
    final.fill_(float('nan'))
    size = flash_kda.get_workspace_size(q.shape[0] * q.shape[1], H, 1)
    protected_workspace = torch.full((256 + size,), 85, device=q.device, dtype=torch.uint8)
    workspace = protected_workspace[128:-128]
    flash_kda_C.fwd(q, k, v, gate, beta, D ** -.5, out, workspace,
                    torch.zeros(H, device=q.device), torch.zeros(H, D, device=q.device), -5.,
                    initial_state=initial, final_state=final, cu_seqlens=cu)
    return out, final, (protected_out, protected_state, protected_workspace)


def _check_canaries(buffers):
    for buffer, sentinel in zip(buffers, [13, 13, 85]):
        assert bool((buffer[:128] == sentinel).all())
        assert bool((buffer[-128:] == sentinel).all())


def _check_reference(actual, expected, limit=.025):
    for result, reference in zip(actual, expected):
        result = result.detach().double().cpu()
        assert bool(torch.isfinite(result).all())
        relative = (result - reference).norm() / reference.norm().clamp_min(1e-12)
        assert float(relative) < limit


@requires_ppu
@pytest.mark.parametrize('tokens', [4097, 6593])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('varlen', [False, True])
def test_irregular_prefill_matches_independent_fp64(tokens, dtype, packed, varlen):
    data = _data(tokens, dtype)
    saved = data[-1].clone()
    cu = torch.tensor([0, tokens], device='cuda', dtype=torch.int64) if varlen else None
    out, final, guards = _native_call(data, packed=packed, cu=cu)
    expected = recurrent_reference(data, True)
    _check_reference((out, final), expected)
    _check_canaries(guards)
    assert torch.equal(data[-1], saved)


@requires_ppu
@pytest.mark.parametrize('tokens', [102401, 262144, 262145])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_long_prefill_keeps_live_state_columns(tokens, dtype):
    data = _data(tokens, dtype)
    for tensor in data[:3]:
        tensor.zero_()
    data[3].fill_(-100.)
    data[3][..., :64].fill_(8.)
    saved = data[-1].clone()
    expected = saved.bfloat16().to(dtype).clone()
    expected[..., :64] = 0
    cu = torch.tensor([0, tokens], device='cuda', dtype=torch.int64)
    out, final, guards = _native_call(data, cu=cu)
    assert torch.count_nonzero(out) == 0
    assert torch.equal(final, expected)
    assert torch.equal(data[-1], saved)
    _check_canaries(guards)


@requires_ppu
@pytest.mark.xfail(
    strict=True,
    reason=("Inherited cafa9f5 BF16 inverse/carry precision for repeated keys and weak decay: "
            "FP64 output error about3%, final-state error up to4.36% in the original serial path. "
            "Keep the2.5% assertion visible until that separate precision limitation is fixed."),
)
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('gate_value', [-8., -18.])
def test_irregular_prefill_correlated_keys(dtype, packed, gate_value):
    data = _data(6593, dtype)
    data[1].copy_(data[1][:, :1].clone().expand_as(data[1]))
    data[3].fill_(gate_value)
    out, final, guards = _native_call(data, packed=packed,
                                    cu=torch.tensor([0, 6593], device='cuda', dtype=torch.int64))
    _check_reference((out, final), recurrent_reference(data, True))
    _check_canaries(guards)


@requires_ppu
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_irregular_graph_updates_full_empty_and_padded_ranges(dtype):
    tokens = 6593
    data = _data(tokens, dtype)
    saved = data[-1].clone()
    cu = torch.tensor([0, tokens], device='cuda', dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _native_call(data, cu=cu)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual_out, actual_state, guards = _native_call(data, cu=cu)
    for bounds in [(0, tokens), (0, 0), (17, 33), (0, 1025), (0, tokens)]:
        cu.copy_(torch.tensor(bounds, device='cuda', dtype=torch.int64))
        graph.replay()
        eager_out, eager_state, eager_guards = _native_call(data, cu=cu)
        torch.cuda.synchronize()
        begin, end = bounds
        assert torch.equal(actual_out[:, begin:end], eager_out[:, begin:end])
        assert torch.equal(actual_state, eager_state)
        if begin == end:
            assert torch.equal(actual_state, saved)
        _check_canaries(guards)
        _check_canaries(eager_guards)
        assert torch.equal(data[-1], saved)
