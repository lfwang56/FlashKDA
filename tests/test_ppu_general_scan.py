"""Independent numerical, boundary, workspace and graph checks for general scan."""
import pytest
import torch
import flash_kda
import flash_kda_C
from test_ppu import recurrent_reference, requires_ppu

D = 128


def fixture(lengths, heads, dtype, packed, offset=0, physical=None, fixed=False):
    total = physical or max(1, offset + sum(lengths))
    torch.manual_seed(2026102923 + total + heads)
    projection = torch.randn(1, total, 3 * heads * D, device='cuda', dtype=torch.bfloat16) * .2
    q, k, v = [x.unflatten(-1, (heads, D)) for x in projection.split(heads * D, -1)]
    if not packed or fixed:
        q, k, v = [x.contiguous() for x in (q, k, v)]
    gate = torch.randn_like(q) * .25 - 2
    beta = torch.randn(1, total, heads, device='cuda', dtype=torch.bfloat16)
    initial = (torch.randn(len(lengths), heads, D, D, device='cuda') * .01).to(dtype)
    bounds = [offset]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    cu = torch.tensor(bounds, device='cuda', dtype=torch.int64)
    data = [q, k, v, gate, beta, initial]
    if fixed:
        assert offset == 0 and len(set(lengths)) == 1 and total == sum(lengths)
        data[:5] = [x.view(len(lengths), lengths[0], *x.shape[2:]) for x in data[:5]]
    return data, cu, bounds


def forward(data, cu, *, initial=True, final=True, capture=False):
    q, k, v, gate, beta, state = data
    heads = q.shape[-2]
    sequences = len(cu) - 1 if cu is not None else q.shape[0]
    out_guard = torch.full((256 + q.numel(),), 13, device='cuda', dtype=q.dtype)
    out = out_guard[128:-128].view_as(q)
    state_guard = torch.full((256 + state.numel(),), 13, device='cuda', dtype=state.dtype)
    final_state = state_guard[128:-128].view_as(state)
    final_state.fill_(float('nan'))
    size = flash_kda.get_workspace_size(q.shape[0] * q.shape[1], heads, sequences)
    ws_guard = torch.full((256 + size,), 85, device='cuda', dtype=torch.uint8)
    alog=torch.zeros(heads,device='cuda')
    bias=torch.zeros(heads,D,device='cuda')
    workspace=ws_guard[128:-128]
    def call():
        flash_kda_C.fwd(q,k,v,gate,beta,D**-.5,out,workspace,alog,bias,-5.,
                        initial_state=state if initial else None,
                        final_state=final_state if final else None,cu_seqlens=cu)
    if capture:
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            call()
        graph.replay()
        torch.cuda.synchronize()
    else:
        call()
    return out, final_state, (out_guard, state_guard, ws_guard)


def canaries(guards):
    for guard, value in zip(guards, [13, 13, 85]):
        assert bool((guard[:128] == value).all())
        assert bool((guard[-128:] == value).all())


def expected(data, bounds, initial=True):
    flat = [x.flatten(0, 1).unsqueeze(0) for x in data[:5]]
    out = torch.full(flat[2].shape, 13., dtype=torch.float64)
    final = data[-1].double().cpu().clone() if initial else torch.zeros_like(data[-1], device='cpu', dtype=torch.float64)
    for i, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        if begin != end:
            o, s = recurrent_reference([*[x[:, begin:end] for x in flat], data[-1][i:i+1]], initial)
            out[:, begin:end] = o
            final[i:i+1] = s
    return out.view_as(data[2].cpu()), final


def close(actual, reference, bounds=None):
    if bounds is not None:
        out, state = actual
        golden_out, golden_state = reference
        flat = out.flatten(0, 1).unsqueeze(0)
        golden_flat = golden_out.flatten(0, 1).unsqueeze(0)
        mask = torch.zeros(flat.shape[1], dtype=torch.bool)
        for begin, end in zip(bounds, bounds[1:]):
            mask[begin:end] = True
        assert bool((flat.detach().cpu()[:, ~mask] == 13).all())
        actual = (flat[:, mask.to(flat.device)], state)
        reference = (golden_flat[:, mask], golden_state)
    for result, golden in zip(actual, reference):
        result = result.detach().cpu().double()
        assert bool(torch.isfinite(result).all())
        if result.numel():
            assert float((result - golden).norm() / golden.norm().clamp_min(1e-12)) < .025


@requires_ppu
@pytest.mark.parametrize('heads,lengths,fixed', [(3, [4097], False), (12, [4097], False), (6, [4097, 4097], True), (6, [4097, 4097], False)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('use_initial', [False, True])
@pytest.mark.parametrize('capture', [False, True])
def test_general_scan_against_tokenwise_fp64(heads, lengths, fixed, dtype, use_initial, capture):
    data, cu, bounds = fixture(lengths, heads, dtype, True, fixed=fixed)
    saved = data[-1].clone()
    out, state, guards = forward(data, None if fixed else cu, initial=use_initial, capture=capture)
    close((out, state), expected(data, bounds, use_initial))
    canaries(guards)
    assert torch.equal(saved, data[-1])


@requires_ppu
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('capture', [False, True])
def test_general_scan_ragged_empty_tail_and_optional_final(dtype, packed, capture):
    data, cu, bounds = fixture([8193, 0, 4097], 6, dtype, packed, offset=17, physical=13001)
    saved = data[-1].clone()
    out, state, guards = forward(data, cu, capture=capture)
    close((out, state), expected(data, bounds), bounds)
    assert torch.equal(state[1], saved[1])
    without_final, _, omitted_guards = forward(data, cu, capture=capture, final=False)
    assert torch.equal(without_final, out)
    assert torch.equal(saved, data[-1])
    canaries(guards)
    canaries(omitted_guards)


@requires_ppu
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('heads,lengths,physical,ranges', [
    (3, [8193], None, [(0, 8193), (17, 34), (17, 17), (17, 272),
                      (17, 273), (17, 274), (17, 1040), (0, 4097), (0, 8193)]),
    (6, [4001, 4192], None, [(0, 4001, 8193), (17, 17, 4099),
                            (17, 50, 8193), (0, 0, 0), (0, 1, 18), (0, 4001, 8193)]),
    (12, [8193], None, [(0, 8193), (17, 34), (17, 17), (17, 528),
                       (17, 1040), (17, 1041), (17, 2064), (0, 8193)]),
    (5, [0, 33, 8193], 20000, [(0, 0, 33, 8226), (17, 17, 17, 17),
                              (17, 17, 50, 561), (0, 4001, 4001, 8193),
                              (0, 1, 18, 51), (0, 0, 33, 8226)]),
])
def test_general_scan_graph_metadata_updates(dtype, heads, lengths, physical, ranges):
    data, cu, _ = fixture(lengths, heads, dtype, True, physical=physical)
    saved = data[-1].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward(data, cu)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, state, guards = forward(data, cu)
    for bounds in ranges:
        cu.copy_(torch.tensor(bounds, device='cuda', dtype=torch.int64))
        graph.replay()
        torch.cuda.synchronize()
        close((out, state), expected(data, bounds), bounds)
        for i, (begin, end) in enumerate(zip(bounds, bounds[1:])):
            if begin == end:
                assert torch.equal(state[i], saved[i])
        canaries(guards)
        assert torch.equal(saved, data[-1])


@requires_ppu
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('heads', [3, 12])
@pytest.mark.parametrize('middle', [0, 33])
@pytest.mark.parametrize('capture', [False, True])
def test_general_scan_preserves_live_transition_columns(heads, dtype, middle, capture):
    data, cu, _ = fixture([8193, middle, 4097], heads, dtype, True)
    for x in data[:3]:
        x.zero_()
    data[3].fill_(-100.)
    data[3][..., :64].fill_(8.)
    saved = data[-1].clone()
    out, state, guards = forward(data, cu, capture=capture)
    golden = saved.bfloat16().to(dtype).clone()
    golden[..., :64] = 0
    if middle == 0:
        golden[1] = saved[1]
    assert torch.count_nonzero(out) == 0
    assert torch.equal(state, golden)
    assert torch.equal(saved, data[-1])
    canaries(guards)


@requires_ppu
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('heads,lengths,physical,shifted', [
    (6, [0, 8193], None, False),
    (6, [33, 8193], None, True),
    (3, [0], 8193, False),
    (3, [255], 8193, False),
    (3, [4097], 8193, False),
])
@pytest.mark.parametrize('capture', [False, True])
def test_general_scan_state_aliases(dtype, heads, lengths, physical, shifted, capture):
    data, cu, bounds = fixture(lengths, heads, dtype, True, physical=physical)
    for tensor in data[:3]:
        tensor.zero_()
    data[3].fill_(-100.)
    n = len(lengths)
    pool = torch.randn(n + 1, heads, D, D, device='cuda').to(dtype)
    initial = pool[:n]
    final = pool[1:] if shifted else initial
    saved = initial.clone()
    out = torch.full_like(data[2], 13.)
    workspace = torch.empty(flash_kda.get_workspace_size(out.shape[1], heads, n), device='cuda', dtype=torch.uint8)
    alog=torch.zeros(heads,device='cuda');bias=torch.zeros(heads,D,device='cuda')
    def call():
        flash_kda_C.fwd(*data[:5],D**-.5,out,workspace,alog,bias,-5.,
                        initial_state=initial,final_state=final,cu_seqlens=cu)
    if capture:
        snapshot=pool.clone()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):call()
        stream.synchronize();pool.copy_(snapshot);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):call()
        graph.replay();torch.cuda.synchronize()
    else:call()
    golden = saved.bfloat16().to(dtype).clone()
    for i, length in enumerate(lengths):
        if length == 0:
            golden[i] = saved[i]
    valid = torch.zeros(out.shape[1], dtype=torch.bool, device=out.device)
    for begin, end in zip(bounds, bounds[1:]):
        valid[begin:end] = True
    assert torch.count_nonzero(out[:, valid]) == 0
    assert bool((out[:, ~valid] == 13).all())
    assert torch.equal(final, golden)


@requires_ppu
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('heads', [3, 12])
@pytest.mark.parametrize('capture', [False, True])
def test_general_scan_dense_live_transition(heads, dtype, capture):
    data, cu, bounds = fixture([4097], heads, dtype, True)
    data[1].copy_(torch.nn.functional.normalize(data[1].float(), dim=-1).mul_(.8))
    data[3].fill_(-7.)
    saved = data[-1].clone()
    out, state, guards = forward(data, cu, capture=capture)
    close((out, state), expected(data, bounds))
    assert torch.equal(saved, data[-1])
    canaries(guards)


@requires_ppu
@pytest.mark.parametrize('length', [0, 17, 255, 256, 257, 4097])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('use_initial', [False, True])
@pytest.mark.parametrize('capture', [False, True])
def test_general_sparse_single_optional_states(length, dtype, use_initial, capture):
    data, cu, bounds = fixture([length], 3, dtype, True, offset=17, physical=8193)
    saved = data[-1].clone()
    out, state, guards = forward(data, cu, capture=capture, initial=use_initial)
    close((out, state), expected(data, bounds, use_initial), bounds)
    without_final, _, omitted_guards = forward(data, cu, capture=capture, initial=use_initial, final=False)
    assert torch.equal(without_final, out)
    assert torch.equal(saved, data[-1])
    canaries(guards)
    canaries(omitted_guards)


@requires_ppu
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('use_initial', [False, True])
def test_legacy_scan_prepare_replays_device_bounds(packed, use_initial):
    data, cu, _ = fixture([511], 6, torch.float32, packed, physical=32768)
    saved = [tensor.clone() for tensor in data]
    out, state, guards = forward(data, cu, initial=use_initial)
    alog = torch.zeros(6, device='cuda')
    bias = torch.zeros(6, D, device='cuda')
    workspace = guards[2][128:-128]
    def call():
        flash_kda_C.fwd(*data[:5], D**-.5, out, workspace, alog, bias, -5.,
                        initial_state=data[-1] if use_initial else None,
                        final_state=state, cu_seqlens=cu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    for bounds in ([0, 17], [17, 528], [0, 0], [0, 3073]):
        cu.copy_(torch.tensor(bounds, device='cuda', dtype=torch.int64))
        out.fill_(13)
        state.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        close((out, state), expected(data, bounds, use_initial), bounds)
        canaries(guards)
        assert all(torch.equal(before, after) for before, after in zip(saved, data))


@requires_ppu
@pytest.mark.parametrize('heads', [1, 2])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('use_initial', [False, True])
def test_low_head_native_graph_bounds(heads, dtype, use_initial):
    data, cu, _ = fixture([511], heads, dtype, True, physical=8193)
    saved = [tensor.clone() for tensor in data]
    out, state, guards = forward(data, cu, initial=use_initial)
    alog = torch.zeros(heads, device='cuda')
    bias = torch.zeros(heads, D, device='cuda')
    workspace = guards[2][128:-128]
    def call():
        flash_kda_C.fwd(*data[:5], D**-.5, out, workspace, alog, bias, -5.,
                        initial_state=data[-1] if use_initial else None,
                        final_state=state, cu_seqlens=cu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    for bounds in ([0, 17], [17, 528], [0, 0], [0, 1023], [0, 1024], [0, 2047], [0, 4099]):
        cu.copy_(torch.tensor(bounds, device='cuda', dtype=torch.int64))
        out.fill_(13)
        state.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        close((out, state), expected(data, bounds, use_initial), bounds)
        canaries(guards)
        assert all(torch.equal(before, after) for before, after in zip(saved, data))
