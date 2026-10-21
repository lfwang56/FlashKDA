"""PPU regressions: references, layouts, precision, pools, graphs and ABI.

Run: python -m pytest -q tests/test_ppu.py
"""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

try:
    import flash_kda_C

    import flash_kda
except ImportError:
    flash_kda = flash_kda_C = None

requires_ppu = pytest.mark.skipif(
    not hasattr(flash_kda_C, "decode"),
    reason="Requires the PPU extension",
)
DEVICE, D, LOWER_BOUND = "cuda", 128, -5.0


def load_wrapper(native):
    path = Path(__file__).resolve().parents[1] / "flash_kda"
    path = path / "__init__.py"
    spec = importlib.util.spec_from_file_location("flash_kda_abi_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"flash_kda_C": native}):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("capability", [None, False, True])
def test_decode_capability_and_dispatch(capability):
    calls = []
    native = SimpleNamespace(
        fwd=None, get_workspace_size=None, decode=lambda *args: calls.append(args)
    )
    if capability is not None:
        native.DECODE_SUPPORTS_LOWER_BOUND = capability
    module = load_wrapper(native)
    assert module.DECODE_SUPPORTS_LOWER_BOUND is (capability is True)
    assert module.supports_decode_safe_gate is module.DECODE_SUPPORTS_LOWER_BOUND
    args = (
        object(),
        object(),
        object(),
        object(),
        object(),
        0.5,
        object(),
        object(),
        object(),
    )
    assert module.decode(*args) is args[-1]
    assert calls == [args]
    if capability:
        for bound in (-5.0, -1.0, 0.0):
            assert module.decode(*args, lower_bound=bound) is args[-1]
            assert calls[-1] == (*args, bound)
    else:
        with pytest.raises(RuntimeError, match="rebuild/reinstall"):
            module.decode(*args, lower_bound=-5.0)
        assert calls == [args]


def test_no_decode_binary_never_advertises_support():
    module = load_wrapper(
        SimpleNamespace(
            fwd=None, get_workspace_size=None, DECODE_SUPPORTS_LOWER_BOUND=True
        )
    )
    assert not module.DECODE_SUPPORTS_LOWER_BOUND
    with pytest.raises(RuntimeError, match="PPU only"):
        module.decode(None, None, None, None, None, 1.0, None, None, None)


def decode_args(batch=2, heads=6, dtype=torch.float32, device="cuda"):
    d = 128
    rand = lambda *shape: torch.randn(*shape, device=device, dtype=torch.bfloat16)
    return [
        rand(batch, 3 * heads * d),
        rand(batch, heads * d),
        rand(batch, heads),
        torch.zeros(heads, device=device),
        torch.zeros(heads, d, device=device),
        d**-0.5,
        torch.zeros(batch + 1, heads, d, d, device=device, dtype=dtype),
        torch.arange(batch, device=device, dtype=torch.int32),
        torch.empty(batch, 1, heads, d, device=device, dtype=torch.bfloat16),
    ]


def verify_args(batch=2, heads=6, steps=3, dtype=torch.float32, device="cuda"):
    rand = lambda *shape: torch.randn(*shape, device=device, dtype=torch.bfloat16)
    return [
        *(rand(batch, steps, heads, 128) for _ in range(4)),
        rand(batch, steps, heads),
        torch.zeros(heads, device=device),
        torch.zeros(heads, 128, device=device),
        128**-0.5,
        -5.0,
        torch.zeros(batch + 1, heads, 128, 128, device=device, dtype=dtype),
        torch.arange(batch, device=device, dtype=torch.int32),
        torch.full(
            (batch + 1, steps + 1, heads, 128, 128), 7, device=device, dtype=dtype
        ),
        torch.arange(batch, device=device, dtype=torch.int32),
        rand(batch, steps, heads, 128),
    ]


@requires_ppu
@pytest.mark.parametrize("bound", [math.nan, math.inf, -math.inf, -1e100, 1.0])
@pytest.mark.parametrize("operation", ["decode", "verify"])
def test_rejects_invalid_bound(operation, bound):
    if operation == "decode":
        with pytest.raises(RuntimeError, match="finite and non-positive"):
            flash_kda.decode(*decode_args(), lower_bound=bound)
    else:
        args = verify_args()
        args[8] = bound
        with pytest.raises(RuntimeError, match="finite and non-positive"):
            flash_kda.verify(*args)


@requires_ppu
@pytest.mark.parametrize("operation", ["decode", "verify"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_invalid_state_slots_do_not_access_pools(operation, dtype):
    if operation == "decode":
        args = decode_args(dtype=dtype)
        args[-2][:] = torch.tensor(
            [-1, args[6].shape[0]], device="cuda", dtype=torch.int32
        )
        state = args[6].clone()
        flash_kda.decode(*args, lower_bound=-5)
        torch.cuda.synchronize()
        assert torch.equal(args[6], state)
    else:
        args = verify_args(dtype=dtype)
        args[10][:] = torch.tensor(
            [-1, args[9].shape[0]], device="cuda", dtype=torch.int32
        )
        state, checkpoints = args[9].clone(), args[11].clone()
        flash_kda.verify(*args)
        torch.cuda.synchronize()
        assert torch.equal(args[9], state)
        assert torch.equal(args[11], checkpoints)
    assert torch.count_nonzero(args[-1]) == 0


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_checkpoint_slots_and_sentinels(dtype):
    args = verify_args(dtype=dtype)
    committed = args[9].clone()
    args[12][0] = args[11].shape[0]
    args[12][1] = -1  # A valid source can request output without saving checkpoints.
    checkpoints = args[11].clone()
    flash_kda.verify(*args)
    torch.cuda.synchronize()
    assert torch.count_nonzero(args[-1][0]) == 0
    assert torch.count_nonzero(args[-1][1]) > 0
    assert torch.equal(args[9], committed)
    assert torch.equal(args[11], checkpoints)


@requires_ppu
@pytest.mark.parametrize("operation", ["decode", "verify"])
def test_cuda_graph_replay(operation):
    args = decode_args() if operation == "decode" else verify_args()
    fn = flash_kda.decode if operation == "decode" else flash_kda.verify
    state_index = 6 if operation == "decode" else 9
    initial = args[state_index].clone()
    fn(*args)
    torch.cuda.synchronize()
    expected_out = args[-1].clone()
    expected_state = args[state_index].clone()
    args[state_index].copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args)
    args[state_index].copy_(initial)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(args[-1], expected_out)
    assert torch.equal(args[state_index], expected_state)


@requires_ppu
@pytest.mark.parametrize("operation", ["decode", "verify"])
def test_uses_input_device_and_stream(operation):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible devices")
    device = torch.device("cuda:1")
    make = decode_args if operation == "decode" else verify_args
    fn = flash_kda.decode if operation == "decode" else flash_kda.verify
    args = make(device=device)
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    previous = torch.cuda.current_device()
    torch.cuda.set_device(0)
    try:
        torch.cuda.set_stream(stream)
        torch.cuda.set_device(0)
        fn(*args)
        stream.synchronize()
        assert torch.cuda.current_device() == 0
        assert bool(args[-1].isfinite().all())
        assert torch.count_nonzero(args[-1]) > 0
    finally:
        torch.cuda.set_device(previous)


@requires_ppu
@pytest.mark.parametrize("operation", ["decode", "verify"])
def test_rejects_mixed_devices(operation):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible devices")
    args = decode_args() if operation == "decode" else verify_args()
    args[1] = args[1].to("cuda:1")
    fn = flash_kda.decode if operation == "decode" else flash_kda.verify
    with pytest.raises(RuntimeError, match="one device"):
        fn(*args)


@requires_ppu
@pytest.mark.parametrize("operation", ["decode", "verify"])
def test_rejects_unaligned_vectors(operation):
    args = decode_args() if operation == "decode" else verify_args()
    tensor = args[0]
    args[0] = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)[
        1:
    ].view(tensor.shape)
    fn = flash_kda.decode if operation == "decode" else flash_kda.verify
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        fn(*args)


@requires_ppu
@pytest.mark.parametrize(
    "shape", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1), (2**62, 96, 1)]
)
def test_workspace_rejects_unsupported_dimensions(shape):
    with pytest.raises(RuntimeError, match="positive|index range"):
        flash_kda.get_workspace_size(*shape)


@requires_ppu
@pytest.mark.parametrize("operation", ["decode", "verify"])
@pytest.mark.parametrize("scale", [math.nan, math.inf, -math.inf, 1e100])
def test_rejects_nonfinite_fp32_scale(operation, scale):
    args = decode_args() if operation == "decode" else verify_args()
    args[5 if operation == "decode" else 7] = scale
    fn = flash_kda.decode if operation == "decode" else flash_kda.verify
    with pytest.raises(RuntimeError, match="scale must be finite"):
        fn(*args)


def inputs(tokens, heads, batches=1, state_dtype=torch.bfloat16):
    torch.manual_seed(928)
    packed = torch.randn(
        batches, tokens, 3 * heads * 128, device="cuda", dtype=torch.bfloat16
    )
    q, k, v = [
        part.unflatten(-1, (heads, 128)) for part in packed.split(heads * 128, -1)
    ]
    gate = torch.randn(q.shape, device="cuda", dtype=q.dtype) * 0.25 - 2
    beta = torch.randn(batches, tokens, heads, device="cuda", dtype=q.dtype)
    initial = (
        torch.randn(batches, heads, 128, 128, device="cuda", dtype=state_dtype) * 0.01
    )
    return q, k, v, gate, beta, initial


def forward(data, *, packed=True, initial=True, final=True, cu=None):
    q, k, v, gate, beta, state = data
    if not packed:
        q, k, v = (tensor.contiguous() for tensor in (q, k, v))
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    last = torch.empty_like(state) if final else None
    heads = q.shape[2]
    flash_kda.fwd(
        q,
        k,
        v,
        gate,
        beta,
        128**-0.5,
        out,
        torch.zeros(heads, device=q.device),
        torch.zeros(heads, 128, device=q.device),
        -5,
        initial_state=state if initial else None,
        final_state=last,
        cu_seqlens=cu,
    )
    return out, last


def recurrent_reference(data, use_initial):
    """Tokenwise FP64 delta rule; no chunk inverse or native tile arithmetic."""
    q, k, v, gate, beta, initial = [tensor.cpu().double() for tensor in data]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    state = initial.clone() if use_initial else torch.zeros_like(initial)
    out = torch.empty_like(v)
    for token in range(q.shape[1]):
        state *= (-5 * gate[:, token].sigmoid()).exp().unsqueeze(-2)
        key = k[:, token]
        prediction = (state * key.unsqueeze(-2)).sum(-1)
        delta = beta[:, token].sigmoid().unsqueeze(-1) * (v[:, token] - prediction)
        state += delta.unsqueeze(-1) * key.unsqueeze(-2)
        out[:, token] = (state * q[:, token].unsqueeze(-2)).sum(-1) * 128**-0.5
    return out, state


@requires_ppu
@pytest.mark.parametrize("heads", [1, 6, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("initial", [False, True])
def test_prefill_matches_tokenwise_reference(heads, dtype, initial):
    data = inputs(33, heads, state_dtype=dtype)
    actual = forward(data, initial=initial)
    expected = recurrent_reference(data, initial)
    for value, reference in zip(actual, expected):
        error = (value.cpu().double() - reference).square().sum().sqrt()
        relative = error / reference.square().sum().sqrt().clamp_min(1e-12)
        assert float(relative) < 0.025


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "gate_value,initial_value,expected_value",
    [(-8.0, 1.3046875, 1.2734375), (-5.0, 1.6640625, 0.97265625)],
)
def test_prefill_state_decay_preserves_fp32_until_store(
    dtype, gate_value, initial_value, expected_value
):
    # With zero keys/values, one 16-token chunk is pure exponential decay.
    # The closed-form result is rounded to BF16 only at the chunk boundary.
    # These inputs distinguish that result from rounding the decay first.
    data = inputs(16, 1, state_dtype=dtype)
    for tensor in data[:3]:
        tensor.zero_()
    data[3].fill_(gate_value)
    data[-1].fill_(initial_value)
    initial = data[-1].clone()
    out, final = forward(data, packed=False)
    assert torch.count_nonzero(out) == 0
    assert torch.equal(data[-1], initial)
    assert torch.equal(final, torch.full_like(final, expected_value))


@requires_ppu
@pytest.mark.parametrize("heads", [6, 12])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prefill_optional_final_state_preserves_output(
    heads, varlen, packed, initial, dtype
):
    data = inputs(8192, heads, state_dtype=dtype)
    saved_initial = data[-1].clone()
    cu = torch.tensor([0, 8192], device="cuda", dtype=torch.int64) if varlen else None
    expected, final_state = forward(
        data, packed=packed, initial=initial, final=True, cu=cu
    )
    actual, omitted_state = forward(
        data, packed=packed, initial=initial, final=False, cu=cu
    )
    assert omitted_state is None
    assert torch.equal(actual, expected)
    assert torch.equal(data[-1], saved_initial)
    assert bool(final_state.isfinite().all())


@requires_ppu
@pytest.mark.parametrize("heads", [6, 12])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prefill_partial_zero_summary_preserves_live_columns(
    heads, varlen, packed, dtype
):
    data = inputs(8192, heads, state_dtype=dtype)
    for tensor in data[:3]:
        tensor.zero_()
    # The first half of the homogeneous rows underflows to zero. Gate -100
    # makes the other half's FP32 decay exactly one, so those rows stay live.
    # A block-wide stop must include every MMA warp and the load warp.
    data[3].fill_(-100.0)
    data[3][..., :64].fill_(8.0)
    initial = data[-1].clone()
    expected = initial.bfloat16().to(dtype).clone()
    expected[..., :64] = 0
    cu = torch.tensor([0, 8192], device="cuda", dtype=torch.int64) if varlen else None
    out, final = forward(data, packed=packed, cu=cu)
    assert torch.count_nonzero(out) == 0
    assert torch.equal(data[-1], initial)
    assert torch.equal(final, expected)


@requires_ppu
@pytest.mark.parametrize(
    "heads,tokens,gate_mode",
    [
        (6, 8192, "normal"),
        (6, 1024, "weak"),
        (12, 4096, "mixed"),
        (16, 16384, "weak"),
        (16, 32768, "mixed"),
    ],
)
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prefill_scan_matches_tokenwise_reference(
    heads, tokens, gate_mode, varlen, packed, dtype
):
    data = inputs(tokens, heads, state_dtype=dtype)
    if gate_mode != "normal":
        data[3].fill_(-8.0)
        if gate_mode == "mixed":
            data[3][:, :, 0].fill_(0.0)
    initial = data[-1].clone()
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int64) if varlen else None
    actual = forward(data, packed=packed, cu=cu)
    expected = recurrent_reference(data, True)
    assert torch.equal(data[-1], initial)
    for value, reference in zip(actual, expected):
        relative = (
            value.cpu().double() - reference
        ).norm() / reference.norm().clamp_min(1e-12)
        assert float(relative) < 0.025

    if dtype == torch.float32:
        assert torch.equal(actual[1], actual[1].bfloat16().float())


@requires_ppu
@pytest.mark.parametrize("heads", [6, 12, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bounds", [(0, 0), (17, 33), (0, 1025)])
@pytest.mark.parametrize("use_initial", [False, True])
def test_prefill_padded_single_sequence(heads, dtype, bounds, use_initial):
    capacity = 32768 if heads == 16 else 8192
    data = inputs(capacity, heads, state_dtype=dtype)
    saved = data[-1].clone()
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int64)
    actual = forward(data, cu=cu, initial=use_initial)
    begin, end = bounds
    if begin == end:
        expected = saved if use_initial else torch.zeros_like(saved)
        assert torch.equal(actual[1], expected)
    else:
        piece = tuple(x[:, begin:end].contiguous() for x in data[:5]) + (saved,)
        expected = recurrent_reference(piece, use_initial)
        for value, reference in zip((actual[0][:, begin:end], actual[1]), expected):
            relative = (
                value.cpu().double() - reference
            ).norm() / reference.norm().clamp_min(1e-12)
            assert float(relative) < 0.025
        if dtype == torch.float32:
            assert torch.equal(actual[1], actual[1].bfloat16().float())
    assert torch.equal(data[-1], saved)


@requires_ppu
@pytest.mark.parametrize("heads", [6, 12, 16])
def test_prefill_scan_range_guard_updates_during_graph_replay(heads):
    capacity = 32768 if heads == 16 else 8192
    data = inputs(capacity, heads, state_dtype=torch.float32)
    cu = torch.tensor([0, capacity], device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward(data, cu=cu)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = forward(data, cu=cu)
    for bounds in ((0, capacity), (0, 0), (17, 33), (0, capacity)):
        cu.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int64))
        graph.replay()
        expected = forward(data, cu=cu)
        torch.cuda.synchronize()
        begin, end = bounds
        assert torch.equal(actual[0][:, begin:end], expected[0][:, begin:end])
        assert torch.equal(actual[1], expected[1])


@requires_ppu
@pytest.mark.parametrize("heads", [6, 12, 16, 24])
@pytest.mark.parametrize("tokens", [1023, 1024, 1025])
def test_packed_views_across_recurrence_boundary(heads, tokens):
    data = inputs(tokens, heads)
    packed, packed_state = forward(data)
    dense, dense_state = forward(data, packed=False)
    assert torch.equal(packed.view(torch.int16), dense.view(torch.int16))
    assert torch.equal(packed_state.view(torch.int16), dense_state.view(torch.int16))
    assert bool(packed.isfinite().all()) and bool(packed_state.isfinite().all())


@requires_ppu
@pytest.mark.parametrize("final", [False, True])
def test_fixed_and_variable_length_batches(final):
    data = inputs(65, 6, batches=2)
    fixed = forward(data, packed=False, final=final)
    flattened = tuple(
        tensor.contiguous().flatten(0, 1).unsqueeze(0) for tensor in data[:5]
    ) + (data[-1],)
    cu = torch.tensor([0, 65, 130], device="cuda", dtype=torch.int64)
    variable = forward(flattened, packed=False, final=final, cu=cu)
    assert torch.equal(fixed[0].flatten(0, 1), variable[0].squeeze(0))
    if final:
        assert torch.equal(fixed[1], variable[1])


@requires_ppu
def test_rejects_incomplete_workspace_before_launch():
    q, k, v, gate, beta, state = inputs(1024, 6)
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    with pytest.raises(RuntimeError, match="workspace is too small"):
        flash_kda_C.fwd(
            q,
            k,
            v,
            gate,
            beta,
            128**-0.5,
            out,
            torch.empty(1, device=q.device, dtype=torch.uint8),
            torch.zeros(6, device=q.device),
            torch.zeros(6, 128, device=q.device),
            -5.0,
            state,
            torch.empty_like(state),
            None,
        )


@requires_ppu
@pytest.mark.parametrize("gate_value", [-8.0, 8.0])
def test_prefill_extreme_gates_match_independent_reference(gate_value):
    data = list(inputs(1025, 1))
    data[3].fill_(gate_value)
    actual = forward(data)
    expected = recurrent_reference(data, True)
    for value, reference in zip(actual, expected):
        error = (value.cpu().double() - reference).norm()
        relative = error / reference.norm().clamp_min(1e-12)
        assert float(relative) < 0.025


@requires_ppu
@pytest.mark.parametrize("use_initial", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ragged_batch_with_empty_sequence(use_initial, dtype):
    data = inputs(67, 6, state_dtype=dtype)
    initial = data[-1].expand(3, -1, -1, -1).contiguous()
    flat = data[:5] + (initial,)
    cu = torch.tensor([0, 17, 17, 67], device="cuda", dtype=torch.int64)
    out, final = forward(flat, initial=use_initial, cu=cu)
    for index, (start, end) in enumerate([(0, 17), (17, 17), (17, 67)]):
        if start == end:
            expected = (
                initial[index] if use_initial else torch.zeros_like(initial[index])
            )
            assert torch.equal(final[index], expected)
            continue
        piece = tuple(tensor[:, start:end].contiguous() for tensor in data[:5]) + (
            initial[index : index + 1],
        )
        expected_out, expected_final = forward(piece, initial=use_initial)
        assert torch.equal(out[:, start:end], expected_out)
        assert torch.equal(final[index : index + 1], expected_final)


@requires_ppu
@pytest.mark.parametrize(
    "bound", [float("nan"), float("inf"), float("-inf"), -5.1, 0.1]
)
def test_prefill_rejects_unsupported_gate_bound(bound):
    q, k, v, gate, beta, state = inputs(17, 1)
    with pytest.raises(RuntimeError, match=r"finite and in \[-5, 0\]"):
        flash_kda.fwd(
            q,
            k,
            v,
            gate,
            beta,
            128**-0.5,
            torch.empty_like(q),
            torch.zeros(1, device=q.device),
            torch.zeros(1, 128, device=q.device),
            bound,
            initial_state=state,
        )


@requires_ppu
@pytest.mark.parametrize("tokens", [33, 1025])
def test_prefill_cuda_graph_replay(tokens):
    data = inputs(tokens, 6)
    expected_out, expected_final = forward(data)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, final = forward(data)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected_out)
    assert torch.equal(final, expected_final)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_mode", ["normal", "mixed"])
@pytest.mark.parametrize("tokens, heads, batches", [(4096, 6, 4), (2048, 12, 4), (2048, 6, 8)])
def test_prefill_batched_segment_scan_matches_reference(dtype, gate_mode, tokens, heads, batches):
    data = inputs(tokens, heads, batches=batches, state_dtype=dtype)
    if gate_mode == "mixed":
        data[3].fill_(-8.0)
        data[3][:, :, 0].fill_(0.0)
    saved = data[-1].clone()
    actual = forward(data, packed=False)
    flattened = tuple(x.flatten(0, 1).unsqueeze(0) for x in data[:5]) + (data[-1],)
    cu = torch.arange(batches+1, device="cuda", dtype=torch.int64) * tokens
    varlen = forward(flattened, cu=cu)
    assert torch.equal(actual[0].flatten(0, 1), varlen[0].squeeze(0))
    assert torch.equal(actual[1], varlen[1])
    omitted, _ = forward(flattened, cu=cu, final=False)
    assert torch.equal(omitted, varlen[0])
    expected = recurrent_reference(data, True)
    for value, reference in zip(actual, expected):
        relative = (value.cpu().double() - reference).norm() / reference.norm().clamp_min(1e-12)
        assert float(relative) < 0.025
    assert torch.equal(data[-1], saved)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batches", [4, 8])
def test_prefill_batched_scan_guard_changes_during_graph(dtype, batches):
    data = inputs(16384, 6, state_dtype=dtype)
    data = data[:5] + (data[-1].expand(batches, -1, -1, -1).contiguous(),)
    uniform = [i * (16384 // batches) for i in range(batches+1)]
    ragged = uniform.copy()
    ragged[1] = ragged[2] = 16
    cu = torch.tensor(uniform, device="cuda", dtype=torch.int64)
    forward(data, cu=cu)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward(data, cu=cu)
    for bounds in (uniform, ragged, uniform):
        cu.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int64))
        graph.replay()
        expected = forward(data, cu=cu)
        torch.cuda.synchronize()
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        if bounds[1] == bounds[2]:
            assert torch.equal(actual[1][1], data[-1][1])


def _check_decode(actual, expected, name):
    diff = (actual.float() - expected.float()).abs()
    rel = diff.norm() / expected.float().norm().clamp_min(1e-8)
    assert rel < 0.01 and diff.max() < 0.02, (name, rel.item(), diff.max().item())


def _decode_case(batch, heads, dtype, bound, steps):
    torch.manual_seed(913)
    d = 128
    device = "cuda"
    state = torch.randn(batch + 2, heads, d, d, device=device, dtype=dtype) * 0.05
    expected = state.clone()
    untouched = state.clone()
    indices = torch.arange(batch, device=device, dtype=torch.int32).flip(0) + 1
    if batch > 1:
        indices[-1] = -1
    valid = indices >= 0
    slots = indices[valid].long()
    alog = torch.linspace(-3, 3, heads, device=device)
    dt = torch.randn(heads, d, device=device) * 0.1
    out = torch.empty(batch, 1, heads, d, device=device, dtype=torch.bfloat16)
    for step in range(steps):
        mixed = torch.randn(batch, 3 * heads * d, device=device, dtype=torch.bfloat16)
        gate = torch.randn(batch, heads * d, device=device, dtype=torch.bfloat16)
        if step == 0:
            gate[:, : d // 2] = -80
            gate[:, d // 2 : d] = 80
        beta = torch.randn(batch, heads, device=device, dtype=torch.bfloat16)
        q, k, v = (x.reshape(batch, heads, d).float() for x in mixed.chunk(3, -1))
        q = F.normalize(q, dim=-1, eps=1e-6) * d**-0.5
        k = F.normalize(k, dim=-1, eps=1e-6)
        x = gate.reshape(batch, heads, d).float() + dt
        if bound is None:
            decay = torch.exp(-alog.exp()[None, :, None] * F.softplus(x))
        else:
            decay = torch.exp(bound * torch.sigmoid(alog.exp()[None, :, None] * x))
        h = expected[slots].float() * decay[valid, :, None, :]
        delta = (v[valid] - (h * k[valid, :, None, :]).sum(-1)) * beta[
            valid
        ].float().sigmoid()[..., None]
        h = h + delta[..., None] * k[valid, :, None, :]
        ref = (h * q[valid, :, None, :]).sum(-1).bfloat16()
        expected[slots] = h.to(dtype)
        kwargs = {} if bound is None else {"lower_bound": bound}
        flash_kda.decode(
            mixed, gate, beta, alog, dt, d**-0.5, state, indices, out, **kwargs
        )
        torch.cuda.synchronize()
        _check_decode(out[valid, 0], ref, "output")
        _check_decode(state, expected, "state")
        assert torch.equal(state[0], untouched[0])
        assert torch.equal(state[-1], untouched[-1])
        if batch > 1:
            assert torch.count_nonzero(out[-1]) == 0
    print(
        f"PASS B={batch} H={heads} state={dtype} bound={bound} steps={steps}",
        flush=True,
    )


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bound", [None, -5.0, -1.0, 0.0])
@pytest.mark.parametrize(
    "batch,heads", [(1, 6), (4, 12), (32, 12), (64, 12), (128, 16), (512, 12)]
)
def test_decode_reference(dtype, bound, batch, heads):
    _decode_case(batch, heads, dtype, bound, 4)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_decode_long_rollout(dtype):
    _decode_case(4, 12, dtype, -5.0, 512)


def _verify_reference(q, k, v, gate, beta, a_log, dt_bias, state, state_indices):
    """Dense KDA chain _verify_reference with the same per-step state dtype contract."""
    batch, steps, heads, _ = q.shape
    out = torch.empty_like(q)
    checkpoints = torch.empty(
        batch, steps, heads, D, D, device=q.device, dtype=state.dtype
    )
    for bi in range(batch):
        h = state[state_indices[bi]].clone()
        for step in range(steps):
            q_t = F.normalize(q[bi, step].float(), p=2, dim=-1, eps=1e-6)
            q_t.mul_(D**-0.5)
            k_t = F.normalize(k[bi, step].float(), p=2, dim=-1, eps=1e-6)
            gate_t = gate[bi, step].float() + dt_bias
            decay = torch.exp(
                LOWER_BOUND * torch.sigmoid(torch.exp(a_log)[:, None] * gate_t)
            )
            h = h.float() * decay[:, None, :]
            v_new = v[bi, step].float() - (h * k_t[:, None, :]).sum(dim=-1)
            v_new.mul_(torch.sigmoid(beta[bi, step].float())[:, None])
            h = h + v_new[:, :, None] * k_t[:, None, :]
            # Match target_verify semantics: the committed state is loaded from
            # its pool dtype once, then the private chain remains FP32. Only the
            # speculative checkpoint written at each step is rounded to the
            # intermediate pool dtype.
            checkpoints[bi, step].copy_(h)
            out[bi, step].copy_(
                (h.float() * q_t[:, None, :]).sum(dim=-1).to(torch.bfloat16)
            )
    return out, checkpoints


def _check_verify(name, actual, expected, max_rel=4e-2, max_abs=8e-2):
    diff = (actual.float() - expected.float()).abs()
    rel = diff.norm() / (expected.float().norm() + 1e-8)
    print(
        f"{name}: rel_frob={rel.item():.4e} max_abs={diff.max().item():.4e} "
        f"mean_abs={diff.mean().item():.4e}"
    )
    assert rel.item() < max_rel, f"{name} relative error {rel.item():.4e}"
    assert diff.max().item() < max_abs, f"{name} max error {diff.max().item():.4e}"


def _verify_case(state_dtype, batch=2, steps=8, heads=6, strided_qkv=False):
    torch.manual_seed(20260901 + (state_dtype == torch.bfloat16))
    if strided_qkv:
        mixed_qkv = torch.randn(
            batch,
            steps,
            3 * heads * D,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        q_flat, k_flat, v_flat = mixed_qkv.split(heads * D, dim=-1)
        q = q_flat.unflatten(-1, (heads, D))
        k = k_flat.unflatten(-1, (heads, D))
        v = v_flat.unflatten(-1, (heads, D))
        if batch * steps > 1:
            assert (
                not q.is_contiguous()
                and not k.is_contiguous()
                and not v.is_contiguous()
            )
        verify_fn = flash_kda.verify_strided
    else:
        q = torch.randn(batch, steps, heads, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        verify_fn = flash_kda.verify
    gate = torch.randn_like(q)
    beta = torch.randn(batch, steps, heads, device=DEVICE, dtype=torch.bfloat16)
    a_log = torch.randn(heads, device=DEVICE, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(heads, D, device=DEVICE, dtype=torch.float32) * 0.1
    state = torch.randn(batch + 3, heads, D, D, device=DEVICE, dtype=state_dtype) * 0.05
    state_indices = torch.arange(batch, device=DEVICE, dtype=torch.int32)
    intermediate = torch.empty(
        batch + 3, steps + 2, heads, D, D, device=DEVICE, dtype=state_dtype
    )
    intermediate_indices = torch.arange(batch, device=DEVICE, dtype=torch.int32) + 1
    out = torch.empty_like(q)
    committed_before = state.clone()

    expected_out, expected_states = _verify_reference(
        q, k, v, gate, beta, a_log, dt_bias, state, state_indices.long()
    )
    verify_fn(
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        D**-0.5,
        LOWER_BOUND,
        state,
        state_indices,
        intermediate,
        intermediate_indices,
        out,
    )
    torch.cuda.synchronize()

    assert torch.equal(state, committed_before), "verify mutated committed state"
    layout = "strided" if strided_qkv else "contiguous"
    _check_verify(f"output ({state_dtype}, {layout})", out, expected_out)
    got_states = intermediate[intermediate_indices.long(), :steps]
    _check_verify(f"checkpoints ({state_dtype}, {layout})", got_states, expected_states)

    # Invalid state rows are padding: they must neither read/write a pool slot
    # nor leak an uninitialised output into the acceptance path.
    padded_out = torch.empty(1, steps, heads, D, device=DEVICE, dtype=torch.bfloat16)
    padded_state = state.clone()
    verify_fn(
        q[:1],
        k[:1],
        v[:1],
        gate[:1],
        beta[:1],
        a_log,
        dt_bias,
        D**-0.5,
        LOWER_BOUND,
        padded_state,
        torch.tensor([-1], device=DEVICE, dtype=torch.int32),
        intermediate,
        torch.tensor([-1], device=DEVICE, dtype=torch.int32),
        padded_out,
    )
    torch.cuda.synchronize()
    assert torch.count_nonzero(padded_out).item() == 0
    assert torch.equal(padded_state, state)

    return (
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        state,
        state_indices,
        intermediate,
        intermediate_indices,
        out,
    )


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,heads", [(1, 6), (4, 12), (32, 12)])
@pytest.mark.parametrize("steps", [1, 8, 17])
@pytest.mark.parametrize("strided", [False, True])
def test_verify_reference(dtype, batch, heads, steps, strided):
    _verify_case(dtype, batch=batch, steps=steps, heads=heads, strided_qkv=strided)


def test_cuda_wrapper_preserves_forward_abi():
    calls = []
    native = SimpleNamespace(
        fwd=lambda *args, **kwargs: calls.append((args, kwargs)),
        get_workspace_size=lambda total, heads, sequences: 8,
    )
    module = load_wrapper(native)
    q = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
    beta = torch.zeros(1, 1, 1, dtype=q.dtype)
    out = torch.empty_like(q)
    args = (q, q, q, q, beta, 0.5, out, torch.zeros(1), torch.zeros(1, 128), -5.0)
    module.fwd(*args)
    assert len(calls) == 1
    positional, keyword = calls[0]
    assert len(positional) == 11
    assert positional[:7] == args[:7]
    assert positional[7].dtype == torch.uint8 and positional[7].numel() == 8
    assert keyword == {"initial_state": None, "final_state": None, "cu_seqlens": None}
    with pytest.raises(RuntimeError, match="PPU only"):
        module.decode(None, None, None, None, None, 1.0, None, None, None)
    assert not module.DECODE_SUPPORTS_LOWER_BOUND
    assert module.fwd.__module__ == module.__name__
    assert module.decode.__module__ == module.__name__


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_forward_shared_abi(dtype):
    q, k, v, gate, beta, state = inputs(33, 6, state_dtype=dtype)
    alog = torch.zeros(6, device=q.device)
    bias = torch.zeros(6, 128, device=q.device)
    workspace = torch.empty(
        flash_kda_C.get_workspace_size(33, 6, 1), dtype=torch.uint8, device=q.device
    )
    expected = forward((q, k, v, gate, beta, state))
    out, last = torch.empty_like(q), torch.empty_like(state)
    flash_kda_C.fwd(
        q,
        k,
        v,
        gate,
        beta,
        128**-0.5,
        out,
        workspace,
        alog,
        bias,
        -5.0,
        initial_state=state,
        final_state=last,
        cu_seqlens=None,
    )
    assert torch.equal(out, expected[0]) and torch.equal(last, expected[1])


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_mode", ["saturated", "mixed_columns"])
@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("final", [False, True])
def test_prefill_saturated_gate_tail_matches_independent_reference(dtype, gate_mode, initial, final):
    # Full saturated blocks exercise the inverse-decay range, and a one-token
    # tail checks that padding does not change a column's total gate.
    data = inputs(16384, 6, state_dtype=dtype)
    data = data[:5] + (data[-1].expand(8, -1, -1, -1).contiguous(),)
    data[3].fill_(20.0)
    if gate_mode == "mixed_columns":
        data[3][..., 64:].fill_(-80.0)
    saved = data[-1].clone()
    cu = torch.tensor([0] + [97] * 8, device="cuda", dtype=torch.int64)
    out, state = forward(data, cu=cu, initial=initial, final=final)
    piece = tuple(x[:, :97] for x in data[:5]) + (data[-1][:1],)
    expected = recurrent_reference(piece, initial)
    comparisons = [(out[:, :97], expected[0])]
    if state is not None:
        comparisons.append((state[:1], expected[1]))
    for value, reference in comparisons:
        assert bool(value.isfinite().all())
        relative = (value.cpu().double()-reference).norm()/reference.norm().clamp_min(1e-12)
        assert float(relative) < 0.025
    if state is not None:
        empty = saved[1:] if initial else torch.zeros_like(saved[1:])
        assert torch.equal(state[1:], empty)
    assert torch.equal(data[-1], saved)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prefill_centered_dispatch_graph_uses_dynamic_boundaries(dtype):
    data = inputs(16384, 6, state_dtype=dtype)
    data = data[:5] + (data[-1].expand(8, -1, -1, -1).contiguous(),)
    cu = torch.arange(9, device="cuda", dtype=torch.int64) * 2048
    forward(data, cu=cu)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward(data, cu=cu)
    for bounds in ([0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384],
                   [0, 16, 16, 8192, 8192, 8192, 12288, 16384, 16384],
                   [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384]):
        cu.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int64))
        graph.replay()
        expected = forward(data, cu=cu)
        torch.cuda.synchronize()
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        for i in range(8):
            if bounds[i] == bounds[i+1]:
                assert torch.equal(actual[1][i], data[-1][i])


@requires_ppu
@pytest.mark.parametrize("batch", [16, 64])
def test_decode_large_batch_long_rollout(batch):
    _decode_case(batch, 12, torch.bfloat16, -5.0, 512)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_mode", ["weak", "mixed_columns"])
def test_prefill_centered_full_batch_matches_tokenwise_reference(dtype, gate_mode):
    data = inputs(2048, 6, batches=8, state_dtype=dtype)
    data[3].fill_(-8.0)
    if gate_mode == "mixed_columns":
        data[3][..., :64].fill_(0.0)
    saved = data[-1].clone()
    expected = recurrent_reference(data, True)
    flat = tuple(x.flatten(0, 1).unsqueeze(0) for x in data[:5]) + (data[-1],)
    cu = torch.arange(9, device="cuda", dtype=torch.int64) * 2048
    actual = forward(flat, cu=cu)
    values = (actual[0].reshape(expected[0].shape), actual[1])
    for value, reference in zip(values, expected):
        relative = (value.cpu().double() - reference).norm() / reference.norm().clamp_min(1e-12)
        assert float(relative) < 0.025
    assert torch.equal(data[-1], saved)


@requires_ppu
def test_decode_mid_batch_fp32_long_rollout():
    _decode_case(64, 12, torch.float32, -5.0, 512)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prefill_scan_reuses_poisoned_workspace_across_gate_regimes(dtype):
    data = inputs(8192, 6, state_dtype=dtype)
    q, k, v, gate, beta, state = data
    for tensor in (q, k, v):
        tensor.zero_()
    gate.fill_(8.0)
    saved = state.clone()
    workspace = torch.full(
        (flash_kda_C.get_workspace_size(8192, 6, 1),), 255,
        dtype=torch.uint8, device=q.device,
    )
    out, last = torch.empty_like(q), torch.empty_like(state)
    alog, bias = torch.zeros(6, device=q.device), torch.zeros(6, 128, device=q.device)
    cu = torch.tensor([0, 8192], dtype=torch.int64, device=q.device)

    def native():
        flash_kda_C.fwd(
            q, k, v, gate, beta, 128**-0.5, out, workspace, alog, bias, -5.0,
            initial_state=state, final_state=last, cu_seqlens=cu,
        )

    native()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native()
    for regime in ("strong", "identity", "mixed", "strong"):
        gate.fill_(-100.0 if regime == "identity" else 8.0)
        if regime == "mixed":
            gate[..., :64].fill_(-100.0)
        # Both unused matrix bytes and flags start as all-ones/NaNs on every
        # replay. The operator must overwrite its metadata and every live value.
        workspace.fill_(255)
        out.fill_(float("nan"))
        last.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = saved.bfloat16().to(dtype).clone()
        if regime == "strong":
            expected.zero_()
        elif regime == "mixed":
            expected[..., 64:] = 0
        assert torch.count_nonzero(out) == 0
        assert torch.equal(last, expected)
        assert torch.equal(state, saved)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_mode", ["native", "alternating_segments"])
@pytest.mark.parametrize("tokens, heads, segments, batches", [(16384, 6, 26, 1), (8192, 12, 13, 1), (4096, 6, 6, 4), (2048, 6, 3, 8)])
def test_prefill_exact_replay_reuse_full_tokenwise_reference(dtype, gate_mode, tokens, heads, segments, batches):
    data = inputs(tokens, heads, batches=batches, state_dtype=dtype)
    if gate_mode == "alternating_segments":
        data[3].fill_(8.)
        for segment in range(segments):
            first = segment * ((tokens//16) // segments) + min(segment, (tokens//16) % segments)
            count = (tokens//16) // segments + (segment < (tokens//16) % segments)
            if segment % 2:
                data[3][:, first*16:(first+count)*16].fill_(-8.)
    saved = data[-1].clone()
    flat = tuple(t.flatten(0, 1).unsqueeze(0) for t in data[:5]) + (data[-1],)
    cu = torch.arange(batches+1, device="cuda", dtype=torch.int64) * tokens
    actual = forward(flat, cu=cu)
    actual = (actual[0].reshape(data[2].shape), actual[1])
    expected = recurrent_reference(data, True)
    for value, reference in zip(actual, expected):
        relative = (value.cpu().double() - reference).norm() / reference.norm().clamp_min(1e-12)
        assert float(relative) < 0.025
    assert torch.equal(data[-1], saved)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_initial", [False, True])
@pytest.mark.parametrize("with_final", [False, True])
@pytest.mark.parametrize("tokens, heads, segments, batches", [(16384, 6, 26, 1), (8192, 12, 13, 1), (4096, 6, 6, 4), (2048, 6, 3, 8)])
def test_prefill_exact_replay_reuse_poisoned_graph_mixed_heads(dtype, with_initial, with_final, tokens, heads, segments, batches):
    data = inputs(tokens, heads, batches=batches, state_dtype=dtype)
    q, k, v, gate, beta, state = data
    for tensor in (q, k, v):
        tensor.zero_()
    saved = state.clone()
    workspace = torch.full(
        (flash_kda_C.get_workspace_size(tokens*batches, heads, batches),), 255,
        dtype=torch.uint8, device=q.device,
    )
    out, last = torch.empty_like(q), torch.empty_like(state)
    alog = torch.zeros(heads, device=q.device)
    bias = torch.zeros(heads, 128, device=q.device)
    cu = torch.arange(batches+1, device=q.device, dtype=torch.int64) * tokens
    qf, kf, vf, gf, bf = (t.flatten(0, 1).unsqueeze(0) for t in data[:5])
    of = out.flatten(0, 1).unsqueeze(0)

    def native():
        flash_kda_C.fwd(
            qf, kf, vf, gf, bf, 128**-0.5, of, workspace, alog, bias, -5.,
            initial_state=state if with_initial else None,
            final_state=last if with_final else None, cu_seqlens=cu,
        )

    gate.fill_(8.)
    native()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native()
    for regime in ("strong", "mixed", "identity", "strong"):
        gate.fill_(-100. if regime == "identity" else 8.)
        expected = saved.bfloat16().to(dtype).clone() if with_initial else torch.zeros_like(state)
        if regime == "strong":
            expected.zero_()
        elif regime == "mixed":
            # Strong heads converge; weak head and weak half-columns must run
            # the complete history. Head2 has zero final A but an unequal state
            # at its first 64-token checkpoint, so it must process the tail.
            gate[:, :, 1].fill_(-100.)
            gate[:, :, 3, :64].fill_(-100.)
            for segment in range(segments):
                first_tile = segment * ((tokens//16) // segments) + min(segment, (tokens//16) % segments)
                gate[:, first_tile*16:first_tile*16+64, 2].fill_(-100.)
            expected[:, 0] = 0
            expected[:, 2] = 0
            expected[:, 4:] = 0
            expected[:, 3, :, 64:] = 0
        workspace.fill_(255)
        out.fill_(float("nan"))
        last.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(out) == 0
        if with_final:
            assert torch.equal(last, expected)
        assert torch.equal(state, saved)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_initial", [False, True])
@pytest.mark.parametrize("with_final", [False, True])
def test_prefill_single_sequence_prefix_poisoned_dynamic_graph(dtype, with_initial, with_final):
    data = inputs(16384, 6, state_dtype=dtype)
    q, k, v, gate, beta, state = data
    for tensor in (q, k, v):
        tensor.zero_()
    gate.fill_(-100.)
    saved = state.clone()
    workspace = torch.full(
        (flash_kda_C.get_workspace_size(16384, 6, 1),), 255,
        dtype=torch.uint8, device=q.device,
    )
    out, last = torch.empty_like(q), torch.empty_like(state)
    alog = torch.zeros(6, device=q.device)
    bias = torch.zeros(6, 128, device=q.device)
    cu = torch.tensor([0, 16384], device=q.device, dtype=torch.int64)

    def native():
        flash_kda_C.fwd(
            q, k, v, gate, beta, 128**-0.5, out, workspace, alog, bias, -5.,
            initial_state=state if with_initial else None,
            final_state=last if with_final else None, cu_seqlens=cu,
        )

    native()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native()
    # An empty sequence must publish fresh metadata even when every prepare
    # CTA returns before its first load; the full-sequence flag must also reset.
    for bos, eos in ((0, 16384), (33, 130), (23, 23), (0, 16384)):
        cu.copy_(torch.tensor([bos, eos], device=q.device, dtype=torch.int64))
        workspace.fill_(255)
        out.fill_(float("nan"))
        last.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        if eos > bos:
            assert torch.count_nonzero(out[:, bos:eos]) == 0
        if with_final:
            expected = saved if eos == bos else saved.bfloat16().to(dtype)
            if not with_initial:
                expected = torch.zeros_like(state)
            assert torch.equal(last, expected)
        assert torch.equal(state, saved)


@requires_ppu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens, heads, batches", [(16384, 6, 1), (8192, 12, 1), (4096, 6, 4), (2048, 6, 8)])
def test_prefill_replay_graph_allows_inplace_initial_final_state(dtype, tokens, heads, batches):
    data = inputs(tokens, heads, batches=batches, state_dtype=dtype)
    q, k, v, gate, beta, state = data
    for tensor in (q, k, v):
        tensor.zero_()
    saved = state.clone()
    workspace = torch.full(
        (flash_kda_C.get_workspace_size(tokens*batches, heads, batches),), 255,
        dtype=torch.uint8, device=q.device,
    )
    out = torch.empty_like(q)
    alog, bias = torch.zeros(heads, device=q.device), torch.zeros(heads, 128, device=q.device)
    cu = torch.arange(batches+1, device=q.device, dtype=torch.int64)*tokens
    qf, kf, vf, gf, bf = (t.flatten(0, 1).unsqueeze(0) for t in data[:5])
    of = out.flatten(0, 1).unsqueeze(0)

    def native():
        flash_kda_C.fwd(
            qf, kf, vf, gf, bf, 128**-0.5, of, workspace, alog, bias, -5.,
            initial_state=state, final_state=state, cu_seqlens=cu,
        )

    gate.fill_(8.)
    native()
    torch.cuda.synchronize()
    state.copy_(saved)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native()
    for regime in ("strong", "mixed", "identity"):
        gate.fill_(-100. if regime == "identity" else 8.)
        expected = saved.bfloat16().to(dtype).clone()
        if regime == "strong":
            expected.zero_()
        elif regime == "mixed":
            gate[:, :, 1].fill_(-100.)
            gate[:, :, 3, :64].fill_(-100.)
            expected[:, 0] = 0
            expected[:, 2] = 0
            expected[:, 4:] = 0
            expected[:, 3, :, 64:] = 0
        state.copy_(saved)
        workspace.fill_(255)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(out) == 0
        assert torch.equal(state, expected)


@requires_ppu
@pytest.mark.parametrize("bound", [None, -5.0, -1.0, 0.0])
def test_decode_bulk_fp32_b256_long_rollout(bound):
    _decode_case(256, 12, torch.float32, bound, 512)
