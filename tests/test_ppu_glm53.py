"""GLM precision and compact-cache contracts, including general shapes and graphs."""

import pytest
import torch
import flash_kda
import flash_kda_C
from test_ppu import (
    recurrent_reference,
    verify_args,
    _verify_reference,
    _check_verify,
    requires_ppu,
)

pytestmark = requires_ppu
D = 128


def _prefill(tokens, heads, dtype, *, batch=1):
    torch.manual_seed(1109 + tokens + heads)
    q, k, v = [
        torch.randn(batch, tokens, heads, D, device="cuda", dtype=torch.bfloat16) * 0.2
        for _ in range(3)
    ]
    gate = torch.randn_like(q) * 0.25 - 2
    beta = torch.randn(batch, tokens, heads, device="cuda", dtype=torch.bfloat16)
    state = (torch.randn(batch, heads, D, D, device="cuda") * 0.01).to(dtype)
    return q, k, v, gate, beta, state


def _forward(data, *, cu=None, initial=True, final=True):
    q, k, v, gate, beta, state = data
    out, last = torch.empty_like(q), torch.empty_like(state)
    flash_kda.fwd(
        q,
        k,
        v,
        gate,
        beta,
        D**-0.5,
        out,
        torch.zeros(q.shape[2], device="cuda"),
        torch.zeros(q.shape[2], D, device="cuda"),
        -5.0,
        initial_state=state if initial else None,
        final_state=last if final else None,
        cu_seqlens=cu,
        state_accum_dtype=torch.float32,
    )
    return out, last


@pytest.mark.parametrize(
    "tokens,heads,batch",
    [(1, 3, 1), (17, 7, 2), (33, 8, 1), (63, 5, 3), (129, 11, 1), (1025, 8, 1)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("initial,final", [(True, True), (False, True), (True, False)])
def test_fp32_prefill_reference(tokens, heads, batch, dtype, initial, final):
    data = _prefill(tokens, heads, dtype, batch=batch)
    before = data[-1].clone()
    actual = _forward(data, initial=initial, final=final)
    expected = recurrent_reference(data, initial)
    for result, reference in zip(actual[: 2 if final else 1], expected):
        relative = (
            result.cpu().double() - reference
        ).norm() / reference.norm().clamp_min(1e-12)
        assert float(relative) < 0.025
    assert torch.equal(before, data[-1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp32_prefill_ragged_graph(dtype):
    tokens, heads = 197, 7
    data = _prefill(tokens, heads, dtype)
    state = data[-1].expand(4, -1, -1, -1).contiguous()
    data = (*data[:-1], state)
    cu = torch.tensor([0, 0, 17, 80, tokens], device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _forward(data, cu=cu)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, last = _forward(data, cu=cu)
    for bounds in ([0, 0, 17, 80, tokens], [3, 36, 36, 149, tokens], [7, 7, 7, 7, 7]):
        cu.copy_(torch.tensor(bounds, device="cuda", dtype=torch.int64))
        graph.replay()
        for i, (begin, end) in enumerate(zip(bounds, bounds[1:])):
            if begin == end:
                assert torch.equal(last[i], state[i])
                continue
            piece = tuple(x[:, begin:end].contiguous() for x in data[:5]) + (
                state[i : i + 1],
            )
            expected = recurrent_reference(piece, True)
            for result, reference in zip(
                (out[:, begin:end], last[i : i + 1]), expected
            ):
                relative = (
                    result.cpu().double() - reference
                ).norm() / reference.norm().clamp_min(1e-12)
                assert float(relative) < 0.025


@pytest.mark.parametrize("tokens", [17, 4097])
def test_fp32_storage_retains_resident_precision(tokens):
    data = _prefill(tokens, 3, torch.float32)
    for x in data[:3]:
        x.zero_()
    data[3].fill_(-100.0)
    data[-1].fill_(1.001)
    out, last = _forward(data)
    assert torch.count_nonzero(out) == 0
    assert torch.equal(last, data[-1])
    assert not torch.equal(last, last.bfloat16().float())


def _compact(batch, heads, steps, dtype, *, strided=False):
    args = verify_args(batch, heads, steps, dtype)
    args[9].normal_(0.0, 0.01)
    if strided:
        fused = torch.randn(
            batch, steps, heads * D * 3, device="cuda", dtype=torch.bfloat16
        )
        args[:3] = [x.unflatten(-1, (heads, D)) for x in fused.split(heads * D, -1)]
    end = torch.full_like(args[9], 7)
    shape = (end.shape[0], steps, heads, D)
    rings = [
        torch.full(shape, 7.0, dtype=torch.float32, device="cuda") for _ in range(2)
    ]
    rings += [
        torch.full(shape, 7.0, dtype=torch.bfloat16, device="cuda"),
        torch.full(shape[:-1], 7.0, dtype=torch.float32, device="cuda"),
    ]
    return args, end, rings


def _verify_compact(args, end, rings):
    return flash_kda.verify_compact(
        *args[:7],
        args[9],
        args[10],
        end,
        args[12],
        args[-1],
        *rings,
        args[8],
        scale=args[7]
    )


@pytest.mark.parametrize(
    "batch,heads,steps",
    [
        (1, 3, 1),
        (3, 7, 5),
        (8, 8, 14),
        (17, 11, 9),
        (64, 8, 8),
        (128, 8, 6),
        (256, 8, 6),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_compact_accepts_every_prefix(batch, heads, steps, dtype, strided):
    args, end, rings = _compact(batch, heads, steps, dtype, strided=strided)
    before = args[9].clone()
    # Dense FP32 checkpoints retain the same resident recurrence, independently
    # of the compact external storage format.
    dense = [*args]
    dense[9] = args[9].float()
    dense[11] = torch.empty_like(args[11], dtype=torch.float32)
    dense[-1] = torch.empty_like(args[-1])
    flash_kda.verify_strided(*dense)
    expected_out, expected_states = _verify_reference(*args[:7], args[9], args[10])
    for count in range(-1, steps + 2):
        _verify_compact(args, end, rings)
        _check_verify("compact output", args[-1], expected_out)
        assert torch.equal(args[-1], dense[-1])
        terminal = end[:batch].clone()
        accepted = torch.full((batch,), count, device="cuda", dtype=torch.int32)
        # Mix acceptance lengths in a single launch as well.
        accepted += torch.arange(batch, device="cuda", dtype=torch.int32) % (steps + 1)
        accepted.clamp_(max=steps + 1)
        flash_kda.commit_compact(args[9], args[10], end, args[12], *rings, accepted)
        for bi, length in enumerate(accepted.cpu().tolist()):
            expected = (
                before[bi]
                if length <= 0
                else dense[11][bi, min(length, steps) - 1].to(dtype)
            )
            assert torch.equal(end[bi], expected), (bi, length, dtype)
            if length >= steps:
                assert torch.equal(end[bi], terminal[bi])
        assert torch.equal(end[batch], torch.full_like(end[batch], 7))
    assert torch.equal(before, args[9])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("heads", [7, 149])
def test_compact_padding_graph_and_shared_pool(dtype, heads):
    args, end, rings = _compact(4, heads, 5, dtype, strided=True)
    pool = torch.cat((args[9], end))
    args[9] = end = pool
    args[12].add_(5)
    accepted = torch.zeros(4, device="cuda", dtype=torch.int32)

    def run():
        _verify_compact(args, end, rings)
        flash_kda.commit_compact(args[9], args[10], end, args[12], *rings, accepted)

    # Rings refer to destinations in the common pool.
    shape = (pool.shape[0], 5, heads, D)
    rings = [
        torch.full(shape, 7.0, device="cuda", dtype=torch.float32) for _ in range(2)
    ] + [
        torch.full(shape, 7.0, device="cuda", dtype=torch.bfloat16),
        torch.full(shape[:-1], 7.0, device="cuda", dtype=torch.float32),
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    saved = pool.clone()
    for ids, destinations, lengths in [
        ([0, -1, 2, 99], [5, 6, -1, 8], [0, 2, 5, 3]),
        ([0, 1, 2, 3], [5, 6, 7, 99], [3, 0, 5, 1]),
    ]:
        args[10].copy_(torch.tensor(ids, device="cuda", dtype=torch.int32))
        args[12].copy_(torch.tensor(destinations, device="cuda", dtype=torch.int32))
        accepted.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(pool[:5], saved[:5])
        for bi, (src, dst, length) in enumerate(zip(ids, destinations, lengths)):
            if src < 0 or src >= len(pool) or dst >= len(pool):
                assert torch.count_nonzero(args[-1][bi]) == 0
            elif length <= 0 and dst >= 0:
                assert torch.equal(pool[dst], pool[src])


def test_fp32_resident_retains_small_updates_across_chunks():
    import math

    tokens = 4097
    data = _prefill(tokens, 3, torch.float32)
    for x in data[:3]:
        x.zero_()
    data[3].fill_(-18.0)
    data[-1].fill_(1.001)
    out, last = _forward(data)
    expected = data[-1].double() * math.exp(-5 * tokens / (1 + math.exp(18)))
    assert torch.count_nonzero(out) == 0
    assert float((last.double() - expected).abs().max()) < 2.5e-5
    assert not torch.equal(last, last.bfloat16().float())


@pytest.mark.parametrize("bounds", [(17, 33), (17, 34), (3, 36)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp32_prefill_ignores_nonfinite_sequence_padding(bounds, dtype):
    data = _prefill(197, 7, dtype)
    begin, end = bounds
    piece = tuple(x[:, begin:end].contiguous() for x in data[:5]) + (data[-1].clone(),)
    expected = recurrent_reference(piece, True)
    for x in data[:5]:
        x[:, :begin] = float("nan")
        x[:, end:] = float("nan")
    cu = torch.tensor(bounds, device="cuda", dtype=torch.int64)
    out, last = _forward(data, cu=cu)
    for result, reference in zip((out[:, begin:end], last), expected):
        relative = (
            result.cpu().double() - reference
        ).norm() / reference.norm().clamp_min(1e-12)
        assert float(relative) < 0.025
