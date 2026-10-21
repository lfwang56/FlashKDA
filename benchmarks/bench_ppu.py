"""Standalone PPU prefill/decode/verify latency benchmark. See docs/PPU.md."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_kda


def make_case(op, args):
    batch, heads, dim = args.batch, args.heads, 128
    steps = args.tokens if op == "prefill" else (args.steps if op == "verify" else 1)
    dtype = torch.float32 if args.state_dtype == "fp32" else torch.bfloat16
    rand = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    lengths = args.seq_lens if op == "prefill" else None
    if lengths:
        batch, steps = 1, sum(lengths)
    shape = (batch, steps, heads, dim)
    packed = rand(batch, steps, 3 * heads * dim)
    q, k, v = (x.unflatten(-1, (heads, dim)) for x in packed.chunk(3, -1))
    if not args.strided:
        q, k, v = (x.contiguous() for x in (q, k, v))
    gate, beta = rand(*shape), rand(batch, steps, heads)
    alog = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    out = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    state = (
        torch.randn(
            len(lengths) if lengths else batch,
            heads,
            dim,
            dim,
            device="cuda",
            dtype=dtype,
        )
        * 0.05
    )
    indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    if op == "prefill":
        last = torch.empty_like(state)
        cu = (
            torch.tensor([0, *lengths], device="cuda", dtype=torch.int64).cumsum(0)
            if lengths
            else None
        )

        def run():
            flash_kda.fwd(
                q,
                k,
                v,
                gate,
                beta,
                dim**-0.5,
                out,
                alog,
                bias,
                -5.0,
                state,
                last,
                cu,
            )
    elif op == "decode":
        initial = state.clone()

        def reset():
            state.copy_(initial)

        def run():
            flash_kda.decode(
                packed.view(batch, -1),
                gate.view(batch, -1),
                beta.view(batch, heads),
                alog,
                bias,
                dim**-0.5,
                state,
                indices,
                out,
                lower_bound=-5.0,
            )

        return run, reset, steps
    else:
        checkpoints = torch.empty(
            batch, steps, heads, dim, dim, device="cuda", dtype=dtype
        )
        verify = flash_kda.verify_strided if args.strided else flash_kda.verify

        def run():
            verify(
                q,
                k,
                v,
                gate,
                beta,
                alog,
                bias,
                dim**-0.5,
                -5.0,
                state,
                indices,
                checkpoints,
                indices,
                out,
            )

    return run, lambda: None, steps


def measure(run, reset, args):
    for _ in range(args.warmup):
        reset()
        run()
    torch.cuda.synchronize()
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        run = graph.replay
    samples = []
    for _ in range(args.repeats):
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(args.iters)
        ]
        for start, end in events:
            reset()
            start.record()
            run()
            end.record()
        torch.cuda.synchronize()
        samples.extend(start.elapsed_time(end) for start, end in events)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.mean(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op", choices=("prefill", "decode", "verify", "all"), default="all"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--seq-lens", type=lambda x: [int(n) for n in x.split(",")])
    parser.add_argument("--state-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--strided", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tag", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    for name in ("batch", "tokens", "steps", "heads", "warmup", "iters", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.seq_lens is not None and (
        not args.seq_lens or min(args.seq_lens) < 0 or not sum(args.seq_lens)
    ):
        parser.error(
            "--seq-lens must contain non-negative lengths with a positive total"
        )
    if (
        args.strided
        and args.batch != 1
        and args.op in ("prefill", "all")
        and not args.seq_lens
    ):
        parser.error("strided prefill requires --batch 1 or --seq-lens")
    torch.manual_seed(913)
    results = []
    for op in ("prefill", "decode", "verify") if args.op == "all" else (args.op,):
        run, reset, tokens = make_case(op, args)
        result = {
            "operation": op,
            "tokens": tokens,
            "heads": args.heads,
            **measure(run, reset, args),
        }
        results.append(result)
        print(
            f"{op}: {result['median_ms']:.6f} ms median ({'graph' if args.graph else 'eager'})"
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps({"configuration": vars(args), "results": results}, indent=2)
            + "\n"
        )


if __name__ == "__main__":
    main()
