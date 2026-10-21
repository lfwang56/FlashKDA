# PPU decode and MTP verification architecture

PPU provides packed single-token decode and dense, linear-chain verification
for MTP/topk=1 speculative decoding, including DSpark-style chains. Both use
the shared `flash_kda` Python package and `flash_kda_C` extension on
ZW-M890P (tc02/ppu0015), with K=V=128. See [PPU APIs](PPU.md) for tensor shapes,
layout requirements and installation, and [prefill architecture](PREFILL_PPU.md)
for the separate chunked prefill algorithm.

| Component | Responsibility |
| --- | --- |
| [`flash_kda/__init__.py`](../flash_kda/__init__.py) | Public `decode`, `verify` and `verify_strided`; loaded-binary capability checks |
| [`csrc/flash_kda.cpp`](../csrc/flash_kda.cpp) | Shared module registration; delegates PPU extensions to `bind_ppu` |
| [`csrc/ppu/torch_api.cpp`](../csrc/ppu/torch_api.cpp) | PPU bindings, tensor validation, device/stream selection and host dispatch |
| [`csrc/ppu/launch.h`](../csrc/ppu/launch.h) | Private decode and verification launcher declarations |
| [`csrc/ppu/decode_launch_ppu.cu`](../csrc/ppu/decode_launch_ppu.cu) | Packed decode recurrence, state traffic and launch selection |
| [`csrc/ppu/verify_launch_ppu.cu`](../csrc/ppu/verify_launch_ppu.cu) | Private multi-token recurrence and per-step checkpoints |
| [`csrc/ppu/ppu_aiu.cuh`](../csrc/ppu/ppu_aiu.cuh), [`ppu_common.cuh`](../csrc/ppu/ppu_common.cuh) | Asynchronous copies, synchronization and PPU primitives |

## Recurrence and precision

Each head stores state as `S[V,K]`. For one token, the kernels normalize raw
BF16 Q/K in FP32, scale Q, activate beta, and form a per-K-channel decay:

```text
q = scale * Q / sqrt(sum(Q * Q) + 1e-6)
k = K / sqrt(sum(K * K) + 1e-6)
b = sigmoid(beta)
x = gate + dt_bias
d = exp(lower_bound * sigmoid(exp(A_log) * x))  # bounded gate

Sd = S * d[None, :]
r = b * (V - Sd @ k)
Snext = Sd + r[:, None] * k[None, :]
out = Snext @ q
```

Q/K/V, gate and beta enter as BF16; `A_log` and `dt_bias` are FP32. Decode
also supports `lower_bound=None`, selecting
`d = exp(-exp(A_log) * softplus(x))`. Verification requires a finite,
non-positive bound. Output is written as BF16.

State storage and the recurrent arithmetic have different roles:

| Operation | BF16 state storage | FP32 state storage |
| --- | --- | --- |
| Ordinary decode | Load as FP32, compute the update/output in FP32, round the committed state on writeback | Load, update and write back in FP32 |
| MTP verification | Load once as FP32; retain FP32 private state across tokens; round each checkpoint independently | Load once and retain FP32 private state; write FP32 checkpoints |

Decode computes output from the FP32 updated row before the storage rounding.
A later decode call reloads the state written by the previous call. Within one
verification call, a rounded BF16 checkpoint is never reloaded into the private
chain. Repeating BF16-state decode T times therefore has different rounding
points from a single T-token verification call.

FP32 decode state is not narrowed to BF16 as it is in the prefill
resident-state policy; the two operations have different recurrence contracts.

## Ordinary decode

`flash_kda.decode` reads packed Q/K/V from `[B,3*H*128]`, loads each selected
state slot, and updates that slot in place. One native kernel launch fuses
Q/K normalization, gate and beta activation, the recurrent update, state
writeback and state/query readout. The first four warps prepare Q, K,
decay/beta and V while asynchronous state loading is in flight. State rows are
independent, allowing multiple CTAs to own disjoint V rows of a request/head.

The launcher selects the state-loading path using B, H and storage dtype.
These are internal choices in `launch_decode_typed`, not public API options:

| Condition | State loading | CTAs per request/head | Warps per CTA |
| --- | --- | ---: | ---: |
| BF16, `B*H <= 12` | Bulk AIU | 8 | 4 |
| BF16, `13 <= B*H <= 3072` | Bulk AIU | 2 | 8 |
| BF16, `B*H > 3072` | Bulk AIU | 1 | 8 |
| FP32, `B*H <= 96` | Four-stage rolling | 4 | 4 |
| FP32, `B >= 128`, `B*H <= 3072` | Bulk AIU | 2 | 8 |
| FP32, `B >= 128`, `B*H > 3072` | Bulk AIU | 1 | 16 |
| FP32, `B < 128`, `B*H == 384` | Four-stage rolling | 1 | 4 |
| FP32, `B < 128`, `B*H == 768` | Four-stage rolling | 4 | 4 |
| Other FP32 cases with `B < 128`, `B*H > 96` | Four-stage rolling | 1 | 8 |

Bulk loading stages each CTA's state rows in shared memory. Rolling loading
seeds four stages with AIU transfers and refills them with vector asynchronous
copies as recurrence consumes the rows. Warp and half-warp reductions compute
the state/key and state/query dot products; there is no chunk-inverse stage.

The caller supplies unique non-negative state indices within a decode call.
Negative or out-of-range indices produce zero output and leave the pool
untouched. The host validates tensor metadata; slot masking happens on the
device without copying the index tensor to the host.

## MTP and DSpark-style verification

`flash_kda.verify` and `verify_strided` share a native implementation. Both
accept dense head/channel axes and a constant flattened token-row stride,
including views into packed QKV projections. The committed state is read only;
each draft step's post-state goes to
`intermediate_states[intermediate_state_indices[b], t]`.

Each CTA loads its state rows once through AIU, widens them into FP32 registers,
then processes T tokens serially. Different row CTAs run independently. The
initial shared-state allocation is reused for per-token preparation scratch
after all state rows have moved to registers. Each token prepares Q/K,
decay/beta and V, updates the private state, and writes its output/checkpoint.
The BF16 four-split specialization alternates two scratch buffers, allowing the
end-of-token barrier to be omitted; other specializations use one scratch buffer.

All verification CTAs use four warps. Small request/head counts use more row
splits to expose parallelism:

| State storage | `B*H` | CTAs per request/head | V rows per CTA |
| --- | --- | ---: | ---: |
| BF16 or FP32 | `<= 12` | 16 | 8 |
| BF16 | `13..96` | 8 | 16 |
| BF16 | `> 96` | 4 | 32 |
| FP32 | `13..192` | 8 | 16 |
| FP32 | `> 192` | 4 | 32 |

Checkpoint destinations must be unique within a call; read-only source slots
may repeat. Keep committed state and checkpoint storage separate. A negative
checkpoint index suppresses checkpoint writes while still producing output
for a valid source. Invalid source indices or an out-of-range non-negative
checkpoint index produce zero output and no checkpoint writes.

Verification supports a dense linear chain, not tree branches or ragged draft
lengths. These kernels fuse KDA normalization, activations, recurrence, output
and checkpoint stores. Convolution, ReplaySSM ring maintenance, accepted-length
selection and accepted-state commit/exact-fold are outside these APIs. Serving
integration must select the accepted checkpoint and coordinate the other
state updates.

## Build, regression and measurement

Decode and verification use the same cudafy preprocessing, ACTLIZE submodule
and PPU build as prefill. Both operations run entirely in the native extension.
See [installation](PPU.md#installation).

The existing consolidated test and benchmark files cover both operations:

```bash
python -m pytest -q tests/test_ppu.py -k 'decode or verify or checkpoint'
python benchmarks/bench_ppu.py --op decode --batch 64 --heads 12 --state-dtype fp32 --graph
python benchmarks/bench_ppu.py --op verify --batch 4 --steps 8 --heads 12 --state-dtype bf16 --strided --graph
```

Regression coverage includes BF16/FP32 pools, independent tokenwise references,
bounded/unbounded decode, dispatch boundaries, a 512-step decode rollout,
dense/packed verification, checkpoints, committed-state preservation, slot
sentinels, metadata validation and device/stream isolation.

The benchmark uses bounded gating with `lower_bound=-5`. Decode state resets
are outside timed events; verification reads the same committed state on each
iteration. Use the same state dtype, gate, input layout, cache writes and timing
boundary when comparing implementations. Graph replay excludes Python host
dispatch time. Comparisons against a larger convolution/ring/acceptance pipeline
must account for those additional operations separately.
