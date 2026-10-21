# PPU prefill architecture

PPU prefill uses the existing `flash_kda.fwd` API and the `flash_kda_C`
extension. All preparation, recurrence, workspace and matrix-load stages live
in the native PPU backend. There are no model-specific
source trees, separate prefill extension builds or serving adapters.

| Component | Responsibility |
| --- | --- |
| `flash_kda/__init__.py::fwd` | Original public forward API and external state buffers |
| `csrc/flash_kda.cpp` | Shared module registration and original forward/workspace bindings |
| `csrc/fwd.h` | Original shared forward launcher contract |
| `csrc/ppu/torch_api.cpp` | PPU input validation, workspace sizing, host dispatch and extension bindings |
| `csrc/ppu/launch.h` | Private PPU launcher arguments and internal stage declarations |
| `csrc/ppu/fwd_launch_ppu.cu` | Launch configurations and workspace views |
| `csrc/ppu/fwd_kernel1_ppu.cuh` | Q/K normalization, gate/beta activation and chunk preparation |
| `csrc/ppu/fwd_kernel2_ppu.cuh` | Serial, summary and replay recurrences using the same PPU helpers |
| `csrc/ppu/fwd_scan_ppu.cuh` | BF16 segment carry composition and external state conversion |
| `csrc/ppu/ppu_layouts.cuh` | Shared layouts and swizzled-load operations, including uniform-address overloads |

## API and precision

Prefill has one resident-state policy for BF16 and FP32 external state
storage. Matrix products accumulate in FP32. State updates keep both `g_total`
and the MMA delta in FP32 through an FP32 FMA, then round the updated state to
BF16, matching the CUDA update sequence. Neither update operand is narrowed
before the FMA. Affine factorization and parallel carry composition can still
change rounding relative to the CUDA implementation.

```python
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=initial, final_state=final, cu_seqlens=cu)
```

BF16 state buffers feed the resident fragments directly. FP32 initial state is
narrowed to BF16 on load; final state is widened back to FP32 on publication.
FP32 storage therefore does not select a different recurrent arithmetic mode.
Empty sequences preserve the original state exactly, including FP32 values
that cannot be represented by BF16.

Q/K/V may be views of a packed projection on PPU: heads and channels must be
dense, token strides must be non-overlapping multiples of 64 elements, and
strided input uses a flattened batch of one. Gates, beta, output, workspace,
parameters and state tensors retain their contiguous layout contracts. The
supported head dimension remains 128; the number of heads is not fixed.

## Preparation, recurrence and scan

The 16-token prepare kernel solves the lower-triangular inverse in FP32 using
forward substitution. It then computes the state-independent factors
`W = diag(beta) INV^T Kr` and `M = Mqk INV diag(beta)` in parallel across chunks.
The recurrent kernel uses residual `r = V^T - S Kd^T`, updates
`S = S diag(g_total) + r W`, and emits `Qd S^T + M r^T`. The factors occupy the
existing Kr/Mqk workspace slots; the default recurrence skips raw INV/beta loads.
Both BF16 and FP32 external state use these affine transforms. Preparation
operates on 16-token chunks; native launchers select serial or segmented
recurrence without a separate Python scheduling module.

Selected single-sequence H6/H12 shapes compute segment summaries
`A = F(I, 0)` and `B = F(0, V)` in separate recurrences, followed by carry
composition and parallel replay. Eight MMA warps split the resident state into
16-row blocks. A homogeneous warp skips arithmetic after all of its BF16 state
bits are zero (ignoring the sign of zero, preserving subnormal bits), while
continuing every pipeline wait/release. Shared staging and CTA thread counts
include only the buffers and output warp required by each specialization.

For selected sequences of at least 8192 tokens, H6 uses 26 segments and H12 uses
13, targeting 156 CTAs on the measured 39-CU tc02 device. Shorter scan shapes
retain 8/16 segments. The bounds distribute complete 16-token prepare tiles by
quotient and remainder, so uneven segments cover every token without overlap.
The public workspace still reserves 64 segments and its layout is unchanged.
Segment boundaries can change BF16 carry-composition rounding; each chunk retains
the same FP32 update followed by one BF16 state store.
If every A matrix of a head is bitwise zero, its carries can be copied in parallel:
the first carry is the caller's initial state narrowed to BF16 (or zero), and `carry[j] = B[j-1]`
for later segments. Any nonzero BF16 bit, including subnormals, retains the full
affine scan for that head. This uses current-call device metadata and works during
graph replay without host reads. Metadata temporarily occupies replay-final
storage before replay; the public workspace size and export interfaces are unchanged.

Prepare and replay preserve each Q/K/V token stride, so dense and packed QKV
use the same segment selection without materializing dense copies. Short fixed
H12 sequences use one chunk per input stage through 512 tokens and two through
2048, matching the single-sequence varlen policy. Other shapes and ragged batches
retain their stride-aware serial recurrence.

## Build and validation

Initialize `actlize`, preprocess a separate PPU build checkout with
`cudafy.py flashkda`, then build the native extension as described in
[PPU installation and API contracts](PPU.md#installation). Run the consolidated
regression from that converted checkout:

```bash
python -m pytest -q tests/test_ppu.py
```

Decode and speculative verification use the same extension;
see [decode and MTP verification architecture](DECODE_PPU.md).

Tests cover a tokenwise FP64 reference, BF16/FP32 state storage, optional states,
packed views, fixed/ragged batches including empty sequences, the 1023/1024/1025
scan boundary, and workspace rejection before launch.
Default BF16 scan tests also cover strong/weak decay and heads taking different
carry paths in the same launch against the independent FP64 tokenwise reference.
Closed-form decay cases also check that decay remains FP32 until the updated
state is stored, for both BF16 and FP32 external state buffers.

Closed-form mixed-column cases cover uneven H6/P26 and H12/P13 segments with
partly zero homogeneous rows and live columns in dense/packed fixed/varlen calls.

BF16 affine preparation distributes the 16 inverse rows over 16 half-warps.
Each row uses a right-side triangular solve with FP32 pivot broadcasts and FMAs;
the CTA barrier completes all rows before the existing BF16 affine factors are
formed. This changes FP32 summation order and can change subsequent BF16 rounding.

Eight-warp segmented replay writes packed output pairs directly from the MMA
warps, avoiding its separate store warp and full shared output staging. This is
a private compile-time choice; non-segmented recurrences retain the coalescing
store warp. The measured tc02 replay avoids local register spills with this
choice. The existing global direct-output ablation still defaults to zero.

Only the last replay segment writes the requested public final state; all other
replay final states are unused and omitted. Calls without a final-state output
omit every replay state store. The summary/carry workspace reservation is
unchanged, and no final device-to-device state copy is needed. Optional initial
and final states are covered by the consolidated PPU regression file.

## FP32 external state

FP32 storage uses the same selected H6/H12 segmented schedule as BF16 storage.
The carry loaders narrow the input directly into BF16 carries. Replay reads
those carries as BF16 and only its final segment widens the final state directly
into the caller's FP32 buffer. Conversion is fused into existing kernels and
requires neither an intermediate FP32 carry tensor nor a separate conversion
or final-state copy kernel.

Selected single-sequence FP32-state H16 shapes also use segmented replay:
16384 tokens use five segments; 32768, 65536 and 131072 use ten. These shapes
reuse the inverse workspace after affine preparation, when no recurrence reads
it. Three compact summary/carry arrays and their metadata fit the existing
inverse allocation, checked at compile time. Other shapes retain serial
recurrence. The public workspace size and CUDA implementation are unchanged.

A single-sequence variable-length call can contain unused capacity or start at
a nonzero token offset. The prefix kernel writes a full-range flag into the
existing aligned prefix space. Summaries, carry kernels and replay run only for
`cu_seqlens=[0,T]`; an oppositely guarded serial recurrence handles other ranges.
Both branches use current device metadata during graph replay, without reading
sequence offsets on the host. This also preserves exact identity for an empty
FP32-state sequence. BF16 storage shares this range correctness fix.


On ZW-M890P/tc02 with SDK 2.2, the original `ppu-kernel-regression` native
prefill matrix (two models, 16K/64K/100K history, batch 1/4/8) passed its original
full-history reference checks. Across its 18 FP32-state cases, the optimized
default path measured 1.871x PLA and 1.297x the frozen `e79d3e7` default path
geometrically; every case beat both, and the largest gain versus the frozen
path was 3.171x. Both FlashKDA versions used the BF16 resident-state policy.
The original warmup=2 and
repeats=5 were preserved, including required PLA packed-input materialization
inside timing. These measurements apply to that matrix and hardware.

## Uniform TP16 B4 batches

Four equal-length sequences with H=6 and T_total=16384 use six independent
segments per sequence. Summary, carry and replay indexing keep each sequence
isolated; only its last segment publishes that sequence's final state. The
segment count is compiled into the recurrence to avoid runtime divide/modulo
in output addressing. Single-sequence dispatch retains its original indexing.

For varlen calls, the prefix kernel checks device offsets against the full
uniform partition. Ragged, padded or empty sequences take the guarded original
recurrence. Both branches observe current device metadata during graph replay.
The native workspace query reserves four [24,H,128,128] BF16 scan arrays for
this shape (18 MiB plus alignment); callers must use get_workspace_size rather
than caching an allocation size from an older binary. CUDA workspace is unchanged.

The FP32 prepare scan keeps its sequential summation order. Forward decay
remains exp2(cumulative_gate); the inverse factor reuses that result through
the tc02 native reciprocal. Bounded gating keeps both factors representable.
Normalization stores and independent gate preparation share the existing
subsequent CTA barrier.
