# PPU backend

FlashKDA provides prefill, packed single-token decode and dense chain verification
on T-Head Zhenwu ZW-M890P (tc02/ppu0015). Head and value dimensions are both 128.
Other PPU generations are rejected at build time.

## Installation

The PPU sources are stored in HGGC-original format, as in
FlashAttention-for-sail and FlashMLA-for-sail. Before compiling with the PPU
SDK's CUDA-compatible `nvcc`, preprocess a separate PPU build checkout with
[cudafy-for-sail](https://github.com/lfwang56/cudafy-for-sail). Keep the original
checkout for development and CUDA builds; cudafy modifies its target in place.

PPU headers come from the pinned `actlize` submodule
([ACTLIZE](https://github.com/t-head/actlize), revision
`0146d9e9462e38da4f455df1f71eae96465c45ff`, matching FlashMLA-for-sail).
Its `include/` provides `cutlass/`, `cute/` and `ppu/`. The original NVIDIA
`cutlass/` submodule is used only by the CUDA build. No headers from another
FlashMLA checkout or external CUTLASS path are needed.

Use a PPU-enabled PyTorch environment and PPU SDK 2.2 or a compatible version.
The validation environment uses HGGC `2.2.0-663c22` (Runtime API v3), the
SDK's CUDA-compatible compiler reporting CUDA 13.0, and PyTorch `2.11.0`.
All PPU compute stages are compiled into the native
extension; no Triton package is required.

```bash
# Create a separate checkout from the desired FlashKDA branch/commit.
git clone /path/to/FlashKDA FlashKDA-ppu-build
cd FlashKDA-ppu-build
git submodule update --init actlize

# Convert both the PPU implementation and its local ACTLIZE headers.
python /path/to/cudafy-for-sail/cudafy.py flashkda . --dry-run
python /path/to/cudafy-for-sail/cudafy.py flashkda .

export PPU_SDK=/usr/local/PPU_SDK
export CUDA_HOME="$PPU_SDK/CUDA_SDK"
export PATH="$CUDA_HOME/bin:$PATH"
FLASH_KDA_BACKEND=ppu FLASH_KDA_PPU_TARGET=tc02 MAX_JOBS=4 \
  python -m pip install -v --no-build-isolation .
```

For development in the converted checkout, use
`FLASH_KDA_BACKEND=ppu python setup.py build_ext --inplace` with the same
environment. Recreate the build checkout after editing the original sources;
do not commit generated CUDA-compatible files to the source branch. The default
backend remains CUDA, and both extensions are named `flash_kda_C`.

### Preprocessing rules

FlashKDA uses cudafy's general rules; no private SDK map or residual patch is
needed. The `flashkda` command validates the repository and local ACTLIZE
checkout before converting the tree. It rejects a symlink for `actlize`
to avoid modifying shared dependency sources.

- SDK headers/types use HGGC spellings in the PPU source, for example
  `hggc_awbarrier.h`, `hggcStream_t` and `__ppu_bfloat162`. Cudafy maps them to
  the CUDA-compatible SDK spellings.
- The tc02 flag `-arch=ppu_15` becomes the SDK's
  `-gencode=arch=compute_89,code=sm_89`; `-Xllvm` becomes `-mllvm`.
- PPU inline assembly, intrinsics, `__HGGC_ARCH__ == 150`,
  `ACOMPUTE_VERSION=10500` and HGGC pragmas retain their PPU meaning.
- The same general conversion applies to the initialized ACTLIZE headers.
  Repeating preprocessing produces no further changes. `--dry-run` reports
  changes without writing files.

This is source compatibility for the PPU SDK, not a translation of the kernels
to NVIDIA hardware. PyTorch's `torch.cuda` and `c10::cuda` APIs remain the
framework interface in the PPU environment. The common exports and original
CUDA sources keep their upstream format.

## Shared exports and backend selection

Both builds use `csrc/flash_kda.cpp` for module registration and the original
launcher contract in `csrc/fwd.h`. The build selects `csrc/smxx/fwd_launch.cu`
for CUDA or the implementation under `csrc/ppu/` for PPU.

PPU tensor validation, workspace sizing and host dispatch live in
`csrc/ppu/torch_api.cpp`; private launcher arguments live in `csrc/ppu/launch.h`.
The shared binding registers the original `fwd` and `get_workspace_size`
signatures for both builds. PPU-only operations and compatibility overloads are
registered by the PPU implementation. CUDA keeps its upstream implementation,
arguments, beta layout and workspace size.

There is one native module, `flash_kda_C`, and one public Python interface,
`flash_kda/__init__.py`. PPU prefill uses the original forward signature and
one BF16 resident-state policy, with FP32 matrix accumulation and state-update
arithmetic. BF16 and FP32 external state buffers are both supported.
Unsupported operations report a capability error through the public interface.

## APIs and tensor contracts

All tensors in a call must be on the same device. Operations use that device's
current stream and restore the caller's current device. BF16 vector buffers must
be aligned to 16 bytes. Native workspace buffers must be aligned to 128 bytes;
normal PyTorch allocations satisfy these requirements. Dimensions exceeding
native launch or indexing limits are rejected. Scale values must remain finite when converted to FP32.

`flash_kda.fwd` retains the API described in the main README. Prefill requires a
finite gate bound in `[-5, 0]` to keep the 16-token inverse operands representable.
`cu_seqlens` is contiguous device int64, starts at zero, ends at the total token
count, and is nondecreasing. Empty sequences preserve their initial state or
produce a zero final state. Callers provide valid cumulative lengths without a
host synchronization in the hot path. See [prefill architecture](PREFILL_PPU.md)
for packed QKV layouts and FP32 accumulation.

`flash_kda.decode(mixed_qkv, gate, beta, A_log, dt_bias, scale, state,
state_indices, out, lower_bound=None)` updates selected state slots in place:

| Tensor | Dtype | Shape |
| --- | --- | --- |
| `mixed_qkv` | BF16 | `[B, 3*H*128]` |
| `gate`, `beta` | BF16 | `[B,H*128]`, `[B,H]` raw logits |
| `A_log`, `dt_bias` | FP32 | `H`, `H*128` elements |
| `state` | BF16 or FP32 | `[slots,H,128,128]`, contiguous |
| `state_indices` | int32 | `[B]`, contiguous |
| `out` | BF16 | `[B,1,H,128]`, contiguous |

All decode tensors are contiguous. Non-negative source slots must be unique
within a decode call. Negative slots pad requests. Slots past the pool size
produce zero output without accessing the pool. `lower_bound=None` uses the
unbounded gate; finite non-positive bounds select bounded gating. Capability
flags are read from the loaded extension, so an old binary cannot advertise the
new bounded ABI through Python source alone.

`flash_kda.verify(q, k, v, gate, beta, A_log, dt_bias, scale, lower_bound,
state, state_indices, intermediate_states, intermediate_state_indices, out)`
verifies a dense speculative chain. Q/K/V/gate and output use `[B,T,H,128]`, beta
uses `[B,T,H]`, and checkpoints use `[slots,cache_steps,H,128,128]` with
`cache_steps >= T`. State and checkpoints have the same BF16 or FP32 dtype.
Checkpoint destinations must be unique within a call. The committed state stays
unchanged; the private chain accumulates in FP32 and each checkpoint is rounded
to its storage dtype. The bound must be finite and non-positive.

A negative source slot produces zero output and no checkpoint writes. A negative
checkpoint slot suppresses checkpoint writes while still computing output from
a valid source. Out-of-range source or checkpoint slots produce zero output and
leave both pools unchanged. `verify_strided` accepts dense head/channel axes
with a constant, non-overlapping token stride across the flattened batch. The
same layouts are accepted by `verify`. Tree and ragged verification require a
separate implementation. See [decode and MTP verification architecture](DECODE_PPU.md)
for state precision, kernel dispatch and fusion boundaries.

## Regression tests

Build the extension in the checkout being tested. No FLA or SGLang installation
is required for these correctness tests:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q tests/test_ppu.py
```

Two devices enable device/stream isolation tests; those cases skip with one
visible device.
The decode and verify suites compare against independent tokenwise recurrences,
cover both state dtypes and the runtime dispatch boundaries, and check padding,
long rollouts, speculative checkpoints and committed-state preservation. Prefill
also checks graph capture, packed inputs, ragged batches and segmented replay.
CUDA retains its original exact reference and comparisons.

The single PPU benchmark covers prefill, decode and verify without FLA or SGLang:

```bash
python benchmarks/bench_ppu.py --op all --batch 1 --tokens 8192 --heads 6 --graph --json results.json
python benchmarks/bench_ppu.py --op verify --batch 4 --steps 8 --heads 12 --state-dtype fp32
```

Use `--state-dtype bf16` or `--state-dtype fp32` for external state storage,
`--strided` for packed QKV prefill/verify inputs, and `--seq-lens 65,17,0` for
ragged prefill. Prefill uses the same BF16 resident-state policy for both storage types.
GPU event timing reports median/min/mean latency; `--graph` measures graph replay.
Warmup, iterations and repeats are configurable. Decode resets its state pool
outside the timed interval so repeated measurements start from identical states.
