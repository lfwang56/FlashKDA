#pragma once
#include "prefill_chunk_ppu.h"

// Kernel 2 -- recurrence. Grid (N, H), one CTA per sequence per head.
//
// Carries a [D_v, D_k] running state across the sequence's chunks. Per chunk:
//   u = (V - Kd @ S^T) * beta ; U = INV @ u
//   out = Qd @ S^T + Mqk @ U
//   S   = S * exp2(g_total) + Kr^T @ U
//
// Two things make this a redesign rather than a translation of the CUDA kernel.
//
// 1. There is no movmatrix. The CUDA kernel's headline optimisation was keeping U
//    in registers and transposing it in place to feed the next MMA as a B operand
//    (fwd_kernel2.cuh:609-612, 634-637). A _TN atom computes
//    C[m,n] = sum_k A[m,k]*B[n,k], so a fragment always denotes the same matrix,
//    never its transpose -- reinterpreting C as A or B is free, transposing is
//    not. The fix is to choose each GEMM's output orientation so the transpose
//    never arises: u and U are produced already transposed as [d_v, t], while out
//    stays in its natural [t, d_v]. Then every reuse is a free relabel and the
//    only two transposing reads, V^T and Kr^T, are hardware transposing loads.
//
// 2. The state cannot live in shared memory. It is the one buffer that must be
//    both MMA-read and register-written, and an AIU/matrix-load buffer may not be
//    element-addressed. Keeping it in registers dissolves that, and also removes
//    32 KB of shared memory, 16 matrix loads per k-loop per chunk, and a 32 KB
//    per-chunk write-back that would have been 8-way bank conflicted. It is legal
//    because the state is already warp-private in the CUDA kernel: each warp reads
//    and writes only its own 32 columns (fwd_kernel2.cuh:541, :723).
//
// Both rest on tc02's CLayout == ALayout and BLayout == ALayout-with-value-modes-
// 1-and-2-exchanged, which csrc/ppu/ppu_frag.cuh proves at compile time.

#include "epilogue.cuh"
#include "ppu_aiu.cuh"
#include "ppu_common.cuh"
#include "ppu_frag.cuh"
#include "ppu_layouts.cuh"
#include "ppu_pipeline.cuh"

namespace flash_kda_ppu {

// Global direct-output ablation. The default launcher selects direct stores
// only for eight-warp segmented replay, where they avoid local spills. Other
// recurrences keep the aggregating store warp for coalesced global writes.
#ifndef FLASH_KDA_PPU_DIRECT_OUT
#define FLASH_KDA_PPU_DIRECT_OUT 0
#endif

// P1 accumulates u in fp32, but P4 consumes bf16. Pairing the intervening
// (V-u)*beta epilogue is numerically valid, but measured 1-3% slower on the long
// TP16 cases because P4 must narrow the value again; retain it as an ablation.

// Move the compact per-chunk metadata with PPU's 16-byte global-to-shared
// async copy.  The AIU remains responsible for the large matrix tiles; both
// engines are committed before the same mbarrier arrival, so MMA consumers see
// a complete stage.  This replaces a serial stream of global loads plus shared
// stores on the only load warp.
#ifndef FLASH_KDA_PPU_ASYNC_SMALL_ARRAYS
#define FLASH_KDA_PPU_ASYNC_SMALL_ARRAYS 1
#endif

template <int D, int CHUNK, int InputStages, int OutputStages, int NumMmaWarps, bool HasStateIn,
          int DvSplit, int ChunksPerStage, bool ProduceOutput, bool ZeroValue, bool AffinePrepared,
          bool DirectOutput>
struct SharedStorageK2 {
    using L = K2Layouts<D, CHUNK, D / DvSplit>;
    static constexpr int kStageRows = CHUNK * ChunksPerStage;
    using StageTile = AiuTile<BF16, kStageRows, L::kCubeW, L::kCubesPerTile>;
    using VStageTile = AiuTile<BF16, kStageRows, L::kCubeW, L::kVCubesPerTile>;
    using StageOut = PaddedRowMajor<BF16, kStageRows, D / DvSplit + 8>;
    static constexpr int kBetaElems = kStageRows;

    struct InStage {
        std::conditional_t<ZeroValue, AiuTile<BF16, 1, L::kCubeW, 1>, VStageTile> v;
        AiuTile<BF16, 3 * kStageRows, L::kCubeW, L::kCubesPerTile> transforms;
        alignas(
            128) BF16 inv[AffinePrepared ? 1 : ChunksPerStage][AffinePrepared ? 1 : L::LM::kElems];
        alignas(
            128) BF16 mqk[ProduceOutput ? ChunksPerStage : 1][ProduceOutput ? L::LM::kElems : 1];
        alignas(128) float g_total[ChunksPerStage][D];
        // Raw recurrence stages prepare's sigmoid(beta). Affine W/M already
        // incorporate it and need only the placeholder storage.
        alignas(128) float beta[AffinePrepared ? 1 : kBetaElems];
    };

    struct OutStage {
        alignas(128) BF16 out[ProduceOutput && !DirectOutput ? StageOut::kElems : 1];
    };

    // The state staging buffer is only alive in the prologue, so it shares storage
    // with the pipeline. Two things keep it from setting the occupancy ceiling:
    //
    //   * Only an incoming state needs it. The final state is written straight from
    //     the MMA warps' registers to gmem, so HasStateOut stages nothing.
    //   * It is always bf16, even for an fp32 state, because the fragments it feeds
    //     are bf16 -- converting during the gmem -> smem copy rounds at exactly the
    //     same point as converting out of an fp32 staging buffer would.
    //
    // Sizing it as fp32 [D, D+8] unconditionally cost 69,632 B, which exceeded the
    // 64,384 B pipeline and silently held every instantiation to 3 CTAs/CU -- so
    // dropping the input stages from 4 to 3 had bought no occupancy at all. At
    // 34,816 B the pipeline is the larger member again and 4 CTAs/CU fit.
    static constexpr int kStateStageBytes = HasStateIn ? L::StateStage::kBytes : 0;

    union {
        struct {
            InStage in[InputStages];
            OutStage out[OutputStages];
        };
        alignas(128) char state_stage[kStateStageBytes > 0 ? kStateStageBytes : 1];
    };

    AsyncInPipeline<InputStages, 32, 32 * NumMmaWarps> in_pipe;
    SyncOutPipeline<OutputStages, 32 * NumMmaWarps, 32> out_pipe;
};

// The ungrouped shape uses 64,384 B/CTA and can reach four CTAs/CU. Grouped TP16
// stages use about 166 KB and run one CTA/CU, but this launch-bounds hint still
// gives the best register allocation in measurement. It is overridable for sweeps.
#ifndef FLASH_KDA_PPU_K2_CTAS_PER_CU
#define FLASH_KDA_PPU_K2_CTAS_PER_CU 4
#endif

// gmem -> padded shared staging, one 16-byte packet per thread per step.
//
// This runs once per sequence, so it is invisible at 512 chunks per sequence and
// dominant at one. Element-at-a-time it was D*D/kThreads = 85 two-byte global
// loads per thread; the assembly census counted 250 two-byte and 82 four-byte
// global loads per instantiation, and measured cost tracked it: at H=96 the same
// total work took 6.4 ms spread over 32-chunk sequences but 19.6 ms over 1-chunk
// ones.
template <class T, int ROWS, int COLS, int PAD, int NumThreads>
CUTE_DEVICE void stage_state_2d(T *dst, T const *src, int tid) {
    constexpr int kPer = 16 / int(sizeof(T));
    constexpr int kPacketsPerRow = COLS / kPer;
    static_assert(COLS % kPer == 0, "the row must split into whole 16-byte packets");
    static_assert(PAD * int(sizeof(T)) % 16 == 0,
                  "the padded row stride must keep packets 16-byte aligned");
    for (int p = tid; p < ROWS * kPacketsPerRow; p += NumThreads) {
        int const row = p / kPacketsPerRow;
        int const col = (p % kPacketsPerRow) * kPer;
        *reinterpret_cast<cute::uint128_t *>(&dst[row * PAD + col]) =
            *reinterpret_cast<cute::uint128_t const *>(&src[row * COLS + col]);
    }
}

// Same, narrowing fp32 -> bf16 on the way. Reads a 16-byte packet of four floats
// and writes the eight bytes they become, so the source side stays fully coalesced.
template <int ROWS, int COLS, int PAD, int NumThreads>
CUTE_DEVICE void stage_state_2d_cvt(BF16 *dst, float const *src, int tid) {
    constexpr int kPer = 4;
    constexpr int kPacketsPerRow = COLS / kPer;
    static_assert(COLS % kPer == 0, "the row must split into whole 16-byte float packets");
    static_assert(PAD * int(sizeof(BF16)) % 8 == 0,
                  "the padded row stride must keep 8-byte bf16 runs aligned");
    for (int p = tid; p < ROWS * kPacketsPerRow; p += NumThreads) {
        int const row = p / kPacketsPerRow;
        int const col = (p % kPacketsPerRow) * kPer;
        float f[kPer];
        *reinterpret_cast<cute::uint128_t *>(f) =
            *reinterpret_cast<cute::uint128_t const *>(&src[row * COLS + col]);
        BF16 b[kPer];
        CUTE_UNROLL
        for (int i = 0; i < kPer; ++i)
            b[i] = BF16(f[i]);
        *reinterpret_cast<uint64_t *>(&dst[row * PAD + col]) =
            *reinterpret_cast<uint64_t const *>(b);
    }
}

// Publish one completed segment head from a compact internal BF16 matrix.
// All CTA threads participate in the copy; only the last segment is public.
template <int D, bool OutputStateFP32, int SequenceBatches, int SegmentsPerSequence,
          bool AllSegments = false>
CUTE_DEVICE void publish_reused_segment_final(BF16 const *source, void *final_state, int H, int N,
                                              int logical_segment = -1) {
    int const segment = logical_segment >= 0 ? logical_segment : int(blockIdx.x);
    int const head = blockIdx.y;
    bool const last =
        SequenceBatches == 1
            ? segment + 1 == N
            : (segment + 1) % (SegmentsPerSequence > 0 ? SegmentsPerSequence : 1) == 0;
    if (!AllSegments && !last)
        return;
    int const sequence =
        AllSegments ? segment
                    : (SequenceBatches == 1
                           ? 0
                           : segment / (SegmentsPerSequence > 0 ? SegmentsPerSequence : 1));
    source += int64_t(segment * H + head) * D * D;
    int64_t const dst = int64_t(sequence * H + head) * D * D;
    if constexpr (OutputStateFP32) {
        for (int i = threadIdx.x; i < D * D / 4; i += blockDim.x) {
            float4 const value = make_float4(float(source[4 * i]), float(source[4 * i + 1]),
                                             float(source[4 * i + 2]), float(source[4 * i + 3]));
            reinterpret_cast<float4 *>(static_cast<float *>(final_state) + dst)[i] = value;
        }
    } else {
        for (int i = threadIdx.x; i < D * D / 8; i += blockDim.x)
            reinterpret_cast<cute::uint128_t *>(static_cast<BF16 *>(final_state) + dst)[i] =
                reinterpret_cast<cute::uint128_t const *>(source)[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2
// ---------------------------------------------------------------------------
template <int CHUNK, int D, int InputStages, int OutputStages, int NumMmaWarps, bool HasStateIn,
          bool HasStateOut, bool StateFP32, bool IsVarlen, int DvSplit, int ChunksPerStage,
          bool ProduceOutput = true, bool IdentityState = false, int PathFlagMode = 0,
          bool ZeroValue = false, bool DualSummary = false, bool AffinePrepared = false,
          bool Segmented = false, bool DirectOutput = FLASH_KDA_PPU_DIRECT_OUT,
          bool OutputStateFP32 = StateFP32, int SegmentSequences = 1, int SegmentsPerSequence = 0,
          int PrefixMode = 0, int PrefixChunks = 4, int StaticHeads = 0, int StaticTokens = 0,
          int StaticSequences = 0, bool DirectZeroCarry = false, bool ExternalHasStateIn = false,
          bool ExternalStateFP32 = false, bool PrepublishFinal = false,
          bool PublicHasStateOut = false, int SerialSegmentIndex = -1, bool SegmentCarryOut = false,
          bool FirstSegmentTrueB = false>
__global__ void __launch_bounds__(32 * (NumMmaWarps + 1 + (ProduceOutput && !DirectOutput ? 1 : 0)),
                                  (DualSummary || PrefixMode == 4 || PrefixMode == 6 ||
                                   PrefixMode == 7)
                                      ? 2
                                      : FLASH_KDA_PPU_K2_CTAS_PER_CU)
    flash_kda_ppu_fwd_recurrence(
        BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
        BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd,
        BF16 const *__restrict__ ws_kr, float const *__restrict__ ws_gt,
        BF16 const *__restrict__ ws_inv, BF16 const *__restrict__ ws_mqk,
        void const *__restrict__ initial_state_ptr, void *__restrict__ final_state_ptr,
        BF16 *__restrict__ out_ptr, int T_total_runtime, int H_runtime, int N_runtime,
        int64_t const *__restrict__ cu_seqlens, int total_tiles,
        int const *__restrict__ tile_prefix, int const *__restrict__ path_flag,
        void *__restrict__ auxiliary_final_state_ptr, int v_stride,
        int *__restrict__ summary_nonzero, BF16 const *zero_summary_b = nullptr,
        void const *external_initial_state = nullptr, void *external_final_state = nullptr) {
    if constexpr (PrefixMode == 7) {
        static_assert(PathFlagMode == 1 && SegmentSequences == 8 && SegmentsPerSequence == 3 &&
                      StaticHeads == 6 && StaticSequences == 24);
        if (*path_flag == 0)
            return;
        int const base = int(blockIdx.x) * SegmentsPerSequence;
        bool const repair_prior_matched =
            summary_nonzero[StaticSequences * StaticHeads + (base + 1) * StaticHeads +
                            int(blockIdx.y)] != 0;
        bool const final_prefix_matched =
            summary_nonzero[StaticSequences * StaticHeads + (base + 2) * StaticHeads +
                            int(blockIdx.y)] != 0;
        if (repair_prior_matched && final_prefix_matched)
            return;
        // Every dependency is inside this CTA. No cross-CTA readiness wait.
        for (int repair_segment_index = repair_prior_matched ? 2 : 1;
             repair_segment_index < SegmentsPerSequence; ++repair_segment_index) {
#include "fwd_recurrence_body_ppu.cuh"
        }
    } else {
        constexpr int repair_segment_index = 0;
        constexpr bool repair_prior_matched = true;
#include "fwd_recurrence_body_ppu.cuh"
    }
}

} // namespace flash_kda_ppu
