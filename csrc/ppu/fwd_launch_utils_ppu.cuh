#pragma once

// PPU forward dispatch and workspace layout. See torch_api.cpp for validation.

#include "launch.h"
#include <cstdint>
#include <atomic>
#include <stdexcept>

#include "fwd_kernel1_ppu.cuh"
#include "fwd_prepare32_ppu.cuh"
#include "prefill_chunk_ppu.h"
#include "fwd_kernel2_ppu.cuh"
#include "fwd_hybrid_ppu.cuh"
#include "fwd_scan_ppu.cuh"
#include "ppu_common.cuh"
#include "ppu_frag.cuh"
#include "ppu_layouts.cuh"

namespace {

// Three input stages overlap AIU loads with recurrence. Small grids group
// adjacent chunks to amortize transfers, trading shared memory for latency.
#ifndef FLASH_KDA_PPU_INPUT_STAGES
#define FLASH_KDA_PPU_INPUT_STAGES 3
#endif
#ifndef FLASH_KDA_PPU_OUTPUT_STAGES
#define FLASH_KDA_PPU_OUTPUT_STAGES 2
#endif
#ifndef FLASH_KDA_PPU_MMA_WARPS
#define FLASH_KDA_PPU_MMA_WARPS 4
#endif
// The CTA count below which the d_v axis gets split across two CTAs. Tunable
// because it is a hardware crossover, not an algorithmic constant; see the
// measurements at the dispatch site in launch_fwd.
#ifndef FLASH_KDA_PPU_DV_SPLIT_CTA_TARGET
#define FLASH_KDA_PPU_DV_SPLIT_CTA_TARGET 64
#endif

constexpr int kInputStages = FLASH_KDA_PPU_INPUT_STAGES;
constexpr int kOutputStages = FLASH_KDA_PPU_OUTPUT_STAGES;

// On the small-grid TP16 path, stage adjacent 16-token transforms in one wider
// AIU transaction and consume the group before releasing the stage.  Fixed-length
// prefill benefits from four chunks per stage; varlen is faster at two because its
// shorter sequences do not amortize the larger shared-memory footprint.
#ifndef FLASH_KDA_PPU_GROUPED_STAGES
#define FLASH_KDA_PPU_GROUPED_STAGES 1
#endif
constexpr int kSmallGridFixedChunksPerStage = FLASH_KDA_PPU_GROUPED_STAGES ? 4 : 1;
constexpr int kSmallGridVarlenChunksPerStage = FLASH_KDA_PPU_GROUPED_STAGES ? 2 : 1;

constexpr int kNumMmaWarps = FLASH_KDA_PPU_MMA_WARPS; // each owns D/kNumMmaWarps d_v rows
constexpr int kTp8MmaWarps = 2;
// One load warp owns the AIU descriptors. Segmented replay selects direct
// MMA-warp output below; other output paths retain the coalescing store warp.
constexpr int kNumLoadWarps = 1;
constexpr int kNumStoreWarps = FLASH_KDA_PPU_DIRECT_OUT ? 0 : 1;

#ifndef FLASH_KDA_PPU_SEGMENT_SCAN
#define FLASH_KDA_PPU_SEGMENT_SCAN 1
#endif
// Reserve enough matrix workspace for the largest specialization.  Long
// sequences keep P fixed at 64 and grow the number of 16-token chunks within
// each segment, so scan storage stays constant instead of growing with T.
constexpr int kSegmentScanWorkspaceSegments = 64;

// Forward substitution keeps the 16x16 triangular solve in FP32 registers.
// The exact FP32 Neumann variant remains available for controlled comparisons.
// Both avoid the invalid BF16 transpose shortcut described below.
//
// FP32_EXACT uses the true fp32 k=8 atom, whose C -> A relabel is not free, so
// each of the six Neumann products costs a 16x16 shared round-trip. It is the
// correctness path: each B operand is loaded through a transposed shared-memory
// view because tc02's _TN atom computes A @ B^T.
//
// BF16_A_FP32_ACC reinterprets an A/C fragment directly as B. That is cheap but
// computes A @ B^T, not A @ B. For iid keys L is small enough to hide the error;
// for correlated keys its supposed L^2 becomes L @ L^T, is no longer nilpotent,
// and repeated squaring overflows. Do not re-enable it until it has a real
// transpose implementation.
#ifndef FLASH_KDA_PPU_INV_PREC
#define FLASH_KDA_PPU_INV_PREC FP32_FORWARD_SUBSTITUTION
#endif
constexpr flash_kda_ppu::InvPrec kInvPrec = flash_kda_ppu::InvPrec::FLASH_KDA_PPU_INV_PREC;
static_assert(kInvPrec != flash_kda_ppu::InvPrec::BF16_A_FP32_ACC,
              "BF16_A_FP32_ACC computes A @ B^T in the inverse; use an FP32 inverse");

} // namespace

// One K2 launch at a fixed d_v split. Separate from launch_fwd so the split can be
// selected at runtime without duplicating the workspace arithmetic.
template <int D, int CHUNK, int DvSplit, int ChunksPerStage, int NumMmaWarps, bool HasStateIn,
          bool HasStateOut, bool StateFP32, bool IsVarlen, bool ProduceOutput = true,
          bool IdentityState = false, int PathFlagMode = 0, bool ZeroValue = false,
          bool DualSummary = false, bool Segmented = false, bool OutputStateFP32 = StateFP32,
          int SegmentSequences = 1, int SegmentsPerSequence = 0, int InputStagesOverride = 0,
          int PrefixMode = 0, int PrefixChunks = 4, int PrefixHeads = 6, int PrefixTokens = 16384,
          int PrefixSequences = 26, bool DirectZeroCarry = false, bool ExternalHasStateIn = false,
          bool ExternalStateFP32 = false, bool PrepublishFinal = false,
          bool PublicHasStateOut = false, int SerialSegmentIndex = -1, bool SegmentCarryOut = false,
          bool FirstSegmentTrueB = false>
static void launch_recurrence(
    cutlass::bfloat16_t const *v_ptr, float const *ws_beta, cutlass::bfloat16_t const *ws_kd,
    cutlass::bfloat16_t const *ws_qd, cutlass::bfloat16_t const *ws_kr, float const *ws_gt,
    cutlass::bfloat16_t const *ws_inv, cutlass::bfloat16_t const *ws_mqk,
    void const *initial_state_ptr, void *final_state_ptr, cutlass::bfloat16_t *out_ptr, int T_total,
    int H, int N, int64_t const *cu_seqlens_ptr, int total_tiles, int *ws_tile_prefix,
    cudaStream_t stream, int const *path_flag = nullptr, void *auxiliary_final_state_ptr = nullptr,
    int v_stride = 0, int *summary_nonzero = nullptr,
    cutlass::bfloat16_t const *zero_summary_b = nullptr,
    void const *external_initial_state = nullptr, void *external_final_state = nullptr) {
    static_assert(D % (DvSplit * NumMmaWarps * 16) == 0,
                  "D / (DvSplit * MMA warps) must be whole 16-row state blocks");
    static_assert(InputStagesOverride >= 0);
    constexpr int kLaunchInputStages =
        InputStagesOverride > 0
            ? InputStagesOverride
            : (CHUNK == 32 ? (Segmented ? 1 : 2) : (ChunksPerStage == 1 ? kInputStages : 2));
    // Only balanced eight-warp segment replay benefits from direct stores.
    // Other shapes retain their coalescing store warp unless the ablation is set.
    constexpr bool kDirectOutput = FLASH_KDA_PPU_DIRECT_OUT || (Segmented && ProduceOutput);
    constexpr int kLaunchStoreWarps = Segmented ? 0 : kNumStoreWarps;
    constexpr int kLaunchThreads =
        32 * (NumMmaWarps + kNumLoadWarps + (ProduceOutput ? kLaunchStoreWarps : 0));
    flash_kda_ppu::flash_kda_ppu_fwd_recurrence<
        CHUNK, D, kLaunchInputStages, kOutputStages, NumMmaWarps, HasStateIn, HasStateOut,
        StateFP32, IsVarlen, DvSplit, ChunksPerStage, ProduceOutput, IdentityState, PathFlagMode,
        ZeroValue, DualSummary, true, Segmented, kDirectOutput, OutputStateFP32, SegmentSequences,
        SegmentsPerSequence, PrefixMode, PrefixChunks, PrefixMode == 0 ? 0 : PrefixHeads,
        PrefixMode == 0 ? 0 : PrefixTokens, PrefixMode == 0 ? 0 : PrefixSequences, DirectZeroCarry,
        ExternalHasStateIn, ExternalStateFP32, PrepublishFinal, PublicHasStateOut,
        SerialSegmentIndex, SegmentCarryOut, FirstSegmentTrueB>
        <<<dim3(PrefixMode == 5 ? SegmentSequences * (SegmentsPerSequence - 1)
                                : (SerialSegmentIndex < 0 ? N : SegmentSequences),
                H, DvSplit),
           dim3(kLaunchThreads), 0, stream>>>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
            final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles, ws_tile_prefix,
            path_flag, auxiliary_final_state_ptr, v_stride, summary_nonzero, zero_summary_b,
            external_initial_state, external_final_state);
}

template <int D, int P, int CHUNK, bool HasStateIn, bool HasStateOut, bool StateFP32 = false,
          bool Guarded = false, int SequenceBatches = 1, bool PrefixReuse = false,
          int ReuseHeads = 6, int ReuseTokens = 16384>
static void launch_segment_scan_path(
    cutlass::bfloat16_t const *v_ptr, float const *ws_beta, cutlass::bfloat16_t const *ws_kd,
    cutlass::bfloat16_t const *ws_qd, cutlass::bfloat16_t const *ws_kr, float const *ws_gt,
    cutlass::bfloat16_t const *ws_inv, cutlass::bfloat16_t const *ws_mqk,
    void const *initial_state_ptr, void *final_state_ptr, cutlass::bfloat16_t *out_ptr,
    cutlass::bfloat16_t *summary_b, cutlass::bfloat16_t *summary_apb, cutlass::bfloat16_t *carries,
    cutlass::bfloat16_t *replay_finals, int T_total, int H, int total_tiles, int *ws_tile_prefix,
    cudaStream_t stream, int v_stride, int const *path_flag, bool allow_early_final = false) {
    constexpr int kChunk = CHUNK;
    constexpr int kScanPathFlagMode = Guarded ? 1 : 0;
    using BF16 = cutlass::bfloat16_t;
    constexpr int kScanMmaWarps = 8;
    static_assert(P > 0 && P <= kSegmentScanWorkspaceSegments,
                  "segment count must fit the reserved scan workspace");

    // Replay finals are not live until after the carry scan. Use their first
    // P*H words for exact-zero summary metadata, without growing the workspace.
    static_assert(!PrefixReuse ||
                  (D == 128 && CHUNK == 16 &&
                   ((ReuseHeads == 6 && ReuseTokens == 16384 && P == 26 && SequenceBatches == 1) ||
                    (ReuseHeads == 12 && ReuseTokens == 8192 && P == 13 && SequenceBatches == 1) ||
                    (ReuseHeads == 6 && ReuseTokens == 16384 && P == 6 && SequenceBatches == 4))));
    // Selected full-row single-sequence shapes have enough dead affine beta
    // space for two P*H integer flag tables.
    static_assert(!PrefixReuse ||
                  2 * P * SequenceBatches * int(sizeof(int)) <= (ReuseTokens / CHUNK) * 128);
    constexpr bool kDirectZeroCarry = PrefixReuse;
    int *summary_nonzero = PrefixReuse ? reinterpret_cast<int *>(const_cast<float *>(ws_beta))
                                       : reinterpret_cast<int *>(replay_finals);
    // Separate the bias chain from the homogeneous chain. The latter has
    // whole-stage arithmetic skip for exact-zero rows; all input stages are
    // still consumed, without branching inside individual MMAs.
    auto launch_homogeneous = [&]() {
        launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, false, true, false, false, false, true,
                          kScanPathFlagMode, true, false, true, false, SequenceBatches,
                          SequenceBatches == 1 ? 0 : P>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, summary_apb,
            nullptr, T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix, stream,
            path_flag, nullptr, v_stride, summary_nonzero);
    };
    if constexpr (PrefixReuse) {
        launch_homogeneous();
        launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, false, true, false, false, true, false,
                          kScanPathFlagMode, false, false, true, false, SequenceBatches,
                          SequenceBatches == 1 ? 0 : P, 2, 1, 4, ReuseHeads, ReuseTokens,
                          P * SequenceBatches>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, summary_b, out_ptr,
            T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix, stream,
            path_flag, summary_apb, v_stride, summary_nonzero);
    } else {
        launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, false, true, false, false, false, false,
                          kScanPathFlagMode, false, false, true, false, SequenceBatches,
                          SequenceBatches == 1 ? 0 : P>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, summary_b, nullptr,
            T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix, stream,
            path_flag, nullptr, v_stride);
        launch_homogeneous();
    }

    if constexpr (!kDirectZeroCarry) {
        flash_kda_ppu::flash_kda_ppu_copy_zero_summary_carries<D, P, HasStateIn, StateFP32,
                                                               SequenceBatches>
            <<<dim3(P, H, SequenceBatches), 128, 0, stream>>>(summary_b, initial_state_ptr, carries,
                                                              H, summary_nonzero, path_flag);
    }

    if constexpr (P <= 64) {
        // The carry update is independent across d_v rows. Four row slices
        // expose 24/48 CTAs at H6/H12 while retaining the single launch and
        // avoiding group-map materialization. DualSummary writes homogeneous A
        // directly, so the carry scan does not subtract B from its operand.
        if (H != 12) {
            constexpr int kCarryDvSplit = 4;
            constexpr int kCarryMmaWarps = (D / 16) / kCarryDvSplit;
            flash_kda_ppu::flash_kda_ppu_segment_scan<D, P, HasStateIn, 2, kCarryMmaWarps, true,
                                                      kCarryDvSplit, StateFP32, SequenceBatches>
                <<<dim3(H, kCarryDvSplit, SequenceBatches), dim3(32 * (kCarryMmaWarps + 1)), 0,
                   stream>>>(summary_apb, summary_b, initial_state_ptr, carries, H, summary_nonzero,
                             path_flag);
        } else {
            constexpr int kCarryDvSplit = 1;
            constexpr int kCarryMmaWarps = D / 16;
            flash_kda_ppu::flash_kda_ppu_segment_scan<D, P, HasStateIn, 2, kCarryMmaWarps, true,
                                                      kCarryDvSplit, StateFP32, SequenceBatches>
                <<<dim3(H, 1, SequenceBatches), dim3(32 * (kCarryMmaWarps + 1)), 0, stream>>>(
                    summary_apb, summary_b, initial_state_ptr, carries, H, summary_nonzero,
                    path_flag);
        }
    }

    if constexpr (PrefixReuse) {
        auto launch_prefix_and_tail = [&](auto publish_tag) {
            constexpr bool kPublish = decltype(publish_tag)::value;
            constexpr int kPrefixChunks = 4;
            int *equal_flags = summary_nonzero + P * SequenceBatches * H;
            launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, true, true, false, false, true, false,
                              kScanPathFlagMode, false, false, true, false, SequenceBatches,
                              SequenceBatches == 1 ? 0 : P, 0, 2, kPrefixChunks, ReuseHeads,
                              ReuseTokens, P * SequenceBatches, kDirectZeroCarry, HasStateIn,
                              StateFP32, kPublish, HasStateOut>(
                v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, carries, replay_finals,
                out_ptr, T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix,
                stream, path_flag, summary_apb, v_stride, summary_nonzero,
                kDirectZeroCarry ? summary_b : nullptr,
                kDirectZeroCarry ? initial_state_ptr : nullptr,
                kPublish ? final_state_ptr : nullptr);
            launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, true, HasStateOut, false, false, true,
                              false, kScanPathFlagMode, false, false, true, StateFP32,
                              SequenceBatches, SequenceBatches == 1 ? 0 : P, 0, 3, kPrefixChunks,
                              ReuseHeads, ReuseTokens, P * SequenceBatches, false, false, false,
                              kPublish, HasStateOut>(
                v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, replay_finals,
                final_state_ptr, out_ptr, T_total, H, P * SequenceBatches, nullptr, total_tiles,
                ws_tile_prefix, stream, path_flag, summary_b, v_stride, equal_flags);
        };
        if constexpr (HasStateOut) {
            if (allow_early_final)
                launch_prefix_and_tail(std::true_type{});
            else
                launch_prefix_and_tail(std::false_type{});
        } else {
            launch_prefix_and_tail(std::false_type{});
        }
    } else {
        // Replay all segments concurrently from the scanned carry states.
        launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, true, HasStateOut, false, false, true,
                          false, kScanPathFlagMode, false, false, true, StateFP32, SequenceBatches,
                          SequenceBatches == 1 ? 0 : P>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, carries, final_state_ptr,
            out_ptr, T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix, stream,
            path_flag, nullptr, v_stride);
    }
}

#include "fwd_general_scan_ppu.cuh"

// Three finite temporal segments. Eligibility flags mean only that an exact
// checkpoint comparison is allowed; no homogeneous-A property is asserted.
// Parallel hypotheses are checked, then two ordered repairs supply true carries.
template <int D, bool StateFP32>
static void launch_serial_verify_path(
    cutlass::bfloat16_t const *v_ptr, float const *ws_beta, cutlass::bfloat16_t const *ws_kd,
    cutlass::bfloat16_t const *ws_qd, cutlass::bfloat16_t const *ws_kr, float const *ws_gt,
    cutlass::bfloat16_t const *ws_inv, cutlass::bfloat16_t const *ws_mqk, void const *initial_state,
    void *final_state, cutlass::bfloat16_t *out, cutlass::bfloat16_t *b_final,
    cutlass::bfloat16_t *checkpoint, cutlass::bfloat16_t *prefix_state,
    cutlass::bfloat16_t *true_carry, int T, int H, int total_tiles, int *tile_prefix,
    int const *path_flag, int v_stride, cudaStream_t stream) {
    constexpr int kSegments = 3, kSequences = 8, kTotal = 24, kChunk = 16;
    int *eligible = reinterpret_cast<int *>(const_cast<float *>(ws_beta));
    // Active private B CTAs initialize their own comparison flags.
    flash_kda_ppu::flash_kda_ppu_fwd_recurrence_hybrid_b<D, StateFP32>
        <<<dim3(kTotal, H, 1), dim3(288), 0, stream>>>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, b_final, out, T, H,
            kTotal, nullptr, total_tiles, tile_prefix, path_flag, checkpoint, v_stride, eligible,
            nullptr, initial_state, final_state);
    // Check the first48 tokens of both nonfirst segments concurrently.
    // Only segment1 is known to have a true carry at this point.
    launch_recurrence<D, kChunk, 1, 1, 8, true, true, false, false, true, false, 1, false, false,
                      true, false, kSequences, kSegments, 3, 5, 3, 6, 16384, kTotal, true, true,
                      StateFP32, false, false, -1, false>(
        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, prefix_state, out, T,
        H, kTotal, nullptr, total_tiles, tile_prefix, stream, path_flag, checkpoint, v_stride,
        eligible, b_final, initial_state, nullptr);
    // One finite CTA repairs segment1 then segment2 when necessary. Both
    // full-state matches authorize the already-published B2 and return early.
    flash_kda_ppu::flash_kda_ppu_fwd_recurrence_hybrid_repair<D, StateFP32>
        <<<dim3(kSequences, H, 1), dim3(288), 0, stream>>>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, prefix_state, true_carry,
            out, T, H, kTotal, nullptr, total_tiles, tile_prefix, path_flag, checkpoint, v_stride,
            eligible, b_final, initial_state, final_state);
}
