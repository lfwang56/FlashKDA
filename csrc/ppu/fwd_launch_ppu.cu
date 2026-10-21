// PPU forward dispatch and workspace layout. See torch_api.cpp for validation.

#include "launch.h"
#include <cstdint>
#include <stdexcept>

#include "fwd_kernel1_ppu.cuh"
#include "fwd_prepare32_ppu.cuh"
#include "prefill_chunk_ppu.h"
#include "fwd_kernel2_ppu.cuh"
#include "fwd_hybrid_b_ppu.cuh"
#include "fwd_hybrid_repair_ppu.cuh"
#include "fwd_h12_n4_hybrid_b_ppu.cuh"
#include "fwd_h12_n4_hybrid_repair_ppu.cuh"
#include "fwd_h12_n4_compact_prefix_ppu.cuh"
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
          int PrefixMode = 0, int PrefixChunks = 4, int PrefixHeads = 6,
          int PrefixTokens = 16384, int PrefixSequences = 26,
          bool DirectZeroCarry = false, bool ExternalHasStateIn = false,
          bool ExternalStateFP32 = false, bool PrepublishFinal = false,
          bool PublicHasStateOut = false, int SerialSegmentIndex = -1,
          bool SegmentCarryOut = false, bool FirstSegmentTrueB = false>
static void launch_recurrence(cutlass::bfloat16_t const *v_ptr, float const *ws_beta,
                              cutlass::bfloat16_t const *ws_kd, cutlass::bfloat16_t const *ws_qd,
                              cutlass::bfloat16_t const *ws_kr, float const *ws_gt,
                              cutlass::bfloat16_t const *ws_inv, cutlass::bfloat16_t const *ws_mqk,
                              void const *initial_state_ptr, void *final_state_ptr,
                              cutlass::bfloat16_t *out_ptr, int T_total, int H, int N,
                              int64_t const *cu_seqlens_ptr, int total_tiles, int *ws_tile_prefix,
                              cudaStream_t stream, int const *path_flag = nullptr,
                              void *auxiliary_final_state_ptr = nullptr, int v_stride = 0,
                              int *summary_nonzero = nullptr,
                              cutlass::bfloat16_t const *zero_summary_b = nullptr,
                              void const *external_initial_state = nullptr,
                              void *external_final_state = nullptr) {
    static_assert(D % (DvSplit * NumMmaWarps * 16) == 0,
                  "D / (DvSplit * MMA warps) must be whole 16-row state blocks");
    static_assert(InputStagesOverride >= 0);
    constexpr int kLaunchInputStages = InputStagesOverride > 0 ? InputStagesOverride
        : (CHUNK == 32 ? (Segmented ? 1 : 2)
                       : (ChunksPerStage == 1 ? kInputStages : 2));
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
        SegmentsPerSequence, PrefixMode, PrefixChunks,
        PrefixMode == 0 ? 0 : PrefixHeads, PrefixMode == 0 ? 0 : PrefixTokens,
        PrefixMode == 0 ? 0 : PrefixSequences, DirectZeroCarry,
        ExternalHasStateIn, ExternalStateFP32, PrepublishFinal, PublicHasStateOut,
        SerialSegmentIndex, SegmentCarryOut, FirstSegmentTrueB>
        <<<dim3(PrefixMode == 5 ? SegmentSequences * (SegmentsPerSequence - 1) :
                    (SerialSegmentIndex < 0 ? N : SegmentSequences), H, DvSplit),
           dim3(kLaunchThreads), 0, stream>>>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
            final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles, ws_tile_prefix,
            path_flag, auxiliary_final_state_ptr, v_stride, summary_nonzero,
            zero_summary_b, external_initial_state, external_final_state);
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
    cudaStream_t stream, int v_stride, int const *path_flag,
    bool allow_early_final = false) {
    constexpr int kChunk = CHUNK;
    constexpr int kScanPathFlagMode = Guarded ? 1 : 0;
    using BF16 = cutlass::bfloat16_t;
    constexpr int kScanMmaWarps = 8;
    static_assert(P > 0 && P <= kSegmentScanWorkspaceSegments,
                  "segment count must fit the reserved scan workspace");

    // Replay finals are not live until after the carry scan. Use their first
    // P*H words for exact-zero summary metadata, without growing the workspace.
    static_assert(!PrefixReuse || (D == 128 && CHUNK == 16 &&
        ((ReuseHeads == 6 && ReuseTokens == 16384 && P == 26 && SequenceBatches == 1) ||
         (ReuseHeads == 12 && ReuseTokens == 8192 && P == 13 && SequenceBatches == 1) ||
         (ReuseHeads == 6 && ReuseTokens == 16384 && P == 6 && SequenceBatches == 4))));
    // Selected full-row single-sequence shapes have enough dead affine beta
    // space for two P*H integer flag tables.
    static_assert(!PrefixReuse || 2 * P * SequenceBatches * int(sizeof(int)) <= (ReuseTokens / CHUNK) * 128);
    constexpr bool kDirectZeroCarry = PrefixReuse;
    int *summary_nonzero = PrefixReuse
        ? reinterpret_cast<int *>(const_cast<float *>(ws_beta))
        : reinterpret_cast<int *>(replay_finals);
    // Separate the bias chain from the homogeneous chain. The latter has
    // whole-stage arithmetic skip for exact-zero rows; all input stages are
    // still consumed, without branching inside individual MMAs.
    auto launch_homogeneous = [&]() {
    launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, false, true, false, false, false, true,
                      kScanPathFlagMode, true, false, true, false, SequenceBatches,
                      SequenceBatches == 1 ? 0 : P>(
        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, summary_apb, nullptr,
        T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix, stream, path_flag,
        nullptr, v_stride, summary_nonzero);
    };
    if constexpr (PrefixReuse) {
        launch_homogeneous();
        launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, false, true, false, false, true, false,
                          kScanPathFlagMode, false, false, true, false, SequenceBatches,
                          SequenceBatches == 1 ? 0 : P, 2, 1, 4, ReuseHeads, ReuseTokens, P * SequenceBatches>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, summary_b,
            out_ptr, T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix,
            stream, path_flag, summary_apb, v_stride, summary_nonzero);
    } else {
    launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, false, true, false, false, false, false,
                      kScanPathFlagMode, false, false, true, false, SequenceBatches,
                      SequenceBatches == 1 ? 0 : P>(
        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, nullptr, summary_b, nullptr,
        T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix, stream, path_flag,
        nullptr, v_stride);
        launch_homogeneous();
    }

    if constexpr (!kDirectZeroCarry) {
    flash_kda_ppu::flash_kda_ppu_copy_zero_summary_carries<D, P, HasStateIn, StateFP32,
                                                         SequenceBatches>
        <<<dim3(P, H, SequenceBatches), 128, 0, stream>>>(
            summary_b, initial_state_ptr, carries, H, summary_nonzero, path_flag);
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
                <<<dim3(H, kCarryDvSplit, SequenceBatches),
                   dim3(32 * (kCarryMmaWarps + 1)), 0, stream>>>(
                    summary_apb, summary_b, initial_state_ptr, carries, H, summary_nonzero,
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
                          SequenceBatches == 1 ? 0 : P, 0, 2, kPrefixChunks,
                          ReuseHeads, ReuseTokens, P * SequenceBatches,
                          kDirectZeroCarry, HasStateIn, StateFP32, kPublish, HasStateOut>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, carries, replay_finals,
            out_ptr, T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix,
            stream, path_flag, summary_apb, v_stride, summary_nonzero,
            kDirectZeroCarry ? summary_b : nullptr,
            kDirectZeroCarry ? initial_state_ptr : nullptr,
            kPublish ? final_state_ptr : nullptr);
        launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, true, HasStateOut, false, false, true, false,
                          kScanPathFlagMode, false, false, true, StateFP32, SequenceBatches,
                          SequenceBatches == 1 ? 0 : P, 0, 3, kPrefixChunks,
                          ReuseHeads, ReuseTokens, P * SequenceBatches,
                          false, false, false, kPublish, HasStateOut>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, replay_finals, final_state_ptr,
            out_ptr, T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix,
            stream, path_flag, summary_b, v_stride, equal_flags);
        };
        if constexpr (HasStateOut) {
            if (allow_early_final) launch_prefix_and_tail(std::true_type{});
            else launch_prefix_and_tail(std::false_type{});
        } else {
            launch_prefix_and_tail(std::false_type{});
        }
    } else {
    // Replay all segments concurrently from the scanned carry states.
    launch_recurrence<D, kChunk, 1, 1, kScanMmaWarps, true, HasStateOut, false, false, true, false,
                      kScanPathFlagMode, false, false, true, StateFP32, SequenceBatches,
                      SequenceBatches == 1 ? 0 : P>(
        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, carries, final_state_ptr,
        out_ptr, T_total, H, P * SequenceBatches, nullptr, total_tiles, ws_tile_prefix, stream,
        path_flag, nullptr, v_stride);
    }
}

// Three finite temporal segments. Eligibility flags mean only that an exact
// checkpoint comparison is allowed; no homogeneous-A property is asserted.
// Parallel hypotheses are checked, then two ordered repairs supply true carries.
template<int D, bool StateFP32>
static void launch_serial_verify_path(
    cutlass::bfloat16_t const *v_ptr,float const *ws_beta,
    cutlass::bfloat16_t const *ws_kd,cutlass::bfloat16_t const *ws_qd,
    cutlass::bfloat16_t const *ws_kr,float const *ws_gt,
    cutlass::bfloat16_t const *ws_inv,cutlass::bfloat16_t const *ws_mqk,
    void const *initial_state,void *final_state,cutlass::bfloat16_t *out,
    cutlass::bfloat16_t *b_final,cutlass::bfloat16_t *checkpoint,
    cutlass::bfloat16_t *prefix_state,cutlass::bfloat16_t *true_carry,
    int T,int H,int total_tiles,int *tile_prefix,int const *path_flag,
    int v_stride,cudaStream_t stream) {
    constexpr int kSegments = 3, kSequences = 8, kTotal = 24, kChunk = 16;
    int *eligible = reinterpret_cast<int *>(const_cast<float *>(ws_beta));
    // Active private B CTAs initialize their own comparison flags.
    flash_kda_ppu::flash_kda_ppu_fwd_recurrence_hybrid_b<D,StateFP32>
        <<<dim3(kTotal,H,1),dim3(288),0,stream>>>(
        v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,ws_gt,ws_inv,ws_mqk,nullptr,b_final,out,
        T,H,kTotal,nullptr,total_tiles,tile_prefix,path_flag,checkpoint,v_stride,
        eligible,nullptr,initial_state,final_state);
    // Check the first48 tokens of both nonfirst segments concurrently.
    // Only segment1 is known to have a true carry at this point.
    launch_recurrence<D,kChunk,1,1,8,true,true,false,false,true,false,1,
        false,false,true,false,kSequences,kSegments,3,5,3,6,16384,kTotal,
        true,true,StateFP32,false,false,-1,false>(
        v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,ws_gt,ws_inv,ws_mqk,nullptr,prefix_state,out,
        T,H,kTotal,nullptr,total_tiles,tile_prefix,stream,path_flag,checkpoint,v_stride,
        eligible,b_final,initial_state,nullptr);
    // One finite CTA repairs segment1 then segment2 when necessary. Both
    // full-state matches authorize the already-published B2 and return early.
    flash_kda_ppu::flash_kda_ppu_fwd_recurrence_hybrid_repair<D,StateFP32>
        <<<dim3(kSequences,H,1),dim3(288),0,stream>>>(
        v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,ws_gt,ws_inv,ws_mqk,prefix_state,true_carry,out,
        T,H,kTotal,nullptr,total_tiles,tile_prefix,path_flag,checkpoint,v_stride,
        eligible,b_final,initial_state,final_state);
}

template <int D, int CHUNK, bool HasStateIn, bool HasStateOut, bool StateFP32, bool IsVarlen>
static void launch_fwd_impl(cutlass::bfloat16_t const *q_ptr, cutlass::bfloat16_t const *k_ptr,
                            cutlass::bfloat16_t const *v_ptr, cutlass::bfloat16_t const *g_bf16_ptr,
                            cutlass::bfloat16_t const *beta_ptr, void const *initial_state_ptr,
                            float scale, void *final_state_ptr, cutlass::bfloat16_t *out_ptr,
                            void *workspace_ptr, int total_tiles, int T_total, int H, int N,
                            int64_t const *cu_seqlens_ptr, float const *A_log_ptr,
                            float const *dt_bias_ptr, float gate_scale, cudaStream_t stream,
                            int q_stride, int k_stride, int v_stride) {
    constexpr int kChunk = CHUNK;
    using namespace flash_kda_ppu;
    using BF16 = cutlass::bfloat16_t;
    using K1L = K1Layouts<D, kChunk>;
    using K2L = K2Layouts<D, kChunk>;
    using WS = WorkspaceSizes<kChunk, D>;
    constexpr int kSmallGridChunksPerStage = kChunk == 32
        ? (IsVarlen ? 1 : (FLASH_KDA_PPU_GROUPED_STAGES ? 2 : 1))
        : (IsVarlen ? kSmallGridVarlenChunksPerStage : kSmallGridFixedChunksPerStage);

    static_assert(K2L::Tile::kBytes == WS::kKDecayed);
    static_assert(K1L::QK::kRowStrideBytes % 128 != 0);

    // Keep the PPU workspace layout paired with the private host allocator
    // in csrc/ppu/torch_api.cpp.
    int64_t const n_ht = int64_t(H) * total_tiles;
    char *ws = reinterpret_cast<char *>(workspace_ptr);
    auto *ws_kd = reinterpret_cast<BF16 *>(ws);
    auto *ws_qd = reinterpret_cast<BF16 *>(ws + WS::kKDecayed);
    auto *ws_kr = reinterpret_cast<BF16 *>(ws + WS::kKDecayed + WS::kQDecayed);
    auto *ws_gt =
        reinterpret_cast<float *>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored));
    auto *ws_inv = reinterpret_cast<BF16 *>(
        ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal));
    auto *ws_mqk = reinterpret_cast<BF16 *>(
        ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal + WS::kINV));
    auto *ws_beta = reinterpret_cast<float *>(ws + n_ht * (WS::kPerTile - WS::kBeta));
    auto *ws_tile_prefix = reinterpret_cast<int *>(ws + n_ht * WS::kPerTile);

    // Publishing a last-segment final during prefix must not overwrite an
    // initial state that another prefix CTA has not staged yet. Preserve the
    // original post-prefix publication for every intersecting memory span.
    bool allow_early_final = false;
    if constexpr (HasStateOut && kChunk == 16) {
        bool const qualified = (N == 1 && H == 6 && T_total == 16384) ||
                               (N == 1 && H == 12 && T_total == 8192) ||
                               (N == 4 && H == 6 && T_total == 16384);
        if (qualified) {
            uint64_t const state_bytes = uint64_t(N) * H * D * D * (StateFP32 ? 4 : 2);
            uintptr_t const begin = reinterpret_cast<uintptr_t>(final_state_ptr);
            uintptr_t const end = begin + state_bytes;
            auto intersects_final = [&](void const *ptr, uint64_t bytes) {
                uintptr_t const a = reinterpret_cast<uintptr_t>(ptr);
                return ptr && begin < a + bytes && a < end;
            };
            int64_t const allocated_tiles = (T_total + kChunk - 1) / kChunk + N;
            uint64_t const prefix_bytes = ((uint64_t(N + 1) * 4 + 127) / 128) * 128;
            int const reserved_segments = N == 4 ? 24 : kSegmentScanWorkspaceSegments;
            uint64_t const active_ws_bytes = uint64_t(H) * allocated_tiles * WS::kPerTile +
                prefix_bytes + uint64_t(4) * reserved_segments * H * D * D * sizeof(BF16);
            uint64_t const v_bytes = (uint64_t(T_total - 1) *
                (v_stride ? v_stride : H * D) + H * D) * sizeof(BF16);
            allow_early_final = final_state_ptr &&
                !intersects_final(initial_state_ptr, state_bytes) &&
                !intersects_final(v_ptr, v_bytes) &&
                !intersects_final(out_ptr, uint64_t(T_total) * H * D * sizeof(BF16)) &&
                !intersects_final(workspace_ptr, active_ws_bytes);
        }
    }

    bool allow_serial_verify = false;
    if constexpr (kChunk == 16 && IsVarlen && HasStateIn && HasStateOut &&
                  kInvPrec == InvPrec::FP32_FORWARD_SUBSTITUTION) {
        if ((H == 6 && N == 8 && T_total == 16384) ||
            (H == 12 && N == 4 && T_total == 8192)) {
                int64_t const allocated_tiles = (T_total+kChunk-1)/kChunk+N;
                int64_t const prefix_bytes = ((int64_t(N+1)*4+127)/128)*128;
                int64_t const elements = int64_t(H == 12 ? 12 : 24)*H*D*D;
                uint64_t const workspace_bytes = uint64_t(H)*allocated_tiles*WS::kPerTile +
                    prefix_bytes + uint64_t(H == 12 ? 2 : 4)*elements*sizeof(BF16)+128;
                uint64_t const state_bytes = uint64_t(N)*H*D*D*(StateFP32?4:2);
                uint64_t const output_bytes = uint64_t(T_total)*H*D*sizeof(BF16);
                uint64_t const v_bytes = (uint64_t(T_total-1)*(v_stride?v_stride:H*D)+H*D)*sizeof(BF16);
                auto overlap = [](void const *a,uint64_t na,void const *b,uint64_t nb) {
                    uintptr_t const x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
                    return a && b && x<y+nb && y<x+na;
                };
                bool const disjoint =
                    !overlap(workspace_ptr,workspace_bytes,initial_state_ptr,state_bytes) &&
                    !overlap(workspace_ptr,workspace_bytes,final_state_ptr,state_bytes) &&
                    !overlap(workspace_ptr,workspace_bytes,v_ptr,v_bytes) &&
                    !overlap(workspace_ptr,workspace_bytes,out_ptr,output_bytes) &&
                    !overlap(workspace_ptr,workspace_bytes,cu_seqlens_ptr,uint64_t(N+1)*8) &&
                    !overlap(final_state_ptr,state_bytes,initial_state_ptr,state_bytes) &&
                    !overlap(final_state_ptr,state_bytes,v_ptr,v_bytes) &&
                    !overlap(final_state_ptr,state_bytes,out_ptr,output_bytes) &&
                    !overlap(final_state_ptr,state_bytes,cu_seqlens_ptr,uint64_t(N+1)*8) &&
                    !overlap(out_ptr,output_bytes,initial_state_ptr,state_bytes) &&
                    !overlap(out_ptr,output_bytes,v_ptr,v_bytes) &&
                    !overlap(out_ptr,output_bytes,cu_seqlens_ptr,uint64_t(N+1)*8);
            allow_serial_verify = disjoint;
        }
    }

    int *scan_flag = nullptr;
    if constexpr (IsVarlen) {
        if (N == 1 || ((N == 4 || N == 8) && H == 6 && T_total == 16384) ||
            (N == 4 && H == 12 && T_total == 8192))
            scan_flag = ws_tile_prefix + N + 1;
        if (!(kChunk == 16 && N == 1))
            flash_kda_ppu_build_tile_prefix<<<1, 32, 0, stream>>>(cu_seqlens_ptr, N, kChunk,
                                                               ws_tile_prefix, scan_flag, T_total, allow_serial_verify);
    }

    bool const scan_shape =
        gate_scale <= 0.0f && N == 1 &&
        ((H == 6 &&
          (T_total == 1024 || T_total == 2048 || T_total == 4096 || T_total == 8192 ||
           T_total == 16384 || T_total == 32768 || T_total == 65536 || T_total == 131072)) ||
         (H == 12 && (T_total == 4096 || T_total == 8192 || T_total == 16384 || T_total == 32768 ||
                      T_total == 65536 || T_total == 131072)) ||
         (StateFP32 && H == 16 &&
          (T_total == 16384 || T_total == 32768 || T_total == 65536 || T_total == 131072)));
    auto launch_guarded_prepare = [&]() {
        if constexpr (kChunk == 16 && IsVarlen && HasStateIn && HasStateOut &&
                      kInvPrec == InvPrec::FP32_FORWARD_SUBSTITUTION) {
            auto guarded_prepare = [&](auto heads_tag) {
                constexpr int kHeads = decltype(heads_tag)::value;
                flash_kda_ppu_fwd_prepare<kChunk,D,128,IsVarlen,kInvPrec,
                                         false,true,kHeads,false,true>
                    <<<dim3(total_tiles,H),dim3(128),0,stream>>>(
                        q_ptr,k_ptr,g_bf16_ptr,beta_ptr,A_log_ptr,dt_bias_ptr,ws_kd,ws_qd,
                        ws_kr,ws_gt,ws_inv,ws_mqk,ws_beta,scale,gate_scale,T_total,H,N,
                        cu_seqlens_ptr,total_tiles,ws_tile_prefix,nullptr,0.0f,
                        q_stride,k_stride,scan_flag);
            };
            if (H == 6 && q_stride == 3*H*D && k_stride == 3*H*D)
                guarded_prepare(std::integral_constant<int,6>{});
            else guarded_prepare(std::integral_constant<int,0>{});
        }
    };
    if (allow_serial_verify) launch_guarded_prepare();
    else
    {
        constexpr InvPrec kPrepareInvPrec = kInvPrec;
        auto prepare = [&](auto heads_tag, auto prefix_tag) {
            constexpr int kHeads = decltype(heads_tag)::value;
            constexpr bool kBuildSinglePrefix = decltype(prefix_tag)::value;
            if constexpr (kChunk == 32) {
                flash_kda_ppu_fwd_prepare32<D, 512, IsVarlen, kHeads>
                    <<<dim3(total_tiles, H), dim3(512), 0, stream>>>(
                        q_ptr, k_ptr, g_bf16_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, ws_kd, ws_qd,
                        ws_kr, ws_gt, ws_inv, ws_mqk, ws_beta, scale, gate_scale, T_total, H, N,
                        cu_seqlens_ptr, total_tiles, ws_tile_prefix, nullptr, 0.0f, q_stride, k_stride);
            } else {
                auto prepare_threads = [&](auto threads_tag) {
                    constexpr int kPrepareThreads = decltype(threads_tag)::value;
                    flash_kda_ppu_fwd_prepare<kChunk, D, kPrepareThreads, IsVarlen, kPrepareInvPrec,
                                              false, true, kHeads, kBuildSinglePrefix>
                        <<<dim3(total_tiles, H), dim3(kPrepareThreads), 0, stream>>>(
                            q_ptr, k_ptr, g_bf16_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, ws_kd, ws_qd,
                            ws_kr, ws_gt, ws_inv, ws_mqk, ws_beta, scale, gate_scale, T_total, H, N,
                            cu_seqlens_ptr, total_tiles, ws_tile_prefix, nullptr, 0.0f, q_stride, k_stride, scan_flag);
                };
                bool const large_shape = (H == 6 && T_total == 16384) ||
                                         (H == 12 && T_total == 8192);
                if (large_shape)
                    prepare_threads(std::integral_constant<int, 128>{});
                else
                    prepare_threads(std::integral_constant<int, 256>{});
            }
        };
        auto prepare_heads = [&](auto prefix_tag) {
            bool const packed_qk = q_stride == 3 * H * D && k_stride == 3 * H * D;
            if (H == 6 && packed_qk) {
                prepare(std::integral_constant<int, 6>{}, prefix_tag);
            } else if constexpr (kChunk == 16) {
                if (H == 12 && packed_qk)
                    prepare(std::integral_constant<int, 12>{}, prefix_tag);
                else prepare(std::integral_constant<int, 0>{}, prefix_tag);
            } else {
                prepare(std::integral_constant<int, 0>{}, prefix_tag);
            }
        };
        if constexpr (kChunk == 16 && IsVarlen) {
            if (N == 1) prepare_heads(std::true_type{});
            else prepare_heads(std::false_type{});
        } else {
            prepare_heads(std::false_type{});
        }
    }

    if constexpr (kChunk == 16 && IsVarlen && HasStateIn && HasStateOut &&
                  kInvPrec == InvPrec::FP32_FORWARD_SUBSTITUTION) {
        if (H == 12 && N == 4 && T_total == 8192 && allow_serial_verify) {
            int64_t const allocated_tiles = (T_total+kChunk-1)/kChunk+N;
            int64_t const prefix_bytes = ((int64_t(N+1)*4+127)/128)*128;
            int64_t const elements = int64_t(12)*H*D*D;
            char *scratch=ws+int64_t(H)*allocated_tiles*WS::kPerTile+prefix_bytes;
            auto *b_final=reinterpret_cast<BF16 *>(scratch);
            // B states remain live through all prefix CTAs, then become true
            // carries. Each checkpoint slot becomes prefix state only on a
            // mismatch, after that same CTA completed its full comparison.
            auto *prefix_slot=b_final+elements;
            int *eligible=reinterpret_cast<int *>(const_cast<float *>(ws_beta));
            flash_kda_ppu::flash_kda_ppu_h12_n4_hybrid_b<D,StateFP32>
                <<<dim3(12,12,1),dim3(288),0,stream>>>(
                v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,ws_gt,ws_inv,ws_mqk,nullptr,b_final,out_ptr,
                T_total,H,12,nullptr,total_tiles,ws_tile_prefix,scan_flag,prefix_slot,v_stride,
                eligible,nullptr,initial_state_ptr,final_state_ptr);
            flash_kda_ppu::flash_kda_ppu_h12_n4_compact_prefix<D,StateFP32>
                <<<dim3(8,12,1),dim3(288),0,stream>>>(
                v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,ws_gt,ws_inv,ws_mqk,prefix_slot,b_final,out_ptr,
                T_total,H,12,total_tiles,ws_tile_prefix,scan_flag,v_stride,eligible);
            flash_kda_ppu::flash_kda_ppu_h12_n4_hybrid_repair<D,StateFP32>
                <<<dim3(4,12,1),dim3(288),0,stream>>>(
                v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,ws_gt,ws_inv,ws_mqk,prefix_slot,b_final,out_ptr,
                T_total,H,12,nullptr,total_tiles,ws_tile_prefix,scan_flag,v_stride,
                eligible,initial_state_ptr,final_state_ptr);
            flash_kda_ppu::flash_kda_ppu_fwd_recurrence<
                16,D,2,kOutputStages,2,HasStateIn,HasStateOut,StateFP32,true,
                4,1,true,false,-2,false,false,true,false,false,
                StateFP32,1,0,0,4,12,8192,4>
                <<<dim3(4,12,4),dim3(128),0,stream>>>(
                v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,ws_gt,ws_inv,ws_mqk,
                initial_state_ptr,final_state_ptr,out_ptr,T_total,H,N,
                cu_seqlens_ptr,total_tiles,ws_tile_prefix,scan_flag,nullptr,
                v_stride,nullptr,nullptr,nullptr,nullptr);
            return;
        }
    }

    if constexpr (kChunk == 16 && IsVarlen) {
        if (H == 6 && N == 8 && T_total == 16384) {
            if constexpr (HasStateIn && HasStateOut &&
                          kInvPrec == InvPrec::FP32_FORWARD_SUBSTITUTION) {
                int64_t const allocated_tiles = (T_total+kChunk-1)/kChunk+N;
                int64_t const prefix_bytes = ((int64_t(N+1)*4+127)/128)*128;
                int64_t const elements = int64_t(24)*H*D*D;
                if (allow_serial_verify) {
                    char *scratch=ws+int64_t(H)*allocated_tiles*WS::kPerTile+prefix_bytes;
                    auto *b_final=reinterpret_cast<BF16 *>(scratch);
                    auto *checkpoint=b_final+elements;
                    auto *prefix_state=checkpoint+elements;
                    auto *true_carry=prefix_state+elements;
                    launch_serial_verify_path<D,StateFP32>(v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,
                        ws_gt,ws_inv,ws_mqk,initial_state_ptr,final_state_ptr,out_ptr,
                        b_final,checkpoint,prefix_state,true_carry,T_total,H,total_tiles,
                        ws_tile_prefix,scan_flag,v_stride,stream);
                    launch_recurrence<D,kChunk,4,1,kTp8MmaWarps,HasStateIn,HasStateOut,StateFP32,true,
                        true,false,-2,false,false,false,StateFP32,1,0,2>(
                        v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,ws_gt,ws_inv,ws_mqk,initial_state_ptr,
                        final_state_ptr,out_ptr,T_total,H,N,cu_seqlens_ptr,total_tiles,ws_tile_prefix,
                        stream,scan_flag,nullptr,v_stride);
                    return;
                }
            }
            // Match the TP8 multi-sequence geometry:192 row-split CTAs,
            // two math warps perCTA and one16-token chunk per stage.
            launch_recurrence<D,kChunk,4,1,kTp8MmaWarps,HasStateIn,HasStateOut,StateFP32,true,
                              true,false,0,false,false,false,StateFP32,1,0,2>(
                v_ptr,ws_beta,ws_kd,ws_qd,ws_kr,ws_gt,ws_inv,ws_mqk,initial_state_ptr,
                final_state_ptr,out_ptr,T_total,H,N,cu_seqlens_ptr,total_tiles,ws_tile_prefix,
                stream,nullptr,nullptr,v_stride);
            return;
        }
    }

#if FLASH_KDA_PPU_SEGMENT_SCAN
    // Uniform TP16 B4 batches need time parallelism too. Keep every sequence's
    // summaries and carries independent. Device metadata guards this path so
    // ragged or empty sequences keep the original recurrence during graph replay.
    if (H == 6 && N == 4 && T_total == 16384) {
        constexpr int kBatchSegments = 6;
        constexpr int kBatchSequences = 4;
        int64_t const allocated_tiles = (T_total + kChunk - 1) / kChunk + N;
        int64_t const prefix_bytes = ((int64_t(N + 1) * 4 + 127) / 128) * 128;
        char *scan_ws = ws + int64_t(H) * allocated_tiles * WS::kPerTile + prefix_bytes;
        int64_t const elems = int64_t(kBatchSequences * kBatchSegments) * H * D * D;
        auto *scan_b = reinterpret_cast<BF16 *>(scan_ws);
        auto *scan_a = scan_b + elems;
        auto *scan_carries = scan_a + elems;
        auto *scan_finals = scan_carries + elems;
        auto launch_batch_scan = [&](auto reuse_tag) {
            constexpr bool kReuse = decltype(reuse_tag)::value;
            launch_segment_scan_path<D, kBatchSegments, kChunk, HasStateIn, HasStateOut, StateFP32,
                                     IsVarlen, kBatchSequences, kReuse>(
                v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                final_state_ptr, out_ptr, scan_b, scan_a, scan_carries, scan_finals, T_total, H,
                total_tiles, ws_tile_prefix, stream, v_stride, scan_flag, allow_early_final);
        };
        if constexpr (kChunk == 16) launch_batch_scan(std::true_type{});
        else launch_batch_scan(std::false_type{});
        if constexpr (IsVarlen) {
            launch_recurrence<D, kChunk, 2, kSmallGridChunksPerStage, kNumMmaWarps, HasStateIn,
                              HasStateOut, StateFP32, true, true, false, -1>(
                v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles,
                ws_tile_prefix, stream, scan_flag, nullptr, v_stride);
        }
        return;
    }
    // The scan supports selected single-sequence H6/H12 shapes with default
    // BF16-resident recurrence. FP32 external state is converted by carry loads
    // and final replay stores; prepare and V loaders preserve packed token strides.
    // Segment counts fit the public 64-segment reserve and expose time
    // parallelism; other shapes use the serial recurrence.
    {
        if (scan_shape) {
            // The public allocator uses this upper-bound tile count, whereas fixed-mode
            // launch_fwd receives the exact count.  Place scan storage after the public
            // allocation's base workspace so it never aliases the one-tile-per-sequence
            // slack or the aligned prefix table.
            int64_t const allocated_tiles = (T_total + kChunk - 1) / kChunk + N;
            int64_t const prefix_bytes = ((int64_t(N + 1) * 4 + 127) / 128) * 128;
            // Default affine recurrence never reads INV after prepare. For
            // FP32 external H16 state, selected long shapes fit three compact
            // summary/carry matrices plus metadata inside that existing region.
            // H6/H12 retain their public scan reserve. No workspace is enlarged.
            int const storage_segments =
                H == 16 ? (T_total == 16384 ? 5 : 10) : kSegmentScanWorkspaceSegments;
            char *scan_ws = H == 16
                                ? reinterpret_cast<char *>(ws_inv)
                                : ws + int64_t(H) * allocated_tiles * WS::kPerTile + prefix_bytes;
            int64_t const scan_matrix_elems = int64_t(storage_segments) * H * D * D;
            auto *scan_b = reinterpret_cast<BF16 *>(scan_ws);
            auto *scan_apb = scan_b + scan_matrix_elems;
            auto *scan_carries = scan_apb + scan_matrix_elems;
            auto *scan_finals = scan_carries + scan_matrix_elems;
            // P26/H6 and P13/H12 make 156 CTAs, targeting four per CU on
            // the measured 39-CU tc02 part. Whole-tile remainder partitioning keeps
            // every token covered while avoiding partial CTA waves.
            if (H == 16 && T_total == 16384) {
                static_assert(5 * (3 * D * D * int(sizeof(BF16)) + int(sizeof(int))) <=
                                  (16384 / kChunk) * WS::kINV,
                              "H16 scan must fit the dead inverse workspace");
                launch_segment_scan_path<D, 5, kChunk, HasStateIn, HasStateOut, StateFP32, IsVarlen>(
                    v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                    final_state_ptr, out_ptr, scan_b, scan_apb, scan_carries, scan_finals, T_total,
                    H, total_tiles, ws_tile_prefix, stream, v_stride, scan_flag, allow_early_final);
            } else if (H == 16) {
                static_assert(10 * (3 * D * D * int(sizeof(BF16)) + int(sizeof(int))) <=
                                  (32768 / kChunk) * WS::kINV,
                              "H16 scan must fit the dead inverse workspace");
                launch_segment_scan_path<D, 10, kChunk, HasStateIn, HasStateOut, StateFP32, IsVarlen>(
                    v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                    final_state_ptr, out_ptr, scan_b, scan_apb, scan_carries, scan_finals, T_total,
                    H, total_tiles, ws_tile_prefix, stream, v_stride, scan_flag, allow_early_final);
            } else if (H == 6 && T_total == 1024) {
                launch_segment_scan_path<D, 8, kChunk, HasStateIn, HasStateOut, StateFP32, IsVarlen>(
                    v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                    final_state_ptr, out_ptr, scan_b, scan_apb, scan_carries, scan_finals, T_total,
                    H, total_tiles, ws_tile_prefix, stream, v_stride, scan_flag, allow_early_final);
            } else if (T_total <= 4096) {
                launch_segment_scan_path<D, 16, kChunk, HasStateIn, HasStateOut, StateFP32, IsVarlen>(
                    v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                    final_state_ptr, out_ptr, scan_b, scan_apb, scan_carries, scan_finals, T_total,
                    H, total_tiles, ws_tile_prefix, stream, v_stride, scan_flag, allow_early_final);
            } else if (H == 6) {
                auto launch_h6_scan = [&](auto reuse_tag) {
                    constexpr bool kReuse = decltype(reuse_tag)::value;
                    launch_segment_scan_path<D, 26, kChunk, HasStateIn, HasStateOut, StateFP32,
                                             IsVarlen, 1, kReuse>(
                        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                        final_state_ptr, out_ptr, scan_b, scan_apb, scan_carries, scan_finals, T_total,
                        H, total_tiles, ws_tile_prefix, stream, v_stride, scan_flag, allow_early_final);
                };
                if constexpr (kChunk == 16) {
                    if (T_total == 16384) launch_h6_scan(std::true_type{});
                    else launch_h6_scan(std::false_type{});
                } else {
                    launch_h6_scan(std::false_type{});
                }
            } else {
                auto launch_h12_scan = [&](auto reuse_tag) {
                    constexpr bool kReuse = decltype(reuse_tag)::value;
                    launch_segment_scan_path<D, 13, kChunk, HasStateIn, HasStateOut, StateFP32,
                                             IsVarlen, 1, kReuse, 12, 8192>(
                        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                        final_state_ptr, out_ptr, scan_b, scan_apb, scan_carries, scan_finals, T_total,
                        H, total_tiles, ws_tile_prefix, stream, v_stride, scan_flag, allow_early_final);
                };
                if constexpr (kChunk == 16) {
                    if (T_total == 8192) launch_h12_scan(std::true_type{});
                    else launch_h12_scan(std::false_type{});
                } else {
                    launch_h12_scan(std::false_type{});
                }
            }

            if constexpr (IsVarlen) {
                // The fast path assumes cu=[0,T]. A device flag makes scan and
                // fallback mutually exclusive, including exact FP32 identity
                // for an empty sequence. Both launches remain graph-capturable.
                if (H == 12) {
                    launch_recurrence<D, kChunk, 4, 2, kTp8MmaWarps, HasStateIn, HasStateOut,
                                      StateFP32, true, true, false, -1>(
                        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                        initial_state_ptr, final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr,
                        total_tiles, ws_tile_prefix, stream, scan_flag, nullptr, v_stride);
                } else {
                    launch_recurrence<D, kChunk, 2, 2, kNumMmaWarps, HasStateIn, HasStateOut,
                                      StateFP32, true, true, false, -1>(
                        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                        initial_state_ptr, final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr,
                        total_tiles, ws_tile_prefix, stream, scan_flag, nullptr, v_stride);
                }
            }
            return;
        }
    }
#endif

    // How many CTAs the d_v split should produce per (sequence, head), chosen at
    // launch time because the right answer depends on the shape, not the build.
    //
    // The recurrence is row-local in d_v, so splitting it multiplies the grid with
    // no communication -- but it replicates the kd/qd/kr loads, so it only pays when
    // the grid is short of work. Measured against SGLang's Triton chunk_kda at
    // T=8192, ratio > 1 meaning FlashKDA faster:
    //
    //             H=6      H=32     H=96
    //   split 1   0.51x    1.12x    2.15x
    //   split 2   0.82x    1.77x    2.02x
    //
    // and on the standard shapes split 2 is 16-17% faster at H=16 but 6% slower at
    // H=96. So the crossover is around 64 CTAs, which also matches the earlier
    // finding that H=96 (96 CTAs) already hides the chunk-to-chunk chain while
    // H=16 does not. Splits beyond 2 were measured and gain almost nothing -- the
    // split adds parallelism but does not shorten the chain, and each CTA still
    // walks every chunk in sequence.
#ifdef FLASH_KDA_PPU_DV_SPLIT
    // Experimental fixed dispatch used by the tuning sweep.  The production
    // default remains the measured runtime split below; keeping this hook makes
    // it possible to retest wider d_v partitioning after changes to the input
    // pipeline without editing the launcher for every build.
    constexpr int kFixedDvSplit = FLASH_KDA_PPU_DV_SPLIT;
    static_assert(kFixedDvSplit == 1 || kFixedDvSplit == 2 || kFixedDvSplit == 4 ||
                      kFixedDvSplit == 8,
                  "fixed d_v split must be 1, 2, 4, or 8");
    launch_recurrence<D, kChunk, kFixedDvSplit, kSmallGridChunksPerStage, kNumMmaWarps, HasStateIn,
                      HasStateOut, StateFP32, IsVarlen>(
        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
        final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles, ws_tile_prefix,
        stream, nullptr, nullptr, v_stride);
#else
    constexpr int64_t kDvSplitCtaTarget = FLASH_KDA_PPU_DV_SPLIT_CTA_TARGET;
    // As on TP8, a very short single sequence does not amortize a two-chunk AIU
    // stage. Keep the long H6 recurrence unchanged, but use group1 through 256
    // tokens to reduce fixed stage/barrier work at the TP16 latency floor.
    if constexpr (IsVarlen) {
        if (H == 6 && N == 1 && T_total <= 256) {
            launch_recurrence<D, kChunk, 2, 1, kNumMmaWarps, HasStateIn, HasStateOut, StateFP32,
                              IsVarlen>(v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                                        initial_state_ptr, final_state_ptr, out_ptr, T_total, H, N,
                                        cu_seqlens_ptr, total_tiles, ws_tile_prefix, stream,
                                        nullptr, nullptr, v_stride);
            return;
        }
    }
    // H12/N8 ordinary C16: keep16 state rows per MMA warp while avoiding
    // four independent copies of the transform stream. Actual CU bounds remain
    // device inputs; static factors describe only the host-checked dimensions.
    if constexpr (IsVarlen && kChunk == 16) {
        if (H == 12 && N == 8 && T_total == 8192) {
            flash_kda_ppu::flash_kda_ppu_fwd_recurrence<
                16, D, 2, kOutputStages, 8, HasStateIn, HasStateOut, StateFP32, true,
                1, 1, true, false, 0, false, false, true, false, false,
                StateFP32, 1, 0, 0, 4, 12, 8192, 8>
                <<<dim3(8,12,1), dim3(320), 0, stream>>>(
                    v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                    initial_state_ptr, final_state_ptr, out_ptr, T_total, H, N,
                    cu_seqlens_ptr, total_tiles, ws_tile_prefix, nullptr, nullptr,
                    v_stride, nullptr, nullptr, nullptr, nullptr);
            return;
        }
    }

    // Kimi K3 TP8 has 12 local KDA heads.  Four d_v slices expose 48 CTAs and
    // two MMA warps per CTA preserve the same 16 state rows per warp as the
    // TP16 split2/4-warp path.  The 64-column V staging in K2Layouts avoids the
    // previous full-D load for these 32-column slices.  Keep this shape-specific
    // so the established TP16/H=6 and many-head dispatches remain untouched.
    if (H == 12) {
        if constexpr (IsVarlen) {
            // The serving API passes cu_seqlens even for one fixed-length
            // sequence.  Group1 wins through 512 tokens; from 1024 onward that
            // N=1 case amortizes group2.  True multi-sequence varlen favors
            // group1 because short/tail chunks waste a wide stage.
            if (N == 1 && T_total > 512) {
                launch_recurrence<D, kChunk, 4, kSmallGridVarlenChunksPerStage, kTp8MmaWarps,
                                  HasStateIn, HasStateOut, StateFP32, IsVarlen>(
                    v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                    final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles,
                    ws_tile_prefix, stream, nullptr, nullptr, v_stride);
            } else {
                auto launch_group1 = [&](auto stages_tag) {
                    constexpr int kStages = decltype(stages_tag)::value;
                    launch_recurrence<D, kChunk, 4, 1, kTp8MmaWarps, HasStateIn, HasStateOut,
                                      StateFP32, IsVarlen, true, false, 0, false, false, false,
                                      StateFP32, 1, 0, kStages>(
                        v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                        final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles,
                        ws_tile_prefix, stream, nullptr, nullptr, v_stride);
                };
                if (N == 4 && T_total == 8192)
                    launch_group1(std::integral_constant<int, 2>{});
                else
                    launch_group1(std::integral_constant<int, 0>{});
            }
        } else {
            {
                // A short fixed H12 sequence has the same recurrence as N=1
                // varlen. Avoid a four-chunk stage that cannot amortize its
                // wider AIU footprint on these latency-sensitive shapes.
                if (H == 12 && N == 1 && T_total <= 2048) {
                    if (T_total <= 512) {
                        launch_recurrence<D, kChunk, 4, 1, kTp8MmaWarps, HasStateIn, HasStateOut,
                                          StateFP32, false>(
                            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                            initial_state_ptr, final_state_ptr, out_ptr, T_total, H, N,
                            cu_seqlens_ptr, total_tiles, ws_tile_prefix, stream, nullptr, nullptr,
                            v_stride);
                    } else {
                        launch_recurrence<D, kChunk, 4, 2, kTp8MmaWarps, HasStateIn, HasStateOut,
                                          StateFP32, false>(
                            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                            initial_state_ptr, final_state_ptr, out_ptr, T_total, H, N,
                            cu_seqlens_ptr, total_tiles, ws_tile_prefix, stream, nullptr, nullptr,
                            v_stride);
                    }
                    return;
                }
            }
            launch_recurrence<D, kChunk, 4, kSmallGridFixedChunksPerStage, kTp8MmaWarps, HasStateIn,
                              HasStateOut, StateFP32, IsVarlen>(
                v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
                final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles,
                ws_tile_prefix, stream, nullptr, nullptr, v_stride);
        }
    } else if (int64_t(N) * H >= kDvSplitCtaTarget) {
        launch_recurrence<D, kChunk, 1, 1, kNumMmaWarps, HasStateIn, HasStateOut, StateFP32,
                          IsVarlen>(v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
                                    initial_state_ptr, final_state_ptr, out_ptr, T_total, H, N,
                                    cu_seqlens_ptr, total_tiles, ws_tile_prefix, stream, nullptr,
                                    nullptr, v_stride);
    } else {
        launch_recurrence<D, kChunk, 2, kSmallGridChunksPerStage, kNumMmaWarps, HasStateIn,
                          HasStateOut, StateFP32, IsVarlen>(
            v_ptr, ws_beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk, initial_state_ptr,
            final_state_ptr, out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles, ws_tile_prefix,
            stream, nullptr, nullptr, v_stride);
    }
#endif
}

// Explicit instantiations. The (false, false, *) state combination collapses to
// StateFP32 = false, so there are 7 per varlen setting rather than 8.

template <int D, bool HasStateIn, bool HasStateOut, bool StateFP32, bool IsVarlen>
void launch_fwd(cutlass::bfloat16_t const *q_ptr, cutlass::bfloat16_t const *k_ptr,
                cutlass::bfloat16_t const *v_ptr, cutlass::bfloat16_t const *g_bf16_ptr,
                cutlass::bfloat16_t const *beta_ptr, void const *initial_state_ptr, float scale,
                void *final_state_ptr, cutlass::bfloat16_t *out_ptr, void *workspace_ptr,
                int total_tiles, int T_total, int H, int N, int64_t const *cu_seqlens_ptr,
                float const *A_log_ptr, float const *dt_bias_ptr, float gate_scale,
                cudaStream_t stream, int q_stride, int k_stride, int v_stride) {
    if constexpr (IsVarlen) {
        if (flash_kda_ppu_use_centered32(T_total, H, N)) {
    launch_fwd_impl<D, 32, HasStateIn, HasStateOut, StateFP32, IsVarlen>(
        q_ptr, k_ptr, v_ptr, g_bf16_ptr, beta_ptr, initial_state_ptr, scale, final_state_ptr,
        out_ptr, workspace_ptr, total_tiles, T_total, H, N, cu_seqlens_ptr, A_log_ptr, dt_bias_ptr,
        gate_scale, stream, q_stride, k_stride, v_stride);

            return;
        }
    }
    launch_fwd_impl<D, 16, HasStateIn, HasStateOut, StateFP32, IsVarlen>(
        q_ptr, k_ptr, v_ptr, g_bf16_ptr, beta_ptr, initial_state_ptr, scale, final_state_ptr,
        out_ptr, workspace_ptr, total_tiles, T_total, H, N, cu_seqlens_ptr, A_log_ptr, dt_bias_ptr,
        gate_scale, stream, q_stride, k_stride, v_stride);
}

template <int D, bool HasStateIn, bool HasStateOut, bool StateFP32, bool IsVarlen>
void launch_fwd(cutlass::bfloat16_t const *q_ptr, cutlass::bfloat16_t const *k_ptr,
                cutlass::bfloat16_t const *v_ptr, cutlass::bfloat16_t const *g_bf16_ptr,
                cutlass::bfloat16_t const *beta_ptr, void const *initial_state_ptr, float scale,
                void *final_state_ptr, cutlass::bfloat16_t *out_ptr, void *workspace_ptr,
                int total_tiles, int T_total, int H, int N, int64_t const *cu_seqlens_ptr,
                float const *A_log_ptr, float const *dt_bias_ptr, float gate_scale,
                cudaStream_t stream) {
    launch_fwd<D, HasStateIn, HasStateOut, StateFP32, IsVarlen>(
        q_ptr, k_ptr, v_ptr, g_bf16_ptr, beta_ptr, initial_state_ptr, scale, final_state_ptr,
        out_ptr, workspace_ptr, total_tiles, T_total, H, N, cu_seqlens_ptr, A_log_ptr, dt_bias_ptr,
        gate_scale, stream, H * D, H * D, H * D);
}

#define INSTANTIATE_LAUNCH_FWD(D, HI, HO, FP32, VL)                                                \
    template void launch_fwd<D, HI, HO, FP32, VL>(                                                 \
        cutlass::bfloat16_t const *, cutlass::bfloat16_t const *, cutlass::bfloat16_t const *,     \
        cutlass::bfloat16_t const *, cutlass::bfloat16_t const *, void const *, float, void *,     \
        cutlass::bfloat16_t *, void *, int, int, int, int, int64_t const *, float const *,         \
        float const *, float, cudaStream_t, int, int, int);                                        \
    template void launch_fwd<D, HI, HO, FP32, VL>(                                                 \
        cutlass::bfloat16_t const *, cutlass::bfloat16_t const *, cutlass::bfloat16_t const *,     \
        cutlass::bfloat16_t const *, cutlass::bfloat16_t const *, void const *, float, void *,     \
        cutlass::bfloat16_t *, void *, int, int, int, int, int64_t const *, float const *,         \
        float const *, float, cudaStream_t);

#define INSTANTIATE_STATE_VARIANTS(VL)                                                             \
    INSTANTIATE_LAUNCH_FWD(128, true, true, false, VL)                                             \
    INSTANTIATE_LAUNCH_FWD(128, true, true, true, VL)                                              \
    INSTANTIATE_LAUNCH_FWD(128, false, false, false, VL)                                           \
    INSTANTIATE_LAUNCH_FWD(128, false, true, false, VL)                                            \
    INSTANTIATE_LAUNCH_FWD(128, true, false, false, VL)                                            \
    INSTANTIATE_LAUNCH_FWD(128, false, true, true, VL)                                             \
    INSTANTIATE_LAUNCH_FWD(128, true, false, true, VL)

INSTANTIATE_STATE_VARIANTS(true)  // varlen
INSTANTIATE_STATE_VARIANTS(false) // fixed length
