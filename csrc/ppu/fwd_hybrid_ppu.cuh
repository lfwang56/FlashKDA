#pragma once

#include "fwd_kernel2_ppu.cuh"

// Private mutually exclusive P3/P2 B family. Include after fwd_kernel2.
namespace flash_kda_ppu {
template <int D, bool ExternalStateFP32>
__global__ void __launch_bounds__(288, 4) flash_kda_ppu_fwd_recurrence_hybrid_b(
    BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
    BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd, BF16 const *__restrict__ ws_kr,
    float const *__restrict__ ws_gt, BF16 const *__restrict__ ws_inv,
    BF16 const *__restrict__ ws_mqk, void const *__restrict__ initial_state_ptr,
    void *__restrict__ final_state_ptr, BF16 *__restrict__ out_ptr, int T_total_runtime,
    int H_runtime, int N_runtime, int64_t const *__restrict__ cu_seqlens, int total_tiles,
    int const *__restrict__ tile_prefix, int const *__restrict__ path_flag,
    void *__restrict__ auxiliary_final_state_ptr, int v_stride, int *__restrict__ summary_nonzero,
    BF16 const *zero_summary_b = nullptr, void const *external_initial_state = nullptr,
    void *external_final_state = nullptr) {
    using HybridStorage =
        SharedStorageK2<D, 16, 2, 2, 8, !ExternalStateFP32, 1, 1, true, false, true, true>;
    __shared__ HybridStorage hybrid_b_storage;
    // Device flags were initialized and conservatively cleared in this call.
    // Family selection is uniform across every warp in this CTA.
#define FLASH_KDA_EXTERNAL_B_STORAGE hybrid_b_storage
#define FLASH_KDA_HYBRID_B_LOGICAL_RANGES
    if (path_flag[0] != 0) {
        constexpr int CHUNK = 16;
        constexpr int InputStages = 2;
        constexpr int OutputStages = 2;
        constexpr int NumMmaWarps = 8;
        constexpr int DvSplit = 1;
        constexpr int ChunksPerStage = 1;
        constexpr int PathFlagMode = 0;
        constexpr int SegmentSequences = 8;
        constexpr int PrefixMode = 1;
        constexpr int PrefixChunks = 3;
        constexpr int StaticHeads = 6;
        constexpr int StaticTokens = 16384;
        constexpr int SerialSegmentIndex = -1;
        constexpr int repair_segment_index = 0;
        constexpr int SegmentsPerSequence = 3;
        constexpr int StaticSequences = 24;
        constexpr bool HasStateIn = false;
        constexpr bool HasStateOut = true;
        constexpr bool StateFP32 = false;
        constexpr bool IsVarlen = false;
        constexpr bool ProduceOutput = true;
        constexpr bool IdentityState = false;
        constexpr bool ZeroValue = false;
        constexpr bool DualSummary = false;
        constexpr bool AffinePrepared = true;
        constexpr bool Segmented = true;
        constexpr bool DirectOutput = true;
        constexpr bool OutputStateFP32 = false;
        constexpr bool DirectZeroCarry = false;
        constexpr bool ExternalHasStateIn = true;
        constexpr bool PrepublishFinal = false;
        constexpr bool SegmentCarryOut = false;
        constexpr bool FirstSegmentTrueB = true;
        constexpr bool repair_prior_matched = true;
        constexpr bool PublicHasStateOut = true;
#include "fwd_recurrence_body_ppu.cuh"
    } else if (path_flag[1] != 0) {
        // Keep all 96 useful P2 CTAs contiguous in the physical launch order.
        // The final 48 CTAs return without touching shared/global state.
        if (int(blockIdx.x) + 24 * int(blockIdx.y) >= 16 * 6)
            return;
        constexpr int CHUNK = 16;
        constexpr int InputStages = 2;
        constexpr int OutputStages = 2;
        constexpr int NumMmaWarps = 8;
        constexpr int DvSplit = 1;
        constexpr int ChunksPerStage = 1;
        constexpr int PathFlagMode = 0;
        constexpr int SegmentSequences = 8;
        constexpr int PrefixMode = 1;
        constexpr int PrefixChunks = 3;
        constexpr int StaticHeads = 6;
        constexpr int StaticTokens = 16384;
        constexpr int SerialSegmentIndex = -1;
        constexpr int repair_segment_index = 0;
        constexpr int SegmentsPerSequence = 2;
        constexpr int StaticSequences = 16;
        constexpr bool HasStateIn = false;
        constexpr bool HasStateOut = true;
        constexpr bool StateFP32 = false;
        constexpr bool IsVarlen = false;
        constexpr bool ProduceOutput = true;
        constexpr bool IdentityState = false;
        constexpr bool ZeroValue = false;
        constexpr bool DualSummary = false;
        constexpr bool AffinePrepared = true;
        constexpr bool Segmented = true;
        constexpr bool DirectOutput = true;
        constexpr bool OutputStateFP32 = false;
        constexpr bool DirectZeroCarry = false;
        constexpr bool ExternalHasStateIn = true;
        constexpr bool PrepublishFinal = false;
        constexpr bool SegmentCarryOut = false;
        constexpr bool FirstSegmentTrueB = true;
        constexpr bool repair_prior_matched = true;
        constexpr bool PublicHasStateOut = false;
#define FLASH_KDA_HYBRID_B_P2_MAPPING
#include "fwd_recurrence_body_ppu.cuh"
#undef FLASH_KDA_HYBRID_B_P2_MAPPING
    }
#undef FLASH_KDA_HYBRID_B_LOGICAL_RANGES
#undef FLASH_KDA_EXTERNAL_B_STORAGE
}
} // namespace flash_kda_ppu

// One selected repair family per complete CTA; no cross-CTA polling.
namespace flash_kda_ppu {
template <int D, bool ExternalStateFP32>
__global__ void __launch_bounds__(288, 2) flash_kda_ppu_fwd_recurrence_hybrid_repair(
    BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
    BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd, BF16 const *__restrict__ ws_kr,
    float const *__restrict__ ws_gt, BF16 const *__restrict__ ws_inv,
    BF16 const *__restrict__ ws_mqk, void *__restrict__ prefix_state_base,
    void *__restrict__ true_carry_base, BF16 *__restrict__ out_ptr, int T_total_runtime,
    int H_runtime, int N_runtime, int64_t const *__restrict__ cu_seqlens, int total_tiles,
    int const *__restrict__ tile_prefix, int const *__restrict__ path_flag,
    void *__restrict__ auxiliary_final_state_ptr, int v_stride, int *__restrict__ summary_nonzero,
    BF16 const *zero_summary_b = nullptr, void const *external_initial_state = nullptr,
    void *external_final_state = nullptr) {
    static_assert(D == 128, "private hybrid repair is D128 only");
    using HybridStorage = SharedStorageK2<D, 16, 2, 2, 8, true, 1, 1, true, false, true, true>;
    // One aligned aggregate for AIU pipeline and Mode4 checkpoint flag.
    struct HybridRepairStorage {
        HybridStorage pipeline;
        unsigned mismatch;
    };
    __shared__ HybridRepairStorage hybrid_repair_storage;
#define FLASH_KDA_EXTERNAL_REPAIR_STORAGE hybrid_repair_storage
    if (path_flag[0] != 0) {
        constexpr int CHUNK = 16;
        constexpr int InputStages = 2;
        constexpr int OutputStages = 2;
        constexpr int NumMmaWarps = 8;
        constexpr int DvSplit = 1;
        constexpr int ChunksPerStage = 1;
        constexpr int PathFlagMode = 0;
        constexpr int SegmentSequences = 8;
        constexpr int PrefixChunks = 3;
        constexpr int StaticHeads = 6;
        constexpr int StaticTokens = 16384;
        constexpr int SegmentsPerSequence = 3;
        constexpr int StaticSequences = 24;
        constexpr int PrefixMode = 7;
        constexpr int SerialSegmentIndex = 0;
        constexpr bool HasStateIn = (D == 128);
        constexpr bool HasStateOut = (D == 128);
        constexpr bool StateFP32 = (D != 128);
        constexpr bool IsVarlen = (D != 128);
        constexpr bool ProduceOutput = (D == 128);
        constexpr bool IdentityState = (D != 128);
        constexpr bool ZeroValue = (D != 128);
        constexpr bool DualSummary = (D != 128);
        constexpr bool AffinePrepared = (D == 128);
        constexpr bool Segmented = (D == 128);
        constexpr bool DirectOutput = (D == 128);
        constexpr bool OutputStateFP32 = (D != 128);
        constexpr bool PublicHasStateOut = (D == 128);
        constexpr bool SegmentCarryOut = (D == 128);
        constexpr bool FirstSegmentTrueB = (D != 128);
        constexpr bool DirectZeroCarry = (D != 128);
        constexpr bool ExternalHasStateIn = (D != 128);
        constexpr bool PrepublishFinal = (D == 128);
        void const *initial_state_ptr = prefix_state_base;
        // Both the first true-carry store and second-phase carry read derive
        // from this same restricted origin, preserving the169 alias repair.
        void *final_state_ptr = true_carry_base;
        int const base = int(blockIdx.x) * 3;
        bool const repair_prior_matched =
            summary_nonzero[24 * 6 + (base + 1) * 6 + int(blockIdx.y)] != 0;
        bool const final_prefix_matched =
            summary_nonzero[24 * 6 + (base + 2) * 6 + int(blockIdx.y)] != 0;
        if (repair_prior_matched && final_prefix_matched)
            return;
        for (int repair_segment_index = repair_prior_matched ? 2 : 1; repair_segment_index < 3;
             ++repair_segment_index) {
#include "fwd_recurrence_body_ppu.cuh"
        }
    } else if (path_flag[1] != 0) {
        constexpr int CHUNK = 16;
        constexpr int InputStages = 2;
        constexpr int OutputStages = 2;
        constexpr int NumMmaWarps = 8;
        constexpr int DvSplit = 1;
        constexpr int ChunksPerStage = 1;
        constexpr int PathFlagMode = 0;
        constexpr int SegmentSequences = 8;
        constexpr int PrefixChunks = 3;
        constexpr int StaticHeads = 6;
        constexpr int StaticTokens = 16384;
        constexpr int SegmentsPerSequence = 2;
        constexpr int StaticSequences = 16;
        constexpr int PrefixMode = 4;
        constexpr int SerialSegmentIndex = 1;
        constexpr bool HasStateIn = (D == 128);
        constexpr bool HasStateOut = (D == 128);
        constexpr bool StateFP32 = (D != 128);
        constexpr bool IsVarlen = (D != 128);
        constexpr bool ProduceOutput = (D == 128);
        constexpr bool IdentityState = (D != 128);
        constexpr bool ZeroValue = (D != 128);
        constexpr bool DualSummary = (D != 128);
        constexpr bool AffinePrepared = (D == 128);
        constexpr bool Segmented = (D == 128);
        constexpr bool DirectOutput = (D == 128);
        constexpr bool OutputStateFP32 = (D != 128);
        constexpr bool PublicHasStateOut = (D == 128);
        constexpr bool SegmentCarryOut = (D == 128);
        constexpr bool FirstSegmentTrueB = (D != 128);
        constexpr bool DirectZeroCarry = (D == 128);
        constexpr bool ExternalHasStateIn = (D == 128);
        constexpr bool PrepublishFinal = (D != 128);
        void const *initial_state_ptr = true_carry_base;
        void *final_state_ptr = prefix_state_base;
        constexpr int repair_segment_index = 0;
        constexpr bool repair_prior_matched = (D == 128);
#include "fwd_recurrence_body_ppu.cuh"
    }
#undef FLASH_KDA_EXTERNAL_REPAIR_STORAGE
}
} // namespace flash_kda_ppu

// Private mutually exclusive P3/P2 B family. Include after fwd_kernel2.
namespace flash_kda_ppu {
template <int D, bool ExternalStateFP32>
__global__ void __launch_bounds__(288, 4) flash_kda_ppu_h12_n4_hybrid_b(
    BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
    BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd, BF16 const *__restrict__ ws_kr,
    float const *__restrict__ ws_gt, BF16 const *__restrict__ ws_inv,
    BF16 const *__restrict__ ws_mqk, void const *__restrict__ initial_state_ptr,
    void *__restrict__ final_state_ptr, BF16 *__restrict__ out_ptr, int T_total_runtime,
    int H_runtime, int N_runtime, int64_t const *__restrict__ cu_seqlens, int total_tiles,
    int const *__restrict__ tile_prefix, int const *__restrict__ path_flag,
    void *__restrict__ auxiliary_final_state_ptr, int v_stride, int *__restrict__ summary_nonzero,
    BF16 const *zero_summary_b = nullptr, void const *external_initial_state = nullptr,
    void *external_final_state = nullptr) {
    using HybridStorage =
        SharedStorageK2<D, 16, 2, 2, 8, !ExternalStateFP32, 1, 1, true, false, true, true>;
    __shared__ HybridStorage hybrid_b_storage;
    // Device flags were initialized and conservatively cleared in this call.
    // Family selection is uniform across every warp in this CTA.
#define FLASH_KDA_EXTERNAL_B_STORAGE hybrid_b_storage
#define FLASH_KDA_HYBRID_B_LOGICAL_RANGES
    if (path_flag[0] != 0) {
        constexpr int CHUNK = 16;
        constexpr int InputStages = 2;
        constexpr int OutputStages = 2;
        constexpr int NumMmaWarps = 8;
        constexpr int DvSplit = 1;
        constexpr int ChunksPerStage = 1;
        constexpr int PathFlagMode = 0;
        constexpr int SegmentSequences = 4;
        constexpr int PrefixMode = 1;
        constexpr int PrefixChunks = 3;
        constexpr int StaticHeads = 12;
        constexpr int StaticTokens = 8192;
        constexpr int SerialSegmentIndex = -1;
        constexpr int repair_segment_index = 0;
        constexpr int SegmentsPerSequence = 3;
        constexpr int StaticSequences = 12;
        constexpr bool HasStateIn = false;
        constexpr bool HasStateOut = true;
        constexpr bool StateFP32 = false;
        constexpr bool IsVarlen = false;
        constexpr bool ProduceOutput = true;
        constexpr bool IdentityState = false;
        constexpr bool ZeroValue = false;
        constexpr bool DualSummary = false;
        constexpr bool AffinePrepared = true;
        constexpr bool Segmented = true;
        constexpr bool DirectOutput = true;
        constexpr bool OutputStateFP32 = false;
        constexpr bool DirectZeroCarry = false;
        constexpr bool ExternalHasStateIn = true;
        constexpr bool PrepublishFinal = false;
        constexpr bool SegmentCarryOut = false;
        constexpr bool FirstSegmentTrueB = true;
        constexpr bool repair_prior_matched = true;
        constexpr bool PublicHasStateOut = true;
#include "fwd_recurrence_body_ppu.cuh"
    } else if (path_flag[1] != 0) {
        // Keep all 96 useful P2 CTAs contiguous in the physical launch order.
        // The final 48 CTAs return without touching shared/global state.
        if (int(blockIdx.x) + 12 * int(blockIdx.y) >= 8 * 12)
            return;
        constexpr int CHUNK = 16;
        constexpr int InputStages = 2;
        constexpr int OutputStages = 2;
        constexpr int NumMmaWarps = 8;
        constexpr int DvSplit = 1;
        constexpr int ChunksPerStage = 1;
        constexpr int PathFlagMode = 0;
        constexpr int SegmentSequences = 4;
        constexpr int PrefixMode = 1;
        constexpr int PrefixChunks = 3;
        constexpr int StaticHeads = 12;
        constexpr int StaticTokens = 8192;
        constexpr int SerialSegmentIndex = -1;
        constexpr int repair_segment_index = 0;
        constexpr int SegmentsPerSequence = 2;
        constexpr int StaticSequences = 8;
        constexpr bool HasStateIn = false;
        constexpr bool HasStateOut = true;
        constexpr bool StateFP32 = false;
        constexpr bool IsVarlen = false;
        constexpr bool ProduceOutput = true;
        constexpr bool IdentityState = false;
        constexpr bool ZeroValue = false;
        constexpr bool DualSummary = false;
        constexpr bool AffinePrepared = true;
        constexpr bool Segmented = true;
        constexpr bool DirectOutput = true;
        constexpr bool OutputStateFP32 = false;
        constexpr bool DirectZeroCarry = false;
        constexpr bool ExternalHasStateIn = true;
        constexpr bool PrepublishFinal = false;
        constexpr bool SegmentCarryOut = false;
        constexpr bool FirstSegmentTrueB = true;
        constexpr bool repair_prior_matched = true;
        constexpr bool PublicHasStateOut = false;
#define FLASH_KDA_H12_N4_HYBRID_B_P2_MAPPING
#include "fwd_recurrence_body_ppu.cuh"
#undef FLASH_KDA_H12_N4_HYBRID_B_P2_MAPPING
    }
#undef FLASH_KDA_HYBRID_B_LOGICAL_RANGES
#undef FLASH_KDA_EXTERNAL_B_STORAGE
}
} // namespace flash_kda_ppu

// One selected repair family per complete CTA; no cross-CTA polling.
namespace flash_kda_ppu {
template <int D, bool ExternalStateFP32>
__global__ void __launch_bounds__(288, 2) flash_kda_ppu_h12_n4_hybrid_repair(
    BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
    BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd, BF16 const *__restrict__ ws_kr,
    float const *__restrict__ ws_gt, BF16 const *__restrict__ ws_inv,
    BF16 const *__restrict__ ws_mqk, void *__restrict__ prefix_state_base,
    BF16 *__restrict__ b_state, BF16 *__restrict__ out_ptr, int T_total_runtime, int H_runtime,
    int N_runtime, int64_t const *__restrict__ cu_seqlens, int total_tiles,
    int const *__restrict__ tile_prefix, int const *__restrict__ path_flag, int v_stride,
    int *__restrict__ summary_nonzero, void const *external_initial_state = nullptr,
    void *external_final_state = nullptr) {
    // The opaque checkpoint/prefix slot has one restricted origin.
    void *auxiliary_final_state_ptr = prefix_state_base;
    static_assert(D == 128, "private hybrid repair is D128 only");
    using HybridStorage = SharedStorageK2<D, 16, 2, 2, 8, true, 1, 1, true, false, true, true>;
    // One aligned aggregate for AIU pipeline and Mode4 checkpoint flag.
    struct HybridRepairStorage {
        HybridStorage pipeline;
        unsigned mismatch;
    };
    __shared__ HybridRepairStorage hybrid_repair_storage;
#define FLASH_KDA_EXTERNAL_REPAIR_STORAGE hybrid_repair_storage
    if (path_flag[0] != 0) {
        constexpr int CHUNK = 16;
        constexpr int InputStages = 2;
        constexpr int OutputStages = 2;
        constexpr int NumMmaWarps = 8;
        constexpr int DvSplit = 1;
        constexpr int ChunksPerStage = 1;
        constexpr int PathFlagMode = 0;
        constexpr int SegmentSequences = 4;
        constexpr int PrefixChunks = 3;
        constexpr int StaticHeads = 12;
        constexpr int StaticTokens = 8192;
        constexpr int SegmentsPerSequence = 3;
        constexpr int StaticSequences = 12;
        constexpr int PrefixMode = 7;
        constexpr int SerialSegmentIndex = 0;
        constexpr bool HasStateIn = (D == 128);
        constexpr bool HasStateOut = (D == 128);
        constexpr bool StateFP32 = (D != 128);
        constexpr bool IsVarlen = (D != 128);
        constexpr bool ProduceOutput = (D == 128);
        constexpr bool IdentityState = (D != 128);
        constexpr bool ZeroValue = (D != 128);
        constexpr bool DualSummary = (D != 128);
        constexpr bool AffinePrepared = (D == 128);
        constexpr bool Segmented = (D == 128);
        constexpr bool DirectOutput = (D == 128);
        constexpr bool OutputStateFP32 = (D != 128);
        constexpr bool PublicHasStateOut = (D == 128);
        constexpr bool SegmentCarryOut = (D == 128);
        constexpr bool FirstSegmentTrueB = (D != 128);
        constexpr bool DirectZeroCarry = (D != 128);
        constexpr bool ExternalHasStateIn = (D != 128);
        constexpr bool PrepublishFinal = (D == 128);
        void const *initial_state_ptr = prefix_state_base;
        // All prefix hypotheses finished before this kernel. B1/B2 now serve
        // as true carries; both phases derive reads/writes from one origin.
        // The original169 fence/full-CTA join remains in the shared body.
        void *final_state_ptr = b_state;
        BF16 const *zero_summary_b = nullptr;
        int const base = int(blockIdx.x) * 3;
        bool const repair_prior_matched =
            summary_nonzero[12 * 12 + (base + 1) * 12 + int(blockIdx.y)] != 0;
        bool const final_prefix_matched =
            summary_nonzero[12 * 12 + (base + 2) * 12 + int(blockIdx.y)] != 0;
        if (repair_prior_matched && final_prefix_matched)
            return;
        // Two finite phases, with independent compile-time segment indices.
        // The original body retains its fence/full-CTA join between phases.
        if (!repair_prior_matched) {
            constexpr int repair_segment_index = 1;
#include "fwd_recurrence_body_ppu.cuh"
        }
        {
            constexpr int repair_segment_index = 2;
#include "fwd_recurrence_body_ppu.cuh"
        }
    } else if (path_flag[1] != 0) {
        BF16 const *zero_summary_b = b_state;
        constexpr int CHUNK = 16;
        constexpr int InputStages = 2;
        constexpr int OutputStages = 2;
        constexpr int NumMmaWarps = 8;
        constexpr int DvSplit = 1;
        constexpr int ChunksPerStage = 1;
        constexpr int PathFlagMode = 0;
        constexpr int SegmentSequences = 4;
        constexpr int PrefixChunks = 3;
        constexpr int StaticHeads = 12;
        constexpr int StaticTokens = 8192;
        constexpr int SegmentsPerSequence = 2;
        constexpr int StaticSequences = 8;
        constexpr int PrefixMode = 4;
        constexpr int SerialSegmentIndex = 1;
        constexpr bool HasStateIn = (D == 128);
        constexpr bool HasStateOut = (D == 128);
        constexpr bool StateFP32 = (D != 128);
        constexpr bool IsVarlen = (D != 128);
        constexpr bool ProduceOutput = (D == 128);
        constexpr bool IdentityState = (D != 128);
        constexpr bool ZeroValue = (D != 128);
        constexpr bool DualSummary = (D != 128);
        constexpr bool AffinePrepared = (D == 128);
        constexpr bool Segmented = (D == 128);
        constexpr bool DirectOutput = (D == 128);
        constexpr bool OutputStateFP32 = (D != 128);
        constexpr bool PublicHasStateOut = (D == 128);
        constexpr bool SegmentCarryOut = (D == 128);
        constexpr bool FirstSegmentTrueB = (D != 128);
        constexpr bool DirectZeroCarry = (D == 128);
        constexpr bool ExternalHasStateIn = (D == 128);
        constexpr bool PrepublishFinal = (D != 128);
        void const *initial_state_ptr = nullptr;
        void *final_state_ptr = nullptr;
        constexpr int repair_segment_index = 0;
        constexpr bool repair_prior_matched = (D == 128);
#define FLASH_KDA_COMPACT_P2_REPAIR
#include "fwd_recurrence_body_ppu.cuh"
#undef FLASH_KDA_COMPACT_P2_REPAIR
    }
#undef FLASH_KDA_EXTERNAL_REPAIR_STORAGE
}
} // namespace flash_kda_ppu

// Exact H12/N4 P3 prefix: checkpoint read and sparse prefix write share
// one restricted slot origin, after the complete-state comparison/CTA join.
namespace flash_kda_ppu {
template <int D, bool ExternalStateFP32>
__global__ void __launch_bounds__(288, FLASH_KDA_PPU_K2_CTAS_PER_CU)
    flash_kda_ppu_h12_n4_compact_prefix(
        BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
        BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd,
        BF16 const *__restrict__ ws_kr, float const *__restrict__ ws_gt,
        BF16 const *__restrict__ ws_inv, BF16 const *__restrict__ ws_mqk,
        BF16 *__restrict__ prefix_slot, BF16 const *__restrict__ b_state,
        BF16 *__restrict__ out_ptr, int T_total_runtime, int H_runtime, int N_runtime,
        int total_tiles, int const *__restrict__ tile_prefix, int const *__restrict__ path_flag,
        int v_stride, int *__restrict__ summary_nonzero) {
    static_assert(D == 128, "private compact prefix is D128 only");
    using Storage = SharedStorageK2<D, 16, 3, 2, 8, true, 1, 1, true, false, true, true>;
    struct PrefixStorage {
        Storage pipeline;
        unsigned mismatch;
    };
    __shared__ PrefixStorage prefix_storage;
    // Both local aliases below derive from prefix_slot, never two restrict
    // parameters. The body compares every opaque word and joins the whole CTA
    // before a mismatch overwrites this CTA's slot with row-major state.
    void *final_state_ptr = prefix_slot;
    void *auxiliary_final_state_ptr = prefix_slot;
    void const *initial_state_ptr = nullptr;
    BF16 const *zero_summary_b = b_state;
    int64_t const *cu_seqlens = nullptr;
    void const *external_initial_state = nullptr;
    void *external_final_state = nullptr;
    constexpr int CHUNK = 16, InputStages = 3, OutputStages = 2;
    constexpr int NumMmaWarps = 8, DvSplit = 1, ChunksPerStage = 1;
    constexpr int PathFlagMode = 1, SegmentSequences = 4, SegmentsPerSequence = 3;
    constexpr int PrefixMode = 5, PrefixChunks = 3, StaticHeads = 12;
    constexpr int StaticTokens = 8192, StaticSequences = 12;
    constexpr int SerialSegmentIndex = -1, repair_segment_index = 0;
    constexpr bool HasStateIn = (D == 128), HasStateOut = (D == 128), StateFP32 = (D != 128);
    constexpr bool IsVarlen = (D != 128), ProduceOutput = (D == 128), IdentityState = (D != 128);
    constexpr bool ZeroValue = (D != 128), DualSummary = (D != 128), AffinePrepared = (D == 128);
    constexpr bool Segmented = (D == 128), DirectOutput = (D == 128), OutputStateFP32 = (D != 128);
    constexpr bool DirectZeroCarry = (D == 128), ExternalHasStateIn = (D == 128);
    constexpr bool PrepublishFinal = (D != 128), SegmentCarryOut = (D != 128);
    constexpr bool FirstSegmentTrueB = (D != 128), repair_prior_matched = (D == 128);
    constexpr bool PublicHasStateOut = (D != 128);
#define FLASH_KDA_EXTERNAL_REPAIR_STORAGE prefix_storage
#include "fwd_recurrence_body_ppu.cuh"
#undef FLASH_KDA_EXTERNAL_REPAIR_STORAGE
}
} // namespace flash_kda_ppu
