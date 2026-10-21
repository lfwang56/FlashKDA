// One selected repair family per complete CTA; no cross-CTA polling.
#pragma once
namespace flash_kda_ppu {
template <int D, bool ExternalStateFP32>
__global__ void __launch_bounds__(288, 2)
    flash_kda_ppu_fwd_recurrence_hybrid_repair(BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
                                 BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd,
                                 BF16 const *__restrict__ ws_kr, float const *__restrict__ ws_gt,
                                 BF16 const *__restrict__ ws_inv, BF16 const *__restrict__ ws_mqk,
                                 void *__restrict__ prefix_state_base,
                                 void *__restrict__ true_carry_base, BF16 *__restrict__ out_ptr,
                                 int T_total_runtime, int H_runtime, int N_runtime,
                                 int64_t const *__restrict__ cu_seqlens,
                                 int total_tiles, int const *__restrict__ tile_prefix,
                                 int const *__restrict__ path_flag,
                                 void *__restrict__ auxiliary_final_state_ptr, int v_stride,
                                 int *__restrict__ summary_nonzero,
                                 BF16 const *zero_summary_b = nullptr,
                                 void const *external_initial_state = nullptr,
                                 void *external_final_state = nullptr)
{
    static_assert(D == 128, "private hybrid repair is D128 only");
    using HybridStorage = SharedStorageK2<D,16,2,2,8,true,1,1,true,false,true,true>;
    // One aligned aggregate for AIU pipeline and Mode4 checkpoint flag.
    struct HybridRepairStorage { HybridStorage pipeline; unsigned mismatch; };
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
        bool const repair_prior_matched = summary_nonzero[
            24 * 6 + (base + 1) * 6 + int(blockIdx.y)] != 0;
        bool const final_prefix_matched = summary_nonzero[
            24 * 6 + (base + 2) * 6 + int(blockIdx.y)] != 0;
        if (repair_prior_matched && final_prefix_matched) return;
        for (int repair_segment_index = repair_prior_matched ? 2 : 1;
             repair_segment_index < 3; ++repair_segment_index) {
#include "fwd_recurrence_body_ppu.cuh"
        }
    }
    else if (path_flag[1] != 0) {
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
