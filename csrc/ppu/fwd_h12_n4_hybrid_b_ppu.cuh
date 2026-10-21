// Private mutually exclusive P3/P2 B family. Include after fwd_kernel2.
#pragma once
namespace flash_kda_ppu {
template <int D, bool ExternalStateFP32>
__global__ void __launch_bounds__(288, 4)
    flash_kda_ppu_h12_n4_hybrid_b(BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
                                 BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd,
                                 BF16 const *__restrict__ ws_kr, float const *__restrict__ ws_gt,
                                 BF16 const *__restrict__ ws_inv, BF16 const *__restrict__ ws_mqk,
                                 void const *__restrict__ initial_state_ptr,
                                 void *__restrict__ final_state_ptr, BF16 *__restrict__ out_ptr,
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
    using HybridStorage = SharedStorageK2<D,16,2,2,8,!ExternalStateFP32,
                                         1,1,true,false,true,true>;
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
    }
    else if (path_flag[1] != 0) {
        // Keep all 96 useful P2 CTAs contiguous in the physical launch order.
        // The final 48 CTAs return without touching shared/global state.
        if (int(blockIdx.x) + 12 * int(blockIdx.y) >= 8 * 12) return;
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
