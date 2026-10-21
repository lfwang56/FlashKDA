// Exact H12/N4 P3 prefix: checkpoint read and sparse prefix write share
// one restricted slot origin, after the complete-state comparison/CTA join.
#pragma once
namespace flash_kda_ppu {
template <int D, bool ExternalStateFP32>
__global__ void __launch_bounds__(288, FLASH_KDA_PPU_K2_CTAS_PER_CU)
flash_kda_ppu_h12_n4_compact_prefix(
    BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
    BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd,
    BF16 const *__restrict__ ws_kr, float const *__restrict__ ws_gt,
    BF16 const *__restrict__ ws_inv, BF16 const *__restrict__ ws_mqk,
    BF16 *__restrict__ prefix_slot, BF16 const *__restrict__ b_state,
    BF16 *__restrict__ out_ptr, int T_total_runtime, int H_runtime,
    int N_runtime, int total_tiles, int const *__restrict__ tile_prefix,
    int const *__restrict__ path_flag, int v_stride,
    int *__restrict__ summary_nonzero)
{
    static_assert(D == 128, "private compact prefix is D128 only");
    using Storage = SharedStorageK2<D,16,3,2,8,true,1,1,true,false,true,true>;
    struct PrefixStorage { Storage pipeline; unsigned mismatch; };
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
