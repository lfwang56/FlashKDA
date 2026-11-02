#include "verify_kernel_ppu.cuh"

template <typename StateT, int RowSplits>
void launch_verify_typed(cutlass::bfloat16_t const *q, cutlass::bfloat16_t const *k,
                         cutlass::bfloat16_t const *v, cutlass::bfloat16_t const *gate,
                         cutlass::bfloat16_t const *beta, float const *A_log, float const *dt_bias,
                         float scale, float lower_bound, StateT const *state,
                         int32_t const *state_indices, StateT *intermediate_states,
                         int32_t const *intermediate_state_indices, cutlass::bfloat16_t *out, int B,
                         int T, int H, int64_t q_token_stride, int64_t k_token_stride,
                         int64_t v_token_stride, int64_t gate_token_stride,
                         int64_t state_slot_stride, int64_t intermediate_slot_stride,
                         int64_t intermediate_step_stride, int state_slots, int intermediate_slots,
                         hggcStream_t stream) {
    constexpr int kRows = 128 / RowSplits;
    constexpr int state_bytes = sizeof(StateT) * kRows * 128;
    constexpr bool double_scratch =
        sizeof(StateT) == sizeof(flash_kda_ppu_verify::BF16) && RowSplits == 4;
    constexpr int scratch_bytes =
        (double_scratch ? 2 : 1) * flash_kda_ppu_verify::VerifyScratch<kRows>::kStrideBytes;
    constexpr int smem_bytes = state_bytes > scratch_bytes ? state_bytes : scratch_bytes;
    flash_kda_ppu_verify::flash_kda_ppu_verify_kernel<StateT, RowSplits>
        <<<dim3(RowSplits, H, B), dim3(32, flash_kda_ppu_verify::kWarps), smem_bytes, stream>>>(
            q, k, v, gate, beta, A_log, dt_bias, scale, lower_bound, state, state_indices,
            intermediate_states, intermediate_state_indices, out, T, H, q_token_stride,
            k_token_stride, v_token_stride, gate_token_stride, state_slot_stride,
            intermediate_slot_stride, intermediate_step_stride, state_slots, intermediate_slots);
}

void launch_verify(cutlass::bfloat16_t const *q, cutlass::bfloat16_t const *k,
                   cutlass::bfloat16_t const *v, cutlass::bfloat16_t const *gate,
                   cutlass::bfloat16_t const *beta, float const *A_log, float const *dt_bias,
                   float scale, float lower_bound, void const *state, bool state_bf16,
                   int32_t const *state_indices, void *intermediate_states,
                   int32_t const *intermediate_state_indices, cutlass::bfloat16_t *out, int B,
                   int T, int H, int64_t q_token_stride, int64_t k_token_stride,
                   int64_t v_token_stride, int64_t gate_token_stride, int64_t state_slot_stride,
                   int64_t intermediate_slot_stride, int64_t intermediate_step_stride,
                   int state_slots, int intermediate_slots, hggcStream_t stream) {
    // Small DSpark batches benefit from extra row CTAs; wider batches already
    // expose enough request/head parallelism and amortize a larger state tile.
    int const bh = B * H;
#define LAUNCH_VERIFY(TY, SPLIT)                                                                   \
    launch_verify_typed<TY, SPLIT>(                                                                \
        q, k, v, gate, beta, A_log, dt_bias, scale, lower_bound,                                   \
        reinterpret_cast<TY const *>(state), state_indices,                                        \
        reinterpret_cast<TY *>(intermediate_states), intermediate_state_indices, out, B, T, H,     \
        q_token_stride, k_token_stride, v_token_stride, gate_token_stride, state_slot_stride,      \
        intermediate_slot_stride, intermediate_step_stride, state_slots, intermediate_slots,       \
        stream)
    if (state_bf16) {
        if (bh <= 12)
            LAUNCH_VERIFY(flash_kda_ppu_verify::BF16, 16);
        else if (bh <= 96)
            LAUNCH_VERIFY(flash_kda_ppu_verify::BF16, 8);
        else
            LAUNCH_VERIFY(flash_kda_ppu_verify::BF16, 4);
    } else {
        if (bh <= 12)
            LAUNCH_VERIFY(float, 16);
        else if (bh <= 192)
            LAUNCH_VERIFY(float, 8);
        else
            LAUNCH_VERIFY(float, 4);
    }
#undef LAUNCH_VERIFY
}
