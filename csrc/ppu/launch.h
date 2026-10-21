#pragma once
#include <cstdint>
#include "../fwd.h"

// Packed token-row strides are private to the PPU implementation.
template <int D, bool HasStateIn, bool HasStateOut, bool StateFP32, bool IsVarlen>
void launch_fwd(cutlass::bfloat16_t const *q_ptr, cutlass::bfloat16_t const *k_ptr,
                cutlass::bfloat16_t const *v_ptr, cutlass::bfloat16_t const *g_bf16_ptr,
                cutlass::bfloat16_t const *beta_ptr, void const *initial_state_ptr, float scale,
                void *final_state_ptr, cutlass::bfloat16_t *out_ptr, void *workspace_ptr,
                int total_tiles, int T_total, int H, int N, int64_t const *cu_seqlens_ptr,
                float const *A_log_ptr, float const *dt_bias_ptr, float gate_scale,
                hggcStream_t stream, int q_stride, int k_stride, int v_stride);

void launch_decode(cutlass::bfloat16_t const *mixed_qkv, cutlass::bfloat16_t const *gate,
                   cutlass::bfloat16_t const *beta, float const *A_log, float const *dt_bias,
                   float scale, float lower_bound, bool use_lower_bound, void *state,
                   bool state_bf16, int32_t const *state_indices, cutlass::bfloat16_t *out, int B,
                   int H, int64_t state_slot_stride, int state_slots, hggcStream_t stream);

// Dense, linear-chain speculative verification.  Unlike decode this keeps the
// committed state read-only and writes every draft token's post-state to the
// caller-provided checkpoint cache.  The state and checkpoint tensors are both
// laid out [slot, H, V=128, K=128]. q/k/v/gate keep each [H, 128]
// token row contiguous but may use a larger token-row stride, as happens for
// views into a fused QKV projection.
void launch_verify(cutlass::bfloat16_t const *q, cutlass::bfloat16_t const *k,
                   cutlass::bfloat16_t const *v, cutlass::bfloat16_t const *gate,
                   cutlass::bfloat16_t const *beta, float const *A_log, float const *dt_bias,
                   float scale, float lower_bound, void const *state, bool state_bf16,
                   int32_t const *state_indices, void *intermediate_states,
                   int32_t const *intermediate_state_indices, cutlass::bfloat16_t *out, int B,
                   int T, int H, int64_t q_token_stride, int64_t k_token_stride,
                   int64_t v_token_stride, int64_t gate_token_stride, int64_t state_slot_stride,
                   int64_t intermediate_slot_stride, int64_t intermediate_step_stride,
                   int state_slots, int intermediate_slots, hggcStream_t stream);
