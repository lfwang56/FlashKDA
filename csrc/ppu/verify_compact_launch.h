#pragma once
#include "launch.h"

// FP32 Q, K, decay, plus an aligned beta record per request/token/head.
constexpr int kCompactOperandStride = 3 * 128 + 4;

// Compact verification publishes one terminal state and per-token operands.
// Source and destination indices refer to disjoint live slots owned by serving.
struct CompactStateArgs {
    void const *state;
    int32_t const *state_indices;
    void *end_state;
    int32_t const *end_indices;
    float *ring_k;
    float *ring_decay;
    cutlass::bfloat16_t *ring_v;
    float *ring_beta;
    int state_slots;
    int end_slots;
};

void launch_verify_compact(cutlass::bfloat16_t const *q, cutlass::bfloat16_t const *k,
                           cutlass::bfloat16_t const *v, cutlass::bfloat16_t const *gate,
                           cutlass::bfloat16_t const *beta, float const *A_log,
                           float const *dt_bias, CompactStateArgs state, cutlass::bfloat16_t *out,
                           float *prepared, float scale, float lower_bound, int B, int T, int H,
                           int64_t qs, int64_t ks, int64_t vs, int64_t gs, bool state_bf16,
                           hggcStream_t stream);

void launch_commit_compact(CompactStateArgs state, int32_t const *accepted, int B, int T, int H,
                           bool state_bf16, hggcStream_t stream);
