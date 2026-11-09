#pragma once
#include "launch.h"

// FP32 accumulation is independent of the external state storage dtype.
void launch_fwd_fp32(cutlass::bfloat16_t const *q, cutlass::bfloat16_t const *k,
                     cutlass::bfloat16_t const *v, cutlass::bfloat16_t const *g,
                     cutlass::bfloat16_t const *beta, void const *initial, float scale, void *final,
                     cutlass::bfloat16_t *out, void *workspace, int total_tiles, int tokens, int H,
                     int N, int64_t const *cu, float const *A_log, float const *bias,
                     float gate_scale, hggcStream_t stream, int qs, int ks, int vs, bool has_in,
                     bool has_out, bool state_fp32);
