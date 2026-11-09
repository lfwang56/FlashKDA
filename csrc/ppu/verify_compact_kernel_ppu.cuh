#pragma once

#include "verify_kernel_ppu.cuh"
#include "verify_compact_launch.h"

namespace flash_kda_ppu_verify {
namespace {

// The compact cache stores recurrence operands instead of a full matrix at
// every step. Its row decomposition and reductions match FP32 dense verify.
constexpr int kCompactRows = kD / 8;
constexpr int kCompactRowsPerWarp = kCompactRows / kWarps;

template <int Rows>
CUTE_DEVICE void advance_compact_rows(float4 (&state)[Rows], float4 key, float4 decay,
                                      BF16 const *values, float beta, int row_base, int warp) {
    float prediction[Rows];
    CUTE_UNROLL
    for (int row = 0; row < Rows; ++row) {
        float4 &s = state[row];
        s.x *= decay.x;
        s.y *= decay.y;
        s.z *= decay.z;
        s.w *= decay.w;
        prediction[row] = s.x * key.x + s.y * key.y + s.z * key.z + s.w * key.w;
    }
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
        CUTE_UNROLL
        for (int row = 0; row < Rows; ++row)
            prediction[row] += __shfl_xor_sync(0xffffffffu, prediction[row], offset);
    }
    CUTE_UNROLL
    for (int row = 0; row < Rows; ++row) {
        float const delta =
            (float(values[row_base + warp + row * kWarps]) - prediction[row]) * beta;
        float4 &s = state[row];
        s.x += key.x * delta;
        s.y += key.y * delta;
        s.z += key.z * delta;
        s.w += key.w * delta;
    }
}

template <typename StateT>
__global__ void flash_kda_ppu_verify_compact_kernel(
    BF16 const *q, BF16 const *k, BF16 const *v, BF16 const *gate, BF16 const *beta,
    float const *A_log, float const *dt_bias, StateT const *state, int const *state_indices,
    StateT *end_state, int const *end_indices, BF16 *out, float *ring_k, float *ring_decay,
    BF16 *ring_v, float *ring_beta, float scale, float lower_bound, int T, int H, int64_t qs,
    int64_t ks, int64_t vs, int64_t gs, int state_slots, int end_slots) {
    int const batch = blockIdx.z, head = blockIdx.y;
    int const row_base = blockIdx.x * kCompactRows;
    int const warp = threadIdx.y, lane = threadIdx.x, tid = warp * 32 + lane;
    int const col = lane * 4;
    int const source = state_indices[batch], destination = end_indices[batch];
    if (source < 0 || source >= state_slots || destination >= end_slots) {
        for (int t = 0; t < T; ++t) {
            for (int row = tid; row < kCompactRows; row += kWarps * 32)
                out[((batch * T + t) * H + head) * kD + row_base + row] = BF16(0.f);
        }
        return;
    }
    StateT const *initial = state + (int64_t(source) * H + head) * kD * kD + row_base * kD;
    float4 resident[kCompactRowsPerWarp];
    CUTE_UNROLL
    for (int row = 0; row < kCompactRowsPerWarp; ++row)
        resident[row] = load_state4(initial + (warp + row * kWarps) * kD + col);

    extern __shared__ unsigned char scratch_bytes[];
    auto scratch = get_verify_scratch<kCompactRows>(scratch_bytes, 0);
    float const exp_A = __shfl_sync(0xffffffffu, lane == 0 ? __expf(A_log[head]) : 0.f, 0);
    for (int t = 0; t < T; ++t) {
        produce_token<kCompactRows>(q, k, v, gate, beta, dt_bias, exp_A, scale, lower_bound,
                                    batch * T + t, head, H, qs, ks, vs, gs, row_base, warp, lane,
                                    scratch.q, scratch.k, scratch.decay, scratch.beta, scratch.v);
        __syncthreads();
        if (destination >= 0) {
            int64_t const offset = ((int64_t(destination) * T + t) * H + head) * kD;
            if (blockIdx.x == 0) {
                ring_k[offset + tid] = scratch.k[tid];
                ring_decay[offset + tid] = scratch.decay[tid];
                if (tid == 0)
                    ring_beta[offset / kD] = scratch.beta[0];
            }
            if (tid < kCompactRows)
                ring_v[offset + row_base + tid] = scratch.v[tid];
        }
        float4 const key = *reinterpret_cast<float4 const *>(scratch.k + col);
        float4 const decay = *reinterpret_cast<float4 const *>(scratch.decay + col);
        float4 const query = *reinterpret_cast<float4 const *>(scratch.q + col);
        advance_compact_rows(resident, key, decay, scratch.v, scratch.beta[0], 0, warp);
        float result[kCompactRowsPerWarp];
        CUTE_UNROLL
        for (int row = 0; row < kCompactRowsPerWarp; ++row) {
            float4 const s = resident[row];
            result[row] = s.x * query.x + s.y * query.y + s.z * query.z + s.w * query.w;
            if (destination >= 0 && t + 1 == T) {
                StateT *last = end_state + (int64_t(destination) * H + head) * kD * kD;
                store_state4_global(last + (row_base + warp + row * kWarps) * kD + col, s);
            }
        }
        CUTE_UNROLL
        for (int offset = 16; offset > 0; offset >>= 1) {
            CUTE_UNROLL
            for (int row = 0; row < kCompactRowsPerWarp; ++row)
                result[row] += __shfl_down_sync(0xffffffffu, result[row], offset);
        }
        if (lane == 0) {
            CUTE_UNROLL
            for (int row = 0; row < kCompactRowsPerWarp; ++row)
                out[((batch * T + t) * H + head) * kD + row_base + warp + row * kWarps] =
                    BF16(result[row]);
        }
        __syncthreads();
    }
}

// Large grids prepare operands once per token/head; row CTAs then run without
// CTA barriers or repeated normalization/exponentiation. Scratch is indexed by
// request, while persistent rings are indexed by the caller's destination slot.
__global__ void prepare_compact_operands(BF16 const *q, BF16 const *k, BF16 const *v,
                                         BF16 const *gate, BF16 const *beta, float const *A_log,
                                         float const *dt_bias, CompactStateArgs a, float *prepared,
                                         float scale, float lower_bound, int B, int T, int H,
                                         int64_t qs, int64_t ks, int64_t vs, int64_t gs) {
    int const token = blockIdx.x * kWarps + threadIdx.y;
    if (token >= B * T)
        return;
    int const batch = token / T, head = blockIdx.y, t = token % T;
    int const source = a.state_indices[batch], dest = a.end_indices[batch];
    if (source < 0 || source >= a.state_slots || dest >= a.end_slots)
        return;
    int const lane = threadIdx.x, half = lane / 16, col = (lane % 16) * 8;
    int64_t const record = int64_t(token) * H + head;
    float *p = prepared + record * kCompactOperandStride;
    int64_t const cache = ((int64_t(dest) * T + t) * H + head) * kD;
    float const exp_A = __shfl_sync(0xffffffffu, lane == 0 ? __expf(A_log[head]) : 0.f, 0);
    BF16 const *src =
        half == 0 ? q + int64_t(token) * qs + head * kD : k + int64_t(token) * ks + head * kD;
    float4 x0 = load_bf16x4(src + col), x1 = load_bf16x4(src + col + 4);
    float sq = x0.x * x0.x + x0.y * x0.y + x0.z * x0.z + x0.w * x0.w + x1.x * x1.x + x1.y * x1.y +
               x1.z * x1.z + x1.w * x1.w;
    float inv = rsqrtf(half_warp_sum_all(sq) + 1.e-6f);
    inv *= half == 0 ? scale : 1.f;
    x0.x *= inv;
    x0.y *= inv;
    x0.z *= inv;
    x0.w *= inv;
    x1.x *= inv;
    x1.y *= inv;
    x1.z *= inv;
    x1.w *= inv;
    float *operand = p + half * kD + col;
    *reinterpret_cast<float4 *>(operand) = x0;
    *reinterpret_cast<float4 *>(operand + 4) = x1;
    if (half == 1 && dest >= 0) {
        *reinterpret_cast<float4 *>(a.ring_k + cache + col) = x0;
        *reinterpret_cast<float4 *>(a.ring_k + cache + col + 4) = x1;
    }
    CUTE_UNROLL
    for (int j = 0; j < 2; ++j) {
        int const c = lane * 4 + j * 2;
        float2 const g = bf16x2_to_float2(
            *reinterpret_cast<uint32_t const *>(gate + int64_t(token) * gs + head * kD + c));
        float2 const bias = *reinterpret_cast<float2 const *>(dt_bias + head * kD + c);
        float2 const d = make_float2(__expf(lower_bound * __ppu_sgmdf(exp_A * (g.x + bias.x))),
                                     __expf(lower_bound * __ppu_sgmdf(exp_A * (g.y + bias.y))));
        *reinterpret_cast<float2 *>(p + 2 * kD + c) = d;
        if (dest >= 0)
            *reinterpret_cast<float2 *>(a.ring_decay + cache + c) = d;
    }
    if (dest >= 0)
        *reinterpret_cast<uint2 *>(a.ring_v + cache + lane * 4) =
            *reinterpret_cast<uint2 const *>(v + int64_t(token) * vs + head * kD + lane * 4);
    if (lane == 0) {
        float const b = __ppu_sgmdf(float(beta[record]));
        p[3 * kD] = b;
        if (dest >= 0)
            a.ring_beta[cache / kD] = b;
    }
}

template <typename StateT, int Rows>
__global__ void recurrent_compact_operands(CompactStateArgs a, float const *prepared, BF16 const *v,
                                           BF16 *out, int T, int H, int64_t vs) {
    constexpr int R = Rows / 8;
    int const batch = blockIdx.z, head = blockIdx.y, row_base = blockIdx.x * Rows;
    int const lane = threadIdx.x % 16, subwarp = threadIdx.y * 2 + threadIdx.x / 16, col = lane * 4;
    int const source = a.state_indices[batch], dest = a.end_indices[batch];
    if (source < 0 || source >= a.state_slots || dest >= a.end_slots) {
        if (lane == 0) {
            for (int t = 0; t < T; ++t) {
                CUTE_UNROLL
                for (int j = 0; j < R; ++j)
                    out[((batch * T + t) * H + head) * kD + row_base + subwarp + j * 8] = BF16(0.f);
            }
        }
        return;
    }
    StateT const *initial = static_cast<StateT const *>(a.state) +
                            (int64_t(source) * H + head) * kD * kD + row_base * kD;
    float4 lo[R], hi[R];
    CUTE_UNROLL
    for (int j = 0; j < R; ++j) {
        lo[j] = load_state4(initial + (subwarp + j * 8) * kD + col);
        hi[j] = load_state4(initial + (subwarp + j * 8) * kD + col + 64);
    }
    for (int t = 0; t < T; ++t) {
        int const token = batch * T + t;
        float const *p = prepared + (int64_t(token) * H + head) * kCompactOperandStride;
        float4 const kl = *reinterpret_cast<float4 const *>(p + kD + col);
        float4 const kh = *reinterpret_cast<float4 const *>(p + kD + col + 64);
        float4 const dl = *reinterpret_cast<float4 const *>(p + 2 * kD + col);
        float4 const dh = *reinterpret_cast<float4 const *>(p + 2 * kD + col + 64);
        float4 const ql = *reinterpret_cast<float4 const *>(p + col);
        float4 const qh = *reinterpret_cast<float4 const *>(p + col + 64);
        float prediction[R], result[R];
        CUTE_UNROLL
        for (int j = 0; j < R; ++j) {
            lo[j].x *= dl.x;
            lo[j].y *= dl.y;
            lo[j].z *= dl.z;
            lo[j].w *= dl.w;
            hi[j].x *= dh.x;
            hi[j].y *= dh.y;
            hi[j].z *= dh.z;
            hi[j].w *= dh.w;
            float const l = lo[j].x * kl.x + lo[j].y * kl.y + lo[j].z * kl.z + lo[j].w * kl.w;
            float const h = hi[j].x * kh.x + hi[j].y * kh.y + hi[j].z * kh.z + hi[j].w * kh.w;
            // Preserve dense verify's XOR16, then XOR8/4/2/1 reduction order.
            prediction[j] = l + h;
        }
        CUTE_UNROLL
        for (int offset = 8; offset > 0; offset >>= 1) {
            CUTE_UNROLL
            for (int j = 0; j < R; ++j)
                prediction[j] += __shfl_xor_sync(0xffffffffu, prediction[j], offset, 16);
        }
        CUTE_UNROLL
        for (int j = 0; j < R; ++j) {
            float const delta =
                (float(v[int64_t(token) * vs + head * kD + row_base + subwarp + j * 8]) -
                 prediction[j]) *
                p[3 * kD];
            lo[j].x += kl.x * delta;
            lo[j].y += kl.y * delta;
            lo[j].z += kl.z * delta;
            lo[j].w += kl.w * delta;
            hi[j].x += kh.x * delta;
            hi[j].y += kh.y * delta;
            hi[j].z += kh.z * delta;
            hi[j].w += kh.w * delta;
            float const l = lo[j].x * ql.x + lo[j].y * ql.y + lo[j].z * ql.z + lo[j].w * ql.w;
            float const h = hi[j].x * qh.x + hi[j].y * qh.y + hi[j].z * qh.z + hi[j].w * qh.w;
            result[j] = l + h;
        }
        CUTE_UNROLL
        for (int offset = 8; offset > 0; offset >>= 1) {
            CUTE_UNROLL
            for (int j = 0; j < R; ++j)
                result[j] += __shfl_down_sync(0xffffffffu, result[j], offset, 16);
        }
        if (lane == 0) {
            CUTE_UNROLL
            for (int j = 0; j < R; ++j)
                out[((batch * T + t) * H + head) * kD + row_base + subwarp + j * 8] =
                    BF16(result[j]);
        }
    }
    if (dest >= 0) {
        StateT *last = static_cast<StateT *>(a.end_state) + (int64_t(dest) * H + head) * kD * kD +
                       row_base * kD;
        CUTE_UNROLL
        for (int j = 0; j < R; ++j) {
            store_state4_global(last + (subwarp + j * 8) * kD + col, lo[j]);
            store_state4_global(last + (subwarp + j * 8) * kD + col + 64, hi[j]);
        }
    }
}

template <typename StateT>
__global__ void flash_kda_ppu_commit_compact_kernel(StateT const *state, int const *state_indices,
                                                    StateT *end_state, int const *end_indices,
                                                    float const *ring_k, float const *ring_decay,
                                                    BF16 const *ring_v, float const *ring_beta,
                                                    int const *accepted, int T, int H,
                                                    int state_slots, int end_slots) {
    int const batch = blockIdx.z, head = blockIdx.y;
    int const row_base = blockIdx.x * kCompactRows;
    int const warp = threadIdx.y, lane = threadIdx.x, col = lane * 4;
    int const source = state_indices[batch], destination = end_indices[batch];
    int const count = accepted[batch];
    if (source < 0 || source >= state_slots || destination < 0 || destination >= end_slots ||
        count >= T)
        return;
    StateT const *initial = state + (int64_t(source) * H + head) * kD * kD + row_base * kD;
    float4 resident[kCompactRowsPerWarp];
    CUTE_UNROLL
    for (int row = 0; row < kCompactRowsPerWarp; ++row)
        resident[row] = load_state4(initial + (warp + row * kWarps) * kD + col);
    for (int t = 0; t < count; ++t) {
        int64_t const offset = ((int64_t(destination) * T + t) * H + head) * kD;
        float4 const key = *reinterpret_cast<float4 const *>(ring_k + offset + col);
        float4 const decay = *reinterpret_cast<float4 const *>(ring_decay + offset + col);
        advance_compact_rows(resident, key, decay, ring_v + offset, ring_beta[offset / kD],
                             row_base, warp);
    }
    StateT *destination_head = end_state + (int64_t(destination) * H + head) * kD * kD;
    CUTE_UNROLL
    for (int row = 0; row < kCompactRowsPerWarp; ++row)
        store_state4_global(destination_head + (row_base + warp + row * kWarps) * kD + col,
                            resident[row]);
}

} // namespace
} // namespace flash_kda_ppu_verify
