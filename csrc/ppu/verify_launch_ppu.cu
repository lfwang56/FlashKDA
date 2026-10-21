#include "launch.h"

#include "ppu_aiu.cuh"
#include "ppu_common.cuh"

// This is intentionally a separate kernel from packed decode.  A target-verify
// launch must never write the committed state pool: it evolves a private state
// tile in shared memory and checkpoints the tile after every draft token.  The
// dense topk=1/DSpark path is a chain, so a row tile has no dependence on the
// other CTAs that own the same [request, head] state.
namespace flash_kda_ppu_verify {
namespace {

constexpr int kD = 128;
constexpr int kWarps = 4;
using BF16 = flash_kda_ppu::BF16;

#ifdef __HGGC_ARCH__
typedef float v4float __attribute__((ext_vector_type(4)));
#define FLASH_KDA_GLOBAL __attribute__((address_space(1)))
#endif

CUTE_DEVICE float2 bf16x2_to_float2(uint32_t raw) {
    return __bfloat1622float2(*reinterpret_cast<__ppu_bfloat162 const *>(&raw));
}

CUTE_DEVICE float4 load_bf16x4(BF16 const *ptr) {
    uint2 raw = *reinterpret_cast<uint2 const *>(ptr);
    float2 lo = bf16x2_to_float2(raw.x);
    float2 hi = bf16x2_to_float2(raw.y);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
}

template <typename StateT> CUTE_DEVICE float4 load_state4(StateT const *ptr) {
    if constexpr (sizeof(StateT) == sizeof(float)) {
        return *reinterpret_cast<float4 const *>(ptr);
    } else {
        return load_bf16x4(reinterpret_cast<BF16 const *>(ptr));
    }
}

template <typename StateT> CUTE_DEVICE void store_state4_global(StateT *ptr, float4 value) {
    if constexpr (sizeof(StateT) == sizeof(float)) {
#ifdef __HGGC_ARCH__
        *(FLASH_KDA_GLOBAL v4float *)(ptr) = *(v4float *)&value;
#else
        *reinterpret_cast<float4 *>(ptr) = value;
#endif
    } else {
        __ppu_bfloat162 lo = __floats2bfloat162_rn(value.x, value.y);
        __ppu_bfloat162 hi = __floats2bfloat162_rn(value.z, value.w);
        uint2 raw =
            make_uint2(*reinterpret_cast<uint32_t *>(&lo), *reinterpret_cast<uint32_t *>(&hi));
        *reinterpret_cast<uint2 *>(ptr) = raw;
    }
}

CUTE_DEVICE void store_bf16_state8_global(BF16 *ptr, float4 lo, float4 hi) {
    __ppu_bfloat162 x0 = __floats2bfloat162_rn(lo.x, lo.y);
    __ppu_bfloat162 x1 = __floats2bfloat162_rn(lo.z, lo.w);
    __ppu_bfloat162 x2 = __floats2bfloat162_rn(hi.x, hi.y);
    __ppu_bfloat162 x3 = __floats2bfloat162_rn(hi.z, hi.w);
    uint4 raw = make_uint4(*reinterpret_cast<uint32_t *>(&x0), *reinterpret_cast<uint32_t *>(&x1),
                           *reinterpret_cast<uint32_t *>(&x2), *reinterpret_cast<uint32_t *>(&x3));
#ifdef __HGGC_ARCH__
    __stcg(reinterpret_cast<uint4 *>(ptr), raw);
#else
    *reinterpret_cast<uint4 *>(ptr) = raw;
#endif
}

CUTE_DEVICE void half_warp_sum_all_pair(float &x0, float &x1) {
    CUTE_UNROLL
    for (int offset = 8; offset > 0; offset >>= 1) {
        x0 += __shfl_xor_sync(0xffffffff, x0, offset, 16);
        x1 += __shfl_xor_sync(0xffffffff, x1, offset, 16);
    }
}

CUTE_DEVICE void half_warp_sum_lane0_pair(float &x0, float &x1) {
    CUTE_UNROLL
    for (int offset = 8; offset > 0; offset >>= 1) {
        x0 += __shfl_down_sync(0xffffffff, x0, offset, 16);
        x1 += __shfl_down_sync(0xffffffff, x1, offset, 16);
    }
}

CUTE_DEVICE float half_warp_sum_all(float x) {
    CUTE_UNROLL
    for (int offset = 8; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 16);
    }
    return x;
}

CUTE_DEVICE float half_warp_sum_lane0(float x) {
    CUTE_UNROLL
    for (int offset = 8; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset, 16);
    }
    return x;
}

template <int Rows> struct VerifyScratch {
    static constexpr int kFloatStride = Rows == 32 ? kD + 4 : kD;
    static constexpr int kPayloadBytes =
        sizeof(float) * (3 * kFloatStride + 4) + sizeof(BF16) * Rows;
    static constexpr int kStrideBytes = (kPayloadBytes + 15) & ~15;
    float *q;
    float *k;
    float *decay;
    float *beta;
    BF16 *v;
};

template <int Rows>
CUTE_DEVICE VerifyScratch<Rows> get_verify_scratch(unsigned char *raw_smem, int token) {
    unsigned char *base = raw_smem + (token & 1) * VerifyScratch<Rows>::kStrideBytes;
    float *q = reinterpret_cast<float *>(base);
    float *k = q + VerifyScratch<Rows>::kFloatStride;
    float *decay = k + VerifyScratch<Rows>::kFloatStride;
    float *beta = decay + VerifyScratch<Rows>::kFloatStride;
    BF16 *v = reinterpret_cast<BF16 *>(beta + 4);
    return {q, k, decay, beta, v};
}

// State tiles are copied by the ppu0015 AIU into ordinary shared memory.  Do
// not use an AIU swizzled destination here: the recurrent loop needs scalar
// element addressing for the same tile at every verify step.
template <typename StateT, int Rows>
CUTE_DEVICE void aiu_load_rows(StateT *smem, StateT const *gmem) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    constexpr int kDimC = kD;
    constexpr int kDimW = Rows;
    constexpr int kDimN = 1;
    constexpr int kCubeC = kD;
    constexpr int kCubeW = Rows;
    constexpr int kCubeN = 1;
    constexpr int64_t kStrideW = kD * sizeof(StateT);
    constexpr int64_t kStrideN = Rows * kD * sizeof(StateT);
    if constexpr (sizeof(StateT) == sizeof(float)) {
        asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b32"
                     "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, "
                     "{%10, %11, %12}, %13;\n" ::"r"(smem_addr),
                     "l"(gmem), "r"(kDimC), "r"(kDimW), "r"(kDimN), "r"(kCubeC), "r"(kCubeW),
                     "r"(kCubeN), "l"(kStrideW), "l"(kStrideN), "r"(0), "r"(0), "r"(0), "r"(0));
    } else {
        asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
                     "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, "
                     "{%10, %11, %12}, %13;\n" ::"r"(smem_addr),
                     "l"(gmem), "r"(kDimC), "r"(kDimW), "r"(kDimN), "r"(kCubeC), "r"(kCubeW),
                     "r"(kCubeN), "l"(kStrideW), "l"(kStrideN), "r"(0), "r"(0), "r"(0), "r"(0));
    }
}

CUTE_DEVICE float warp_sum_all(float x) {
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset);
    }
    return x;
}

CUTE_DEVICE float warp_sum_lane0(float x) {
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

template <int Rows>
CUTE_DEVICE void
produce_token(BF16 const *q, BF16 const *k, BF16 const *v, BF16 const *gate, BF16 const *beta,
              float const *dt_bias, float exp_A, float scale, float lower_bound, int token,
              int head, int H, int64_t q_token_stride, int64_t k_token_stride,
              int64_t v_token_stride, int64_t gate_token_stride, int row_base, int warp, int lane,
              float *sm_q, float *sm_k, float *sm_decay, float *sm_beta, BF16 *sm_v) {
    // The caller passes token as the already packed B*T token index.
    int64_t const q_base = int64_t(token) * q_token_stride + head * kD;
    int64_t const k_base = int64_t(token) * k_token_stride + head * kD;
    int64_t const v_base = int64_t(token) * v_token_stride + head * kD;
    int64_t const gate_base = int64_t(token) * gate_token_stride + head * kD;
    if (warp == 0) {
        // Run the independent Q and K reductions in the two 16-lane groups.
        // This frees a second warp for the SFU-heavy gate/decay transform.
        int const half = lane >> 4;
        int const half_lane = lane & 15;
        int const col = half_lane * 8;
        BF16 const *src = half == 0 ? q + q_base : k + k_base;
        float *dst = half == 0 ? sm_q : sm_k;
        float4 x0 = load_bf16x4(src + col);
        float4 x1 = load_bf16x4(src + col + 4);
        float sq = x0.x * x0.x + x0.y * x0.y + x0.z * x0.z + x0.w * x0.w + x1.x * x1.x +
                   x1.y * x1.y + x1.z * x1.z + x1.w * x1.w;
        float inv = rsqrtf(half_warp_sum_all(sq) + 1.0e-6f);
        inv *= half == 0 ? scale : 1.0f;
        x0.x *= inv;
        x0.y *= inv;
        x0.z *= inv;
        x0.w *= inv;
        x1.x *= inv;
        x1.y *= inv;
        x1.z *= inv;
        x1.w *= inv;
        *reinterpret_cast<float4 *>(dst + col) = x0;
        *reinterpret_cast<float4 *>(dst + col + 4) = x1;
    } else if (warp == 1 || warp == 2) {
        int const col = (warp - 1) * 64 + lane * 2;
        float2 a = bf16x2_to_float2(*reinterpret_cast<uint32_t const *>(gate + gate_base + col));
        float2 dt = *reinterpret_cast<float2 const *>(dt_bias + head * kD + col);
        float2 d = make_float2(__expf(lower_bound * __ppu_sgmdf(exp_A * (a.x + dt.x))),
                               __expf(lower_bound * __ppu_sgmdf(exp_A * (a.y + dt.y))));
        *reinterpret_cast<float2 *>(sm_decay + col) = d;
        if (warp == 1 && lane == 0) {
            sm_beta[0] = __ppu_sgmdf(float(beta[token * H + head]));
        }
    } else if (warp == 3) {
        // The scratch tail is 16-byte aligned without increasing its rounded
        // allocation, so copy eight BF16 V values per active lane.
        if (lane < Rows / 8) {
            int const row = lane * 8;
            *reinterpret_cast<uint4 *>(sm_v + row) =
                *reinterpret_cast<uint4 const *>(v + v_base + row_base + row);
        }
    }
}

template <typename StateT, int RowSplits>
__global__ void flash_kda_ppu_verify_kernel(
    BF16 const *__restrict__ q, BF16 const *__restrict__ k, BF16 const *__restrict__ v,
    BF16 const *__restrict__ gate, BF16 const *__restrict__ beta, float const *__restrict__ A_log,
    float const *__restrict__ dt_bias, float scale, float lower_bound,
    StateT const *__restrict__ state, int32_t const *__restrict__ state_indices,
    StateT *__restrict__ intermediate_states,
    int32_t const *__restrict__ intermediate_state_indices, BF16 *__restrict__ out, int T, int H,
    int64_t q_token_stride, int64_t k_token_stride, int64_t v_token_stride,
    int64_t gate_token_stride, int64_t state_slot_stride, int64_t intermediate_slot_stride,
    int64_t intermediate_step_stride, int state_slots, int intermediate_slots) {
    constexpr int kRows = kD / RowSplits;
    int const batch = blockIdx.z;
    int const head = blockIdx.y;
    int const row_base = blockIdx.x * kRows;
    int const warp = threadIdx.y;
    int const lane = threadIdx.x;
    int const tid = warp * 32 + lane;
    int const col = lane * 4;
    int const slot = state_indices[batch];
    int const intermediate_slot = intermediate_state_indices[batch];
    constexpr bool kDoubleScratch = sizeof(StateT) == sizeof(BF16) && RowSplits == 4;

    extern __shared__ unsigned char raw_smem[];
    StateT *sm_state = reinterpret_cast<StateT *>(raw_smem);
    // The state tile is needed only for the initial AIU load. Once every warp
    // has moved its rows into registers, reuse the allocation for two aligned
    // producer-scratch stages.

    if (slot < 0 || slot >= state_slots || intermediate_slot >= intermediate_slots) {
        for (int t = 0; t < T; ++t) {
            BF16 *out_token = out + ((batch * T + t) * H + head) * kD;
            for (int d = tid; d < kRows; d += kWarps * 32)
                out_token[row_base + d] = BF16(0.0f);
        }
        return;
    }

    StateT const *state_head =
        state + int64_t(slot) * state_slot_stride + int64_t(head) * kD * kD + row_base * kD;
    if (warp == 0)
        aiu_load_rows<StateT, kRows>(sm_state, state_head);
    flash_kda_ppu::cp_async_fence_all();
    flash_kda_ppu::cp_async_wait_all<0>();
    __syncthreads();

    if constexpr (sizeof(StateT) == sizeof(BF16) && RowSplits == 16) {
        // The eight-row tile gives each warp exactly two rows. Assign one row
        // to each 16-lane group so the register footprint stays identical to
        // the full-warp path while reductions and BF16 stores become wider.
        int const half = lane >> 4;
        int const half_lane = lane & 15;
        int const col0 = half_lane * 8;
        int const row = warp * 2 + half;
        float4 h0 = load_state4(sm_state + row * kD + col0);
        float4 h1 = load_state4(sm_state + row * kD + col0 + 4);
        __syncthreads();

        float const exp_A = __shfl_sync(0xffffffff, lane == 0 ? __expf(A_log[head]) : 0.0f, 0);
        for (int t = 0; t < T; ++t) {
            auto scratch = get_verify_scratch<kRows>(raw_smem, kDoubleScratch ? t : 0);
            float *sm_q = scratch.q;
            float *sm_k = scratch.k;
            float *sm_decay = scratch.decay;
            float *sm_beta = scratch.beta;
            BF16 *sm_v = scratch.v;
            produce_token<kRows>(q, k, v, gate, beta, dt_bias, exp_A, scale, lower_bound,
                                 batch * T + t, head, H, q_token_stride, k_token_stride,
                                 v_token_stride, gate_token_stride, row_base, warp, lane, sm_q,
                                 sm_k, sm_decay, sm_beta, sm_v);
            __syncthreads();

            float4 q0 = *reinterpret_cast<float4 const *>(sm_q + col0);
            float4 q1 = *reinterpret_cast<float4 const *>(sm_q + col0 + 4);
            float4 k0 = *reinterpret_cast<float4 const *>(sm_k + col0);
            float4 k1 = *reinterpret_cast<float4 const *>(sm_k + col0 + 4);
            float4 d0 = *reinterpret_cast<float4 const *>(sm_decay + col0);
            float4 d1 = *reinterpret_cast<float4 const *>(sm_decay + col0 + 4);
            float const beta_value = sm_beta[0];
            h0.x *= d0.x;
            h0.y *= d0.y;
            h0.z *= d0.z;
            h0.w *= d0.w;
            h1.x *= d1.x;
            h1.y *= d1.y;
            h1.z *= d1.z;
            h1.w *= d1.w;
            float hk = h0.x * k0.x + h0.y * k0.y + h0.z * k0.z + h0.w * k0.w + h1.x * k1.x +
                       h1.y * k1.y + h1.z * k1.z + h1.w * k1.w;
            hk = half_warp_sum_all(hk);
            float const v_new = (float(sm_v[row]) - hk) * beta_value;
            h0.x += k0.x * v_new;
            h0.y += k0.y * v_new;
            h0.z += k0.z * v_new;
            h0.w += k0.w * v_new;
            h1.x += k1.x * v_new;
            h1.y += k1.y * v_new;
            h1.z += k1.z * v_new;
            h1.w += k1.w * v_new;
            if (intermediate_slot >= 0) {
                BF16 *checkpoint_head = reinterpret_cast<BF16 *>(
                    intermediate_states + int64_t(intermediate_slot) * intermediate_slot_stride +
                    int64_t(t) * intermediate_step_stride + int64_t(head) * kD * kD +
                    row_base * kD);
                store_bf16_state8_global(checkpoint_head + row * kD + col0, h0, h1);
            }
            float hq = h0.x * q0.x + h0.y * q0.y + h0.z * q0.z + h0.w * q0.w + h1.x * q1.x +
                       h1.y * q1.y + h1.z * q1.z + h1.w * q1.w;
            hq = half_warp_sum_lane0(hq);
            if (half_lane == 0) {
                out[((batch * T + t) * H + head) * kD + row_base + row] = BF16(hq);
            }
            if constexpr (!kDoubleScratch)
                __syncthreads();
        }
    } else if constexpr (sizeof(StateT) == sizeof(BF16) && (RowSplits == 4 || RowSplits == 8)) {
        // Two independent 16-lane groups share each warp. Each lane owns eight
        // contiguous columns for two rows, preserving the original per-thread
        // FP32 state count while halving shuffle depth and issuing natural
        // aligned 16-byte BF16 checkpoint packets.
        constexpr int kRowsPerWarp = kRows / kWarps;
        constexpr int kGroupsPerWarp = kRowsPerWarp / 4;
        int const half = lane >> 4;
        int const half_lane = lane & 15;
        int const col0 = half_lane * 8;
        float4 h00[kGroupsPerWarp];
        float4 h01[kGroupsPerWarp];
        float4 h10[kGroupsPerWarp];
        float4 h11[kGroupsPerWarp];
        CUTE_UNROLL
        for (int group = 0; group < kGroupsPerWarp; ++group) {
            int const row0 = group * 4 * kWarps + warp * 2 + half;
            int const row1 = row0 + 2 * kWarps;
            h00[group] = load_state4(sm_state + row0 * kD + col0);
            h01[group] = load_state4(sm_state + row0 * kD + col0 + 4);
            h10[group] = load_state4(sm_state + row1 * kD + col0);
            h11[group] = load_state4(sm_state + row1 * kD + col0 + 4);
        }
        __syncthreads();

        float const exp_A = __shfl_sync(0xffffffff, lane == 0 ? __expf(A_log[head]) : 0.0f, 0);
        for (int t = 0; t < T; ++t) {
            auto scratch = get_verify_scratch<kRows>(raw_smem, kDoubleScratch ? t : 0);
            float *sm_q = scratch.q;
            float *sm_k = scratch.k;
            float *sm_decay = scratch.decay;
            float *sm_beta = scratch.beta;
            BF16 *sm_v = scratch.v;
            produce_token<kRows>(q, k, v, gate, beta, dt_bias, exp_A, scale, lower_bound,
                                 batch * T + t, head, H, q_token_stride, k_token_stride,
                                 v_token_stride, gate_token_stride, row_base, warp, lane, sm_q,
                                 sm_k, sm_decay, sm_beta, sm_v);
            __syncthreads();

            float4 q0 = *reinterpret_cast<float4 const *>(sm_q + col0);
            float4 q1 = *reinterpret_cast<float4 const *>(sm_q + col0 + 4);
            float4 k0 = *reinterpret_cast<float4 const *>(sm_k + col0);
            float4 k1 = *reinterpret_cast<float4 const *>(sm_k + col0 + 4);
            float4 d0 = *reinterpret_cast<float4 const *>(sm_decay + col0);
            float4 d1 = *reinterpret_cast<float4 const *>(sm_decay + col0 + 4);
            float const beta_value = sm_beta[0];
            BF16 *checkpoint_head = nullptr;
            if (intermediate_slot >= 0) {
                checkpoint_head = reinterpret_cast<BF16 *>(
                    intermediate_states + int64_t(intermediate_slot) * intermediate_slot_stride +
                    int64_t(t) * intermediate_step_stride + int64_t(head) * kD * kD +
                    row_base * kD);
            }

            CUTE_UNROLL
            for (int group = 0; group < kGroupsPerWarp; ++group) {
                int const row0 = group * 4 * kWarps + warp * 2 + half;
                int const row1 = row0 + 2 * kWarps;
                float4 &x00 = h00[group];
                float4 &x01 = h01[group];
                float4 &x10 = h10[group];
                float4 &x11 = h11[group];
                x00.x *= d0.x;
                x00.y *= d0.y;
                x00.z *= d0.z;
                x00.w *= d0.w;
                x01.x *= d1.x;
                x01.y *= d1.y;
                x01.z *= d1.z;
                x01.w *= d1.w;
                x10.x *= d0.x;
                x10.y *= d0.y;
                x10.z *= d0.z;
                x10.w *= d0.w;
                x11.x *= d1.x;
                x11.y *= d1.y;
                x11.z *= d1.z;
                x11.w *= d1.w;
                float hk0 = x00.x * k0.x + x00.y * k0.y + x00.z * k0.z + x00.w * k0.w +
                            x01.x * k1.x + x01.y * k1.y + x01.z * k1.z + x01.w * k1.w;
                float hk1 = x10.x * k0.x + x10.y * k0.y + x10.z * k0.z + x10.w * k0.w +
                            x11.x * k1.x + x11.y * k1.y + x11.z * k1.z + x11.w * k1.w;
                half_warp_sum_all_pair(hk0, hk1);
                float const v_new0 = (float(sm_v[row0]) - hk0) * beta_value;
                float const v_new1 = (float(sm_v[row1]) - hk1) * beta_value;
                x00.x += k0.x * v_new0;
                x00.y += k0.y * v_new0;
                x00.z += k0.z * v_new0;
                x00.w += k0.w * v_new0;
                x01.x += k1.x * v_new0;
                x01.y += k1.y * v_new0;
                x01.z += k1.z * v_new0;
                x01.w += k1.w * v_new0;
                x10.x += k0.x * v_new1;
                x10.y += k0.y * v_new1;
                x10.z += k0.z * v_new1;
                x10.w += k0.w * v_new1;
                x11.x += k1.x * v_new1;
                x11.y += k1.y * v_new1;
                x11.z += k1.z * v_new1;
                x11.w += k1.w * v_new1;
                if (checkpoint_head != nullptr) {
                    store_bf16_state8_global(checkpoint_head + row0 * kD + col0, x00, x01);
                    store_bf16_state8_global(checkpoint_head + row1 * kD + col0, x10, x11);
                }
                float hq0 = x00.x * q0.x + x00.y * q0.y + x00.z * q0.z + x00.w * q0.w +
                            x01.x * q1.x + x01.y * q1.y + x01.z * q1.z + x01.w * q1.w;
                float hq1 = x10.x * q0.x + x10.y * q0.y + x10.z * q0.z + x10.w * q0.w +
                            x11.x * q1.x + x11.y * q1.y + x11.z * q1.z + x11.w * q1.w;
                half_warp_sum_lane0_pair(hq0, hq1);
                if (half_lane == 0) {
                    out[((batch * T + t) * H + head) * kD + row_base + row0] = BF16(hq0);
                    out[((batch * T + t) * H + head) * kD + row_base + row1] = BF16(hq1);
                }
            }
            if constexpr (!kDoubleScratch)
                __syncthreads();
        }
    } else {
        // Each warp owns a fixed set of V rows for the complete verify chain.
        // Keeping those rows in FP32 registers removes one shared read and one
        // shared write per state element per draft step. It also matches Triton's
        // recurrence semantics: BF16 is used at pool/checkpoint boundaries, while
        // the private chain state remains FP32 between steps.
        constexpr int kRowsPerWarp = kRows / kWarps;
        float4 h_regs[kRowsPerWarp];
        CUTE_UNROLL
        for (int owned = 0; owned < kRowsPerWarp; ++owned) {
            int const row = warp + owned * kWarps;
            h_regs[owned] = load_state4(sm_state + row * kD + col);
        }
        __syncthreads();

        float const exp_A = __shfl_sync(0xffffffff, lane == 0 ? __expf(A_log[head]) : 0.0f, 0);

        // Load the committed state once. Every subsequent step updates only the
        // private register tile, then stores it to the speculative checkpoint cache.
        for (int t = 0; t < T; ++t) {
            auto scratch = get_verify_scratch<kRows>(raw_smem, kDoubleScratch ? t : 0);
            float *sm_q = scratch.q;
            float *sm_k = scratch.k;
            float *sm_decay = scratch.decay;
            float *sm_beta = scratch.beta;
            BF16 *sm_v = scratch.v;
            produce_token<kRows>(q, k, v, gate, beta, dt_bias, exp_A, scale, lower_bound,
                                 batch * T + t, head, H, q_token_stride, k_token_stride,
                                 v_token_stride, gate_token_stride, row_base, warp, lane, sm_q,
                                 sm_k, sm_decay, sm_beta, sm_v);
            __syncthreads();

            float4 qv = *reinterpret_cast<float4 const *>(sm_q + col);
            float4 kv = *reinterpret_cast<float4 const *>(sm_k + col);
            float4 decay = *reinterpret_cast<float4 const *>(sm_decay + col);
            float const beta_value = sm_beta[0];
            StateT *checkpoint_head = nullptr;
            if (intermediate_slot >= 0) {
                checkpoint_head =
                    intermediate_states + int64_t(intermediate_slot) * intermediate_slot_stride +
                    int64_t(t) * intermediate_step_stride + int64_t(head) * kD * kD + row_base * kD;
            }

            // Each warp owns disjoint V rows.  The recurrence for one V row is
            // independent of every other row, so no inter-CTA synchronization is
            // needed between draft steps.
            CUTE_UNROLL
            for (int owned = 0; owned < kRowsPerWarp; ++owned) {
                int const row = warp + owned * kWarps;
                float4 &h = h_regs[owned];
                h.x *= decay.x;
                h.y *= decay.y;
                h.z *= decay.z;
                h.w *= decay.w;
                float hk = h.x * kv.x + h.y * kv.y + h.z * kv.z + h.w * kv.w;
                hk = warp_sum_all(hk);
                float const v_new = (float(sm_v[row]) - hk) * beta_value;
                h.x += kv.x * v_new;
                h.y += kv.y * v_new;
                h.z += kv.z * v_new;
                h.w += kv.w * v_new;
                if (checkpoint_head != nullptr) {
                    store_state4_global(checkpoint_head + row * kD + col, h);
                }
                float hq = h.x * qv.x + h.y * qv.y + h.z * qv.z + h.w * qv.w;
                hq = warp_sum_lane0(hq);
                if (lane == 0) {
                    out[((batch * T + t) * H + head) * kD + row_base + row] = BF16(hq);
                }
            }
            if constexpr (!kDoubleScratch)
                __syncthreads();
        }
    }
}

} // namespace
} // namespace flash_kda_ppu_verify

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
