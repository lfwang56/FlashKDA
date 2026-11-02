#pragma once

#include "launch.h"

#include "ppu_aiu.cuh"
#include "ppu_common.cuh"

namespace flash_kda_ppu {
namespace {

constexpr int kD = 128;
constexpr int kBulkUnsplitWarps = 16;
constexpr int kBulkSplitWarps = 8;
constexpr int kBf16UnsplitWarps = 8;
constexpr int kBulkRowSplits = 2;
constexpr int kWideRollingWarps = 8;
constexpr int kNarrowRollingWarps = 4;
constexpr int kTallRollingWarps = 4;
constexpr int kTallRowSplits = 4;
constexpr int kStages = 4;
constexpr int kBulkRows = 32;
constexpr int kStateElems = kD * kD;

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

CUTE_DEVICE void store_float4_global(float *ptr, float4 value) {
#ifdef __HGGC_ARCH__
    *(FLASH_KDA_GLOBAL v4float *)(ptr) = *(v4float *)&value;
#else
    *reinterpret_cast<float4 *>(ptr) = value;
#endif
}

// PPU warp-bulk store: the hardware derives the 16-byte lane offset, so every
// lane supplies the same row base and a distinct four-float payload.
CUTE_DEVICE void store_float4_global_bulk(float *row_ptr, float4 value) {
#ifdef __HGGC_ARCH__
    uint4 raw = *reinterpret_cast<uint4 const *>(&value);
    __ppu_global_store_bulk_b32x4(row_ptr, raw);
#else
    *reinterpret_cast<float4 *>(row_ptr + (threadIdx.x & 31) * 4) = value;
#endif
}

// Cache-global vector store: keep 16 contiguous bytes per lane while bypassing
// the per-row warp-bulk path and its extra ppu0015 store transaction.
CUTE_DEVICE void store_float4_global_cg(float *ptr, float4 value) {
#ifdef __HGGC_ARCH__
    __stcg(reinterpret_cast<float4 *>(ptr), value);
#else
    *reinterpret_cast<float4 *>(ptr) = value;
#endif
}

template <typename StateT> CUTE_DEVICE float4 load_state4(StateT const *ptr) {
    if constexpr (sizeof(StateT) == sizeof(float)) {
        return *reinterpret_cast<float4 const *>(ptr);
    } else {
        static_assert(sizeof(StateT) == sizeof(BF16));
        return load_bf16x4(reinterpret_cast<BF16 const *>(ptr));
    }
}

template <typename StateT> CUTE_DEVICE void store_state4_global(StateT *ptr, float4 value) {
    if constexpr (sizeof(StateT) == sizeof(float)) {
        store_float4_global(reinterpret_cast<float *>(ptr), value);
    } else {
        static_assert(sizeof(StateT) == sizeof(BF16));
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
    *reinterpret_cast<uint4 *>(ptr) = raw;
}

// ppu0015 AIU: load 32 contiguous fp32 state rows.  The height dimension moves
// 16 KiB with one warp-level descriptor instead of 1024 lane-level requests.
template <typename StateT, int Rows, int StatePitch = kD>
CUTE_DEVICE void aiu_load_rows(StateT *smem, StateT const *gmem, int row_offset) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    constexpr int kDimC = kD;
    constexpr int kDimW = kD;
    constexpr int kDimN = 1;
    constexpr int kCubeC = StatePitch;
    constexpr int kCubeW = Rows;
    constexpr int kCubeN = 1;
    constexpr int64_t kStrideW = kD * sizeof(StateT);
    constexpr int64_t kStrideN = kStateElems * sizeof(StateT);
    if constexpr (sizeof(StateT) == sizeof(float)) {
        asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b32"
                     "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, "
                     "{%10, %11, %12}, %13;\n" ::"r"(smem_addr),
                     "l"(gmem), "r"(kDimC), "r"(kDimW), "r"(kDimN), "r"(kCubeC), "r"(kCubeW),
                     "r"(kCubeN), "l"(kStrideW), "l"(kStrideN), "r"(0), "r"(row_offset), "r"(0),
                     "r"(0));
    } else {
        static_assert(sizeof(StateT) == sizeof(BF16));
        asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.padz.linear.b16"
                     "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, "
                     "{%10, %11, %12}, %13;\n" ::"r"(smem_addr),
                     "l"(gmem), "r"(kDimC), "r"(kDimW), "r"(kDimN), "r"(kCubeC), "r"(kCubeW),
                     "r"(kCubeN), "l"(kStrideW), "l"(kStrideN), "r"(0), "r"(row_offset), "r"(0),
                     "r"(0));
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

CUTE_DEVICE void warp_sum_all_quad(float &x0, float &x1, float &x2, float &x3) {
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
        x0 += __shfl_xor_sync(0xffffffff, x0, offset);
        x1 += __shfl_xor_sync(0xffffffff, x1, offset);
        x2 += __shfl_xor_sync(0xffffffff, x2, offset);
        x3 += __shfl_xor_sync(0xffffffff, x3, offset);
    }
}

CUTE_DEVICE void warp_sum_lane0_quad(float &x0, float &x1, float &x2, float &x3) {
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
        x0 += __shfl_down_sync(0xffffffff, x0, offset);
        x1 += __shfl_down_sync(0xffffffff, x1, offset);
        x2 += __shfl_down_sync(0xffffffff, x2, offset);
        x3 += __shfl_down_sync(0xffffffff, x3, offset);
    }
}

CUTE_DEVICE void warp_sum_all_pair(float &x0, float &x1) {
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
        x0 += __shfl_xor_sync(0xffffffff, x0, offset);
        x1 += __shfl_xor_sync(0xffffffff, x1, offset);
    }
}

CUTE_DEVICE void warp_sum_lane0_pair(float &x0, float &x1) {
    CUTE_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1) {
        x0 += __shfl_down_sync(0xffffffff, x0, offset);
        x1 += __shfl_down_sync(0xffffffff, x1, offset);
    }
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

CUTE_DEVICE float decay_factor(float gate, float dt, float exp_A, float lower_bound,
                               bool use_lower_bound) {
    float x = gate + dt;
    if (use_lower_bound) {
        // Kimi K3 safe gate: lower_bound * sigmoid(exp(A_log) * x).
        return __expf(lower_bound * __ppu_sgmdf(exp_A * x));
    }
    float softplus = x <= 20.0f ? __logf(1.0f + __expf(x)) : x;
    return __expf(-exp_A * softplus);
}

template <typename StateT, bool kUseBulkAiu, int NumWarps, int RowSplits = 1,
          bool kFp32HalfWarp = false, int StatePitch = kD, bool MergeBulkJoins = false>
__global__ void
flash_kda_ppu_decode_kernel(BF16 const *__restrict__ mixed_qkv, BF16 const *__restrict__ gate,
                            BF16 const *__restrict__ beta, float const *__restrict__ A_log,
                            float const *__restrict__ dt_bias, float scale, float lower_bound,
                            bool use_lower_bound, StateT *__restrict__ state,
                            int32_t const *__restrict__ state_indices, BF16 *__restrict__ out,
                            int H, int64_t state_slot_stride, int state_slots) {
    static_assert(StatePitch >= kD && (StatePitch * sizeof(StateT)) % 16 == 0);
    static_assert(kUseBulkAiu || StatePitch == kD);
    // The existing FP32 half-warp specialization uniquely selects the measured
    // 16-warp/two-split large-batch route. Keep legacy kernel signatures intact.
    constexpr bool InterleaveHalfLanes = kFp32HalfWarp;
    // Keep BF16's original physical row halves and reduction tree. Only the
    // measured H12 large-batch route changes its three FP32 parameter arrays.
    constexpr bool TwoPlaneParameters =
        kFp32HalfWarp ||
        (sizeof(StateT) == sizeof(BF16) && kUseBulkAiu && NumWarps == 8 &&
         (RowSplits == 1 || RowSplits == 2) && StatePitch == 136 && MergeBulkJoins);
    static_assert(!InterleaveHalfLanes ||
                  (kUseBulkAiu && kFp32HalfWarp && sizeof(StateT) == sizeof(float) &&
                   NumWarps == 16 && RowSplits == 2));
    constexpr int kWarps = NumWarps;
    constexpr int kRowsPerHead = kD / RowSplits;
    constexpr int kRowsPerWarp = kRowsPerHead / kWarps;
    constexpr int kRowsPerGroupWarp = kBulkRows / kWarps;
    constexpr bool kHalfWarpRows = sizeof(StateT) == sizeof(BF16) || kFp32HalfWarp;
    constexpr int kRowsPerBulkGroup = kHalfWarpRows ? 4 * kWarps : kBulkRows;
    constexpr int kBulkGroupsPerHead = kRowsPerHead / kRowsPerBulkGroup;
    int const batch = blockIdx.z;
    int const head = blockIdx.y;
    int const warp = threadIdx.y;
    int const lane = threadIdx.x;
    int const tid = warp * 32 + lane;
    int const k0 = lane * 4;
    // The half-warp consumer owns two four-float packets. Put the low/high
    // packets in separate 64-float planes so each vector load is contiguous
    // across its 16 logical lanes; normalization/gate values stay unchanged.
    int const parameter_k0 = TwoPlaneParameters ? (lane >> 1) * 4 + (lane & 1) * 64 : k0;
    int const row_base = blockIdx.x * kRowsPerHead;
    int const slot = state_indices[batch];
    int const packed_stride = 3 * H * kD;
    int const head_offset = head * kD;

    if (slot < 0 || slot >= state_slots) {
        if (tid < kRowsPerHead) {
            out[(batch * H + head) * kD + row_base + tid] = BF16(0.0f);
        }
        return;
    }

    extern __shared__ float smem[];
    StateT *sm_state = reinterpret_cast<StateT *>(smem);
    constexpr int kStateRows = kUseBulkAiu ? kRowsPerHead : kStages * kWarps;
    float *sm_q = reinterpret_cast<float *>(sm_state + kStateRows * StatePitch);
    float *sm_k = sm_q + kD;
    BF16 *sm_v = reinterpret_cast<BF16 *>(sm_k + kD);
    float *sm_decay = reinterpret_cast<float *>(sm_v + kD);
    float *sm_beta = sm_decay + kD;

    StateT *state_head = state + int64_t(slot) * state_slot_stride + int64_t(head) * kStateElems;

    if constexpr (kUseBulkAiu) {
        // Split the 128 state rows across resident CTAs.  Each CTA issues one
        // contiguous AIU descriptor, halving its shared footprint while
        // preserving full-width row traffic and the total recurrence warps.
        if (warp == 0)
            aiu_load_rows<StateT, kRowsPerHead, StatePitch>(sm_state, state_head, row_base);
        cp_async_fence_all();
    } else {
        // Seed every resident rolling stage with one contiguous AIU transfer.
        if (warp == 0) {
            aiu_load_rows<StateT, kStages * kWarps>(sm_state, state_head, row_base);
        }
        CUTE_UNROLL
        for (int i = 0; i < kStages; ++i)
            cp_async_fence_all();
    }

    BF16 const *packed = mixed_qkv + int64_t(batch) * packed_stride;

    // Four independent producer warps overlap all scalar/vector preparation:
    // Q norm, K norm, KDA decay+beta, and V staging.
    if (warp == 0 || warp == 1) {
        BF16 const *src = packed + (warp == 0 ? head_offset : H * kD + head_offset);
        float4 x = load_bf16x4(src + k0);
        float sq = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
        float inv_norm = rsqrtf(warp_sum_all(sq) + 1.0e-6f);
        if (warp == 0)
            inv_norm *= scale;
        x.x *= inv_norm;
        x.y *= inv_norm;
        x.z *= inv_norm;
        x.w *= inv_norm;
        *reinterpret_cast<float4 *>((warp == 0 ? sm_q : sm_k) + parameter_k0) = x;
    } else if (warp == 2) {
        float exp_A = __shfl_sync(0xffffffff, lane == 0 ? __expf(A_log[head]) : 0.0f, 0);
        float4 g = load_bf16x4(gate + (batch * H + head) * kD + k0);
        float4 dt = *reinterpret_cast<float4 const *>(dt_bias + head_offset + k0);
        float4 d = make_float4(decay_factor(g.x, dt.x, exp_A, lower_bound, use_lower_bound),
                               decay_factor(g.y, dt.y, exp_A, lower_bound, use_lower_bound),
                               decay_factor(g.z, dt.z, exp_A, lower_bound, use_lower_bound),
                               decay_factor(g.w, dt.w, exp_A, lower_bound, use_lower_bound));
        *reinterpret_cast<float4 *>(sm_decay + parameter_k0) = d;
        if (lane == 0) {
            sm_beta[0] = __ppu_sgmdf(float(beta[batch * H + head]));
        }
    } else if (warp == 3 && lane * 4 < kRowsPerHead) {
        int const v_offset = row_base + lane * 4;
        BF16 const *src_v = packed + 2 * H * kD + head_offset + v_offset;
        *reinterpret_cast<uint2 *>(sm_v + v_offset) = *reinterpret_cast<uint2 const *>(src_v);
    }
    if constexpr (MergeBulkJoins) {
        static_assert(kUseBulkAiu && sizeof(StateT) == sizeof(BF16) && NumWarps == 8 &&
                      (RowSplits == 1 || RowSplits == 2) && StatePitch == 136);
        // Only measured large BF16 calls select this specialization. The one
        // common join publishes completed state copies and all parameter stores.
        cp_async_wait_all<0>();
    }
    __syncthreads();

    float4 q = *reinterpret_cast<float4 const *>(sm_q + k0);
    float4 k = *reinterpret_cast<float4 const *>(sm_k + k0);
    float4 decay = *reinterpret_cast<float4 const *>(sm_decay + k0);
    float beta_value = sm_beta[0];

    if constexpr (kUseBulkAiu) {
        if constexpr (!MergeBulkJoins) {
            cp_async_wait_all<0>();
            __syncthreads();
        }
        int half = 0;
        int half_lane = 0;
        int col0 = 0;
        float4 q0_invariant, q1_invariant;
        float4 k0_invariant, k1_invariant;
        float4 d0_invariant, d1_invariant;
        constexpr bool kCacheHalfParameters =
            kHalfWarpRows && (RowSplits == 1 || sizeof(StateT) == sizeof(BF16));
        if constexpr (kCacheHalfParameters) {
            half = InterleaveHalfLanes ? (lane & 1) : (lane >> 4);
            half_lane = InterleaveHalfLanes ? (lane >> 1) : (lane & 15);
            col0 = half_lane * 8;
            q0_invariant =
                *reinterpret_cast<float4 const *>(sm_q + (TwoPlaneParameters ? col0 / 2 : col0));
            q1_invariant = *reinterpret_cast<float4 const *>(
                sm_q + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
            k0_invariant =
                *reinterpret_cast<float4 const *>(sm_k + (TwoPlaneParameters ? col0 / 2 : col0));
            k1_invariant = *reinterpret_cast<float4 const *>(
                sm_k + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
            d0_invariant = *reinterpret_cast<float4 const *>(
                sm_decay + (TwoPlaneParameters ? col0 / 2 : col0));
            d1_invariant = *reinterpret_cast<float4 const *>(
                sm_decay + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
        } else if constexpr (kHalfWarpRows) {
            half = InterleaveHalfLanes ? (lane & 1) : (lane >> 4);
            half_lane = InterleaveHalfLanes ? (lane >> 1) : (lane & 15);
            col0 = half_lane * 8;
        }
        CUTE_UNROLL
        for (int group = 0; group < kBulkGroupsPerHead; ++group) {
            if constexpr (kHalfWarpRows) {
                int const row0 = row_base + group * kRowsPerBulkGroup + warp * 2 + half;
                int const row1 = row0 + 2 * kWarps;
                float4 q0, q1, k0v, k1v, d0, d1;
                if constexpr (kCacheHalfParameters) {
                    q0 = q0_invariant;
                    q1 = q1_invariant;
                    k0v = k0_invariant;
                    k1v = k1_invariant;
                    d0 = d0_invariant;
                    d1 = d1_invariant;
                } else {
                    q0 = *reinterpret_cast<float4 const *>(sm_q +
                                                           (TwoPlaneParameters ? col0 / 2 : col0));
                    q1 = *reinterpret_cast<float4 const *>(
                        sm_q + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
                    k0v = *reinterpret_cast<float4 const *>(sm_k +
                                                            (TwoPlaneParameters ? col0 / 2 : col0));
                    k1v = *reinterpret_cast<float4 const *>(
                        sm_k + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
                    d0 = *reinterpret_cast<float4 const *>(sm_decay +
                                                           (TwoPlaneParameters ? col0 / 2 : col0));
                    d1 = *reinterpret_cast<float4 const *>(
                        sm_decay + (TwoPlaneParameters ? col0 / 2 + 64 : col0 + 4));
                }
                StateT *state_ptr0 =
                    sm_state + (group * kRowsPerBulkGroup + warp * 2 + half) * StatePitch + col0;
                StateT *state_ptr1 = state_ptr0 + 2 * kWarps * StatePitch;
                float4 h00 = load_state4(state_ptr0);
                float4 h01 = load_state4(state_ptr0 + 4);
                float4 h10 = load_state4(state_ptr1);
                float4 h11 = load_state4(state_ptr1 + 4);

                h00.x *= d0.x;
                h00.y *= d0.y;
                h00.z *= d0.z;
                h00.w *= d0.w;
                h01.x *= d1.x;
                h01.y *= d1.y;
                h01.z *= d1.z;
                h01.w *= d1.w;
                h10.x *= d0.x;
                h10.y *= d0.y;
                h10.z *= d0.z;
                h10.w *= d0.w;
                h11.x *= d1.x;
                h11.y *= d1.y;
                h11.z *= d1.z;
                h11.w *= d1.w;

                float hk0 = h00.x * k0v.x + h00.y * k0v.y + h00.z * k0v.z + h00.w * k0v.w +
                            h01.x * k1v.x + h01.y * k1v.y + h01.z * k1v.z + h01.w * k1v.w;
                float hk1 = h10.x * k0v.x + h10.y * k0v.y + h10.z * k0v.z + h10.w * k0v.w +
                            h11.x * k1v.x + h11.y * k1v.y + h11.z * k1v.z + h11.w * k1v.w;
                if constexpr (InterleaveHalfLanes) {
                    // Same logical 16-lane XOR tree: consecutive logical lanes
                    // are two physical lanes apart, with even/odd row owners.
                    CUTE_UNROLL
                    for (int offset = 16; offset > 1; offset >>= 1) {
                        hk0 += __shfl_xor_sync(0xffffffff, hk0, offset);
                        hk1 += __shfl_xor_sync(0xffffffff, hk1, offset);
                    }
                } else {
                    half_warp_sum_all_pair(hk0, hk1);
                }
                float const v_new0 = (float(sm_v[row0]) - hk0) * beta_value;
                float const v_new1 = (float(sm_v[row1]) - hk1) * beta_value;

                h00.x += k0v.x * v_new0;
                h00.y += k0v.y * v_new0;
                h00.z += k0v.z * v_new0;
                h00.w += k0v.w * v_new0;
                h01.x += k1v.x * v_new0;
                h01.y += k1v.y * v_new0;
                h01.z += k1v.z * v_new0;
                h01.w += k1v.w * v_new0;
                h10.x += k0v.x * v_new1;
                h10.y += k0v.y * v_new1;
                h10.z += k0v.z * v_new1;
                h10.w += k0v.w * v_new1;
                h11.x += k1v.x * v_new1;
                h11.y += k1v.y * v_new1;
                h11.z += k1v.z * v_new1;
                h11.w += k1v.w * v_new1;

                if constexpr (sizeof(StateT) == sizeof(BF16)) {
                    store_bf16_state8_global(state_head + row0 * kD + col0, h00, h01);
                    store_bf16_state8_global(state_head + row1 * kD + col0, h10, h11);
                } else {
                    store_state4_global(state_head + row0 * kD + col0, h00);
                    store_state4_global(state_head + row0 * kD + col0 + 4, h01);
                    store_state4_global(state_head + row1 * kD + col0, h10);
                    store_state4_global(state_head + row1 * kD + col0 + 4, h11);
                }

                float hq0 = h00.x * q0.x + h00.y * q0.y + h00.z * q0.z + h00.w * q0.w +
                            h01.x * q1.x + h01.y * q1.y + h01.z * q1.z + h01.w * q1.w;
                float hq1 = h10.x * q0.x + h10.y * q0.y + h10.z * q0.z + h10.w * q0.w +
                            h11.x * q1.x + h11.y * q1.y + h11.z * q1.z + h11.w * q1.w;
                if constexpr (InterleaveHalfLanes) {
                    CUTE_UNROLL
                    for (int offset = 16; offset > 1; offset >>= 1) {
                        hq0 += __shfl_down_sync(0xffffffff, hq0, offset);
                        hq1 += __shfl_down_sync(0xffffffff, hq1, offset);
                    }
                } else {
                    half_warp_sum_lane0_pair(hq0, hq1);
                }
                if (half_lane == 0) {
                    out[(batch * H + head) * kD + row0] = BF16(hq0);
                    out[(batch * H + head) * kD + row1] = BF16(hq1);
                }
            } else if constexpr (kRowsPerGroupWarp == 4) {
                int const row0 = row_base + group * kBulkRows + warp;
                int const row1 = row0 + kWarps;
                int const row2 = row1 + kWarps;
                int const row3 = row2 + kWarps;
                StateT *stage_ptr0 = sm_state + (group * kBulkRows + warp) * StatePitch + k0;
                StateT *stage_ptr1 = stage_ptr0 + kWarps * StatePitch;
                StateT *stage_ptr2 = stage_ptr1 + kWarps * StatePitch;
                StateT *stage_ptr3 = stage_ptr2 + kWarps * kD;
                float4 h0 = load_state4(stage_ptr0);
                float4 h1 = load_state4(stage_ptr1);
                float4 h2 = load_state4(stage_ptr2);
                float4 h3 = load_state4(stage_ptr3);
                h0.x *= decay.x;
                h0.y *= decay.y;
                h0.z *= decay.z;
                h0.w *= decay.w;
                h1.x *= decay.x;
                h1.y *= decay.y;
                h1.z *= decay.z;
                h1.w *= decay.w;
                h2.x *= decay.x;
                h2.y *= decay.y;
                h2.z *= decay.z;
                h2.w *= decay.w;
                h3.x *= decay.x;
                h3.y *= decay.y;
                h3.z *= decay.z;
                h3.w *= decay.w;

                float hk0 = h0.x * k.x + h0.y * k.y + h0.z * k.z + h0.w * k.w;
                float hk1 = h1.x * k.x + h1.y * k.y + h1.z * k.z + h1.w * k.w;
                float hk2 = h2.x * k.x + h2.y * k.y + h2.z * k.z + h2.w * k.w;
                float hk3 = h3.x * k.x + h3.y * k.y + h3.z * k.z + h3.w * k.w;
                warp_sum_all_quad(hk0, hk1, hk2, hk3);
                float const v_new0 = (float(sm_v[row0]) - hk0) * beta_value;
                float const v_new1 = (float(sm_v[row1]) - hk1) * beta_value;
                float const v_new2 = (float(sm_v[row2]) - hk2) * beta_value;
                float const v_new3 = (float(sm_v[row3]) - hk3) * beta_value;

                h0.x += k.x * v_new0;
                h0.y += k.y * v_new0;
                h0.z += k.z * v_new0;
                h0.w += k.w * v_new0;
                h1.x += k.x * v_new1;
                h1.y += k.y * v_new1;
                h1.z += k.z * v_new1;
                h1.w += k.w * v_new1;
                h2.x += k.x * v_new2;
                h2.y += k.y * v_new2;
                h2.z += k.z * v_new2;
                h2.w += k.w * v_new2;
                h3.x += k.x * v_new3;
                h3.y += k.y * v_new3;
                h3.z += k.z * v_new3;
                h3.w += k.w * v_new3;
                store_state4_global(state_head + row0 * kD + k0, h0);
                store_state4_global(state_head + row1 * kD + k0, h1);
                store_state4_global(state_head + row2 * kD + k0, h2);
                store_state4_global(state_head + row3 * kD + k0, h3);

                float hq0 = h0.x * q.x + h0.y * q.y + h0.z * q.z + h0.w * q.w;
                float hq1 = h1.x * q.x + h1.y * q.y + h1.z * q.z + h1.w * q.w;
                float hq2 = h2.x * q.x + h2.y * q.y + h2.z * q.z + h2.w * q.w;
                float hq3 = h3.x * q.x + h3.y * q.y + h3.z * q.z + h3.w * q.w;
                warp_sum_lane0_quad(hq0, hq1, hq2, hq3);
                if (lane == 0) {
                    out[(batch * H + head) * kD + row0] = BF16(hq0);
                    out[(batch * H + head) * kD + row1] = BF16(hq1);
                    out[(batch * H + head) * kD + row2] = BF16(hq2);
                    out[(batch * H + head) * kD + row3] = BF16(hq3);
                }
            } else {
                static_assert(kRowsPerGroupWarp == 2);
                int const row0 = row_base + group * kBulkRows + warp;
                int const row1 = row0 + kWarps;
                StateT *stage_ptr0 = sm_state + (group * kBulkRows + warp) * StatePitch + k0;
                StateT *stage_ptr1 = stage_ptr0 + kWarps * StatePitch;
                float4 h0 = load_state4(stage_ptr0);
                float4 h1 = load_state4(stage_ptr1);
                h0.x *= decay.x;
                h0.y *= decay.y;
                h0.z *= decay.z;
                h0.w *= decay.w;
                h1.x *= decay.x;
                h1.y *= decay.y;
                h1.z *= decay.z;
                h1.w *= decay.w;

                float hk0 = h0.x * k.x + h0.y * k.y + h0.z * k.z + h0.w * k.w;
                float hk1 = h1.x * k.x + h1.y * k.y + h1.z * k.z + h1.w * k.w;
                warp_sum_all_pair(hk0, hk1);
                float const v_new0 = (float(sm_v[row0]) - hk0) * beta_value;
                float const v_new1 = (float(sm_v[row1]) - hk1) * beta_value;

                h0.x += k.x * v_new0;
                h0.y += k.y * v_new0;
                h0.z += k.z * v_new0;
                h0.w += k.w * v_new0;
                h1.x += k.x * v_new1;
                h1.y += k.y * v_new1;
                h1.z += k.z * v_new1;
                h1.w += k.w * v_new1;
                store_state4_global(state_head + row0 * kD + k0, h0);
                store_state4_global(state_head + row1 * kD + k0, h1);

                float hq0 = h0.x * q.x + h0.y * q.y + h0.z * q.z + h0.w * q.w;
                float hq1 = h1.x * q.x + h1.y * q.y + h1.z * q.z + h1.w * q.w;
                warp_sum_lane0_pair(hq0, hq1);
                if (lane == 0) {
                    out[(batch * H + head) * kD + row0] = BF16(hq0);
                    out[(batch * H + head) * kD + row1] = BF16(hq1);
                }
            }
        }
    } else {
        CUTE_UNROLL
        for (int v_pair = 0; v_pair < kRowsPerWarp; v_pair += 2) {
            int const stage0 = v_pair % kStages;
            int const stage1 = (v_pair + 1) % kStages;
            int const row0 = row_base + v_pair * kWarps + warp;
            int const row1 = row0 + kWarps;
            cp_async_wait_all<kStages - 2>();
            if (v_pair == 0) {
                __ppu_barrier_sync_defer(0, kWarps * 32, 1);
            }

            float *stage_ptr0 = sm_state + (stage0 * kWarps + warp) * kD + k0;
            float *stage_ptr1 = sm_state + (stage1 * kWarps + warp) * kD + k0;
            float4 h0 = *reinterpret_cast<float4 const *>(stage_ptr0);
            float4 h1 = *reinterpret_cast<float4 const *>(stage_ptr1);
            h0.x *= decay.x;
            h0.y *= decay.y;
            h0.z *= decay.z;
            h0.w *= decay.w;
            h1.x *= decay.x;
            h1.y *= decay.y;
            h1.z *= decay.z;
            h1.w *= decay.w;

            float hk0 = h0.x * k.x + h0.y * k.y + h0.z * k.z + h0.w * k.w;
            float hk1 = h1.x * k.x + h1.y * k.y + h1.z * k.z + h1.w * k.w;
            warp_sum_all_pair(hk0, hk1);
            float const v_new0 = (float(sm_v[row0]) - hk0) * beta_value;
            float const v_new1 = (float(sm_v[row1]) - hk1) * beta_value;

            int const next0 = v_pair + kStages;
            int const next1 = next0 + 1;
            if (next0 < kRowsPerWarp) {
                float const *src0 = state_head + (row_base + next0 * kWarps + warp) * kD + k0;
                CpAsyncVec16::copy(*reinterpret_cast<cute::uint128_t const *>(src0),
                                   *reinterpret_cast<cute::uint128_t *>(stage_ptr0));
            }
            cp_async_fence_all();
            if (next1 < kRowsPerWarp) {
                float const *src1 = state_head + (row_base + next1 * kWarps + warp) * kD + k0;
                CpAsyncVec16::copy(*reinterpret_cast<cute::uint128_t const *>(src1),
                                   *reinterpret_cast<cute::uint128_t *>(stage_ptr1));
            }
            cp_async_fence_all();

            h0.x += k.x * v_new0;
            h0.y += k.y * v_new0;
            h0.z += k.z * v_new0;
            h0.w += k.w * v_new0;
            h1.x += k.x * v_new1;
            h1.y += k.y * v_new1;
            h1.z += k.z * v_new1;
            h1.w += k.w * v_new1;
            if constexpr (RowSplits == kTallRowSplits) {
                store_float4_global_cg(state_head + row0 * kD + k0, h0);
                store_float4_global_cg(state_head + row1 * kD + k0, h1);
            } else {
                store_float4_global(state_head + row0 * kD + k0, h0);
                store_float4_global(state_head + row1 * kD + k0, h1);
            }

            float hq0 = h0.x * q.x + h0.y * q.y + h0.z * q.z + h0.w * q.w;
            float hq1 = h1.x * q.x + h1.y * q.y + h1.z * q.z + h1.w * q.w;
            warp_sum_lane0_pair(hq0, hq1);
            if (lane == 0) {
                out[(batch * H + head) * kD + row0] = BF16(hq0);
                out[(batch * H + head) * kD + row1] = BF16(hq1);
            }
        }
        cp_async_wait_all<0>();
    }
}

} // namespace
} // namespace flash_kda_ppu
