#pragma once

#include "fwd_kernel1_ppu.cuh"

namespace flash_kda_ppu {

// Prefix storage is dead before the triangular matrices are materialized.
// Centering at half the total log-decay bounds both BF16 factors by the
// original 16-token exponent range, including fully saturated bounded gates.
template <int D> struct SharedStorageK32 {
    using L = K1Layouts<D, 32>;
    union {
        struct {
            alignas(128) BF16 q[L::QK::kElems];
            alignas(128) BF16 k[L::QK::kElems];
            alignas(128) BF16 g_raw[L::QK::kElems];
        } a;
        struct {
            alignas(128) BF16 k_decayed[L::Tile::kElems];
            alignas(128) BF16 q_decayed[L::Tile::kElems];
            alignas(128) BF16 k_inv[L::Tile::kElems];
            alignas(128) BF16 k_restored[L::Tile::kElems];
        } b;
    };
    union {
        alignas(128) float g_cumsum[L::GF32::kElems];
        struct {
            alignas(128) float Lf32[L::LMF32::kElems];
            alignas(128) float INVf32[L::LMF32::kElems];
            alignas(128) float scratch[L::LMF32::kElems];
            alignas(128) BF16 Mqk[L::LMBF16::kElems];
            alignas(128) BF16 INVbf[L::LMBF16::kElems];
        };
    };
    alignas(128) float dt_bias[D];
    alignas(128) float g_total[D];
    alignas(128) float g_center[D];
    alignas(128) float g_center_decay[D];
    alignas(128) float beta[32];
};

template <int D, int NumThreads, bool IsVarlen, int StaticHeads = 0>
__global__ void __launch_bounds__(NumThreads, 4)
flash_kda_ppu_fwd_prepare32(BF16 const *__restrict__ q_ptr, BF16 const *__restrict__ k_ptr,
    BF16 const *__restrict__ g_ptr, BF16 const *__restrict__ beta_ptr,
    float const *__restrict__ A_log_ptr, float const *__restrict__ dt_bias_ptr,
    BF16 *__restrict__ ws_kd, BF16 *__restrict__ ws_qd, BF16 *__restrict__ ws_kr,
    float *__restrict__ ws_gt, BF16 *__restrict__ ws_inv, BF16 *__restrict__ ws_mqk,
    float *__restrict__ ws_beta, float scale, float gate_scale, int T_total, int H_runtime, int N,
    int64_t const *__restrict__ cu_seqlens, int total_tiles,
    int const *__restrict__ tile_prefix, int *__restrict__ scan_flag,
    float scan_threshold, int q_stride_runtime, int k_stride_runtime) {
    constexpr int CHUNK = 32;
    static_assert(D == 128 && NumThreads == 512);
    static_assert(StaticHeads == 0 || StaticHeads == 6 || StaticHeads == 12);
    int const H = StaticHeads ? StaticHeads : H_runtime;
    int const q_stride = StaticHeads ? 3 * StaticHeads * D : q_stride_runtime;
    int const k_stride = StaticHeads ? 3 * StaticHeads * D : k_stride_runtime;
    using namespace cute;
    using L1 = K1Layouts<D, CHUNK>;
    constexpr int kPad = L1::kColsPad;

    __shared__ SharedStorageK32<D> sm;

    int const global_tile = blockIdx.x;
    int const head_idx = blockIdx.y;
    int const tid = threadIdx.x;

    int seq_idx, local_t;
    int64_t bos, eos;
    if constexpr (IsVarlen) {
        // O(log N) instead of the CUDA kernel's per-CTA linear scan.
        int lo = 0, hi = N;
        while (lo + 1 < hi) {
            int const mid = (lo + hi) >> 1;
            if (tile_prefix[mid] <= global_tile)
                lo = mid;
            else
                hi = mid;
        }
        seq_idx = lo;
        local_t = global_tile - tile_prefix[lo];
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
    } else {
        int const T_seq = T_total / N;
        int const tiles_per_seq = (T_seq + CHUNK - 1) / CHUNK;
        seq_idx = global_tile / tiles_per_seq;
        local_t = global_tile - seq_idx * tiles_per_seq;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
    }
    int const seq_len = int(eos - bos);
    int const t_tiles = (seq_len + CHUNK - 1) / CHUNK;
    // total_tiles is an upper bound in varlen mode, so trailing CTAs have no work.
    if (local_t >= t_tiles)
        return;

    int const row0 = int(bos) + local_t * CHUNK;
    int const rows_valid = max(0, min(CHUNK, seq_len - local_t * CHUNK));
    int const gmem_row_stride = H * D;
    BF16 const *q_tile = q_ptr + int64_t(row0) * q_stride + head_idx * D;
    BF16 const *k_tile = k_ptr + int64_t(row0) * k_stride + head_idx * D;
    BF16 const *g_tile = g_ptr + int64_t(row0) * gmem_row_stride + head_idx * D;

    // --- load. cp.async into plain padded destinations; tail rows zero-filled.
    cp_async_tile<CHUNK, D, kPad, NumThreads>(sm.a.q, q_tile, q_stride, rows_valid, tid);
    cp_async_tile<CHUNK, D, kPad, NumThreads>(sm.a.k, k_tile, k_stride, rows_valid, tid);
    cp_async_tile<CHUNK, D, kPad, NumThreads>(sm.a.g_raw, g_tile, gmem_row_stride, rows_valid, tid);
    for (int i = tid; i < D; i += NumThreads)
        sm.dt_bias[i] = dt_bias_ptr[head_idx * D + i];
    // PPU consumes beta in its original [T_total, H] layout, avoiding the host
    // framework's full-tensor transpose/materialisation on every forward.
    for (int i = tid; i < CHUNK; i += NumThreads) {
        sm.beta[i] =
            i < rows_valid ? kda_sigmoid(float(beta_ptr[int64_t(row0 + i) * H + head_idx])) : 0.0f;
    }

    // exp(A_log) overlaps the loads.
    float const a_log_exp = expf(A_log_ptr[head_idx]);

    cp_async_fence_all();
    cp_async_wait_all<0>();
    __syncthreads();

    // --- L2 normalise q and k over D, one row per 16 lanes.
    {
        constexpr int kPerThread = 8;
        constexpr int kThreadsPerRow = D / kPerThread; // 16
        int const row = tid / kThreadsPerRow;
        int const col = (tid % kThreadsPerRow) * kPerThread;

        if (row < CHUNK) {
            float qv[kPerThread], kv[kPerThread];
            float q_sq = 0.0f, k_sq = 0.0f;
            CUTE_UNROLL
            for (int i = 0; i < kPerThread; ++i) {
                qv[i] = float(sm.a.q[row * kPad + col + i]);
                kv[i] = float(sm.a.k[row * kPad + col + i]);
                q_sq += qv[i] * qv[i];
                k_sq += kv[i] * kv[i];
            }
            CUTE_UNROLL
            for (int delta = 8; delta >= 1; delta >>= 1) {
                q_sq += __shfl_xor_sync(0xFFFFFFFFu, q_sq, delta);
                k_sq += __shfl_xor_sync(0xFFFFFFFFu, k_sq, delta);
            }
            float const q_inv = rsqrtf(q_sq + 1e-6f);
            float const k_inv = rsqrtf(k_sq + 1e-6f);
            CUTE_UNROLL
            for (int i = 0; i < kPerThread; ++i) {
                sm.a.q[row * kPad + col + i] = BF16(qv[i] * q_inv);
                sm.a.k[row * kPad + col + i] = BF16(kv[i] * k_inv);
            }
        }
    }

    int const ws_idx = head_idx * total_tiles + global_tile;
    // Eight4-token gate partitions per adjacent channel pair keep all512
    // lanes active. Owner-local packed32-bit decay avoids the FP32 prefix
    // round trip; centered Gram operands and uncentered workspace semantics
    // retain the original FP32-to-BF16 boundaries.
    {
        int const lane = tid%32, part = lane/4;
        int const pair = (tid/32)*4+lane%4, col = pair*2;
        float const dt[2] = {sm.dt_bias[col],sm.dt_bias[col+1]};
        float prefix[2][4], sum[2] = {0.f,0.f};
        auto read_pair = [](BF16 const *p) {
            uint32_t const bits = *reinterpret_cast<uint32_t const *>(p);
            return __bfloat1622float2(*reinterpret_cast<__ppu_bfloat162 const *>(&bits));
        };
        auto write_pair = [](BF16 *p, float x, float y) {
            __ppu_bfloat162 const packed = __floats2bfloat162_rn(x,y);
            *reinterpret_cast<uint32_t *>(p) = *reinterpret_cast<uint32_t const *>(&packed);
        };
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) {
            int const row = part*4+i;
            float2 const raw = read_pair(sm.a.g_raw+row*kPad+col);
            float const g[2] = {raw.x,raw.y};
            CUTE_UNROLL
            for (int c = 0; c < 2; ++c) {
                sum[c] += row < rows_valid
                    ? gate_scale*kda_sigmoid(a_log_exp*(g[c]+dt[c])) : 0.f;
                prefix[c][i] = sum[c];
            }
        }
        float preceding[2], center[2];
        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
            float scan = sum[c];
            CUTE_UNROLL
            for (int delta = 1; delta <= 4; delta *= 2) {
                float const other = __shfl_up_sync(0xffffffffu,scan,delta*4);
                if (part >= delta) scan += other;
            }
            float const previous = __shfl_up_sync(0xffffffffu,scan,4);
            preceding[c] = part ? previous : 0.f;
            float const total = __shfl_sync(0xffffffffu,scan,(lane%4)+28);
            center[c] = total*0.5f;
            if (part == 0) {
                sm.g_total[col+c] = kda_exp2(total);
                sm.g_center[col+c] = center[c];
                sm.g_center_decay[col+c] = kda_exp2(center[c]);
            }
        }
        __syncthreads(); // Normalization and every raw gate read are complete.
        float2 const centered_decay =
            *reinterpret_cast<float2 const *>(sm.g_center_decay+col);
        CUTE_UNROLL
        for (int i = 0; i < 4; ++i) {
            int const row = part*4+i, o = row*kPad+col;
            float2 const q = read_pair(sm.a.q+o), k = read_pair(sm.a.k+o);
            float const dec0 = kda_exp2(prefix[0][i]+preceding[0]-center[0]);
            float const dec1 = kda_exp2(prefix[1][i]+preceding[1]-center[1]);
            float const inv0 = __ppu_rcpf(dec0), inv1 = __ppu_rcpf(dec1);
            float const kd0 = k.x*dec0, kd1 = k.y*dec1;
            float const qd0 = q.x*dec0*scale, qd1 = q.y*dec1*scale;
            write_pair(sm.b.k_decayed+o,kd0,kd1);
            write_pair(sm.b.q_decayed+o,qd0,qd1);
            write_pair(sm.b.k_inv+o,k.x*inv0,k.y*inv1);
            write_pair(sm.b.k_restored+o,k.x*inv0*centered_decay.x,
                                              k.y*inv1*centered_decay.y);
            int64_t const dst = int64_t(ws_idx)*3*CHUNK*D+row*D+col;
            // Restore FP32 centered products before the single BF16 workspace
            // narrowing, matching the original range/rounding contract.
            write_pair(ws_kd+dst,kd0*centered_decay.x,kd1*centered_decay.y);
            write_pair(ws_qd+dst,qd0*centered_decay.x,qd1*centered_decay.y);
        }
    }
    __syncthreads(); // Matrix storage may now replace the dead prefix union.

    int const warp = tid / 32, lane = tid % 32;
    auto mma = make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1,_1,_1>>{}, Tile<_16,_16,_16>{});
    auto thr = mma.get_slice(lane);
    // Only three lower-triangle 16x16 blocks are needed per matrix. Skipping
    // the upper-right block also prevents irrelevant exp(160) accumulations.
    if (warp < 6) {
        int const tile = warp % 3;
        int const rb = tile == 0 ? 0 : 1, cb = tile == 2 ? 1 : 0;
        Tensor ta = make_tensor(make_smem_ptr(warp < 3 ? sm.b.k_decayed : sm.b.q_decayed),
                                typename L1::Tile::Layout{});
        Tensor tb = make_tensor(make_smem_ptr(sm.b.k_inv), typename L1::Tile::Layout{});
        Tensor tc = make_tensor(make_smem_ptr(sm.Mqk), typename L1::LMBF16::Layout{});
        auto acc = thr.make_fragment_C(thr.partition_C(local_tile(tc, Shape<_16,_16>{}, make_coord(0,0))));
        clear(acc);
        CUTE_UNROLL
        for (int kb = 0; kb < D/16; ++kb) {
            auto ak = local_tile(ta, Shape<_16,_16>{}, make_coord(rb,kb));
            auto bk = local_tile(tb, Shape<_16,_16>{}, make_coord(cb,kb));
            auto a = thr.partition_fragment_A(ak); auto b = thr.partition_fragment_B(bk);
            copy(AutoVectorizingCopy{}, thr.partition_A(ak), a);
            copy(AutoVectorizingCopy{}, thr.partition_B(bk), b);
            cute::gemm(thr,a,b,acc);
        }
        CUTE_UNROLL
        for (int v = 0; v < decltype(size(acc))::value; ++v) {
            FragCoord c = frag_coord_mn(lane,v); int const r = rb*16+c.m, n = cb*16+c.n;
            if (warp < 3) sm.Lf32[r*36+n] = r <= n ? 0.f : acc(v)*sm.beta[r];
            else sm.Mqk[r*40+n] = BF16(r < n ? 0.f : acc(v));
        }
    } else if (tid < 448) {
        int const i = tid - 192, r = i/16, c = 16+i%16;
        sm.Lf32[r*36+c] = 0.f;
        sm.Mqk[r*40+c] = BF16(0.f);
        sm.INVf32[r*36+c] = 0.f;
    }
    __syncthreads();

    // Both diagonal blocks use the original FP32 half-warp solve.
    {
        int const row = tid/16, col = tid%16, block = row/16;
        float inverse = row%16 == col ? 1.f : 0.f;
        CUTE_UNROLL
        for (int pivot = 15; pivot > 0; --pivot) {
            float const value = __shfl_sync(0xffffffffu,inverse,(lane&16)+pivot);
            float const factor = col < pivot ? sm.Lf32[(block*16+pivot)*36+block*16+col] : 0.f;
            inverse = fmaf(-factor,value,inverse);
        }
        sm.INVf32[row*36+block*16+col] = inverse;
    }
    store_contiguous<NumThreads>(ws_gt+int64_t(ws_idx)*D,sm.g_total,D*int(sizeof(float)),tid);
    __syncthreads();

    // INV10 = -INV11 * L10 * INV00. True FP32 matrix products preserve
    // the triangular solve's arithmetic; no BF16 inverse iteration is used.
    if (warp == 0) {
        auto fmma = make_tiled_mma(MMA_Atom<MmaF32>{}, Layout<Shape<_1,_1,_1>>{}, Tile<_16,_16,_8>{});
        auto ft = fmma.get_slice(lane);
        auto multiply = [&](float const *a, float const *b, float *out, float sign) {
            auto ta = make_tensor(make_smem_ptr(a), Layout<Shape<_16,_16>,Stride<Int<36>,_1>>{});
            auto tb = make_tensor(make_smem_ptr(b), Layout<Shape<_16,_16>,Stride<_1,Int<36>>>{});
            auto tc = make_tensor(make_smem_ptr(out),Layout<Shape<_16,_16>,Stride<Int<36>,_1>>{});
            auto acc = ft.make_fragment_C(ft.partition_C(tc)); clear(acc);
            CUTE_UNROLL
            for (int k = 0; k < 2; ++k) {
                auto ak = local_tile(ta,Shape<_16,_8>{},make_coord(0,k));
                auto bk = local_tile(tb,Shape<_16,_8>{},make_coord(0,k));
                auto af = ft.partition_fragment_A(ak); auto bf = ft.partition_fragment_B(bk);
                copy(AutoVectorizingCopy{},ft.partition_A(ak),af);
                copy(AutoVectorizingCopy{},ft.partition_B(bk),bf);
                cute::gemm(ft,af,bf,acc);
            }
            auto dst = ft.partition_C(tc);
            CUTE_UNROLL
            for (int v = 0; v < decltype(size(acc))::value; ++v) dst(v) = sign*acc(v);
            __syncwarp();
        };
        multiply(sm.Lf32+16*36,sm.INVf32,sm.scratch+16*36,-1.f);
        multiply(sm.INVf32+16*36+16,sm.scratch+16*36,sm.INVf32+16*36,1.f);
    }
    __syncthreads();
    for (int i = tid; i < CHUNK*CHUNK; i += NumThreads)
        sm.INVbf[(i/CHUNK)*40+i%CHUNK] = BF16(sm.INVf32[(i/CHUNK)*36+i%CHUNK]);
    __syncthreads();

    Tensor ti = make_tensor(make_smem_ptr(sm.INVbf),Layout<Shape<Int<CHUNK>,Int<CHUNK>>,Stride<_1,Int<40>>>{});
    Tensor tr = make_tensor(make_smem_ptr(sm.b.k_restored),Layout<Shape<Int<D>,Int<CHUNK>>,Stride<_1,Int<kPad>>>{});
    Tensor tm = make_tensor(make_smem_ptr(sm.Mqk),typename L1::LMBF16::Layout{});
    int const wr = warp/8, wd = warp%8;
    auto acc = thr.make_fragment_C(thr.partition_C(local_tile(tm,Shape<_16,_16>{},make_coord(0,0))));
    clear(acc);
    CUTE_UNROLL
    for (int kb = 0; kb < 2; ++kb) {
        // INV transpose has an exactly zero lower-left block.
        if (wr == 1 && kb == 0) continue;
        auto ak = local_tile(ti,Shape<_16,_16>{},make_coord(wr,kb));
        auto bk = local_tile(tr,Shape<_16,_16>{},make_coord(wd,kb));
        auto a = thr.partition_fragment_A(ak); auto b = thr.partition_fragment_B(bk);
        copy(AutoVectorizingCopy{},thr.partition_A(ak),a);
        copy(AutoVectorizingCopy{},thr.partition_B(bk),b);
        cute::gemm(thr,a,b,acc);
    }
    CUTE_UNROLL
    for (int v = 0; v < decltype(size(acc))::value/2; ++v) {
        FragCoord c = frag_coord_mn(lane,2*v); int const r = wr*16+c.m;
        BF16 pair[2] = {BF16(acc(2*v)*sm.beta[r]),BF16(acc(2*v+1)*sm.beta[r])};
        *reinterpret_cast<uint32_t *>(ws_kr+int64_t(ws_idx)*3*CHUNK*D+r*D+wd*16+c.n)
            = *reinterpret_cast<uint32_t const *>(pair);
    }
    if (warp < 4) {
        int const mr = warp/2, mn = warp%2;
        clear(acc);
        CUTE_UNROLL
        for (int kb = 0; kb < 2; ++kb) {
            // Product of two lower-triangular block matrices: n <= k <= m.
            if (kb < mn || kb > mr) continue;
            auto ak = local_tile(tm,Shape<_16,_16>{},make_coord(mr,kb));
            auto bk = local_tile(ti,Shape<_16,_16>{},make_coord(mn,kb));
            auto a = thr.partition_fragment_A(ak); auto b = thr.partition_fragment_B(bk);
            copy(AutoVectorizingCopy{},thr.partition_A(ak),a);
            copy(AutoVectorizingCopy{},thr.partition_B(bk),b);
            cute::gemm(thr,a,b,acc);
        }
        CUTE_UNROLL
        for (int v = 0; v < decltype(size(acc))::value/2; ++v) {
            FragCoord c = frag_coord_mn(lane,2*v); int const r = mr*16+c.m, n = mn*16+c.n;
            BF16 pair[2] = {BF16(acc(2*v)*sm.beta[n]),BF16(acc(2*v+1)*sm.beta[n+1])};
            *reinterpret_cast<uint32_t *>(ws_mqk+int64_t(ws_idx)*CHUNK*CHUNK+r*CHUNK+n)
                = *reinterpret_cast<uint32_t const *>(pair);
        }
    }
}

} // namespace flash_kda_ppu
