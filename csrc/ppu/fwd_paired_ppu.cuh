#pragma once
#include "fwd_kernel2_ppu.cuh"

namespace flash_kda_ppu {

// Prefixes keep adjacent transforms inside each sequence, including empty rows.
__global__ void flash_kda_ppu_pair_prefix(int64_t const *cu, int T, int N, int *raw_prefix,
                                          int *pair_prefix) {
    if (threadIdx.x == 0) {
        int raw = 0, paired = 0;
        raw_prefix[0] = pair_prefix[0] = 0;
        for (int seq = 0; seq < N; ++seq) {
            int const len = cu ? int(cu[seq + 1] - cu[seq]) : T / N;
            raw += (len + 15) / 16;
            paired += (len + 31) / 32;
            raw_prefix[seq + 1] = raw;
            pair_prefix[seq + 1] = paired;
        }
    }
}

// True FP32 GEMM: the PPU TN atom needs a transposed B view for A @ B.
CUTE_DEVICE void pair_gemm_fp32(float const *a, float const *b, float *output, int lane) {
    using namespace cute;
    auto mma =
        make_tiled_mma(MMA_Atom<MmaF32>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _8>{});
    auto thr = mma.get_slice(lane);
    Tensor ta = make_tensor(make_smem_ptr(a), Layout<Shape<_16, _16>, Stride<Int<20>, _1>>{});
    Tensor tb = make_tensor(make_smem_ptr(b), Layout<Shape<_16, _16>, Stride<_1, Int<20>>>{});
    Tensor tc = make_tensor(make_smem_ptr(output), Layout<Shape<_16, _16>, Stride<Int<20>, _1>>{});
    auto acc = thr.make_fragment_C(thr.partition_C(tc));
    clear(acc);
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i) {
        Tensor aa = local_tile(ta, Shape<_16, _8>{}, make_coord(0, i));
        Tensor bb = local_tile(tb, Shape<_16, _8>{}, make_coord(0, i));
        auto ra = thr.partition_fragment_A(aa);
        auto rb = thr.partition_fragment_B(bb);
        copy(AutoVectorizingCopy{}, thr.partition_A(aa), ra);
        copy(AutoVectorizingCopy{}, thr.partition_B(bb), rb);
        cute::gemm(thr, ra, rb, acc);
    }
    copy(AutoVectorizingCopy{}, acc, thr.partition_C(tc));
    __syncwarp();
}

__global__ __launch_bounds__(64) void flash_kda_ppu_compose_pairs(
    BF16 const *tr, float const *gt, BF16 const *inv, BF16 const *mqk, float const *beta, BF16 *ptr,
    float *pgt, BF16 *pinv, BF16 *pmqk, float *pbeta, int const *raw_prefix, int const *pair_prefix,
    int raw_tiles, int pair_tiles, int N) {
    using namespace cute;
    int const pair = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    int const lane = tid % 32, warp = tid / 32;
    if (pair >= pair_prefix[N])
        return;
    int lo = 0, hi = N;
    while (lo + 1 < hi) {
        int const mid = (lo + hi) / 2;
        if (pair_prefix[mid] <= pair)
            lo = mid;
        else
            hi = mid;
    }
    int const first = raw_prefix[lo] + 2 * (pair - pair_prefix[lo]);
    bool const has_second = first + 1 < raw_prefix[lo + 1];
    int64_t const raw = int64_t(head) * raw_tiles + first;
    int64_t const dst = int64_t(head) * pair_tiles + pair;
    __shared__ BF16 key[16 * 136], query[16 * 136], restored[16 * 136];
    __shared__ float cross[16 * 20], i0[16 * 20], i1[16 * 20], temp[16 * 20];
    for (int x = tid; x < 16 * 128; x += 64) {
        int const row = x / 128, col = x % 128;
        BF16 const kd0 = tr[raw * 48 * 128 + x];
        BF16 const qd0 = tr[raw * 48 * 128 + 16 * 128 + x];
        BF16 const kr0 = tr[raw * 48 * 128 + 32 * 128 + x];
        BF16 const kd1 = has_second ? tr[(raw + 1) * 48 * 128 + x] : BF16(0.f);
        BF16 const qd1 = has_second ? tr[(raw + 1) * 48 * 128 + 16 * 128 + x] : BF16(0.f);
        BF16 const kr1 = has_second ? tr[(raw + 1) * 48 * 128 + 32 * 128 + x] : BF16(0.f);
        float const g0 = gt[raw * 128 + col];
        float const g1 = has_second ? gt[(raw + 1) * 128 + col] : 1.f;
        BF16 *d = ptr + dst * 96 * 128 + x;
        d[0] = kd0;
        d[16 * 128] = BF16(float(kd1) * g0);
        d[32 * 128] = qd0;
        d[48 * 128] = BF16(float(qd1) * g0);
        d[64 * 128] = BF16(float(kr0) * g1);
        d[80 * 128] = kr1;
        key[row * 136 + col] = kd1;
        query[row * 136 + col] = qd1;
        restored[row * 136 + col] = kr0;
        if (row == 0)
            pgt[dst * 128 + col] = g0 * g1;
    }
    for (int x = tid; x < 256; x += 64) {
        int const row = x / 16, col = x % 16;
        BF16 const a = inv[raw * 256 + x];
        BF16 const b = has_second ? inv[(raw + 1) * 256 + x] : BF16(row == col ? 1.f : 0.f);
        i0[row * 20 + col] = float(a);
        i1[row * 20 + col] = float(b);
        pinv[dst * 1024 + row * 32 + col] = a;
        pinv[dst * 1024 + row * 32 + col + 16] = BF16(0.f);
        pinv[dst * 1024 + (row + 16) * 32 + col + 16] = b;
        pmqk[dst * 1024 + row * 32 + col] = mqk[raw * 256 + x];
        pmqk[dst * 1024 + row * 32 + col + 16] = BF16(0.f);
        pmqk[dst * 1024 + (row + 16) * 32 + col + 16] =
            has_second ? mqk[(raw + 1) * 256 + x] : BF16(0.f);
    }
    if (tid < 16) {
        pbeta[dst * 32 + tid] = beta[raw * 32 + tid];
        pbeta[dst * 32 + 16 + tid] = has_second ? beta[(raw + 1) * 32 + tid] : 0.f;
    }
    __syncthreads();
    auto mma =
        make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr = mma.get_slice(lane);
    Tensor ta = make_tensor(make_smem_ptr(warp == 0 ? key : query),
                            Layout<Shape<_16, _128>, Stride<Int<136>, _1>>{});
    Tensor tb =
        make_tensor(make_smem_ptr(restored), Layout<Shape<_16, _128>, Stride<Int<136>, _1>>{});
    Tensor ref = local_tile(ta, Shape<_16, _16>{}, make_coord(0, 0));
    auto acc = thr.make_fragment_C(thr.partition_C(ref));
    clear(acc);
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        auto aa = local_tile(ta, Shape<_16, _16>{}, make_coord(0, i));
        auto bb = local_tile(tb, Shape<_16, _16>{}, make_coord(0, i));
        auto ra = thr.partition_fragment_A(aa);
        auto rb = thr.partition_fragment_B(bb);
        copy(AutoVectorizingCopy{}, thr.partition_A(aa), ra);
        copy(AutoVectorizingCopy{}, thr.partition_B(bb), rb);
        cute::gemm(thr, ra, rb, acc);
    }
    CUTE_UNROLL
    for (int j = 0; j < decltype(size(acc))::value; ++j) {
        FragCoord const c = frag_coord_mn(lane, j);
        if (warp == 0)
            cross[c.m * 20 + c.n] = acc(j) * pbeta[dst * 32 + 16 + c.m];
        else
            pmqk[dst * 1024 + (c.m + 16) * 32 + c.n] = BF16(acc(j));
    }
    __syncwarp();
    if (warp == 0) {
        pair_gemm_fp32(i1, cross, temp, lane);
        pair_gemm_fp32(temp, i0, cross, lane);
        for (int x = lane; x < 256; x += 32)
            pinv[dst * 1024 + (x / 16 + 16) * 32 + x % 16] = BF16(-cross[x / 16 * 20 + x % 16]);
    }
}

// Paired 16-token transforms, 32-token recurrent steps.
struct SharedStorageK2Paired {
    struct Stage {
        AiuTile<BF16, 96, 64, 2> tr;
        AiuTile<BF16, 32, 64, 1> v;
        AiuTile<BF16, 32, 64, 1> inv, mqk;
        alignas(128) float gt[128], beta[32];
    } in[2];
    AsyncInPipeline<2, 32, 64> pipe;
};
template <bool HasIn, bool HasOut, bool StateFP32>
__global__ __launch_bounds__(96, 3) void flash_kda_ppu_fwd_recurrence_paired(
    BF16 const *tr, float const *gt, BF16 const *inv, BF16 const *mqk, float const *beta,
    BF16 const *v, void const *initial, void *final, BF16 *out, int T, int H, int N, int vs,
    int64_t const *cu, int const *prefix, int total_pairs) {
    using namespace cute;
    __shared__ SharedStorageK2Paired sm;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    int const seq = blockIdx.x;
    int const bos = cu ? int(cu[seq]) : seq * (T / N);
    int const eos = cu ? int(cu[seq + 1]) : (seq + 1) * (T / N);
    int head = blockIdx.y, dv_cta = blockIdx.z * 32, dv = dv_cta + warp * 16;
    int const nt = (eos - bos + 31) / 32, tile_base = prefix[seq];
    auto mma =
        make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr = mma.get_slice(lane);
    Tensor ref = make_tensor(make_smem_ptr(sm.in[0].inv.data()),
                             Layout<Shape<_16, _16>, Stride<Int<40>, _1>>{});
    using AF = decltype(thr.partition_fragment_A(ref));
    using BF = decltype(thr.partition_fragment_B(ref));
    using AC = decltype(thr.make_fragment_C(thr.partition_C(ref)));
    AC state_acc[8];
    if (warp < 2) {
        CUTE_UNROLL
        for (int k = 0; k < 8; ++k) {
            CUTE_UNROLL
            for (int j = 0; j < decltype(size(state_acc[0]))::value; ++j) {
                FragCoord c = frag_coord_mn(lane, j);
                int64_t off =
                    int64_t(seq * H + head) * 128 * 128 + int64_t(dv + c.m) * 128 + k * 16 + c.n;
                if constexpr (HasIn) {
                    if constexpr (StateFP32)
                        state_acc[k](j) = static_cast<float const *>(initial)[off];
                    else
                        state_acc[k](j) = float(static_cast<BF16 const *>(initial)[off]);
                } else {
                    state_acc[k](j) = 0.f;
                }
            }
        }
    }
    sm.pipe.init(tid);
    __syncthreads();
    if (warp == 2) {
        AiuTileLoader<BF16, 96, 64, 2> lt;
        AiuTileLoader<BF16, 32, 64, 1> lv, li, lm;
        lt.init(tr, H * total_pairs * 96, 128, 128);
        // Bound the AIU load by this sequence, so padding (including NaNs)
        // never enters a recurrent MMA through a nominally zero beta.
        lv.init(v + head * 128 + (dv_cta / 64) * 64, eos, 64, vs);
        li.init(inv, H * total_pairs * 32, 32, 32);
        lm.init(mqk, H * total_pairs * 32, 32, 32);
        for (int t = 0; t < nt; ++t) {
            int st = t % 2;
            sm.pipe.producer_acquire(st, t);
            auto &z = sm.in[st];
            int64_t ix = int64_t(head) * total_pairs + tile_base + t;
            lt.load(z.tr.data(), ix * 96);
            lv.load(z.v.data(), bos + t * 32);
            li.load(z.inv.data(), ix * 32);
            lm.load(z.mqk.data(), ix * 32);
            cp_async_contiguous<32>(z.gt, gt + ix * 128, 512, lane);
            cp_async_contiguous<32>(z.beta, beta + ix * 32, 128, lane);
            cp_async_fence_all();
            kda_fence_block();
            sm.pipe.producer_commit(st);
        }
    }
    if (warp < 2) {
        for (int t = 0; t < nt; ++t) {
            int st = t % 2;
            sm.pipe.consumer_wait(st, t);
            auto &z = sm.in[st];
            AC u[2], o[2];
            int const trbase = __ppu_read_firstlane(unsigned(uintptr_t(z.tr.data()) / 16));
            int const vbase = __ppu_read_firstlane(unsigned(uintptr_t(z.v.data()) / 16));
            int const ibase = __ppu_read_firstlane(unsigned(uintptr_t(z.inv.data()) / 16));
            int const mbase = __ppu_read_firstlane(unsigned(uintptr_t(z.mqk.data()) / 16));
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                clear(u[i]);
                clear(o[i]);
            }
            CUTE_UNROLL
            for (int k = 0; k < 8; ++k) {
                AF state_k;
                narrow_into(state_k, state_acc[k]);
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    BF kb;
                    AF qa;
                    TsmOps<96, 64, 2>::load_b(&kb(0), trbase, k, i * 16);
                    TsmOps<96, 64, 2>::load_a(&qa(0), trbase, k, 32 + i * 16);
                    cute::gemm(thr, state_k, kb, u[i]);
                    cute::gemm(thr, qa, as_B(state_k), o[i]);
                }
            }
            AF before[2], U[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                BF vt;
                TsmOps<32, 64, 1>::load_bt(&vt(0), vbase, (dv_cta % 64) / 16 + warp, i * 16);
                auto vv = as_A(vt);
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(u[0]))::value; ++j) {
                    FragCoord c = frag_coord_mn(lane, j);
                    u[i](j) = (float(vv(j)) - u[i](j)) * z.beta[i * 16 + c.n];
                }
                narrow_into(before[i], u[i]);
            }

            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                clear(u[i]);
                CUTE_UNROLL
                for (int j = 0; j <= i; ++j) {
                    BF b;
                    TsmOps<32, 64, 1>::load_b(&b(0), ibase, j, i * 16);
                    cute::gemm(thr, before[j], b, u[i]);
                }
                narrow_into(U[i], u[i]);
            }

            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                CUTE_UNROLL
                for (int j = 0; j <= i; ++j) {
                    AF a;
                    TsmOps<32, 64, 1>::load_a(&a(0), mbase, j, i * 16);
                    cute::gemm(thr, a, as_B(U[j]), o[i]);
                }
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(o[0]))::value / 2; ++j) {
                    FragCoord c = frag_coord_mn(lane, 2 * j);
                    int64_t off =
                        int64_t(bos + t * 32 + i * 16 + c.m) * H * 128 + head * 128 + dv + c.n;
                    BF16 pair[2] = {BF16(o[i](2 * j)), BF16(o[i](2 * j + 1))};
                    if (bos + t * 32 + i * 16 + c.m < eos)
                        *reinterpret_cast<uint32_t *>(out + off) =
                            *reinterpret_cast<uint32_t const *>(pair);
                }
            }
            CUTE_UNROLL
            for (int k = 0; k < 8; ++k) {
                AC ds;
                clear(ds);
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    BF kr;
                    TsmOps<96, 64, 2>::load_bt(&kr(0), trbase, k, 64 + i * 16);
                    cute::gemm(thr, U[i], kr, ds);
                }
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(ds))::value; ++j) {
                    FragCoord c = frag_coord_mn(lane, j);
                    state_acc[k](j) = fmaf(state_acc[k](j), z.gt[k * 16 + c.n], ds(j));
                }
            }
            sm.pipe.consumer_release(st);
        }
    }
    __syncthreads();
    if (HasOut && warp < 2) {
        CUTE_UNROLL
        for (int k = 0; k < 8; ++k) {
            CUTE_UNROLL
            for (int j = 0; j < decltype(size(state_acc[0]))::value / 2; ++j) {
                FragCoord c = frag_coord_mn(lane, 2 * j);
                int64_t off =
                    int64_t(seq * H + head) * 128 * 128 + int64_t(dv + c.m) * 128 + k * 16 + c.n;
                if constexpr (StateFP32) {
                    *reinterpret_cast<float2 *>(static_cast<float *>(final) + off) =
                        make_float2(state_acc[k](2 * j), state_acc[k](2 * j + 1));
                } else {
                    BF16 pair[2] = {BF16(state_acc[k](2 * j)), BF16(state_acc[k](2 * j + 1))};
                    *reinterpret_cast<uint32_t *>(static_cast<BF16 *>(final) + off) =
                        *reinterpret_cast<uint32_t const *>(pair);
                }
            }
        }
    }
}

} // namespace flash_kda_ppu
