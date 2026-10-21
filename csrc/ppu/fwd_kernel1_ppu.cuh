#pragma once

// Kernel 1 -- prepare. Grid (total_tiles, H), one CTA per 16-token chunk per head.
//
// Per chunk: L2-normalise q and k over D, activate and cumulatively sum the gate,
// form the four decayed variants of q/k, build the two 16x16 matrices L and Mqk,
// mask them, invert (I + L) in FP32, and write per-tile transforms to the
// workspace for Kernel 2. Default BF16 recurrence uses precomputed affine factors.
//
// This kernel deliberately uses NEITHER the AIU NOR a matrix load. It is dominated
// by element access -- L2 norm, gate, cumulative sum, decay, triangular masks --
// and an AIU-written or matrix-read buffer may not be element-addressed at all,
// because its bytes are hardware-interleaved. Its only GEMMs are two 16x16x128,
// fed from plain padded smem by AutoVectorizingCopy. That costs roughly what a
// matrix load costs here anyway (a matrix load is ~4 instructions on this part),
// and it keeps every interleave hazard out of the kernel.

#include "epilogue.cuh"
#include "ppu_aiu.cuh"
#include "ppu_common.cuh"
#include "ppu_frag.cuh"
#include "ppu_layouts.cuh"
#include "ppu_pipeline.cuh"
#include <cstddef>
#include <type_traits>

namespace flash_kda_ppu {

// Which arithmetic the 16x16 inverse uses.
//
// BF16-resident recurrence uses FP32_FORWARD_SUBSTITUTION, regardless
// of external state dtype. The alternative k=8 fp32 atom has C
// != A (A wants k columns {t0, t0+4, t0+8, t0+12} while C holds {2t0, 2t0+1, 2t0+8, 2t0+9}), so
// each Neumann product needs a 16x16 fp32 round-trip through smem -- six in total. BF16_A_FP32_ACC
// has C == A and needs none, at the cost of 8-bit input mantissas.
//
// The accuracy argument for BF16_A_FP32_ACC, if it is ever wanted: accumulation is
// fp32 rather than the reference's fp16, L is strictly lower triangular hence
// nilpotent with L^16 = 0 so the series is exact after the L^8 term at CHUNK=16,
// and INV is narrowed to bf16 for the workspace regardless.
enum class InvPrec { FP32_EXACT, BF16_A_FP32_ACC, FP32_FORWARD_SUBSTITUTION };

// Build the prefix sum of per-sequence tile counts once, so Kernel 1 can map a
// global tile index to a sequence with a binary search instead of a linear scan
// per CTA, and Kernel 2 can look up its tile_base directly.
__global__ void flash_kda_ppu_build_tile_prefix(int64_t const *__restrict__ cu_seqlens, int N,
                                                int chunk, int *__restrict__ tile_prefix,
                                                int *full_sequence_flag = nullptr,
                                                int T_total = 0,
                                                bool initialize_serial_pair = false) {
    // This launch has one warp. Use the actual device CU buffer on every
    // invocation and graph replay; all other geometries keep the scalar path.
    if (N == 8 && chunk == 16 && T_total == 16384 &&
        blockDim.x == 32 && blockDim.y == 1 && blockDim.z == 1) {
        int const lane = int(threadIdx.x);
        int64_t const begin = lane < 8 ? cu_seqlens[lane] : int64_t(0);
        int64_t const end = lane < 8 ? cu_seqlens[lane + 1] : int64_t(0);
        int tiles = lane < 8 ? (int(end - begin) + 15) / 16 : 0;
        bool const full = __all_sync(0xffffffffu,
            lane >= 8 || (begin == int64_t(lane) * 2048 &&
                          end == int64_t(lane + 1) * 2048));
        #pragma unroll
        for (int offset = 1; offset < 8; offset <<= 1) {
            int const previous = __shfl_up_sync(0xffffffffu, tiles, offset);
            if (lane >= offset) tiles += previous;
        }
        if (lane < 8) tile_prefix[lane + 1] = tiles;
        if (lane == 0) {
            tile_prefix[0] = 0;
            if (full_sequence_flag) {
                full_sequence_flag[0] = full;
                if (initialize_serial_pair) full_sequence_flag[1] = full;
            }
        }
        return;
    }
    if (threadIdx.x == 0) {
        int acc = 0;
        // The aligned prefix allocation has room for one flag at N=1. The
        // segmented path covers the full input; padded/sliced sequences use
        // the regular varlen recurrence without a host read or synchronization.
        if (full_sequence_flag) {
            bool full = T_total % N == 0;
            for (int i = 0; i <= N; ++i)
                full &= cu_seqlens[i] == int64_t(i) * (T_total / N);
            *full_sequence_flag = full;
            // Only the qualified N8 caller reserves this second flag.
            // Both are current-call eligibility, initialized before prepare.
            if (initialize_serial_pair) full_sequence_flag[1] = full;
        }
        tile_prefix[0] = 0;
        for (int i = 0; i < N; ++i) {
            int const slen = int(cu_seqlens[i + 1] - cu_seqlens[i]);
            acc += (slen + chunk - 1) / chunk;
            tile_prefix[i + 1] = acc;
        }
    }
}

template <int D, int CHUNK, bool NeedF32Work = true> struct SharedStorageK1 {
    using L = K1Layouts<D, CHUNK>;
    static constexpr int kPad = L::kColsPad;

    // Phase A holds q, k and the raw gate; phase B holds the four decayed tiles.
    // Their lifetimes do not overlap, and the union is what keeps K1 small enough
    // to run many CTAs per CU.
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

    alignas(128) float dt_bias[D];
    alignas(128) float g_total[D];
    alignas(128) float beta[32];

    // L and INV in fp32 for the exact inverse; Mqk and the narrowed INV in bf16.
    alignas(128) float Lf32[L::LMF32::kElems];
    // The parallel row solve publishes INVbf directly; other algorithms keep
    // their original FP32 work arrays.
    alignas(128) float INVf32[NeedF32Work ? L::LMF32::kElems : 1];
    alignas(128) float scratch[NeedF32Work ? L::LMF32::kElems : 1]; // Neumann C -> A round-trip
    alignas(128) BF16 Mqk[L::LMBF16::kElems];
    alignas(128) BF16 INVbf[L::LMBF16::kElems];
};

// At128 threads, keep the Gram outputs in k_inv after both reader warps
// finish. Their complete work region fits before the still-live k_restored.
template <int D, int CHUNK> struct SharedStorageK1Packed {
    using L = K1Layouts<D, CHUNK>;
    static_assert(D == 128 && CHUNK == 16);
    static_assert(L::QK::kElems == L::Tile::kElems && L::Tile::kBytes % 128 == 0);
    struct PhaseA {
        alignas(128) BF16 q[L::QK::kElems];
        alignas(128) BF16 k[L::QK::kElems];
        alignas(128) BF16 g_raw[L::QK::kElems];
    };
    struct PhaseB {
        alignas(128) BF16 k_decayed[L::Tile::kElems];
        alignas(128) BF16 q_decayed[L::Tile::kElems];
        alignas(128) BF16 k_inv[L::Tile::kElems];
        alignas(128) BF16 k_restored[L::Tile::kElems];
    };
    struct GramWork {
        alignas(128) BF16 prefix_padding[2 * L::Tile::kElems];
        alignas(128) float Lf32[L::LMF32::kElems];
        alignas(128) BF16 Mqk[L::LMBF16::kElems];
        alignas(128) BF16 INVbf[L::LMBF16::kElems];
    };
    static_assert(offsetof(GramWork, Lf32) == 2 * L::Tile::kBytes);
    static_assert(offsetof(PhaseB, k_restored) == 3 * L::Tile::kBytes);
    static_assert(offsetof(GramWork, INVbf) + L::LMBF16::kBytes <=
                  offsetof(PhaseB, k_restored));
    union { PhaseA a; PhaseB b; GramWork gram; };
    alignas(128) float dt_bias[D];
    alignas(128) float g_total[D];
    alignas(128) float beta[32];
};

// ---------------------------------------------------------------------------
// FP32 inverse of (I + L), one warp
// ---------------------------------------------------------------------------
// The Neumann variant starts with INV = I - L, then INV += INV*L^2,
// INV += INV*L^4, INV += INV*L^8. Exact at
// CHUNK=16 because L is strictly lower triangular, so L^16 = 0.
//
// `A <-> B` is a free register relabel on both atoms, which is what makes
// L^2 = L*L need only one smem read of L rather than two.
template <int D, int CHUNK, InvPrec Prec>
CUTE_DEVICE void chunk_inverse_1warp(SharedStorageK1<D, CHUNK> &sm, int lane) {
    using namespace cute;
    using L1 = K1Layouts<D, CHUNK>;

    if constexpr (Prec == InvPrec::FP32_FORWARD_SUBSTITUTION) {
        static_assert(CHUNK == 16);
        // One lane per inverse column. All 32 lanes execute shuffles; the upper
        // half duplicates 16 columns so the full-warp source mask remains valid.
        float column[16];
        CUTE_UNROLL
        for (int i = 0; i < 16; ++i)
            column[i] = 0.f;
        CUTE_UNROLL
        for (int row = 0; row < 16; ++row) {
            float l = sm.Lf32[row * 20 + (lane & 15)];
            float x = (lane & 15) == row ? 1.f : 0.f;
            CUTE_UNROLL
            for (int k = 0; k < row; ++k)
                x = fmaf(-__shfl_sync(0xffffffffu, l, k), column[k], x);
            column[row] = x;
            if (lane < 16)
                sm.INVf32[row * 20 + lane] = x;
        }
        __syncwarp();
    } else if constexpr (Prec == InvPrec::FP32_EXACT) {
#if FLASH_KDA_PPU_TC02
        auto mma =
            make_tiled_mma(MMA_Atom<MmaF32>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _8>{});
        auto thr = mma.get_slice(lane);

        Tensor tL = make_tensor(make_smem_ptr(sm.Lf32), typename L1::LMF32::Layout{});
        Tensor tS = make_tensor(make_smem_ptr(sm.scratch), typename L1::LMF32::Layout{});
        Tensor tI = make_tensor(make_smem_ptr(sm.INVf32), typename L1::LMF32::Layout{});

        // 16x16 = two k=8 steps. The padded row stride survives local_tile; the
        // logical shape has to be tileable by the atom.
        auto gemm_16x16 = [&](float const *a_smem, float const *b_smem, float *out_smem) {
            Tensor ta = make_tensor(make_smem_ptr(a_smem), typename L1::LMF32::Layout{});
            // The tc02 _TN atom computes A[m,k] * B[n,k].  Present B through a
            // transposed view so this helper implements the ordinary A @ B,
            // rather than A @ B^T.  The latter destroys L's nilpotence and makes
            // the repeated squaring below explode for correlated keys.
            Tensor tb = make_tensor(make_smem_ptr(b_smem),
                                    Layout<Shape<_16, _16>, Stride<_1, Int<CHUNK + 4>>>{});
            Tensor to = make_tensor(make_smem_ptr(out_smem), typename L1::LMF32::Layout{});

            auto acc = thr.make_fragment_C(
                thr.partition_C(local_tile(to, Shape<_16, _16>{}, make_coord(0, 0))));
            clear(acc);

            CUTE_UNROLL
            for (int kb = 0; kb < 2; ++kb) {
                Tensor a_k = local_tile(ta, Shape<_16, _8>{}, make_coord(0, kb));
                Tensor b_k = local_tile(tb, Shape<_16, _8>{}, make_coord(0, kb));
                auto ra = thr.partition_fragment_A(a_k);
                auto rb = thr.partition_fragment_B(b_k);
                copy(AutoVectorizingCopy{}, thr.partition_A(a_k), ra);
                copy(AutoVectorizingCopy{}, thr.partition_B(b_k), rb);
                cute::gemm(thr, ra, rb, acc);
            }

            // C -> A is not free on this atom, so the product has to land in smem
            // before it can be an operand again.
            auto c = thr.partition_C(local_tile(to, Shape<_16, _16>{}, make_coord(0, 0)));
            copy(AutoVectorizingCopy{}, acc, c);
            __syncwarp();
        };

        // Lpow = L^2, in `scratch`.
        gemm_16x16(sm.Lf32, sm.Lf32, sm.scratch);
        CUTE_UNROLL
        for (int step = 0; step < 3; ++step) {
            // INV += INV * Lpow. Accumulate into INV in place via a temporary
            // fragment, so no fourth 16x16 buffer is needed.
            {
                Tensor to = make_tensor(make_smem_ptr(sm.INVf32), typename L1::LMF32::Layout{});
                auto acc = thr.make_fragment_C(
                    thr.partition_C(local_tile(to, Shape<_16, _16>{}, make_coord(0, 0))));
                clear(acc);
                Tensor ti = make_tensor(make_smem_ptr(sm.INVf32), typename L1::LMF32::Layout{});
                Tensor tp = make_tensor(make_smem_ptr(sm.scratch),
                                        Layout<Shape<_16, _16>, Stride<_1, Int<CHUNK + 4>>>{});
                CUTE_UNROLL
                for (int kb = 0; kb < 2; ++kb) {
                    Tensor a_k = local_tile(ti, Shape<_16, _8>{}, make_coord(0, kb));
                    Tensor b_k = local_tile(tp, Shape<_16, _8>{}, make_coord(0, kb));
                    auto ra = thr.partition_fragment_A(a_k);
                    auto rb = thr.partition_fragment_B(b_k);
                    copy(AutoVectorizingCopy{}, thr.partition_A(a_k), ra);
                    copy(AutoVectorizingCopy{}, thr.partition_B(b_k), rb);
                    cute::gemm(thr, ra, rb, acc);
                }
                auto c = thr.partition_C(local_tile(ti, Shape<_16, _16>{}, make_coord(0, 0)));
                CUTE_UNROLL
                for (int v = 0; v < decltype(size(acc))::value; ++v)
                    c(v) += acc(v);
                __syncwarp();
            }
            // Lpow = Lpow^2, except after the last update where it is unused.
            if (step < 2) {
                gemm_16x16(sm.scratch, sm.scratch, sm.Lf32);
                // Lf32 now holds the next power; swap roles by copying back, which
                // is cheaper than carrying a fourth buffer for a 1 KB matrix.
                for (int i = lane; i < L1::LMF32::kElems; i += 32) {
                    float const t = sm.Lf32[i];
                    sm.Lf32[i] = sm.scratch[i];
                    sm.scratch[i] = t;
                }
                __syncwarp();
            }
        }
#endif
    } else {
        // bf16 operands, fp32 accumulate, C == A so no round-trips. Reads L from
        // the fp32 buffer and narrows on the way into the fragment.
        using namespace cute;
        auto mma =
            make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
        auto thr = mma.get_slice(lane);

        Tensor tL = make_tensor(make_smem_ptr(sm.Lf32), typename L1::LMF32::Layout{});
        Tensor tI = make_tensor(make_smem_ptr(sm.INVf32), typename L1::LMF32::Layout{});
        Tensor lv = local_tile(tL, Shape<_16, _16>{}, make_coord(0, 0));
        Tensor iv = local_tile(tI, Shape<_16, _16>{}, make_coord(0, 0));

        // decltype(lv), not Tensor<decltype(lv.engine()), ...>: engine() returns a
        // reference, so that spelling builds Tensor<Engine&, Layout> and fails to
        // instantiate. iv has the same type as lv, so one parameter type serves both.
        auto load_bf16_A = [&](decltype(lv) const &t) {
            auto f32 = thr.partition_fragment_A(t);
            copy(AutoVectorizingCopy{}, thr.partition_A(t), f32);
            auto bf = make_fragment_like<BF16>(f32);
            CUTE_UNROLL
            for (int i = 0; i < decltype(size(f32))::value; ++i)
                bf(i) = BF16(f32(i));
            return bf;
        };

        auto rL = load_bf16_A(lv);
        auto rI = load_bf16_A(iv);

        auto acc = thr.make_fragment_C(thr.partition_C(iv));

        // Lpow = L^2
        clear(acc);
        cute::gemm(thr, rL, as_B(rL), acc);
        auto rP = c_to_a_bf16(acc);

        CUTE_UNROLL
        for (int step = 0; step < 3; ++step) {
            // INV += INV * Lpow
            clear(acc);
            cute::gemm(thr, rI, as_B(rP), acc);
            CUTE_UNROLL
            for (int i = 0; i < decltype(size(rI))::value; ++i) {
                rI(i) = BF16(float(rI(i)) + acc(i));
            }
            if (step < 2) {
                clear(acc);
                cute::gemm(thr, rP, as_B(rP), acc);
                narrow_into(rP, acc);
            }
        }

        // Write the result back to the fp32 buffer so the store path below is
        // shared with the exact variant.
        auto c = thr.partition_C(iv);
        CUTE_UNROLL
        for (int i = 0; i < decltype(size(rI))::value; ++i)
            c(i) = float(rI(i));
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// Kernel 1
// ---------------------------------------------------------------------------
// The first row owned by a 128-thread half warp lies in [0,7]. For a
// finite strictly-lower matrix, its inverse entries at pivots15..8 stay +0;
// those shuffle/FMA operations are exact no-ops. Nonfinite matrices retain
// the original complete operation sequence.
template<int NumThreads,bool PruneUpperRows,int WarpPair = -1>
CUTE_DEVICE void solve_parallel_inverse_rows(
    float (&inverse)[256/NumThreads],float const *lower,int col,int lane) {
    constexpr int Rows=256/NumThreads;
    static_assert(!PruneUpperRows || NumThreads == 128);
    CUTE_UNROLL
    for(int pivot=15;pivot>0;--pivot) {
        // Gram publication explicitly writes +0 to the diagonal and upper
        // triangle. Loading those values preserves the old masked +0 operand
        // while keeping every lane on the same shared-load path.
        float const factor=lower[pivot*20+col];
        CUTE_UNROLL
        for(int r=0;r<Rows;++r) {
            if(!PruneUpperRows || pivot <= (WarpPair < 0 ? (r+1)*(NumThreads/16)-1 : r*8+2*WarpPair+1)) {
                float const value=__shfl_sync(0xffffffffu,inverse[r],(lane&16)+pivot);
                inverse[r]=fmaf(-factor,value,inverse[r]);
            }
        }
    }
}

// Fixed C16 input-copy ownership; the generic helpers remain unchanged.
template <int ROWS, int COLS, int COLS_PAD, int NumThreads, bool FullRows = false>
CUTE_DEVICE void cp_async_c16_prepare_tile(BF16 *__restrict__ smem, BF16 const *__restrict__ gmem,
                               int gmem_row_stride, int rows_valid, int tid) {
    static_assert(COLS % 8 == 0, "16-byte vectors require COLS to be a multiple of 8");
    constexpr int kElemsPerVec = 16 / int(sizeof(BF16)); // 8
    constexpr int kVecsPerRow = COLS / kElemsPerVec;
    constexpr int kVecs = ROWS * kVecsPerRow;

    static_assert(ROWS == 16 && COLS == 128 &&
                  (NumThreads == 128 || NumThreads == 256));
    static_assert(kVecs % NumThreads == 0);
    // The caller masks tid to the legal one-dimensional block range. Every
    // thread therefore owns exactly two (128T) or one (256T) vector slots.
    // Keep its original ascending copy order, including tail zero filling.
    CUTE_UNROLL
    for (int slot = 0; slot < kVecs / NumThreads; ++slot) {
        int const i = tid + slot * NumThreads;
        int const r = i / kVecsPerRow;
        int const c = (i % kVecsPerRow) * kElemsPerVec;
        BF16 *dst = smem + r * COLS_PAD + c;
        if (FullRows || r < rows_valid) {
            CpAsyncVec16::copy(
                *reinterpret_cast<cute::uint128_t const *>(gmem + r * gmem_row_stride + c),
                *reinterpret_cast<cute::uint128_t *>(dst));
        } else {
            CUTE_UNROLL
            for (int e = 0; e < kElemsPerVec; ++e)
                dst[e] = BF16(0);
        }
    }
}

template <int CHUNK, int D, int NumThreads, bool IsVarlen, InvPrec Prec, bool ValidateScan = false,
          bool AffinePrepared = false, int StaticHeads = 0, bool BuildSinglePrefix = false,
          bool GuardWeakP3 = false>
__global__ void __launch_bounds__(NumThreads, 4)
    flash_kda_ppu_fwd_prepare(BF16 const *__restrict__ q_ptr, BF16 const *__restrict__ k_ptr,
                              BF16 const *__restrict__ g_ptr, BF16 const *__restrict__ beta_ptr,
                              float const *__restrict__ A_log_ptr,
                              float const *__restrict__ dt_bias_ptr, BF16 *__restrict__ ws_kd,
                              BF16 *__restrict__ ws_qd, BF16 *__restrict__ ws_kr,
                              float *__restrict__ ws_gt, BF16 *__restrict__ ws_inv,
                              BF16 *__restrict__ ws_mqk, float *__restrict__ ws_beta, float scale,
                              float gate_scale, int T_total, int H_runtime, int N,
                              int64_t const *__restrict__ cu_seqlens, int total_tiles,
                              int *__restrict__ tile_prefix, int *__restrict__ scan_flag,
                              float scan_threshold, int q_stride_runtime, int k_stride_runtime,
                              int *__restrict__ full_sequence_flag = nullptr) {
    static_assert(StaticHeads == 0 || StaticHeads == 6 || StaticHeads == 12);
    int const H = StaticHeads ? StaticHeads : H_runtime;
    int const q_stride = StaticHeads ? 3 * StaticHeads * D : q_stride_runtime;
    int const k_stride = StaticHeads ? 3 * StaticHeads * D : k_stride_runtime;
    using namespace cute;
    using L1 = K1Layouts<D, CHUNK>;
    constexpr int kPad = L1::kColsPad;

    constexpr bool kNeedF32Work = !(AffinePrepared && Prec == InvPrec::FP32_FORWARD_SUBSTITUTION);
    constexpr bool kAliasGram = NumThreads == 128 && !kNeedF32Work;
    constexpr bool kOverlapGramStores = kAliasGram && BuildSinglePrefix;
    constexpr bool kTransposeInverse = NumThreads == 128 && AffinePrepared &&
                                       Prec == InvPrec::FP32_FORWARD_SUBSTITUTION;
    using PreparedStorage = std::conditional_t<kAliasGram, SharedStorageK1Packed<D, CHUNK>,
                                               SharedStorageK1<D, CHUNK, kNeedF32Work>>;
    static_assert(!kAliasGram || sizeof(PreparedStorage) ==
                  4 * L1::Tile::kBytes + 2 * D * sizeof(float) + 32 * sizeof(float));
    __shared__ PreparedStorage sm;
    auto &gram = [&]() -> auto & {
        if constexpr (kAliasGram) return sm.gram;
        else return sm;
    }();

    int const global_tile = blockIdx.x;
    int const head_idx = blockIdx.y;
    // Every C16 launcher uses a one-dimensional NumThreads-sized block.
    // The existing 128/256-thread static constraint makes this an identity,
    // while exposing the actual range to copy/norm/address optimization.
    int const tid = int(threadIdx.x) & (NumThreads - 1);
    // Preserve the converged warp broadcast, then expose the legal 4/8-warp
    // index range of this one-dimensional NumThreads-sized C16 block.
    int const warp_idx = cutlass::canonical_warp_idx_sync() & (NumThreads / 32 - 1);

    int seq_idx, local_t;
    int64_t bos, eos;
    if constexpr (BuildSinglePrefix) {
        static_assert(IsVarlen && CHUNK == 16);
        // No CTA in this specialization reads the prefix being published.
        // Stream completion makes it visible to all later recurrence launches.
        seq_idx = 0;
        local_t = global_tile;
        bos = cu_seqlens[0];
        eos = cu_seqlens[1];
        if (global_tile == 0 && head_idx == 0 && tid == 0) {
            tile_prefix[0] = 0;
            tile_prefix[1] = (int(eos - bos) + CHUNK - 1) / CHUNK;
            if (full_sequence_flag)
                *full_sequence_flag = bos == 0 && eos == T_total;
        }
    } else if constexpr (IsVarlen) {
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
    // This device-side condition is uniform within the CTA and is rechecked
    // on every invocation/replay. Partial tiles keep the original zero-fill path.
    // The256-thread route retains its original normalization packing.
    if (NumThreads == 128 && rows_valid == CHUNK) {
        cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads, true>(sm.a.q, q_tile, q_stride, rows_valid, tid);
        cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads, true>(sm.a.k, k_tile, k_stride, rows_valid, tid);
        cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads, true>(sm.a.g_raw, g_tile, gmem_row_stride, rows_valid, tid);
    } else {
        cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads>(sm.a.q, q_tile, q_stride, rows_valid, tid);
        cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads>(sm.a.k, k_tile, k_stride, rows_valid, tid);
        cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads>(sm.a.g_raw, g_tile, gmem_row_stride, rows_valid, tid);
    }
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
        int const col = (tid % kThreadsPerRow) * kPerThread;

        for (int row = tid / kThreadsPerRow; row < CHUNK; row += NumThreads / kThreadsPerRow) {
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
    // Gate reads a disjoint phase-A buffer; the following barrier also completes
    // normalization before decay reads Q/K and reuses the phase-A storage.

    // Two8-token or four4-token partitions share adjacent gate column pairs.
    // Pack all owner-local phase-A reads and phase-B writes into32-bit words.
    static_assert(CHUNK == 16 && (NumThreads == 128 || NumThreads == 256) && D == 128);
    static_assert(L1::QK::kElems == L1::Tile::kElems);
    {
        constexpr int kParts = NumThreads == 128 ? 2 : 4;
        constexpr int kPairLanes = 32 / kParts, kPartRows = CHUNK / kParts;
        int const lane = tid%32, part = lane/kPairLanes;
        int const pair = (warp_idx)*kPairLanes+lane%kPairLanes, col = pair*2;
        float const dt[2] = {sm.dt_bias[col],sm.dt_bias[col+1]};
        float prefix[2][kPartRows], sum[2] = {0.f,0.f};
        auto read_pair = [](BF16 const *p) {
            uint32_t const bits = *reinterpret_cast<uint32_t const *>(p);
            return __bfloat1622float2(*reinterpret_cast<__ppu_bfloat162 const *>(&bits));
        };
        auto write_pair = [](BF16 *p, float x, float y) {
            __ppu_bfloat162 const packed = __floats2bfloat162_rn(x,y);
            *reinterpret_cast<uint32_t *>(p) = *reinterpret_cast<uint32_t const *>(&packed);
        };
        if constexpr (GuardWeakP3 || BuildSinglePrefix) {
            // Expose the independent SFU work before the ordered accumulation.
            // Reuse the existing prefix slots, storing unscaled sigmoid values
            // so the original scale/multiply/add expression stays in the scan.
            CUTE_UNROLL
            for (int i = 0; i < kPartRows; ++i) {
                int const row = part*kPartRows+i;
                float2 const raw = read_pair(sm.a.g_raw+row*kPad+col);
                float const g[2] = {raw.x,raw.y};
                CUTE_UNROLL
                for (int c = 0; c < 2; ++c)
                    // Padded raw rows are already zero-filled in shared memory.
                    // Their unused activations are discarded by the original
                    // row conditional in the following accumulation stage.
                    prefix[c][i] = kda_sigmoid(a_log_exp*(g[c]+dt[c]));
            }
            CUTE_UNROLL
            for (int i = 0; i < kPartRows; ++i) {
                int const row = part*kPartRows+i;
                CUTE_UNROLL
                for (int c = 0; c < 2; ++c) {
                    float const value = row < rows_valid
                        ? gate_scale*prefix[c][i] : 0.f;
                    sum[c] += value;
                    prefix[c][i] = sum[c];
                }
            }
        } else {
            CUTE_UNROLL
            for (int i = 0; i < kPartRows; ++i) {
                int const row = part*kPartRows+i;
                float2 const raw = read_pair(sm.a.g_raw+row*kPad+col);
                float const g[2] = {raw.x,raw.y};
                CUTE_UNROLL
                for (int c = 0; c < 2; ++c) {
                    float const value = row < rows_valid
                        ? gate_scale*kda_sigmoid(a_log_exp*(g[c]+dt[c])) : 0.f;
                    sum[c] += value;
                    prefix[c][i] = sum[c];
                }
            }
        }
        float preceding[2], total_decay[2];
        CUTE_UNROLL
        for (int c = 0; c < 2; ++c) {
            float scan = sum[c];
            CUTE_UNROLL
            for (int delta = 1; delta < kParts; delta *= 2) {
                float const other = __shfl_up_sync(0xffffffffu,scan,delta*kPairLanes);
                if (part >= delta) scan += other;
            }
            float const previous = __shfl_up_sync(0xffffffffu,scan,kPairLanes);
            preceding[c] = part ? previous : 0.f;
            float const total = __shfl_sync(0xffffffffu,scan,(lane%kPairLanes)+32-kPairLanes);
            total_decay[c] = kda_exp2(total);
            if (part == kParts-1) {
                sm.g_total[col+c] = total_decay[c];
                if constexpr (ValidateScan) {
                    if (!(total_decay[c] >= 0.f && total_decay[c] <= scan_threshold))
                        atomicExch(scan_flag,0);
                }
            }
        }
        if constexpr (GuardWeakP3) {
            static_assert(IsVarlen && CHUNK == 16 && NumThreads == 128 &&
                          !BuildSinglePrefix);
            // Only clear scheduling eligibility. Every enabled family still
            // requires its complete exact checkpoint before reusing a state.
            bool const p3_prefix = (local_t >= 43 && local_t < 46) ||
                                   (local_t >= 86 && local_t < 89);
            bool const p2_prefix = local_t >= 64 && local_t < 67;
            if (p3_prefix || p2_prefix) {
                bool const weak = !(total_decay[0] >= 0.f && total_decay[0] <= 0.125f) ||
                                  !(total_decay[1] >= 0.f && total_decay[1] <= 0.125f);
                if (__any_sync(0xffffffffu, weak) && lane == 0)
                    atomicExch(full_sequence_flag + (p2_prefix ? 1 : 0), 0);
            }
        }
        // Complete normalization/raw gate reads before overwriting their
        // aliased words. Every word has exactly one gate/decay owner.
        __syncthreads();
        CUTE_UNROLL
        for (int i = 0; i < kPartRows; ++i) {
            int const o = (part*kPartRows+i)*kPad+col;
            float2 const q = read_pair(sm.a.q+o), k = read_pair(sm.a.k+o);
            float const dec0 = kda_exp2(prefix[0][i]+preceding[0]);
            float const dec1 = kda_exp2(prefix[1][i]+preceding[1]);
            float const inv0 = __ppu_rcpf(dec0), inv1 = __ppu_rcpf(dec1);
            write_pair(sm.b.q_decayed+o,q.x*dec0*scale,q.y*dec1*scale);
            write_pair(sm.b.k_decayed+o,k.x*dec0,k.y*dec1);
            write_pair(sm.b.k_inv+o,k.x*inv0,k.y*inv1);
            write_pair(sm.b.k_restored+o,k.x*inv0*total_decay[0],k.y*inv1*total_decay[1]);
        }
    }
    __syncthreads();

    int const ws_idx = head_idx * total_tiles + global_tile;

    // --- L = k_decayed @ k_inv^T and Mqk = q_decayed @ k_inv^T, on two warps.
    // Both are 16x16 with K = D, so 8 k-steps of the 16x16x16 atom.
    {
        auto mma =
            make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
        int const warp = warp_idx;
        int const lane = tid % 32;

        if (warp < 2) {
            auto thr = mma.get_slice(lane);
            BF16 const *a_base = (warp == 0) ? sm.b.k_decayed : sm.b.q_decayed;

            Tensor ta = make_tensor(make_smem_ptr(a_base), typename L1::Tile::Layout{});
            Tensor tb = make_tensor(make_smem_ptr(sm.b.k_inv), typename L1::Tile::Layout{});
            Tensor tc = make_tensor(make_smem_ptr(gram.Mqk), typename L1::LMBF16::Layout{});

            auto acc = thr.make_fragment_C(
                thr.partition_C(local_tile(tc, Shape<_16, _16>{}, make_coord(0, 0))));
            clear(acc);

            CUTE_UNROLL
            for (int kb = 0; kb < D / 16; ++kb) {
                Tensor a_k = local_tile(ta, Shape<_16, _16>{}, make_coord(0, kb));
                Tensor b_k = local_tile(tb, Shape<_16, _16>{}, make_coord(0, kb));
                auto ra = thr.partition_fragment_A(a_k);
                auto rb = thr.partition_fragment_B(b_k);
                copy(AutoVectorizingCopy{}, thr.partition_A(a_k), ra);
                copy(AutoVectorizingCopy{}, thr.partition_B(b_k), rb);
                cute::gemm(thr, ra, rb, acc);
            }

            if constexpr (kAliasGram) {
                // Both Gram warps must finish every k_inv read before either
                // publishes L/Mqk into the same bytes. Other warps wait at the
                // original following CTA barrier.
                kda_fence_block();
                asm volatile("" ::: "memory");
                cutlass::arch::NamedBarrier::arrive_and_wait(64, 0);
                asm volatile("" ::: "memory");
            }

            // Masks, beta scaling and INV = I - L are applied in the C fragment
            // via the decoded (m, n), which removes an smem round-trip and a
            // __syncthreads() that the CUDA kernel needed.
            if (warp == 0) {
                uint32_t nonfinite_lower = 0;
                CUTE_UNROLL
                for (int v = 0; v < decltype(size(acc))::value; ++v) {
                    FragCoord const c = frag_coord_mn(lane, v);
                    // Strictly lower triangular, scaled by sigmoid(beta[row]).
                    float const l = (c.m <= c.n) ? 0.0f : acc(v) * sm.beta[c.m];
                    gram.Lf32[c.m * (CHUNK + 4) + c.n] = l;
                    if constexpr (NumThreads == 128 && AffinePrepared &&
                                  Prec == InvPrec::FP32_FORWARD_SUBSTITUTION)
                        nonfinite_lower |= (__float_as_uint(l) & 0x7f800000u) == 0x7f800000u;
                    if constexpr (kNeedF32Work)
                        sm.INVf32[c.m * (CHUNK + 4) + c.n] = (c.m == c.n) ? 1.0f - l : -l;
                }
                if constexpr (NumThreads == 128 && AffinePrepared &&
                              Prec == InvPrec::FP32_FORWARD_SUBSTITUTION) {
                    bool const any_nonfinite = __any_sync(0xffffffffu,nonfinite_lower != 0);
                    // beta[0:16] remains live; slot16 is unused padding.
                    if (lane == 0) reinterpret_cast<int *>(sm.beta)[16] = any_nonfinite;
                }
            } else {
                CUTE_UNROLL
                for (int v = 0; v < decltype(size(acc))::value; ++v) {
                    FragCoord const c = frag_coord_mn(lane, v);
                    gram.Mqk[c.m * (CHUNK + 8) + c.n] = BF16((c.m < c.n) ? 0.0f : acc(v));
                }
            }
        } else if constexpr (kOverlapGramStores) {
            // These two warps otherwise wait for the Gram pair. The first two
            // tiles and g_total remain read-only and disjoint from its overlaid
            // L/Mqk workspace. The existing CTA barrier joins both tasks before
            // any inverse/affine consumer proceeds.
            constexpr int kCopyThreads = 64;
            int const copy_tid = tid - 64;
            store_tile_2d_full<CHUNK,D,kPad,kCopyThreads>(
                ws_kd+int64_t(ws_idx)*3*CHUNK*D,D,sm.b.k_decayed,copy_tid);
            store_tile_2d_full<CHUNK,D,kPad,kCopyThreads>(
                ws_qd+int64_t(ws_idx)*3*CHUNK*D,D,sm.b.q_decayed,copy_tid);
            store_contiguous<kCopyThreads>(ws_gt+int64_t(ws_idx)*D,sm.g_total,
                                          D*int(sizeof(float)),copy_tid);
        }
    }
    __syncthreads();

    // BF16 affine preparation distributes independent FP32 inverse rows over
    // all resident warps. Other inverse algorithms keep their warp-zero inverse,
    // overlapped with Kd/Qd/gate writeback. The affine factors are produced only
    // after the CTA barrier makes every completed inverse row visible.
    if constexpr (AffinePrepared && Prec == InvPrec::FP32_FORWARD_SUBSTITUTION) {
        static_assert((NumThreads == 128 || NumThreads == 256) && CHUNK == 16);
        if constexpr (!kOverlapGramStores) {
        store_tile_2d_full<CHUNK, D, kPad, NumThreads>(ws_kd + int64_t(ws_idx) * 3 * CHUNK * D, D,
                                                       sm.b.k_decayed, tid);
        store_tile_2d_full<CHUNK, D, kPad, NumThreads>(ws_qd + int64_t(ws_idx) * 3 * CHUNK * D, D,
                                                       sm.b.q_decayed, tid);
        store_contiguous<NumThreads>(ws_gt + int64_t(ws_idx) * D, sm.g_total,
                                     D * int(sizeof(float)), tid);
        }

        // Solve each inverse row independently from right to left. A half-warp
        // owns one or two rows and works through 15 FP32 pivots instead
        // of waiting for one warp's 120 dependent dot-product FMAs.
        constexpr int kRowsPerHalfWarp = 256 / NumThreads;
        int const first_row = tid / 16, col = tid % 16, lane = tid % 32;
        float inverse[kRowsPerHalfWarp];
        CUTE_UNROLL
        for (int r = 0; r < kRowsPerHalfWarp; ++r)
            inverse[r] = first_row + r * (NumThreads / 16) == col ? 1.f : 0.f;
        if constexpr (NumThreads == 128) {
            int const nonfinite = __shfl_sync(0xffffffffu,
                reinterpret_cast<int const *>(sm.beta)[16],0);
            if (nonfinite == 0) {
                // Each arm is uniform across a complete warp. Keep the full
                // shuffle mask while specializing the highest live row pair.
                if (warp_idx == 0)
                    solve_parallel_inverse_rows<NumThreads,true,0>(inverse,gram.Lf32,col,lane);
                else if (warp_idx == 1)
                    solve_parallel_inverse_rows<NumThreads,true,1>(inverse,gram.Lf32,col,lane);
                else if (warp_idx == 2)
                    solve_parallel_inverse_rows<NumThreads,true,2>(inverse,gram.Lf32,col,lane);
                else
                    solve_parallel_inverse_rows<NumThreads,true,3>(inverse,gram.Lf32,col,lane);
            } else
                solve_parallel_inverse_rows<NumThreads,false>(inverse,gram.Lf32,col,lane);
        } else {
            solve_parallel_inverse_rows<NumThreads,false>(inverse,gram.Lf32,col,lane);
        }
        CUTE_UNROLL
        for (int r = 0; r < kRowsPerHalfWarp; ++r) {
            int const row = first_row + r * (NumThreads / 16);
            constexpr int kInvPad = CHUNK + 8;
            int const offset = kTransposeInverse ? col * kInvPad + row
                                                : row * kInvPad + col;
            gram.INVbf[offset] = BF16(inverse[r]);
        }
    } else {
        if (tid < 32) {
            chunk_inverse_1warp<D, CHUNK, Prec>(sm, tid);
        } else {
            constexpr int kStoreThreads = NumThreads - 32;
            int const stid = tid - 32;
            store_tile_2d_full<CHUNK, D, kPad, kStoreThreads>(
                ws_kd + int64_t(ws_idx) * 3 * CHUNK * D, D, sm.b.k_decayed, stid);
            store_tile_2d_full<CHUNK, D, kPad, kStoreThreads>(
                ws_qd + int64_t(ws_idx) * 3 * CHUNK * D, D, sm.b.q_decayed, stid);
            if constexpr (!AffinePrepared) {
                store_tile_2d_full<CHUNK, D, kPad, kStoreThreads>(
                    ws_kr + int64_t(ws_idx) * 3 * CHUNK * D, D, sm.b.k_restored, stid);
            }
            store_contiguous<kStoreThreads>(ws_gt + int64_t(ws_idx) * D, sm.g_total,
                                            D * int(sizeof(float)), stid);
            if constexpr (!AffinePrepared) {
                store_tile_2d_full<CHUNK, CHUNK, CHUNK + 8, kStoreThreads>(
                    ws_mqk + int64_t(ws_idx) * CHUNK * CHUNK, CHUNK, gram.Mqk, stid);
                store_contiguous<kStoreThreads>(ws_beta + int64_t(ws_idx) *
                                                              (WorkspaceSizes<CHUNK, D>::kBeta / 4),
                                                sm.beta, CHUNK * int(sizeof(float)), stid);
            }
        }
    }
    if constexpr (!(AffinePrepared && Prec == InvPrec::FP32_FORWARD_SUBSTITUTION)) {
        __syncthreads();
        // Other inverse algorithms publish FP32 rows for this separate narrowing.
        for (int i = tid; i < CHUNK * CHUNK; i += NumThreads) {
            int const r = i / CHUNK, c = i % CHUNK;
            gram.INVbf[r * (CHUNK + 8) + c] = BF16(sm.INVf32[r * (CHUNK + 4) + c]);
        }
    }
    // Every affine consumer sees all completed BF16 inverse rows.
    __syncthreads();
    if constexpr (AffinePrepared) {
        // Hoist the state-independent inverse/beta products out of the serial
        // recurrence. W = diag(beta) INV^T Kr, M = Mqk INV diag(beta).
        // K2 consumes r = V^T - S Kd^T, then S' = S g + r W and
        // out = Qd S^T + M r^T. Reuse the existing Kr/Mqk workspace slots.
        static_assert(CHUNK == 16 && (NumThreads == 128 || NumThreads == 256) && D == 128);
        auto mma =
            make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
        int const warp = warp_idx, lane = tid % 32;
        auto thr = mma.get_slice(lane);
        using InvView = std::conditional_t<kTransposeInverse,
            Layout<Shape<_16, _16>, Stride<Int<CHUNK + 8>, _1>>,
            Layout<Shape<_16, _16>, Stride<_1, Int<CHUNK + 8>>>>;
        Tensor ti = make_tensor(make_smem_ptr(gram.INVbf), InvView{});
        Tensor tr = make_tensor(make_smem_ptr(sm.b.k_restored),
                                Layout<Shape<Int<D>, Int<CHUNK>>, Stride<_1, Int<kPad>>>{});
        Tensor tm = make_tensor(make_smem_ptr(gram.Mqk),
                                Layout<Shape<_16, _16>, Stride<Int<CHUNK + 8>, _1>>{});
        auto a = thr.partition_fragment_A(ti);
        auto acc = thr.make_fragment_C(thr.partition_C(tm));
        copy(AutoVectorizingCopy{}, thr.partition_A(ti), a);
        CUTE_UNROLL
        for (int wb = warp; wb < D / 16; wb += NumThreads / 32) {
            auto rb = local_tile(tr, Shape<_16, _16>{}, make_coord(wb, 0));
            auto b = thr.partition_fragment_B(rb);
            copy(AutoVectorizingCopy{}, thr.partition_B(rb), b);
            clear(acc);
            cute::gemm(thr, a, b, acc);
            CUTE_UNROLL
            for (int j = 0; j < decltype(size(acc))::value / 2; ++j) {
                FragCoord const c = frag_coord_mn(lane, 2 * j);
                BF16 pair[2] = {BF16(acc(2 * j) * sm.beta[c.m]), BF16(acc(2 * j + 1) * sm.beta[c.m])};
                *reinterpret_cast<uint32_t *>(ws_kr + int64_t(ws_idx) * 3 * CHUNK * D + c.m * D +
                                              wb * 16 + c.n) =
                    *reinterpret_cast<uint32_t const *>(pair);
            }
        }
        if (warp == 0) {
            auto ma = thr.partition_fragment_A(tm);
            auto ib = thr.partition_fragment_B(ti);
            copy(AutoVectorizingCopy{}, thr.partition_A(tm), ma);
            copy(AutoVectorizingCopy{}, thr.partition_B(ti), ib);
            clear(acc);
            cute::gemm(thr, ma, ib, acc);
            CUTE_UNROLL
            for (int j = 0; j < decltype(size(acc))::value / 2; ++j) {
                FragCoord const c = frag_coord_mn(lane, 2 * j);
                BF16 pair[2] = {BF16(acc(2 * j) * sm.beta[c.n]),
                                BF16(acc(2 * j + 1) * sm.beta[c.n + 1])};
                *reinterpret_cast<uint32_t *>(ws_mqk + int64_t(ws_idx) * CHUNK * CHUNK +
                                              c.m * CHUNK + c.n) =
                    *reinterpret_cast<uint32_t const *>(pair);
            }
        }
    } else {
        store_tile_2d_full<CHUNK, CHUNK, CHUNK + 8, NumThreads>(
            ws_inv + int64_t(ws_idx) * CHUNK * CHUNK, CHUNK, gram.INVbf, tid);
    }
}

} // namespace flash_kda_ppu
