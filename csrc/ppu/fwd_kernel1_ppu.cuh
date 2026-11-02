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
#include "prefill_chunk_ppu.h"
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
                                                int *full_sequence_flag = nullptr, int T_total = 0,
                                                bool initialize_serial_pair = false) {
    // This launch has one warp. Use the actual device CU buffer on every
    // invocation and graph replay; all other geometries keep the scalar path.
    if (N == 8 && chunk == 16 && T_total == 16384 && blockDim.x == 32 && blockDim.y == 1 &&
        blockDim.z == 1) {
        int const lane = int(threadIdx.x);
        int64_t const begin = lane < 8 ? cu_seqlens[lane] : int64_t(0);
        int64_t const end = lane < 8 ? cu_seqlens[lane + 1] : int64_t(0);
        int tiles = lane < 8 ? (int(end - begin) + 15) / 16 : 0;
        bool const full = __all_sync(0xffffffffu, lane >= 8 || (begin == int64_t(lane) * 2048 &&
                                                                end == int64_t(lane + 1) * 2048));
#pragma unroll
        for (int offset = 1; offset < 8; offset <<= 1) {
            int const previous = __shfl_up_sync(0xffffffffu, tiles, offset);
            if (lane >= offset)
                tiles += previous;
        }
        if (lane < 8)
            tile_prefix[lane + 1] = tiles;
        if (lane == 0) {
            tile_prefix[0] = 0;
            if (full_sequence_flag) {
                full_sequence_flag[0] = full;
                if (initialize_serial_pair)
                    full_sequence_flag[1] = full;
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
            if (initialize_serial_pair)
                full_sequence_flag[1] = full;
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
    static_assert(offsetof(GramWork, INVbf) + L::LMBF16::kBytes <= offsetof(PhaseB, k_restored));
    union {
        PhaseA a;
        PhaseB b;
        GramWork gram;
    };
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
template <int NumThreads, bool PruneUpperRows, int WarpPair = -1>
CUTE_DEVICE void solve_parallel_inverse_rows(float (&inverse)[256 / NumThreads], float const *lower,
                                             int col, int lane) {
    constexpr int Rows = 256 / NumThreads;
    static_assert(!PruneUpperRows || NumThreads == 128);
    CUTE_UNROLL
    for (int pivot = 15; pivot > 0; --pivot) {
        // Gram publication explicitly writes +0 to the diagonal and upper
        // triangle. Loading those values preserves the old masked +0 operand
        // while keeping every lane on the same shared-load path.
        float const factor = lower[pivot * 20 + col];
        CUTE_UNROLL
        for (int r = 0; r < Rows; ++r) {
            if (!PruneUpperRows || pivot <= (WarpPair < 0 ? (r + 1) * (NumThreads / 16) - 1
                                                          : r * 8 + 2 * WarpPair + 1)) {
                float const value = __shfl_sync(0xffffffffu, inverse[r], (lane & 16) + pivot);
                inverse[r] = fmaf(-factor, value, inverse[r]);
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

    static_assert(ROWS == 16 && COLS == 128 && (NumThreads == 128 || NumThreads == 256));
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

template <bool Subgroup> CUTE_DEVICE void flash_kda_ppu_prepare_sync() {
    if constexpr (Subgroup)
        cutlass::arch::NamedBarrier::arrive_and_wait(128, int(threadIdx.x) / 128);
    else
        __syncthreads();
}

template <int CHUNK, int D, int NumThreads, bool IsVarlen, InvPrec Prec, bool ValidateScan = false,
          bool AffinePrepared = false, int StaticHeads = 0, bool BuildSinglePrefix = false,
          bool GuardWeakP3 = false, bool BuildSmallPrefix = false, bool PersistentPrepare = false>
__device__ __forceinline__ bool flash_kda_ppu_prepare_tile(
    BF16 const *__restrict__ q_ptr, BF16 const *__restrict__ k_ptr, BF16 const *__restrict__ g_ptr,
    BF16 const *__restrict__ beta_ptr, float const *__restrict__ A_log_ptr,
    float const *__restrict__ dt_bias_ptr, BF16 *__restrict__ ws_kd, BF16 *__restrict__ ws_qd,
    BF16 *__restrict__ ws_kr, float *__restrict__ ws_gt, BF16 *__restrict__ ws_inv,
    BF16 *__restrict__ ws_mqk, float *__restrict__ ws_beta, float scale, float gate_scale,
    int T_total, int H_runtime, int N, int64_t const *__restrict__ cu_seqlens, int total_tiles,
    int *__restrict__ tile_prefix, int *__restrict__ scan_flag, float scan_threshold,
    int q_stride_runtime, int k_stride_runtime, int *__restrict__ full_sequence_flag,
    int general_tile, void *general_scratch) {
#define FLASH_KDA_PREPARE_TILE general_tile
#define FLASH_KDA_PREPARE_SCRATCH general_scratch
#include "fwd_prepare_body_ppu.cuh"
#undef FLASH_KDA_PREPARE_SCRATCH
#undef FLASH_KDA_PREPARE_TILE
    return true;
}
template <int CHUNK, int D, int NumThreads, bool IsVarlen, InvPrec Prec, bool ValidateScan = false,
          bool AffinePrepared = false, int StaticHeads = 0, bool BuildSinglePrefix = false,
          bool GuardWeakP3 = false>
__global__ void __launch_bounds__(NumThreads, 4) flash_kda_ppu_fwd_prepare(
    BF16 const *__restrict__ q_ptr, BF16 const *__restrict__ k_ptr, BF16 const *__restrict__ g_ptr,
    BF16 const *__restrict__ beta_ptr, float const *__restrict__ A_log_ptr,
    float const *__restrict__ dt_bias_ptr, BF16 *__restrict__ ws_kd, BF16 *__restrict__ ws_qd,
    BF16 *__restrict__ ws_kr, float *__restrict__ ws_gt, BF16 *__restrict__ ws_inv,
    BF16 *__restrict__ ws_mqk, float *__restrict__ ws_beta, float scale, float gate_scale,
    int T_total, int H_runtime, int N, int64_t const *__restrict__ cu_seqlens, int total_tiles,
    int *__restrict__ tile_prefix, int *__restrict__ scan_flag, float scan_threshold,
    int q_stride_runtime, int k_stride_runtime, int *__restrict__ full_sequence_flag = nullptr) {
    constexpr bool BuildSmallPrefix = false, PersistentPrepare = false;
#include "fwd_prepare_body_ppu.cuh"
}

template <int CHUNK, int D, int NumThreads, bool IsVarlen, InvPrec Prec, bool ValidateScan = false,
          bool AffinePrepared = false, int StaticHeads = 0, bool BuildSinglePrefix = false,
          bool GuardWeakP3 = false, bool BuildSmallPrefix = false, bool PersistentPrepare = false>
__global__ void __launch_bounds__(NumThreads, 4) flash_kda_ppu_general_prepare(
    BF16 const *__restrict__ q_ptr, BF16 const *__restrict__ k_ptr, BF16 const *__restrict__ g_ptr,
    BF16 const *__restrict__ beta_ptr, float const *__restrict__ A_log_ptr,
    float const *__restrict__ dt_bias_ptr, BF16 *__restrict__ ws_kd, BF16 *__restrict__ ws_qd,
    BF16 *__restrict__ ws_kr, float *__restrict__ ws_gt, BF16 *__restrict__ ws_inv,
    BF16 *__restrict__ ws_mqk, float *__restrict__ ws_beta, float scale, float gate_scale,
    int T_total, int H_runtime, int N, int64_t const *__restrict__ cu_seqlens, int total_tiles,
    int *__restrict__ tile_prefix, int *__restrict__ scan_flag, float scan_threshold,
    int q_stride_runtime, int k_stride_runtime, int *__restrict__ full_sequence_flag = nullptr) {
    if constexpr (PersistentPrepare && BuildSinglePrefix) {
        static_assert(CHUNK == 16 && NumThreads == 256 && AffinePrepared);
        // Keep one native-wide tile live at a time. The loop stays rolled so
        // sparse calls do not inherit the register footprint of a second tile.
        __shared__ SharedStorageK1<D, CHUNK, false> tile_storage;
        int const real_tiles = int(__ppu_read_firstlane(
            uint32_t((int(cu_seqlens[1] - cu_seqlens[0]) + CHUNK - 1) / CHUNK)));
        if (blockIdx.x != 0 && int(blockIdx.x) >= real_tiles)
            return;
#pragma unroll 1
        for (int round = 0; round < flash_kda_ppu_general_prepare_tiles<true>; ++round) {
            int const tile = int(blockIdx.x) + round * int(gridDim.x);
            if (round > 0 && tile >= real_tiles)
                return;
            if (!flash_kda_ppu_prepare_tile<CHUNK, D, 256, IsVarlen, Prec, ValidateScan,
                                            AffinePrepared, StaticHeads, BuildSinglePrefix,
                                            GuardWeakP3, BuildSmallPrefix>(
                    q_ptr, k_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, ws_kd, ws_qd, ws_kr,
                    ws_gt, ws_inv, ws_mqk, ws_beta, scale, gate_scale, T_total, H_runtime, N,
                    cu_seqlens, total_tiles, tile_prefix, scan_flag, scan_threshold,
                    q_stride_runtime, k_stride_runtime, full_sequence_flag, tile, &tile_storage))
                return;
            if (round + 1 < flash_kda_ppu_general_prepare_tiles<true> &&
                tile + int(gridDim.x) < real_tiles)
                __syncthreads();
        }
    } else if constexpr (PersistentPrepare) {
        static_assert(CHUNK == 16 && (NumThreads == 128 || NumThreads == 256) && AffinePrepared);
        // Tile zero publishes current prefix and dispatch metadata together.
        // Each bounded tile finishes before this CTA reuses its shared root.
        constexpr bool NeedF32 = !(AffinePrepared && Prec == InvPrec::FP32_FORWARD_SUBSTITUTION);
        constexpr bool Alias = NumThreads == 128 && !NeedF32;
        using TileStorage = std::conditional_t<Alias, SharedStorageK1Packed<D, CHUNK>,
                                               SharedStorageK1<D, CHUNK, NeedF32>>;
        __shared__ TileStorage tile_storage;
        constexpr int TileGroup = flash_kda_ppu_general_prepare_tiles<BuildSinglePrefix>;
#pragma unroll 1
        for (int round = 0; round < TileGroup; ++round) {
            int const tile = int(blockIdx.x) + round * int(gridDim.x);
            if (tile >= total_tiles)
                return;
            if (!flash_kda_ppu_prepare_tile<CHUNK, D, NumThreads, IsVarlen, Prec, ValidateScan,
                                            AffinePrepared, StaticHeads, BuildSinglePrefix,
                                            GuardWeakP3, BuildSmallPrefix>(
                    q_ptr, k_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, ws_kd, ws_qd, ws_kr,
                    ws_gt, ws_inv, ws_mqk, ws_beta, scale, gate_scale, T_total, H_runtime, N,
                    cu_seqlens, total_tiles, tile_prefix, scan_flag, scan_threshold,
                    q_stride_runtime, k_stride_runtime, full_sequence_flag, tile, &tile_storage))
                return;
            if (round + 1 < TileGroup)
                __syncthreads();
        }
    } else {
#include "fwd_prepare_body_ppu.cuh"
    }
}

} // namespace flash_kda_ppu
