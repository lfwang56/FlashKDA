#pragma once

// The two gmem -> smem paths, and the one rule that separates them.
//
// AIU (AiuTileLoader): bulk 2D tile fetch, the only engine that can fill a tile
// cheaply enough for K2's inner loop. Its destination is hardware-interleaved, so
// the tile may only ever be read back by a matrix load -- never element-addressed.
// Issue is warp-collective: ONE descriptor per warp, with all 32 lanes executing
// the asm. Issuing from 320 threads instead of 10 warps lands 10 of 64 rows.
//
// cp.async (CpAsync): 16 bytes per lane into a plain, row-padded destination that
// stays element-addressable. This is what K1 uses, because K1 is dominated by
// element access (L2 norm, gate, cumulative sum, decay, masks) and only needs two
// tiny GEMMs.
//
// Neither engine has a store direction: there is no AIU store and no async smem
// -> gmem path at all, so every write-back is a plain vectorized store. See
// epilogue.cuh.

#include "ppu_common.cuh"
#include "ppu_layouts.cuh"

namespace flash_kda_ppu {

#if FLASH_KDA_PPU_TC02

// ---------------------------------------------------------------------------
// AIU bulk tile load
// ---------------------------------------------------------------------------
// Views the source as a 2D array of `dim_h` rows by `dim_w` columns with row
// pitch `stride_w` elements, and fetches CUBE_H consecutive rows starting at
// `coord_h`, in NUM_CUBES horizontal cubes of CUBE_W columns.
//
// `padz` zero-fills coordinates past dim_h, so a short tail tile arrives zeroed
// for free -- strictly better than the CUDA kernel, which left tail rows as
// garbage and relied on k_restored's zeroed rows to annihilate them.
template <class Element, int CUBE_H, int CUBE_W, int NUM_CUBES> struct AiuTileLoader {
    using Atom =
        cute::PPU0015_AIU_LOAD<cute::C<CUBE_H * CUBE_W * cute::sizeof_bits<Element>::value>,
                               Element,
                               /*Trans=*/false, CUBE_H, CUBE_W>;

    static constexpr int kElemsPerCube = CUBE_H * CUBE_W;

    cute::AiuDesc desc;

    // `base` already carries any fixed offset (e.g. head_idx * D for a
    // (H, T, D) tensor); dim_h is what padz clamps against.
    CUTE_DEVICE void init(Element const *base, int dim_h, int dim_w, int stride_w) {
        desc.init(reinterpret_cast<uint8_t const *>(base), dim_h, dim_w, stride_w);
    }

    // Warp-collective. Every lane of the issuing warp must reach this.
    CUTE_DEVICE void load(Element *smem, int coord_h) const {
        CUTE_UNROLL
        for (int c = 0; c < NUM_CUBES; ++c) {
            Atom::copy(smem + c * kElemsPerCube, desc.gmem_ptr, desc,
                       /*coord_w=*/c * CUBE_W, coord_h);
        }
    }
};

#endif // FLASH_KDA_PPU_TC02

// ---------------------------------------------------------------------------
// cp.async into a plain, row-padded destination
// ---------------------------------------------------------------------------
// The atom is gated on CUTE_ARCH_CP_ASYNC_PPU_ENABLED, which the cutlass fork
// never defines; ppu_common.cuh turns it on. M0 confirmed the form assembles,
// lowering to vmem.ld.tsm.b32x4.pref128.kp1 + vmem.acp.commit.grp.
using CpAsyncVec16 = cute::PPU_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

CUTE_DEVICE void cp_async_fence_all() { cute::cp_async_fence(); }

template <int N = 0> CUTE_DEVICE void cp_async_wait_all() { cute::cp_async_wait<N>(); }

// Copy ROWS x COLS bf16 from a strided gmem view into a ROWS x COLS_PAD smem
// tile, zero-filling rows at or past `rows_valid`.
//
// Rows past rows_valid are zeroed with plain stores rather than a zero-filling
// cp.async: the tail is at most CHUNK-1 rows of one tile, and mixing predicated
// async fills into the same commit group buys nothing measurable while making
// the wait protocol harder to reason about.
template <int ROWS, int COLS, int COLS_PAD, int NumThreads>
CUTE_DEVICE void cp_async_tile(BF16 *__restrict__ smem, BF16 const *__restrict__ gmem,
                               int gmem_row_stride, int rows_valid, int tid) {
    static_assert(COLS % 8 == 0, "16-byte vectors require COLS to be a multiple of 8");
    constexpr int kElemsPerVec = 16 / int(sizeof(BF16)); // 8
    constexpr int kVecsPerRow = COLS / kElemsPerVec;
    constexpr int kVecs = ROWS * kVecsPerRow;

    for (int i = tid; i < kVecs; i += NumThreads) {
        int const r = i / kVecsPerRow;
        int const c = (i % kVecsPerRow) * kElemsPerVec;
        BF16 *dst = smem + r * COLS_PAD + c;
        if (r < rows_valid) {
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

// Contiguous 16-byte cp.async of `bytes` bytes. For the small per-tile arrays
// (INV, Mqk, g_total, beta) whose gmem side is already packed.
template <int NumThreads>
CUTE_DEVICE void cp_async_contiguous(void *__restrict__ smem, void const *__restrict__ gmem,
                                     int bytes, int tid) {
    int const vecs = bytes / 16;
    auto const *g = reinterpret_cast<cute::uint128_t const *>(gmem);
    auto *s = reinterpret_cast<cute::uint128_t *>(smem);
    for (int i = tid; i < vecs; i += NumThreads) {
        CpAsyncVec16::copy(g[i], s[i]);
    }
}

} // namespace flash_kda_ppu
