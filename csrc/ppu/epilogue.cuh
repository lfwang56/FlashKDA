#pragma once

// Write-back. There is no async smem -> gmem path on PPU at all: no TMA store, no
// AIU store direction (every AIU mnemonic in the tree is `...shared.global`), and
// no stmatrix. So the CUDA backend's TMA stores, its tma_store_arrive/wait
// protocol, and its STSM register->smem atoms all collapse into plain vectorized
// stores from ordinary padded shared memory.
//
// One thing that gets *better*: the CUDA store warp handled a short tail tile with
// a single-lane scalar loop over all D columns -- 2048 scalar stores for one tile
// (fwd_kernel2.cuh:761-766). Here the tail is just a row predicate on the same
// vectorized path.

#include "ppu_common.cuh"

namespace flash_kda_ppu {

// Padded smem tile -> strided gmem, 16 bytes per store, rows past `rows_valid`
// skipped. `gmem_row_stride` is in elements, so this serves both a packed
// workspace tile (stride D) and an interleaved output tensor (stride H*D).
template <int ROWS, int COLS, int COLS_PAD, int NumThreads>
CUTE_DEVICE void store_tile_2d(BF16 *__restrict__ gmem, int gmem_row_stride,
                               BF16 const *__restrict__ smem, int rows_valid, int tid) {
    static_assert(COLS % 8 == 0, "16-byte stores require COLS to be a multiple of 8");
    constexpr int kElemsPerVec = 16 / int(sizeof(BF16)); // 8
    constexpr int kVecsPerRow = COLS / kElemsPerVec;
    constexpr int kVecs = ROWS * kVecsPerRow;

    for (int i = tid; i < kVecs; i += NumThreads) {
        int const r = i / kVecsPerRow;
        if (r >= rows_valid)
            continue;
        int const c = (i % kVecsPerRow) * kElemsPerVec;
        *reinterpret_cast<cute::uint128_t *>(gmem + r * gmem_row_stride + c) =
            *reinterpret_cast<cute::uint128_t const *>(smem + r * COLS_PAD + c);
    }
}

// Same, for an unpadded source (COLS_PAD == COLS) and no row predicate: the
// workspace arrays K1 writes are always full tiles.
template <int ROWS, int COLS, int COLS_PAD, int NumThreads>
CUTE_DEVICE void store_tile_2d_full(BF16 *__restrict__ gmem, int gmem_row_stride,
                                    BF16 const *__restrict__ smem, int tid) {
    store_tile_2d<ROWS, COLS, COLS_PAD, NumThreads>(gmem, gmem_row_stride, smem, ROWS, tid);
}

// Contiguous `bytes` bytes, 16 at a time. For g_total and the 16x16 matrices,
// whose gmem side is packed.
template <int NumThreads>
CUTE_DEVICE void store_contiguous(void *__restrict__ gmem, void const *__restrict__ smem, int bytes,
                                  int tid) {
    int const vecs = bytes / 16;
    auto *g = reinterpret_cast<cute::uint128_t *>(gmem);
    auto const *s = reinterpret_cast<cute::uint128_t const *>(smem);
    for (int i = tid; i < vecs; i += NumThreads)
        g[i] = s[i];
}

} // namespace flash_kda_ppu
