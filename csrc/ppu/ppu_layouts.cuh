#pragma once

// Shared-memory layouts for both PPU kernels, and the copy atoms that read them.
//
// The rule that shapes this file:
//
//   No shared-memory buffer may be BOTH (a) AIU-written or TSM-read AND
//   (b) element-addressed.
//
// Bytes in an AIU-written / TSM-read buffer are hardware XOR-interleaved. The
// interleave is selected by swzl_mode (derived from CUBE_W * sizeof(Element)),
// not by a CuTe Swizzle<> functor, and the plain `SmemLayoutAtom` typedef that
// production pairs with those loads is a *shape carrier only* -- its strides are
// never used, and ppu/gemm/config/gemm_operands.hpp says so outright. Any CuTe
// element access through such a layout reads the wrong bytes.
//
// So the two kinds of buffer are given two different C++ representations:
//
//   AiuTile<>            raw pointer + cube geometry, no CuTe layout at all, so
//                        element addressing is not expressible.
//   PaddedRowMajor<>     an ordinary CuTe layout, row-padded for bank spread,
//                        never touched by the AIU or a matrix load.
//
// FlashKDA's CUDA kernels violate the rule throughout (fwd_kernel1.cuh:280,
// 312-327, 360-474, 499-508; fwd_kernel2.cuh:718), which is the real reason the
// port is a rewrite rather than a substitution.

#include "ppu_common.cuh"

extern "C" __device__ unsigned __ppu_read_firstlane(unsigned);

namespace flash_kda_ppu {

// ---------------------------------------------------------------------------
// Bank-conflict guard
// ---------------------------------------------------------------------------
// 32 banks x 4 bytes. A row stride that is 0 mod 32 words puts every row on the
// same bank, so a column walk serialises 8 ways. The production fix for exactly
// this was padding a 512-byte stride to 528.
CUTE_HOST_DEVICE constexpr bool bank_safe_row_stride(int stride_bytes) {
    return stride_bytes % 4 == 0 && (stride_bytes / 4) % 32 != 0;
}

// Row-major with the row padded to a bank-safe stride. For element-addressed
// buffers only.
template <class Element, int ROWS, int COLS_PAD> struct PaddedRowMajor {
    using Layout = cute::Layout<cute::Shape<cute::Int<ROWS>, cute::Int<COLS_PAD>>,
                                cute::Stride<cute::Int<COLS_PAD>, cute::_1>>;
    static constexpr int kElems = ROWS * COLS_PAD;
    static constexpr int kBytes = kElems * int(sizeof(Element));
    static constexpr int kRowStrideBytes = COLS_PAD * int(sizeof(Element));

    static_assert(bank_safe_row_stride(kRowStrideBytes),
                  "row stride is 0 mod 32 words: every row lands on one bank, so any "
                  "column-strided access serialises 8 ways. Pad COLS_PAD.");
};

// ---------------------------------------------------------------------------
// AIU / TSM tiles
// ---------------------------------------------------------------------------
// Stored as NUM_CUBES consecutive cubes of CUBE_H x CUBE_W, which is the
// addressing PPU0015_TSM_LD_SWZL_DISPATCH assumes:
//   base += CUBE_H*CUBE_W*(cube_in_stage + stage*InstNum) + coord_h*CUBE_W + coord_w
// and the geometry the AIU writes, one descriptor per cube.
template <class Element, int CUBE_H, int CUBE_W, int NUM_CUBES> struct AiuTile {
    // Forces swzl_mode 0, matching production's bf16 tiles. Anything else selects
    // mode 1 and a different interleave than the loads below assume.
    static_assert(CUBE_W * int(sizeof(Element)) == 128,
                  "CUBE_W * sizeof(Element) must be 128 so swzl_mode == 0");
    // 256/512/576 fault on this hardware.
    static_assert(CUBE_W <= 128, "AIU CUBE_W is capped at 128 elements");

    static constexpr int kElemsPerCube = CUBE_H * CUBE_W;
    static constexpr int kElems = kElemsPerCube * NUM_CUBES;
    static constexpr int kBytes = kElems * int(sizeof(Element));

    // The AIU faults on destination offsets that are not 128-byte aligned.
    alignas(128) Element storage[kElems];

    CUTE_DEVICE Element *data() {
        return storage;
    }
    CUTE_DEVICE Element const *data() const {
        return storage;
    }
};

// ---------------------------------------------------------------------------
// Kernel 1 layouts: all plain and padded (K1 uses neither the AIU nor TSM).
//
// K1 is element-access dominated -- L2 norm, gate activation, cumulative sum,
// decay application, triangular masks -- with only two tiny GEMMs. Feeding those
// two from padded smem via AutoVectorizingCopy costs about what a matrix load
// costs anyway (a matrix load is ~4 instructions here), and removes every
// interleave hazard from the kernel.
// ---------------------------------------------------------------------------
template <int D, int CHUNK = 16> struct K1Layouts {
#ifndef FLASH_KDA_PPU_K1_D_PAD
#define FLASH_KDA_PPU_K1_D_PAD 8
#endif
    // 128 -> 136 elements: 272 B = 68 words = 4 mod 32.
    // Keep the measured default while exposing the padding to compile-only
    // bank-conflict sweeps; it does not change the workspace ABI.
    static constexpr int kColsPad = D + FLASH_KDA_PPU_K1_D_PAD;
    static_assert(FLASH_KDA_PPU_K1_D_PAD >= 0 && FLASH_KDA_PPU_K1_D_PAD % 4 == 0,
                  "K1 D padding must be a non-negative multiple of four");

    using QK = PaddedRowMajor<BF16, CHUNK, kColsPad>;    // q, k, g_bf16, and the
    using Tile = PaddedRowMajor<BF16, CHUNK, kColsPad>;  // four decayed variants
    using GF32 = PaddedRowMajor<float, CHUNK, kColsPad>; // gate cumulative sum

    // 16 -> 20 elements: 80 B = 20 words. Holds L and INV for the fp32 inverse.
    using LMF32 = PaddedRowMajor<float, CHUNK, CHUNK + 4>;
    // 16 -> 24 elements: 48 B = 12 words. Mqk, and INV once narrowed to bf16.
    using LMBF16 = PaddedRowMajor<BF16, CHUNK, CHUNK + 8>;

    using GTotal = cute::Layout<cute::Shape<cute::Int<D>>, cute::Stride<cute::_1>>;
    using Beta = cute::Layout<cute::Shape<cute::Int<32>>, cute::Stride<cute::_1>>;
};

// ---------------------------------------------------------------------------
// Kernel 2 layouts
// ---------------------------------------------------------------------------
// DvPerCta is the slice of the d_v axis one CTA owns. The whole recurrence is
// row-local in d_v -- u^T, U^T, out and dS each touch only the state rows the CTA
// holds -- which is what lets the axis be split across CTAs with no communication
// at all, not just across warps. See fwd_kernel2_ppu.cuh.
template <int D, int CHUNK = 16, int DvPerCta = D> struct K2Layouts {
    static constexpr int kCubeH = CHUNK;                   // 16
    static constexpr int kCubeW = 128 / int(sizeof(BF16)); // 64
    static constexpr int kCubesPerTile = D / kCubeW;       // 2
    // V is indexed by d_v, so a split CTA only needs the 64-column AIU cube(s)
    // intersecting its local rows.  Sub-cube slices share an aligned cube with
    // the adjacent CTA instead of falling back to staging all D columns.
    static constexpr int kVCubesPerTile = (DvPerCta + kCubeW - 1) / kCubeW;
    static_assert(D % kCubeW == 0, "D must split into whole AIU cubes");
    static_assert(DvPerCta % 16 == 0 && D % DvPerCta == 0,
                  "the d_v slice must be whole 16-row blocks and divide D");

    // k_decayed, q_decayed and k_restored are needed over all d_k. V is different:
    // it is consumed only over the CTA-local d_v rows, so split2 stages one
    // 64-column cube, while each pair of split4 CTAs shares one such cube.
    using Tile = AiuTile<BF16, kCubeH, kCubeW, kCubesPerTile>;

    // The recurrent state is NOT here: it lives in the MMA warps' registers.
    // That is what removes 32 KB of shared memory, 16 matrix loads per k-loop per
    // chunk, and an 8-way-conflicted 32 KB/chunk write-back -- and it is legal
    // because the state is already warp-private in the CUDA kernel
    // (fwd_kernel2.cuh:541 reads and :723 writes the same 32 columns).

    // Element-addressed, so plain and padded, and only as wide as the CTA's slice.
    using Out = PaddedRowMajor<BF16, CHUNK, DvPerCta + 8>;
    // Prologue only, and always bf16: the fragments it feeds are bf16, so an fp32
    // initial state is narrowed during the copy rather than after it.
    using StateStage = PaddedRowMajor<BF16, DvPerCta, D + 8>;
    using LM = PaddedRowMajor<BF16, CHUNK, CHUNK + 8>; // INV, Mqk
    using GTotal = cute::Layout<cute::Shape<cute::Int<D>>, cute::Stride<cute::_1>>;
    using Beta = cute::Layout<cute::Shape<cute::Int<32>>, cute::Stride<cute::_1>>;
};

// ---------------------------------------------------------------------------
// Copy atoms
// ---------------------------------------------------------------------------
#if FLASH_KDA_PPU_TC02
// A CHUNK x D tile is stored as D/CUBE_W consecutive cubes, and the load's
// address arithmetic is base + CUBE_H*CUBE_W*cube + coord_h*CUBE_W + coord_w. So a
// 16x16 block index splits into a cube and a within-cube column offset.
template <int CUBE_W> struct TsmBlock {
    static constexpr int kBlocksPerCube = CUBE_W / 16;
    int cube;
    int coord_w;
};

template <int CUBE_W> CUTE_HOST_DEVICE constexpr TsmBlock<CUBE_W> tsm_block(int block_idx) {
    return TsmBlock<CUBE_W>{block_idx / TsmBlock<CUBE_W>::kBlocksPerCube,
                            16 * (block_idx % TsmBlock<CUBE_W>::kBlocksPerCube)};
}

// Swap/Trans select which operand layout the load produces. Derived inside the
// atom: lbo/sbo swap on (Swap != Trans), and .trans is appended on Trans.
//
//   (false, false)  A fragment of a tile stored [m][k]
//   (true,  false)  B fragment of a tile stored [n][k]
//   (true,  true )  B fragment of a tile stored [k][n]   <- the transposing read
//   (false, true )  would be "A from a transposed tile". Production never uses
//                   it and its register order is unverified, so the kernels are
//                   written to never need it: where an A-shaped transposed
//                   fragment is wanted, (true,true) is used and the result is
//                   relabelled with as_A(), which is free.
//
//
// Always select a block through load_a/load_b/load_bt, never by calling the
// atom's copy() directly. PPU0015_TSM_LD_SWZL::copy(frag, smem, coord_w, coord_h,
// cube, stage) forwards positionally, but the two DISPATCH specialisations in the
// fork name their parameters in opposite orders:
//
//   Trans=false : (coord_w, coord_h, stage, cube)
//   Trans=true  : (coord_h, coord_w, stage, cube)
//
// so copy()'s third argument is a column offset for a straight load and a *row*
// offset for a transposing one, and both then compute coord_h*CUBE_W + coord_w.
// Passing the column offset third on a transposing load therefore reads whole
// cubes away: Hardware validation showed 1/8 blocks correct that way
// (only block 0, whose offset is zero) versus 8/8 with the offset passed fourth.
template <int CUBE_H, int CUBE_W, int NUM_CUBES> struct TsmOps {
    template <bool Swap, bool Trans>
    CUTE_DEVICE static void load(void *frag, int base, int block, int row) {
        int addr = base + (block / 4) * (CUBE_H * 8) + (block % 4) * 2 + row * 8;
        int *v = (int *)frag;
        constexpr int lbo = Swap != Trans ? 64 : 1, sbo = Swap != Trans ? 1 : 64;
        if constexpr (Trans) {
            asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans {%0, %1, %2, "
                         "%3}, [%4], %5, %6, %7;"
                         : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                         : "l"(addr), "r"(lbo), "r"(sbo), "n"(0));
        } else {
            asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 {%0, %1, %2, %3}, "
                         "[%4], %5, %6, %7;"
                         : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                         : "l"(addr), "r"(lbo), "r"(sbo), "n"(0));
        }
    }
    CUTE_DEVICE static void load_a(void *p, int base, int block, int row = 0) {
        load<false, false>(p, base, block, row);
    }
    CUTE_DEVICE static void load_b(void *p, int base, int block, int row = 0) {
        load<true, false>(p, base, block, row);
    }
    CUTE_DEVICE static void load_bt(void *p, int base, int block, int row = 0) {
        load<true, true>(p, base, block, row);
    }

    using LdA = cute::PPU0015_TSM_LD_SWZL<BF16, CUBE_H, CUBE_W, /*Swap=*/false,
                                          /*Trans=*/false, NUM_CUBES>;
    using LdB = cute::PPU0015_TSM_LD_SWZL<BF16, CUBE_H, CUBE_W, /*Swap=*/true,
                                          /*Trans=*/false, NUM_CUBES>;
    using LdBT = cute::PPU0015_TSM_LD_SWZL<BF16, CUBE_H, CUBE_W, /*Swap=*/true,
                                           /*Trans=*/true, NUM_CUBES>;

    static_assert(LdA::swzl_mode == 0, "expected swzl_mode 0 for a 128-byte cube row");
    static_assert(LdB::swzl_mode == 0, "expected swzl_mode 0 for a 128-byte cube row");
    static_assert(LdBT::swzl_mode == 0, "expected swzl_mode 0 for a 128-byte cube row");

    CUTE_DEVICE static void load_a(void *frag, void *smem, int block_idx, int row_offset = 0) {
        auto const b = tsm_block<CUBE_W>(block_idx);
        LdA::copy(frag, smem, b.coord_w, row_offset, b.cube);
    }
    CUTE_DEVICE static void load_b(void *frag, void *smem, int block_idx, int row_offset = 0) {
        auto const b = tsm_block<CUBE_W>(block_idx);
        LdB::copy(frag, smem, b.coord_w, row_offset, b.cube);
    }
    CUTE_DEVICE static void load_bt(void *frag, void *smem, int block_idx, int row_offset = 0) {
        auto const b = tsm_block<CUBE_W>(block_idx);
        // Trans=true's dispatch interprets positional argument three as the
        // physical row coordinate and argument four as the column coordinate.
        LdBT::copy(frag, smem, row_offset, b.coord_w, b.cube);
    }
};

// NumBitsPerTMA is the whole tile in bits.
template <int CUBE_H, int CUBE_W>
using AiuLoad = cute::PPU0015_AIU_LOAD<cute::C<CUBE_H * CUBE_W * 16>, BF16,
                                       /*Trans=*/false, CUBE_H, CUBE_W>;
#endif

// Register <-> smem for element-addressed buffers, and gmem stores. There is no
// stmatrix on PPU, so register -> smem is a plain vectorized copy; and there is
// no AIU store, so every gmem write-back is a plain vectorized store.
using VecCopy128 = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, BF16>;
using VecCopy32 = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<32>, BF16>;
using VecCopyF32 = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<32>, float>;

} // namespace flash_kda_ppu
