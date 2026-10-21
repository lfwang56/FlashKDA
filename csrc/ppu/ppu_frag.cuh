#pragma once

// Fragment algebra for the tc02 16x16 atoms, plus compile-time proofs of the
// two layout relations the K2 redesign depends on.
//
// Why this file exists. The _TN atoms compute C[m,n] = sum_k A[m,k]*B[n,k], so A
// and B both index their matrix as (outer, k). On tc02:
//
//   ALayout = ((4,8),(2,2,2)) : ((32,1),(16,  8,128))
//   BLayout = ((4,8),(2,2,2)) : ((32,1),(16,128,  8))
//   CLayout = ALayout
//
// Decoding each against a 16x16 column-major index (idx = outer + 16*k), with
// lane -> (t0 = lane%4, t1 = (lane/4)%8) and value slot v -> (v0,v1,v2):
//
//   A:  m = t1 + 8*v1     k = 2*t0 + v0 + 8*v2
//   B:  n = t1 + 8*v2     k = 2*t0 + v0 + 8*v1
//
// Two consequences, both free at runtime:
//
//   (1) C == A. An accumulator, once cast to bf16, *is* an A operand.
//   (2) B is A with value modes 1 and 2 exchanged. Since the fragment's value
//       shape is (2,2,2) with compact strides (1,2,4), the register index is
//       r = v1 + 2*v2, so the exchange is literally "swap registers 1 and 2" --
//       a compile-time relabel, zero instructions.
//
// Neither is a transpose: a fragment always denotes the same matrix, never its
// transpose. That is why K2 must choose each GEMM's output orientation so a
// transpose never arises (there is no movmatrix on PPU), rather than trying to
// transpose after the fact.
//
// tc01 has CLayout = ((4,8),(4,2)):((16,1),(64,8)), which is neither equal to
// ALayout nor even three-moded, so none of this holds there.

#include "ppu_common.cuh"

namespace flash_kda_ppu {

// ---------------------------------------------------------------------------
// Compile-time proofs
// ---------------------------------------------------------------------------
#if FLASH_KDA_PPU_TC02
namespace detail {

using ABF16 = typename cute::MMA_Traits<MmaBF16>::ALayout;
using BBF16 = typename cute::MMA_Traits<MmaBF16>::BLayout;
using CBF16 = typename cute::MMA_Traits<MmaBF16>::CLayout;

// (1) C == A, stride by stride. Compared elementwise rather than with is_same so
// an equivalent layout spelled with a different tuple type still passes.
static_assert(cute::size(CBF16{}) == cute::size(ABF16{}), "bf16 atom: |C| != |A|");
static_assert(cute::stride<0, 0>(CBF16{}) == cute::stride<0, 0>(ABF16{}), "bf16: C!=A thr0");
static_assert(cute::stride<0, 1>(CBF16{}) == cute::stride<0, 1>(ABF16{}), "bf16: C!=A thr1");
static_assert(cute::stride<1, 0>(CBF16{}) == cute::stride<1, 0>(ABF16{}), "bf16: C!=A val0");
static_assert(cute::stride<1, 1>(CBF16{}) == cute::stride<1, 1>(ABF16{}), "bf16: C!=A val1");
static_assert(cute::stride<1, 2>(CBF16{}) == cute::stride<1, 2>(ABF16{}), "bf16: C!=A val2");

// (2) B == A with value modes 1 and 2 exchanged.
static_assert(cute::stride<0, 0>(BBF16{}) == cute::stride<0, 0>(ABF16{}), "bf16: B/A thr0");
static_assert(cute::stride<0, 1>(BBF16{}) == cute::stride<0, 1>(ABF16{}), "bf16: B/A thr1");
static_assert(cute::stride<1, 0>(BBF16{}) == cute::stride<1, 0>(ABF16{}), "bf16: B/A val0");
static_assert(cute::stride<1, 1>(BBF16{}) == cute::stride<1, 2>(ABF16{}),
              "bf16: B is not A with value modes 1,2 exchanged -- as_B() is invalid");
static_assert(cute::stride<1, 2>(BBF16{}) == cute::stride<1, 1>(ABF16{}),
              "bf16: B is not A with value modes 1,2 exchanged -- as_B() is invalid");

using AF32 = typename cute::MMA_Traits<MmaF32>::ALayout;
using BF32 = typename cute::MMA_Traits<MmaF32>::BLayout;
using CF32 = typename cute::MMA_Traits<MmaF32>::CLayout;

// The k=8 fp32 atom: A <-> B is still a free exchange, but of value modes 0 and
// 1 (its value shape is (2,2), not (2,2,2)).
static_assert(cute::stride<1, 0>(BF32{}) == cute::stride<1, 1>(AF32{}), "fp32: B/A val exchange");
static_assert(cute::stride<1, 1>(BF32{}) == cute::stride<1, 0>(AF32{}), "fp32: B/A val exchange");

// But C -> A is NOT free here: A wants k columns {t0, t0+4, t0+8, t0+12} while C
// holds {2*t0, 2*t0+1, 2*t0+8, 2*t0+9}. Every Neumann product therefore needs a
// 16x16 fp32 smem round-trip. Asserting the inequality so that a future traits
// change which *does* make them equal shows up as a build failure inviting the
// round-trips to be deleted, rather than passing silently.
static_assert(!(cute::stride<1, 0>(CF32{}) == cute::stride<1, 0>(AF32{}) &&
                cute::stride<1, 1>(CF32{}) == cute::stride<1, 1>(AF32{})),
              "fp32 atom now has C == A: the smem round-trips in the Neumann inverse "
              "are no longer needed and should be removed.");

} // namespace detail
#endif // FLASH_KDA_PPU_TC02

// ---------------------------------------------------------------------------
// Where a value slot lives inside a 16x16 C fragment
// ---------------------------------------------------------------------------
struct FragCoord {
    int m; // 0..15, the C fragment's row (M) index
    int n; // 0..15, the C fragment's column (N) index
};

// Used for in-fragment masking, beta scaling and g_total indexing, replacing
// three separate smem round-trips in the CUDA kernels.
CUTE_HOST_DEVICE constexpr FragCoord frag_coord_mn(int lane, int v) {
    int const t0 = lane % 4;
    int const t1 = (lane / 4) % 8;
    int const v0 = v % 2;
    int const v1 = (v / 2) % 2;
    int const v2 = (v / 4) % 2;
    return FragCoord{t1 + 8 * v1, 2 * t0 + v0 + 8 * v2};
}

#if FLASH_KDA_PPU_TC02
namespace detail {

// Check the hand-derived decoding against the traits over all 32 lanes x 8 slots
// instead of trusting the algebra in the header comment.
CUTE_HOST_DEVICE constexpr bool frag_coord_matches_traits() {
    for (int lane = 0; lane < 32; ++lane) {
        for (int v = 0; v < 8; ++v) {
            FragCoord const c = frag_coord_mn(lane, v);
            if (CBF16{}(lane, v) != c.m + 16 * c.n)
                return false;
        }
    }
    return true;
}
static_assert(frag_coord_matches_traits(),
              "frag_coord_mn() disagrees with MMA_Traits<MmaBF16>::CLayout");

} // namespace detail
#endif

// ---------------------------------------------------------------------------
// A/C-shaped fragment -> B-shaped fragment, over the same registers
// ---------------------------------------------------------------------------
namespace detail {

// Exchange two modes of a layout, keeping shape and stride paired.
template <int I, int J, class L> CUTE_HOST_DEVICE constexpr auto exchange_modes(L const &l) {
    static_assert(I < J, "exchange_modes expects I < J");
    return cute::make_layout(cute::get<I>(l), cute::get<J>(l));
}

} // namespace detail

// Reinterpret a fragment that holds matrix X in the A (or C) role as one that
// holds the same X in the B role. Zero instructions: only the value modes are
// relabelled, and the outer modes are singleton for the 16x16 tiles used here.
//
// Rank 3 value mode (bf16, k=16): exchange modes 1 and 2.
// Rank 2 value mode (fp32, k=8):  exchange modes 0 and 1.
template <class Engine, class Layout>
CUTE_HOST_DEVICE auto as_B(cute::Tensor<Engine, Layout> const &f) {
    auto val = cute::get<0>(f.layout());
    constexpr int kValRank = decltype(cute::rank(val))::value;
    static_assert(kValRank == 2 || kValRank == 3,
                  "as_B expects the atom's value mode, of rank 2 (k=8 fp32) or 3 (k=16 bf16)");
    static_assert(cute::rank(Layout{}) == 3, "as_B expects a (value, outer, k) fragment");

    if constexpr (kValRank == 3) {
        auto swapped = cute::make_layout(cute::get<0>(val), cute::get<2>(val), cute::get<1>(val));
        return cute::make_tensor(f.data(), cute::make_layout(swapped, cute::get<1>(f.layout()),
                                                             cute::get<2>(f.layout())));
    } else {
        auto swapped = cute::make_layout(cute::get<1>(val), cute::get<0>(val));
        return cute::make_tensor(f.data(), cute::make_layout(swapped, cute::get<1>(f.layout()),
                                                             cute::get<2>(f.layout())));
    }
}

// The mode exchange is an involution, so the same operation converts B -> A. This
// alias exists so call sites read as their intent: it is what lets a transposed
// matrix reach the A/C orientation through the verified (Swap=true, Trans=true)
// load rather than the unverified (false, true) one.
template <class Engine, class Layout>
CUTE_HOST_DEVICE auto as_A(cute::Tensor<Engine, Layout> const &f) {
    return as_B(f);
}

// Narrow an fp32 accumulator into a bf16 fragment that already exists. Needed
// because cute register tensors have no copy-assignment: `arr[i] = c_to_a_bf16(x)`
// does not compile even though `auto f = c_to_a_bf16(x)` does.
template <class Dst, class Src> CUTE_DEVICE void narrow_into(Dst &dst, Src const &src) {
    CUTE_UNROLL
    for (int i = 0; i < decltype(cute::size(src))::value; ++i) {
        dst(i) = BF16(src(i));
    }
}

// Cast an fp32 accumulator to a bf16 A operand. On tc02 CLayout == ALayout, so
// this is only the narrowing conversion -- there is no data movement to pay for.
template <class AccFrag> CUTE_DEVICE auto c_to_a_bf16(AccFrag const &acc) {
    auto a = cute::make_fragment_like<BF16>(acc);
    // Explicit loop rather than cute::transform: with a const-ref fragment,
    // overload resolution picks cute's tuple-recursing transform, which then
    // calls the element op with a whole Tensor.
    CUTE_UNROLL
    for (int i = 0; i < decltype(cute::size(acc))::value; ++i) {
        a(i) = BF16(acc(i));
    }
    return a;
}

} // namespace flash_kda_ppu
