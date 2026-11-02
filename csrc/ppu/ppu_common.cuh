#pragma once

// Shared PPU backend prelude: toolchain guards, MMA atom selection, the traits
// fix the fp32 atom needs, math shims, and the workspace byte layout.
//
// ---------------------------------------------------------------------------
// Must precede every cute header. The PPU cutlass fork never defines these, so
// PPU_CP_ASYNC_* and PPU_*_LDSM_* otherwise expand to CUTE_RUNTIME_ASSERT and
// never reach the assembler. M0 confirmed both forms do assemble on tc02:
//   cp.async -> vmem.ld.tsm.b32x4.pref128.kp1 + vmem.acp.commit.grp
//   LDSM     -> tsm.ld.mat.b32x4 / tsm.ld.mat.trans.b32x4
// ---------------------------------------------------------------------------
#ifndef CUTE_ARCH_CP_ASYNC_PPU_ENABLED
#define CUTE_ARCH_CP_ASYNC_PPU_ENABLED 1
#endif
#ifndef CUTE_ARCH_LDSM_PPU_ACTIVATED
#define CUTE_ARCH_LDSM_PPU_ACTIVATED 1
#endif

#include <cstdint>

// __mbarrier_t and friends. Without this only the unrelated 32-bit `barrier_t`
// is visible and every barrier call fails to typecheck.
#include <hggc_awbarrier.h>

#include <ppu/ppu_include.hpp>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

#if !defined(ACOMPUTE_VERSION)
#error "ACOMPUTE_VERSION must be set: 10500 for tc02/ppu0015, 10000 for tc01/ppu001."
#endif

// The PPU0015_* atom bodies are guarded on ACOMPUTE_VERSION >= 10500 AND
// __HGGC_ARCH__ == 150. Satisfying only the first yields an *empty* fma() that
// still compiles -- a silent no-op MMA. Catch the mismatch here instead.
#if defined(__HGGC_ARCH__) && ACOMPUTE_VERSION >= 10500 && __HGGC_ARCH__ != 150
#error                                                                                             \
    "ACOMPUTE_VERSION >= 10500 requires -arch=ppu_15 (__HGGC_ARCH__ == 150), else the tc02 MMA silently becomes a no-op."
#endif

namespace flash_kda_ppu {

using BF16 = cutlass::bfloat16_t;

#define FLASH_KDA_PPU_TC02 (ACOMPUTE_VERSION >= 10500)

// ---------------------------------------------------------------------------
// MMA atoms
//
// Both are distinct types rather than using-aliases, so that a private MMA_Traits
// specialisation is legal. Both need one, for the same reason: the fork's traits
// are inherited from a neighbouring atom and carry the wrong operand value type.
//
//   PPU0015_16x16x16_F32BF16BF16F32_TN  inherits the F16F16F32 traits
//                                       -> ValTypeA/B = cutlass::half_t
//   PPU0015_16x16x8_F32F32F32F32_TN     inherits the TF32 traits
//                                       -> ValTypeA/B = cutlass::tfloat32_t
//
// Left uncorrected these are not type errors that stay caught -- they are silent
// numerical corruption. partition_fragment_A would build a half_t fragment and
// round every value to a 5-bit exponent and 10-bit mantissa, while the hardware
// instruction reads the same bits as bf16 with an 8-bit exponent and 7-bit
// mantissa. The tf32 case throws away 13 mantissa bits of the fp32 inverse.
// ---------------------------------------------------------------------------
#if !FLASH_KDA_PPU_TC02
#error "FlashKDA PPU requires tc02/ppu0015 and its FP32 MMA atom."
#endif

#if FLASH_KDA_PPU_TC02
struct MmaBF16 : cute::PPU0015_16x16x16_F32BF16BF16F32_TN {};
struct MmaF32 : cute::PPU0015_16x16x8_F32F32F32F32_TN {};
#else
using MmaBF16 = cute::PPU_16x16x16_F32BF16BF16F32_TN;
#endif

} // namespace flash_kda_ppu

#if FLASH_KDA_PPU_TC02
namespace cute {
template <>
struct MMA_Traits<flash_kda_ppu::MmaBF16> : MMA_Traits<cute::PPU0015_16x16x16_F32BF16BF16F32_TN> {
    using ValTypeD = float;
    using ValTypeA = cutlass::bfloat16_t;
    using ValTypeB = cutlass::bfloat16_t;
    using ValTypeC = float;
};

template <>
struct MMA_Traits<flash_kda_ppu::MmaF32> : MMA_Traits<cute::PPU0015_16x16x8_F32F32F32F32_TN> {
    using ValTypeD = float;
    using ValTypeA = float;
    using ValTypeB = float;
    using ValTypeC = float;
};
} // namespace cute
#endif

namespace flash_kda_ppu {

#if FLASH_KDA_PPU_TC02
// Guard both overrides, so an upstream traits change cannot silently reintroduce
// the wrong operand type.
static_assert(
    cute::is_same<typename cute::MMA_Traits<MmaBF16>::ValTypeA, cutlass::bfloat16_t>::value,
    "MmaBF16 operand type is not bfloat16_t: the traits override was bypassed");
static_assert(cute::is_same<typename cute::MMA_Traits<MmaF32>::ValTypeA, float>::value,
              "MmaF32 operand type is not float: the traits override was bypassed");
#endif

// ---------------------------------------------------------------------------
// Math shims. Keep each call inside ONE source expression: the PPU compiler
// contracts a*b+c into v.fma.f32 only within a single expression, and no flag
// recovers a contraction split across a statement boundary.
// ---------------------------------------------------------------------------

// Replaces the NVIDIA `ex2.approx.ftz.f32` inline asm. --use_fast_math lowers
// this to the hardware exp2; expf/__expf/__exp2f all cost the same 25.2 cyc/op,
// so there is nothing to gain from the "fast" spellings. FTZ handling of
// denormals differs from NVIDIA, which is one reason bit-exactness is gone.
CUTE_DEVICE float kda_exp2(float x) {
    return exp2f(x);
}

// Replaces `tanh.approx.f32`-based sigmoid with the native sigmoid unit.
CUTE_DEVICE float kda_sigmoid(float x) {
    return __ppu_sgmdf(x);
}

// Order this thread's shared-memory writes before anything that follows.
//
// Do NOT use cutlass::arch::fence_view_async_shared() or fence_barrier_init()
// here. They emit `ppu.fence.proxy.async.shared::cta` and
// `ppu.fence.mbarrier_init.release.cluster`, which assemble cleanly but the
// ppu0015 backend rejects at finalisation with "unsupported instruction fence in
// arch ppu0015" and substitutes a trap -- which then fires at run time as a
// device-side assert. They are NVIDIA-derived leftovers in the fork's barrier.h
// and production uses neither. __threadfence_block() is a real device builtin.
CUTE_DEVICE void kda_fence_block() {
    __threadfence_block();
}

// ---------------------------------------------------------------------------
// Warp roles and barrier ids
// ---------------------------------------------------------------------------
enum class WarpRole { MMA, LOAD, STORE, NonParticipant };

// User-visible NamedBarrier ids are offset by ReservedNamedBarrierCount inside
// cutlass, so these are indices, not raw hardware barrier numbers.
enum NamedBarrierId : int {
    kOutReady = 0,    // MMA warps -> store warp, per output stage
    kOutConsumed = 2, // store warp -> MMA warps, per output stage
    kStageFree = 4,   // consumers -> producer, per input stage
};

// ---------------------------------------------------------------------------
// Workspace byte layout. Unchanged from the CUDA backend: INV/Mqk stay 16x16
// and are moved by cp.async rather than the AIU (which would have forced a
// 64-element-wide pad), so get_workspace_size() in csrc/flash_kda.cpp needs no
// change and the two backends remain interchangeable at the ABI level.
// ---------------------------------------------------------------------------
template <int CHUNK, int D> struct WorkspaceSizes {
    static_assert(CHUNK * D * 2 % 128 == 0);
    static_assert(D * 4 % 128 == 0);
    static_assert(CHUNK * CHUNK * 2 % 128 == 0);

    static constexpr int kKDecayed = CHUNK * D * 2;  // 4096
    static constexpr int kQDecayed = CHUNK * D * 2;  // 4096
    static constexpr int kKRestored = CHUNK * D * 2; // 4096
    static constexpr int kGTotal = D * 4;            // 512
    static constexpr int kINV = CHUNK * CHUNK * 2;   // 512
    static constexpr int kMqk = CHUNK * CHUNK * 2;   // 512
    static constexpr int kBeta = 128;                // aligned fp32 sigmoid(beta)
    static constexpr int64_t kPerTile =
        kKDecayed + kQDecayed + kKRestored + kGTotal + kINV + kMqk + kBeta;
};

} // namespace flash_kda_ppu
