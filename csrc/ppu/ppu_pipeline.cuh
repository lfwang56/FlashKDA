#pragma once

// Hand-rolled pipelines. The PPU cutlass fork has no pipeline classes at all --
// cutlass/pipeline/pipeline.hpp is 45 lines of #includes with no Pipeline* type --
// so cutlass::PipelineTmaAsync / PipelineAsync / PipelineState, which the CUDA
// kernels use, have to be rebuilt from primitives.
//
// Two shapes are needed, because only one direction of one pipeline involves an
// asynchronous engine:
//
//   AsyncInPipeline   gmem -> smem. Producer-to-consumer must be an mbarrier,
//                     because the AIU / cp.async completion, not the issuing
//                     thread, is what makes the data visible. The reverse
//                     direction is an ordinary rendezvous.
//   SyncOutPipeline   smem -> gmem. Both directions are ordinary rendezvous:
//                     there is no async store on PPU, so the producing threads'
//                     own stores are the completion event.
//
// NamedBarrier convention used below: arrive() is non-blocking, arrive_and_wait()
// blocks until `total` threads have arrived, and `total` counts producers plus
// consumers so both sides can use the same id.

#include "ppu_common.cuh"

namespace flash_kda_ppu {

// ---------------------------------------------------------------------------
// gmem -> smem, `Stages` buffers
// ---------------------------------------------------------------------------
template <int Stages, int NumProducerThreads, int NumConsumerThreads> struct AsyncInPipeline {
    static constexpr int kTotal = NumProducerThreads + NumConsumerThreads;
    // Any positive depth is correct: phase_of() below works for any Stages, a power
    // of two only lets the compiler turn its division into a shift. Non-powers are
    // allowed because they are what lets the per-CTA shared memory land just under
    // a CTAs/CU boundary -- three input stages is 64,384 B, which fits four CTAs
    // per CU where four stages fits only three.
    static_assert(Stages > 0, "at least one stage");

    // Signalled by copy completion, not by an arriving thread.
    __mbarrier_t full[Stages];

    // One thread, then the caller must __syncthreads() before any use.
    CUTE_DEVICE void init(int tid) {
        if (tid == 0) {
            CUTE_UNROLL
            for (int s = 0; s < Stages; ++s) {
                __mbarrier_init(&full[s], NumProducerThreads);
            }
            kda_fence_block();
        }
    }

    // A stage is reused every `Stages` iterations, so the mbarrier phase alternates
    // on that period. Computed here rather than passed in, so a caller cannot get
    // the parity wrong.
    CUTE_DEVICE static uint32_t phase_of(int t) {
        return uint32_t((t / Stages) & 1);
    }

    // --- producer, all NumProducerThreads lanes ---

    // Wait until the consumers have finished with this stage. Skipped for the
    // first `Stages` iterations, when the buffer has never been read.
    CUTE_DEVICE void producer_acquire(int stage, int t) {
        if (t >= Stages) {
            cutlass::arch::NamedBarrier::arrive_and_wait(kTotal, kStageFree + stage);
        }
    }

    // Issue the copies first, then this: it arms the barrier to fire on their
    // completion without counting as an arrival of its own.
    CUTE_DEVICE void producer_commit(int stage) {
        cutlass::arch::cpasync_barrier_arrive_noinc(&full[stage]);
    }

    // --- consumer, all NumConsumerThreads threads ---

    CUTE_DEVICE void consumer_wait(int stage, int t) {
        uint32_t const phase = phase_of(t);
        while (!cutlass::arch::test_wait(&full[stage], phase, 1)) {
        }
    }

    CUTE_DEVICE void consumer_release(int stage) {
        cutlass::arch::NamedBarrier::arrive(kTotal, kStageFree + stage);
    }
};

// ---------------------------------------------------------------------------
// smem -> gmem, `Stages` buffers, no async engine involved
// ---------------------------------------------------------------------------
template <int Stages, int NumProducerThreads, int NumConsumerThreads> struct SyncOutPipeline {
    static constexpr int kTotal = NumProducerThreads + NumConsumerThreads;

    CUTE_DEVICE void producer_acquire(int stage, int t) {
        if (t >= Stages) {
            cutlass::arch::NamedBarrier::arrive_and_wait(kTotal, kOutConsumed + stage);
        }
    }

    // The producing threads' own smem stores are the completion event, so this
    // only has to order them before the consumer's reads.
    CUTE_DEVICE void producer_commit(int stage) {
        cutlass::arch::NamedBarrier::arrive(kTotal, kOutReady + stage);
    }

    CUTE_DEVICE void consumer_wait(int stage, int /*t*/) {
        cutlass::arch::NamedBarrier::arrive_and_wait(kTotal, kOutReady + stage);
    }

    CUTE_DEVICE void consumer_release(int stage) {
        cutlass::arch::NamedBarrier::arrive(kTotal, kOutConsumed + stage);
    }
};

} // namespace flash_kda_ppu
