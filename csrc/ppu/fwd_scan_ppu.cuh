#pragma once

// Time-parallel affine segment scan for the TP16 long-prefill path.
//
// Separate recurrences evaluate B=F(0,V) and the homogeneous transition
// A=F(I,0) for each segment. The scan evaluates
//
//     carry[j+1] = carry[j] @ A[j] + B[j]
//
// with 128x128 Tensor Cell products, split across independent d_v rows. Heads
// whose A matrices are all exactly zero use parallel carry copies. The resulting
// [segment, head, D, D] states seed a parallel recurrence replay.

#include "ppu_aiu.cuh"
#include "ppu_common.cuh"
#include "ppu_frag.cuh"
#include "ppu_layouts.cuh"
#include "ppu_pipeline.cuh"

namespace flash_kda_ppu {


// Prove prefix convergence before reusing F(0,V)'s remaining outputs. This
// comparison is exact over every BF16 bit, including signed zero. A nonzero A
// never consumes the checkpoint buffer: it still contains the live A matrix.
template <int D, int Segments, int CHUNK, int PrefixChunks, bool HasStateOut,
          bool OutputStateFP32, int SequenceBatches = 1>
__global__ void flash_kda_ppu_compare_replay_prefix(
    BF16 const *f0_prefix, BF16 const *true_prefix, BF16 const *f0_final,
    void *final_state, int const *a_nonzero, int *equal_flags, int T_total, int H,
    int const *path_flag = nullptr) {
    if (path_flag && *path_flag == 0) return;
    int const segment = blockIdx.x, head = blockIdx.y, sequence = blockIdx.z;
    int const tiles = (T_total / SequenceBatches + CHUNK - 1) / CHUNK;
    int const per_segment = tiles / Segments, remainder = tiles % Segments;
    int const count = per_segment + (segment < remainder);
    int const index = (sequence * Segments + segment) * H + head;
    bool const full_prefix = count <= PrefixChunks;
    if (!full_prefix && a_nonzero[index] != 0) {
        if (threadIdx.x == 0) equal_flags[index] = 0;
        return;
    }
    __shared__ unsigned mismatch;
    if (threadIdx.x == 0) mismatch = 0;
    __syncthreads();
    int64_t const base = int64_t(index) * D * D;
    if (!full_prefix) {
        bool different = false;
        for (int packet = threadIdx.x; packet < D * D / 8; packet += blockDim.x) {
            uint4 const a = reinterpret_cast<uint4 const *>(f0_prefix + base)[packet];
            uint4 const b = reinterpret_cast<uint4 const *>(true_prefix + base)[packet];
            different |= a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
        }
        if (__any_sync(0xffffffffu, different) && int(threadIdx.x) % 32 == 0)
            atomicOr(&mismatch, 1u);
    }
    __syncthreads();
    bool const equal = mismatch == 0;
    if (threadIdx.x == 0) equal_flags[index] = equal;
    if constexpr (HasStateOut) {
        if (equal && segment + 1 == Segments) {
            // A full prefix is the actual complete replay state, even when it
            // differs from F(0,V). Otherwise exact equality proves the F0 tail.
            BF16 const *source = (full_prefix ? true_prefix : f0_final) + base;
            int64_t const dst = int64_t(sequence * H + head) * D * D;
            if constexpr (OutputStateFP32) {
                for (int i = threadIdx.x; i < D * D / 4; i += blockDim.x) {
                    float4 value = make_float4(float(source[4*i]), float(source[4*i+1]),
                                              float(source[4*i+2]), float(source[4*i+3]));
                    reinterpret_cast<float4 *>(static_cast<float *>(final_state) + dst)[i] = value;
                }
            } else {
                for (int i = threadIdx.x; i < D * D / 8; i += blockDim.x)
                    reinterpret_cast<cute::uint128_t *>(static_cast<BF16 *>(final_state) + dst)[i] =
                        reinterpret_cast<cute::uint128_t const *>(source)[i];
            }
        }
    }
}

// All lanes inspect the same head's metadata; every warp obtains the same
// result. This is a bitwise-zero check, not a gate/contraction threshold.
template <int Segments>
CUTE_DEVICE bool head_has_nonzero_summary(int const *flags, int head, int H) {
    bool nonzero = false;
    for (int i = int(threadIdx.x) % 32; i < Segments; i += 32)
        nonzero |= flags[i * H + head] != 0;
    return __any_sync(0xffffffffu, nonzero);
}

// When every A[j] is zero, carry[j] = B[j-1] for j>0. Copy these independent
// matrices in parallel instead of serially staging every segment through AIU.
// Heads with a live A are handled by the full affine scan below.
template <int D, int Segments, bool HasStateIn, bool InitialStateFP32 = false,
          int SequenceBatches = 1>
__global__ void flash_kda_ppu_copy_zero_summary_carries(BF16 const *b, void const *initial,
                                                        BF16 *carries, int H, int const *flags,
                                                        int const *path_flag = nullptr) {
    if (path_flag && *path_flag == 0)
        return;
    int const sequence = SequenceBatches == 1 ? 0 : int(blockIdx.z);
    b += int64_t(sequence) * Segments * H * D * D;
    carries += int64_t(sequence) * Segments * H * D * D;
    flags += sequence * Segments * H;
    if constexpr (HasStateIn)
        initial = static_cast<char const *>(initial) +
                  int64_t(sequence) * H * D * D * (InitialStateFP32 ? 4 : 2);
    int const segment = blockIdx.x, head = blockIdx.y;
    if (head_has_nonzero_summary<Segments>(flags, head, H))
        return;
    int64_t const matrix_elems = D * D;
    auto *dst =
        reinterpret_cast<cute::uint128_t *>(carries + (int64_t(segment) * H + head) * matrix_elems);
    if constexpr (HasStateIn && InitialStateFP32) {
        if (segment == 0) {
            // Convert the external input once, directly into the first BF16 carry.
            // The segment summaries and every recurrent chunk remain unchanged.
            auto const *src = static_cast<float const *>(initial) + int64_t(head) * matrix_elems;
            auto *out = carries + int64_t(head) * matrix_elems;
            for (int i = threadIdx.x; i < D * D / 4; i += blockDim.x) {
                float4 const value = reinterpret_cast<float4 const *>(src)[i];
                alignas(8) BF16 converted[4] = {BF16(value.x), BF16(value.y), BF16(value.z),
                                                BF16(value.w)};
                reinterpret_cast<uint64_t *>(out)[i] =
                    *reinterpret_cast<uint64_t const *>(converted);
            }
            return;
        }
    }
    BF16 const *src =
        segment > 0
            ? b + (int64_t(segment - 1) * H + head) * matrix_elems
            : (HasStateIn ? static_cast<BF16 const *>(initial) + int64_t(head) * matrix_elems
                          : nullptr);
    for (int i = threadIdx.x; i < D * D / 8; i += blockDim.x) {
        cute::uint128_t value{};
        if (segment > 0 || HasStateIn)
            value = reinterpret_cast<cute::uint128_t const *>(src)[i];
        dst[i] = value;
    }
}

template <int D, int Stages, int NumMmaWarps> struct SharedStorageSegmentScan {
    static constexpr int kCubeW = 128 / int(sizeof(BF16));
    static constexpr int kCubes = D / kCubeW;
    using MatrixTile = AiuTile<BF16, D, kCubeW, kCubes>;

    struct Stage {
        MatrixTile apb;
        MatrixTile b;
    } stage[Stages];

    AsyncInPipeline<Stages, 32, 32 * NumMmaWarps> pipe;
};

template <int D, int Segments, bool HasStateIn, int Stages = 2, int NumMmaWarps = 8,
          bool HomogeneousA = false, int DvSplit = 1, bool InitialStateFP32 = false,
          int SequenceBatches = 1>
__global__ void __launch_bounds__(32 * (NumMmaWarps + 1), 1)
    flash_kda_ppu_segment_scan(BF16 const *__restrict__ apb_ptr, BF16 const *__restrict__ b_ptr,
                               void const *__restrict__ initial_state_ptr,
                               BF16 *__restrict__ carries_ptr, int H,
                               int const *__restrict__ summary_nonzero,
                               int const *path_flag = nullptr) {
    if (path_flag && *path_flag == 0)
        return;
    int const sequence = SequenceBatches == 1 ? 0 : int(blockIdx.z);
    apb_ptr += int64_t(sequence) * Segments * H * D * D;
    b_ptr += int64_t(sequence) * Segments * H * D * D;
    carries_ptr += int64_t(sequence) * Segments * H * D * D;
    summary_nonzero += sequence * Segments * H;
    if constexpr (HasStateIn)
        initial_state_ptr = static_cast<char const *>(initial_state_ptr) +
                            int64_t(sequence) * H * D * D * (InitialStateFP32 ? 4 : 2);
    using namespace cute;
    static_assert(D == 128, "segment scan is specialized for D=128");
    static_assert(NumMmaWarps * DvSplit == D / 16,
                  "the d_v splits must own all 16-row state blocks");

    constexpr int kCubeW = 128 / int(sizeof(BF16));
    constexpr int kCubes = D / kCubeW;
    constexpr int kBlocks = D / 16;
    using Tsm = TsmOps<D, kCubeW, kCubes>;
    using Loader = AiuTileLoader<BF16, D, kCubeW, kCubes>;

    __shared__ SharedStorageSegmentScan<D, Stages, NumMmaWarps> sm;

    int const head = blockIdx.x;
    if (!head_has_nonzero_summary<Segments>(summary_nonzero, head, H))
        return;
    int const dv_split = blockIdx.y;
    int const dv_base = dv_split * NumMmaWarps * 16;
    int const tid = threadIdx.x;
    int const warp = tid / 32;
    int const lane = tid % 32;
    bool const is_mma = warp < NumMmaWarps;
    bool const is_load = warp == NumMmaWarps;

    auto mma =
        make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr_mma = mma.get_slice(lane);
    Tensor ref = make_tensor(make_smem_ptr(sm.stage[0].apb.data()),
                             Layout<Shape<_16, _16>, Stride<_16, _1>>{});
    auto ref16 = local_tile(ref, Shape<_16, _16>{}, make_coord(0, 0));
    using AFrag = decltype(thr_mma.partition_fragment_A(ref16));
    using BFrag = decltype(thr_mma.partition_fragment_B(ref16));
    using AccFrag = decltype(thr_mma.make_fragment_C(thr_mma.partition_C(ref16)));

    AFrag carry[kBlocks];

    sm.pipe.init(tid);
    __syncthreads();

    if (is_mma) {
        CUTE_UNROLL
        for (int kb = 0; kb < kBlocks; ++kb) {
            if constexpr (HasStateIn) {
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(carry[0]))::value / 2; ++j) {
                    FragCoord const c = frag_coord_mn(lane, 2 * j);
                    int const row = dv_base + warp * 16 + c.m;
                    int const col = kb * 16 + c.n;
                    int64_t const off = int64_t(head) * D * D + int64_t(row) * D + col;
                    BF16 pair[2];
                    if constexpr (InitialStateFP32) {
                        float2 const value = *reinterpret_cast<float2 const *>(
                            &static_cast<float const *>(initial_state_ptr)[off]);
                        pair[0] = BF16(value.x);
                        pair[1] = BF16(value.y);
                    } else {
                        *reinterpret_cast<uint32_t *>(pair) = *reinterpret_cast<uint32_t const *>(
                            &static_cast<BF16 const *>(initial_state_ptr)[off]);
                    }
                    carry[kb](2 * j) = pair[0];
                    carry[kb](2 * j + 1) = pair[1];
                }
            } else {
                clear(carry[kb]);
            }
        }
    }

    if (is_load) {
        Loader load_apb, load_b;
        int const matrix_rows = Segments * H * D;
        load_apb.init(apb_ptr, matrix_rows, D, D);
        load_b.init(b_ptr, matrix_rows, D, D);
        for (int segment = 0; segment < Segments; ++segment) {
            int const stage = segment % Stages;
            sm.pipe.producer_acquire(stage, segment);
            int const matrix_row = (segment * H + head) * D;
            if (summary_nonzero[segment * H + head])
                load_apb.load(sm.stage[stage].apb.data(), matrix_row);
            load_b.load(sm.stage[stage].b.data(), matrix_row);
            kda_fence_block();
            sm.pipe.producer_commit(stage);
        }
    }

    if (is_mma) {
        for (int segment = 0; segment < Segments; ++segment) {
            // Store the start state for this segment.  The fragment's adjacent
            // value pair is adjacent d_k, so each lane emits one packed word.
            CUTE_UNROLL
            for (int kb = 0; kb < kBlocks; ++kb) {
                CUTE_UNROLL
                for (int j = 0; j < decltype(size(carry[0]))::value / 2; ++j) {
                    FragCoord const c = frag_coord_mn(lane, 2 * j);
                    int const row = dv_base + warp * 16 + c.m;
                    int const col = kb * 16 + c.n;
                    int64_t const off =
                        int64_t(segment * H + head) * D * D + int64_t(row) * D + col;
                    BF16 pair[2] = {carry[kb](2 * j), carry[kb](2 * j + 1)};
                    *reinterpret_cast<uint32_t *>(&carries_ptr[off]) =
                        *reinterpret_cast<uint32_t const *>(pair);
                }
            }

            int const stage = segment % Stages;
            sm.pipe.consumer_wait(stage, segment);

            // Eight n-block accumulators provide enough independent MMA chains
            // to hide the 28-cycle tc02 accumulator latency.
            AccFrag next[kBlocks];
            CUTE_UNROLL
            for (int nb = 0; nb < kBlocks; ++nb)
                clear(next[nb]);

            if (summary_nonzero[segment * H + head]) {
                CUTE_UNROLL
                for (int kb = 0; kb < kBlocks; ++kb) {
                    CUTE_UNROLL
                    for (int nb = 0; nb < kBlocks; ++nb) {
                        BFrag apb_bt;
                        Tsm::load_bt(&apb_bt(0), sm.stage[stage].apb.data(), nb, kb * 16);
                        if constexpr (!HomogeneousA) {
                            BFrag b_bt;
                            Tsm::load_bt(&b_bt(0), sm.stage[stage].b.data(), nb, kb * 16);
                            CUTE_UNROLL
                            for (int v = 0; v < decltype(size(apb_bt))::value; ++v) {
                                apb_bt(v) = BF16(float(apb_bt(v)) - float(b_bt(v)));
                            }
                        }
                        cute::gemm(thr_mma, carry[kb], apb_bt, next[nb]);
                    }
                }
            }

            CUTE_UNROLL
            for (int nb = 0; nb < kBlocks; ++nb) {
                AFrag b_add;
                Tsm::load_a(&b_add(0), sm.stage[stage].b.data(), nb, dv_base + warp * 16);
                CUTE_UNROLL
                for (int v = 0; v < decltype(size(next[0]))::value; ++v) {
                    next[nb](v) += float(b_add(v));
                }
                narrow_into(carry[nb], next[nb]);
            }
            sm.pipe.consumer_release(stage);
        }
    }
}

} // namespace flash_kda_ppu
