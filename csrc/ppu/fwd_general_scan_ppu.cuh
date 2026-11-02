#pragma once
#include <hggc_runtime.h>
#include <atomic>
#include <cooperative_groups.h>
namespace flash_kda_ppu {
// A owns IDs 0..3, B/replay 4..6, and carry 7 (user IDs, offset by
// CUTLASS). B drains final consumer arrivals before replay reuses its IDs.
template <int Stages, int Producers, int Consumers, int Base>
struct GeneralAsyncInPipeline : AsyncInPipeline<Stages, Producers, Consumers> {
    static_assert(Base + Stages + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount <= 16);
    CUTE_DEVICE void producer_acquire(int stage, int t) {
        if (t >= Stages)
            cutlass::arch::NamedBarrier::arrive_and_wait(Producers + Consumers, Base + stage);
    }
    CUTE_DEVICE void consumer_release(int stage) {
        cutlass::arch::NamedBarrier::arrive(Producers + Consumers, Base + stage);
    }
};
template <int D, int CHUNK, int InputStages, int OutputStages, int MmaWarps, bool HasInitial,
          int DvSplit, int Chunks, bool Output, bool Zero, bool Affine, bool Direct, int Base>
struct GeneralRecurrenceStorage {
    using Original = SharedStorageK2<D, CHUNK, InputStages, OutputStages, MmaWarps, HasInitial,
                                     DvSplit, Chunks, Output, Zero, Affine, Direct>;
    using InStage = typename Original::InStage;
    using OutStage = typename Original::OutStage;
    union {
        struct {
            InStage in[InputStages];
            OutStage out[OutputStages];
        };
        alignas(
            128) char state_stage[Original::kStateStageBytes > 0 ? Original::kStateStageBytes : 1];
    };
    GeneralAsyncInPipeline<InputStages, 32, 32 * MmaWarps, Base> in_pipe;
    SyncOutPipeline<OutputStages, 32 * MmaWarps, 32> out_pipe;
};
} // namespace flash_kda_ppu
namespace flash_kda_ppu {
// Host admission proves workspace, output and inputs are disjoint. Initial
// and final state deliberately remain unqualified to support exact in-place
// updates; no alias promise is made between those two public state pointers.
template <int D, int CHUNK, int P, int Mode, bool WantInitial, bool WantFinal, bool PublicFP32,
          int LongMmaWarps = 8, int PhysicalThreads = 288, bool DirectSerial = false,
          int SerialChunks = 2, int SerialStages = 2, int SerialMmaWarps = 4, int SerialDvSplit = 2>
__device__ __forceinline__ void general_phase(
    BF16 const *__restrict__ v_ptr, float const *__restrict__ ws_beta,
    BF16 const *__restrict__ ws_kd, BF16 const *__restrict__ ws_qd, BF16 const *__restrict__ ws_kr,
    float const *__restrict__ ws_gt, BF16 const *__restrict__ ws_inv,
    BF16 const *__restrict__ ws_mqk, void const *initial_state_ptr, void *final_state_ptr,
    BF16 *__restrict__ out_ptr, int T_total_runtime, int H_runtime, int public_sequences,
    int64_t const *__restrict__ cu_seqlens, int total_tiles, int const *__restrict__ tile_prefix,
    int v_stride, int *__restrict__ summary_nonzero, void *__restrict__ scratch_ptr) {
    static_assert(Mode >= 0 && Mode <= 3);
    constexpr bool Short = Mode == 0, Replay = Mode == 3, Homogeneous = Mode == 2;
    constexpr int GeneralBarrierBase = Mode == 2 ? 0 : 4;
    static_assert(!Homogeneous ||
                  GeneralBarrierBase + 3 + cutlass::arch::NamedBarrier::ReservedNamedBarrierCount <
                      16);
    int const N_runtime = Short ? public_sequences : public_sequences * P;
    int const *path_flag = nullptr;
    void *auxiliary_final_state_ptr = nullptr;
    BF16 const *zero_summary_b = nullptr;
    void const *external_initial_state = nullptr;
    void *external_final_state = nullptr;
    constexpr int InputStages = Short ? SerialStages : 3;
    constexpr int OutputStages = 2;
    constexpr int NumMmaWarps = Short ? SerialMmaWarps : LongMmaWarps;
    constexpr int DvSplit = Short ? SerialDvSplit : 1;
    constexpr int ChunksPerStage = Short ? SerialChunks : 1;
    constexpr int PathFlagMode = 0;
    constexpr int PrefixMode = 0;
    constexpr int PrefixChunks = 4;
    constexpr int StaticHeads = 0;
    constexpr int StaticTokens = 0;
    constexpr int StaticSequences = 0;
    constexpr int SerialSegmentIndex = -1;
    constexpr int repair_segment_index = 0;
    constexpr bool repair_prior_matched = true;
    constexpr bool HasStateIn = Short ? WantInitial : Replay;
    constexpr bool HasStateOut = (Short || Replay) ? WantFinal : true;
    constexpr bool StateFP32 = Short && PublicFP32;
    constexpr bool IsVarlen = Short;
    constexpr bool ProduceOutput = Short || Replay;
    constexpr bool IdentityState = Homogeneous;
    constexpr bool ZeroValue = Homogeneous;
    constexpr bool DualSummary = false;
    constexpr bool AffinePrepared = true;
    constexpr bool Segmented = !Short;
    constexpr bool DirectOutput = Replay || (Short && DirectSerial);
    constexpr bool OutputStateFP32 = (Short || Replay) && PublicFP32;
    constexpr int SegmentSequences = Short ? 1 : 0;
    constexpr int SegmentsPerSequence = Short ? 0 : P;
    constexpr bool DirectZeroCarry = false;
    constexpr bool ExternalHasStateIn = WantInitial;
    constexpr bool ExternalStateFP32 = PublicFP32;
    constexpr bool PrepublishFinal = false;
    constexpr bool SegmentCarryOut = false;
    constexpr bool FirstSegmentTrueB = false;
    constexpr bool PublicHasStateOut = false;
#define FLASH_KDA_GENERAL_SCRATCH scratch_ptr
#define FLASH_KDA_GENERAL_LOGICAL_SEQUENCE                                                         \
    (Short ? (public_sequences == 1                                                                \
                  ? 0                                                                              \
                  : (int(blockIdx.x) + int(gridDim.x) * int(blockIdx.y)) % public_sequences)       \
           : int(blockIdx.x))
#define FLASH_KDA_GENERAL_FUSED_PHASE
#include "fwd_recurrence_body_ppu.cuh"
#undef FLASH_KDA_GENERAL_FUSED_PHASE
#undef FLASH_KDA_GENERAL_LOGICAL_SEQUENCE
#undef FLASH_KDA_GENERAL_SCRATCH
}
} // namespace flash_kda_ppu
namespace flash_kda_ppu {
template <int D, int Split> struct GeneralCarryStorage {
    static constexpr int W = 128 / int(sizeof(BF16)), C = D / W;
    struct Stage {
        AiuTile<BF16, D, W, C> apb;
        AiuTile<BF16, D / Split, W, C> b;
    } stage[1];
    GeneralAsyncInPipeline<1, 32, 32 * (D / 16 / Split), 7> pipe;
};
template <int D, int P, bool HaveInitial, bool InitialFP32>
__device__ __forceinline__ void general_carry(BF16 const *apb_ptr, BF16 const *b_ptr,
                                              void const *initial_state_ptr, BF16 *carries_ptr,
                                              int H, int const *summary_nonzero, int sequence,
                                              int head, int dv_split, void *scratch) {
    constexpr int Segments = P, Stages = 1, DvSplit = P >= 4 ? 4 : 2,
                  NumMmaWarps = D / 16 / DvSplit, SequenceBatches = 0;
    constexpr bool HasStateIn = HaveInitial, InitialStateFP32 = InitialFP32, HomogeneousA = true;
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
    using BLoader = AiuTileLoader<BF16, D / DvSplit, kCubeW, kCubes>;
    using BTsm = TsmOps<D / DvSplit, kCubeW, kCubes>;

    auto &sm = *reinterpret_cast<GeneralCarryStorage<D, DvSplit> *>(scratch);

    if (!head_has_nonzero_summary<Segments>(summary_nonzero, head, H))
        return;
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
        Loader load_apb;
        BLoader load_b;
        int const matrix_rows = Segments * H * D;
        load_apb.init(apb_ptr, matrix_rows, D, D);
        load_b.init(b_ptr, matrix_rows, D, D);
        for (int segment = 0; segment < Segments; ++segment) {
            int const stage = segment % Stages;
            sm.pipe.producer_acquire(stage, segment);
            int const matrix_row = (segment * H + head) * D;
            if (summary_nonzero[segment * H + head])
                load_apb.load(sm.stage[stage].apb.data(), matrix_row);
            load_b.load(sm.stage[stage].b.data(), matrix_row + dv_base);
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
                BTsm::load_a(&b_add(0), sm.stage[stage].b.data(), nb, warp * 16);
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

namespace flash_kda_ppu {
// A temporal scan must cover at least one CTA per CU before its extra
// phases can amortize their launch and cooperative scheduling costs.
inline bool general_scan_grid_covers_device(int heads, int sequences, int segments) {
    int device = -1;
    if (hggcGetDevice(&device) != hggcSuccess)
        return false;
    static std::atomic<int> cu_count_by_device[64]{};
    int cus =
        device >= 0 && device < 64 ? cu_count_by_device[device].load(std::memory_order_acquire) : 0;
    if (!cus) {
        if (hggcDeviceGetAttribute(&cus, hggcDevAttrMultiProcessorCount, device) != hggcSuccess ||
            cus <= 0)
            return false;
        if (device >= 0 && device < 64)
            cu_count_by_device[device].store(cus, std::memory_order_release);
    }
    return int64_t(heads) * sequences * segments >= cus;
}
// The long grid is admitted only when every block can reside. A current
// device flag bypasses it for serial batches. The ordinary recurrence handles
// those batches and exact empty identities after this launch.
template <int D, int P, int CHUNK, bool HaveInitial, bool WantFinal, bool PublicFP32>
__device__ __forceinline__ void general_fused_body(
    BF16 const *__restrict__ v, float const *__restrict__ beta, BF16 const *__restrict__ kd,
    BF16 const *__restrict__ qd, BF16 const *__restrict__ kr, float const *__restrict__ gt,
    BF16 const *__restrict__ inv, BF16 const *__restrict__ mqk, void const *initial, void *final,
    BF16 *__restrict__ out, BF16 *__restrict__ workspace, int T, int H, int N,
    int64_t const *__restrict__ cu, int total_tiles, int const *__restrict__ prefix, int v_stride) {
    static_assert(P >= 4);
    static_assert(CHUNK == 16, "the general scan consumes C16 affine transforms");
    using RecurrenceStorage =
        SharedStorageK2<D, CHUNK, 3, 2, 8, true, 1, 1, true, false, true, true>;
    constexpr int CarrySplit = P >= 4 ? 4 : 2;
    constexpr size_t RecurrenceBytes = sizeof(RecurrenceStorage);
    constexpr size_t SharedBytes = RecurrenceBytes > sizeof(GeneralCarryStorage<D, CarrySplit>)
                                       ? RecurrenceBytes
                                       : sizeof(GeneralCarryStorage<D, CarrySplit>);
    struct alignas(128) SharedRoot {
        char bytes[SharedBytes];
    };
    __shared__ SharedRoot root;
    void *scratch = root.bytes;
    int const logical = int(blockIdx.x), sequence = N == 1 ? 0 : logical / P, segment = logical % P;
    int const head = int(blockIdx.y), tid = int(threadIdx.x), lane = tid % 32;
    if (cu && prefix[N + 2] == 0)
        return;
    bool const long_sequence =
        !cu || cu[sequence + 1] - cu[sequence] >= flash_kda_ppu_general_serial_tokens;
    int64_t const elements = int64_t(P) * N * H * D * D;
    BF16 *b = workspace, *a = b + elements, *carries = a + elements;
    int *nonzero = reinterpret_cast<int *>(carries + elements);
    if (long_sequence) {
        general_phase<D, CHUNK, P, 1, HaveInitial, WantFinal, PublicFP32>(
            v, beta, kd, qd, kr, gt, inv, mqk, nullptr, b, nullptr, T, H, N, cu, total_tiles,
            prefix, v_stride, nullptr, scratch);
        __syncthreads();
        general_phase<D, CHUNK, P, 2, HaveInitial, WantFinal, PublicFP32>(
            v, beta, kd, qd, kr, gt, inv, mqk, nullptr, a, nullptr, T, H, N, cu, total_tiles,
            prefix, v_stride, nonzero, scratch);
    }
    auto grid = cooperative_groups::this_grid();
    grid.sync();
    if (long_sequence) {
        int const *flags = nonzero + sequence * P * H;
        bool const live = head_has_nonzero_summary<P>(flags, head, H);
        if (!live) {
            int64_t const dst = (int64_t(logical) * H + head) * D * D;
            int64_t const previous = (int64_t(logical - 1) * H + head) * D * D;
            int64_t const public_base = (int64_t(sequence) * H + head) * D * D;
            if (segment == 0 && HaveInitial && PublicFP32) {
                for (int i = tid; i < D * D / 4; i += blockDim.x) {
                    float4 x = reinterpret_cast<float4 const *>(
                        static_cast<float const *>(initial) + public_base)[i];
                    alignas(8) BF16 y[4] = {BF16(x.x), BF16(x.y), BF16(x.z), BF16(x.w)};
                    reinterpret_cast<uint64_t *>(carries + dst)[i] =
                        *reinterpret_cast<uint64_t const *>(y);
                }
            } else {
                BF16 const *source =
                    segment > 0 ? b + previous
                                : (HaveInitial ? static_cast<BF16 const *>(initial) + public_base
                                               : nullptr);
                for (int i = tid; i < D * D / 8; i += blockDim.x) {
                    uint4 x = source ? reinterpret_cast<uint4 const *>(source)[i]
                                     : make_uint4(0, 0, 0, 0);
                    reinterpret_cast<uint4 *>(carries + dst)[i] = x;
                }
            }
        } else if (segment < CarrySplit) {
            general_carry<D, P, HaveInitial, PublicFP32>(a, b, initial, carries, H, nonzero,
                                                         sequence, head, segment, scratch);
        }
    }
    grid.sync();
    if (long_sequence)
        general_phase<D, CHUNK, P, 3, HaveInitial, WantFinal, PublicFP32>(
            v, beta, kd, qd, kr, gt, inv, mqk, carries, final, out, T, H, N, cu, total_tiles,
            prefix, v_stride, nullptr, scratch);
}

// This single-request variant keeps serial fallback in the same launch.
template <int D, int P, int CHUNK, bool HaveInitial, bool WantFinal, bool PublicFP32>
__device__ __forceinline__ void general_small_grid_body(
    BF16 const *__restrict__ v, float const *__restrict__ beta, BF16 const *__restrict__ kd,
    BF16 const *__restrict__ qd, BF16 const *__restrict__ kr, float const *__restrict__ gt,
    BF16 const *__restrict__ inv, BF16 const *__restrict__ mqk, void const *initial, void *final,
    BF16 *__restrict__ out, BF16 *__restrict__ workspace, int T, int H, int N,
    int64_t const *__restrict__ cu, int total_tiles, int const *__restrict__ prefix, int v_stride) {
    static_assert(P >= 4);
    static_assert(CHUNK == 16, "the general scan consumes C16 affine transforms");
    using RecurrenceStorage =
        SharedStorageK2<D, CHUNK, 3, 2, 4, true, 1, 1, true, false, true, true>;
    constexpr int CarrySplit = P >= 4 ? 4 : 2;
    constexpr int SerialMma = 4, SerialDv = 2;
    using SerialStorage =
        SharedStorageK2<D, CHUNK, 2, 2, SerialMma, true, SerialDv, 2, true, false, true, false>;
    constexpr size_t RecurrenceBytes = sizeof(RecurrenceStorage) > sizeof(SerialStorage)
                                           ? sizeof(RecurrenceStorage)
                                           : sizeof(SerialStorage);
    constexpr size_t SharedBytes = RecurrenceBytes > sizeof(GeneralCarryStorage<D, CarrySplit>)
                                       ? RecurrenceBytes
                                       : sizeof(GeneralCarryStorage<D, CarrySplit>);
    struct alignas(128) SharedRoot {
        char bytes[SharedBytes];
    };
    __shared__ SharedRoot root;
    void *scratch = root.bytes;
    int const logical = int(blockIdx.x), sequence = N == 1 ? 0 : logical / P, segment = logical % P;
    int const head = int(blockIdx.y), tid = int(threadIdx.x), lane = tid % 32;
    if (cu && __builtin_expect(prefix[N + 2] == 0, 1)) {
        // Match the native (sequence,head,row-slice) linear block order.
        int const serial_block = int(blockIdx.x) + int(gridDim.x) * int(blockIdx.y);
        if (serial_block < SerialDv * N * H)
            general_phase<D, CHUNK, P, 0, HaveInitial, WantFinal, PublicFP32, 4, 192, false, 2, 2,
                          SerialMma, SerialDv>(v, beta, kd, qd, kr, gt, inv, mqk, initial, final,
                                               out, T, H, N, cu, total_tiles, prefix, v_stride,
                                               nullptr, scratch);
        return;
    }
    bool const long_sequence =
        !cu || cu[sequence + 1] - cu[sequence] >= flash_kda_ppu_general_serial_tokens;
    int64_t const elements = int64_t(P) * N * H * D * D;
    BF16 *b = workspace, *a = b + elements, *carries = a + elements;
    int *nonzero = reinterpret_cast<int *>(carries + elements);
    if (long_sequence) {
        general_phase<D, CHUNK, P, 1, HaveInitial, WantFinal, PublicFP32, 4, 192, false, 2, 2,
                      SerialMma, SerialDv>(v, beta, kd, qd, kr, gt, inv, mqk, nullptr, b, nullptr,
                                           T, H, N, cu, total_tiles, prefix, v_stride, nullptr,
                                           scratch);
        __syncthreads();
        general_phase<D, CHUNK, P, 2, HaveInitial, WantFinal, PublicFP32, 4, 192, false, 2, 2,
                      SerialMma, SerialDv>(v, beta, kd, qd, kr, gt, inv, mqk, nullptr, a, nullptr,
                                           T, H, N, cu, total_tiles, prefix, v_stride, nonzero,
                                           scratch);
    }
    auto grid = cooperative_groups::this_grid();
    grid.sync();
    if (long_sequence) {
        int const *flags = nonzero + sequence * P * H;
        bool const live = head_has_nonzero_summary<P>(flags, head, H);
        if (!live) {
            int64_t const dst = (int64_t(logical) * H + head) * D * D;
            int64_t const previous = (int64_t(logical - 1) * H + head) * D * D;
            int64_t const public_base = (int64_t(sequence) * H + head) * D * D;
            if (segment == 0 && HaveInitial && PublicFP32) {
                for (int i = tid; i < D * D / 4; i += blockDim.x) {
                    float4 x = reinterpret_cast<float4 const *>(
                        static_cast<float const *>(initial) + public_base)[i];
                    alignas(8) BF16 y[4] = {BF16(x.x), BF16(x.y), BF16(x.z), BF16(x.w)};
                    reinterpret_cast<uint64_t *>(carries + dst)[i] =
                        *reinterpret_cast<uint64_t const *>(y);
                }
            } else {
                BF16 const *source =
                    segment > 0 ? b + previous
                                : (HaveInitial ? static_cast<BF16 const *>(initial) + public_base
                                               : nullptr);
                for (int i = tid; i < D * D / 8; i += blockDim.x) {
                    uint4 x = source ? reinterpret_cast<uint4 const *>(source)[i]
                                     : make_uint4(0, 0, 0, 0);
                    reinterpret_cast<uint4 *>(carries + dst)[i] = x;
                }
            }
        } else if (segment < CarrySplit) {
            general_carry<D, P, HaveInitial, PublicFP32>(a, b, initial, carries, H, nonzero,
                                                         sequence, head, segment, scratch);
        }
    }
    grid.sync();
    if (long_sequence)
        general_phase<D, CHUNK, P, 3, HaveInitial, WantFinal, PublicFP32, 4, 192, false, 2, 2,
                      SerialMma, SerialDv>(v, beta, kd, qd, kr, gt, inv, mqk, carries, final, out,
                                           T, H, N, cu, total_tiles, prefix, v_stride, nullptr,
                                           scratch);
}

template <int D, int P, int CHUNK, bool HaveInitial, bool WantFinal, bool PublicFP32>
__global__ void __maxnreg__(168)
    general_fused_scan(BF16 const *__restrict__ v, float const *__restrict__ beta,
                       BF16 const *__restrict__ kd, BF16 const *__restrict__ qd,
                       BF16 const *__restrict__ kr, float const *__restrict__ gt,
                       BF16 const *__restrict__ inv, BF16 const *__restrict__ mqk,
                       void const *initial, void *final, BF16 *__restrict__ out,
                       BF16 *__restrict__ workspace, int T, int H, int N,
                       int64_t const *__restrict__ cu, int total_tiles,
                       int const *__restrict__ prefix, int v_stride) {
    general_fused_body<D, P, CHUNK, HaveInitial, WantFinal, PublicFP32>(
        v, beta, kd, qd, kr, gt, inv, mqk, initial, final, out, workspace, T, H, N, cu, total_tiles,
        prefix, v_stride);
}

template <int D, int P, int CHUNK, bool HaveInitial, bool WantFinal, bool PublicFP32>
__global__ void __maxnreg__(168)
    general_single_scan(BF16 const *__restrict__ v, float const *__restrict__ beta,
                        BF16 const *__restrict__ kd, BF16 const *__restrict__ qd,
                        BF16 const *__restrict__ kr, float const *__restrict__ gt,
                        BF16 const *__restrict__ inv, BF16 const *__restrict__ mqk,
                        void const *initial, void *final, BF16 *__restrict__ out,
                        BF16 *__restrict__ workspace, int T, int H, int N,
                        int64_t const *__restrict__ cu, int total_tiles,
                        int const *__restrict__ prefix, int v_stride) {
    general_fused_body<D, P, CHUNK, HaveInitial, WantFinal, PublicFP32>(
        v, beta, kd, qd, kr, gt, inv, mqk, initial, final, out, workspace, T, H, 1, cu, total_tiles,
        prefix, v_stride);
}

template <int D, int P, int CHUNK, bool HaveInitial, bool WantFinal, bool PublicFP32>
__global__ void __maxnreg__(248)
    general_small_grid_single_scan(BF16 const *__restrict__ v, float const *__restrict__ beta,
                                   BF16 const *__restrict__ kd, BF16 const *__restrict__ qd,
                                   BF16 const *__restrict__ kr, float const *__restrict__ gt,
                                   BF16 const *__restrict__ inv, BF16 const *__restrict__ mqk,
                                   void const *initial, void *final, BF16 *__restrict__ out,
                                   BF16 *__restrict__ workspace, int T, int H, int N,
                                   int64_t const *__restrict__ cu, int total_tiles,
                                   int const *__restrict__ prefix, int v_stride) {
    general_small_grid_body<D, P, CHUNK, HaveInitial, WantFinal, PublicFP32>(
        v, beta, kd, qd, kr, gt, inv, mqk, initial, final, out, workspace, T, H, 1, cu, total_tiles,
        prefix, v_stride);
}
} // namespace flash_kda_ppu

template <int D, int P, int CHUNK, bool HaveInitial, bool WantFinal, bool PublicFP32>
static bool launch_general_scan_path(cutlass::bfloat16_t const *v, float const *__restrict__ beta,
                                     cutlass::bfloat16_t const *kd, cutlass::bfloat16_t const *qd,
                                     cutlass::bfloat16_t const *kr, float const *__restrict__ gt,
                                     cutlass::bfloat16_t const *inv, cutlass::bfloat16_t const *mqk,
                                     void const *initial, void *final, cutlass::bfloat16_t *out,
                                     void *scratch, int T, int H, int N,
                                     int64_t const *__restrict__ cu, int total_tiles,
                                     int *__restrict__ prefix, int v_stride, hggcStream_t stream) {
    if (!flash_kda_ppu::general_scan_grid_covers_device(H, N, P))
        return false;
    int device = 0;
    bool const fused_serial = flash_kda_ppu_fuse_general_serial(T, H, N);
    auto kernel =
        fused_serial ? flash_kda_ppu::general_small_grid_single_scan<D, P, CHUNK, HaveInitial,
                                                                     WantFinal, PublicFP32>
        : N == 1
            ? flash_kda_ppu::general_single_scan<D, P, CHUNK, HaveInitial, WantFinal, PublicFP32>
            : flash_kda_ppu::general_fused_scan<D, P, CHUNK, HaveInitial, WantFinal, PublicFP32>;
    int const mode = fused_serial ? 2 : (N == 1 ? 1 : 0);
    if (hggcGetDevice(&device) != hggcSuccess)
        return false;
    // These private kernels never change their shared-memory or register
    // configuration. Cache admission capacity separately for each device.
    static std::atomic<int> capacity_by_device[3][64]{};
    int capacity = device >= 0 && device < 64
                       ? capacity_by_device[mode][device].load(std::memory_order_acquire)
                       : 0;
    if (!capacity) {
        int cooperative = 0, cus = 0, resident = 0;
        if (hggcDeviceGetAttribute(&cooperative, hggcDevAttrCooperativeLaunch, device) !=
            hggcSuccess)
            return false;
        if (!cooperative)
            capacity = -1;
        else {
            if (hggcDeviceGetAttribute(&cus, hggcDevAttrMultiProcessorCount, device) !=
                    hggcSuccess ||
                hggcOccupancyMaxActiveBlocksPerMultiprocessor(
                    &resident, reinterpret_cast<void const *>(kernel), mode >= 2 ? 192 : 288, 0) !=
                    hggcSuccess)
                return false;
            capacity = cus * resident;
            if (!capacity)
                return false;
        }
        if (device >= 0 && device < 64)
            capacity_by_device[mode][device].store(capacity, std::memory_order_release);
    }
    if (capacity < 0 || int64_t(N) * P * H > capacity)
        return false;
    auto *workspace = static_cast<cutlass::bfloat16_t *>(scratch);
    void *args[] = {&v,   &beta,      &kd, &qd, &kr, &gt, &inv,         &mqk,    &initial, &final,
                    &out, &workspace, &T,  &H,  &N,  &cu, &total_tiles, &prefix, &v_stride};
    auto error = hggcLaunchCooperativeKernel(reinterpret_cast<void const *>(kernel), dim3(P * N, H),
                                             dim3(mode >= 2 ? 192 : 288), args, 0, stream);
    if (error != hggcSuccess)
        throw std::runtime_error(hggcGetErrorString(error));
    return true;
}
