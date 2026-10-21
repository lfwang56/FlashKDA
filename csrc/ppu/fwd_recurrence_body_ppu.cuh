// Intentionally unguarded: each caller retains the original kernel compilation context.
{
    using namespace cute;
    // The private prefix experiment is selected only at H6/N1/T16384,
    // partitioned into26 complete-tile segments. Other recurrences keep their
    // runtime dimensions. Constant address factors avoid repeated spill reloads.
    int const H = StaticHeads > 0 ? StaticHeads : H_runtime;
    int const T_total = StaticTokens > 0 ? StaticTokens : T_total_runtime;
    int const N = StaticSequences > 0 ? StaticSequences : N_runtime;
    constexpr bool kH12N4P2Scope = StaticHeads == 12 && StaticTokens == 8192 &&
        StaticSequences == 8 && SegmentSequences == 4 && SegmentsPerSequence == 2;
    constexpr bool kH12N4P3Scope = StaticHeads == 12 && StaticTokens == 8192 &&
        StaticSequences == 12 && SegmentSequences == 4 && SegmentsPerSequence == 3;
#ifdef FLASH_KDA_COMPACT_P2_REPAIR
    constexpr bool kCompactP2Repair = true;
    static_assert(kH12N4P2Scope && PrefixMode == 4 && SerialSegmentIndex == 1 &&
                  HasStateIn && HasStateOut && PublicHasStateOut && SegmentCarryOut &&
                  !StateFP32 && !OutputStateFP32 && !DualSummary,
                  "compact carry/footer is restricted to fixed true-first H12/N4 repair");
#else
    constexpr bool kCompactP2Repair = false;
#endif
    static_assert(PrefixMode == 0 ||
                  ((StaticHeads == 6 && StaticTokens == 16384 && StaticSequences == 26) ||
                   (StaticHeads == 12 && StaticTokens == 8192 && StaticSequences == 13) ||
                   (StaticHeads == 6 && StaticTokens == 16384 && StaticSequences == 24 &&
                    ((SegmentSequences == 4 && SegmentsPerSequence == 6) ||
                     (SegmentSequences == 8 && SegmentsPerSequence == 3))) ||
                   (StaticHeads == 6 && StaticTokens == 16384 && StaticSequences == 16 &&
                    SegmentSequences == 8 && SegmentsPerSequence == 2) || kH12N4P2Scope || kH12N4P3Scope));
    static_assert(OutputStateFP32 == StateFP32 || (Segmented && ProduceOutput && OutputStateFP32),
                  "mixed state storage is reserved for segmented replay publication");
    // PrefixMode's existing shape assertion covers all three qualified paths.
    static_assert(!DirectZeroCarry || PrefixMode == 2 || PrefixMode == 4 || PrefixMode == 5);
    static_assert(!PrepublishFinal || (PublicHasStateOut &&
                  (PrefixMode == 2 || PrefixMode == 3 || PrefixMode == 6 || PrefixMode == 7)));
    static_assert(PathFlagMode >= -2 && PathFlagMode <= 2);
    static_assert(PrefixMode >= 0 && PrefixMode <= 7 && PrefixChunks > 0);
    static_assert(PrefixMode == 0 ||
                  (CHUNK == 16 && D == 128 && DvSplit == 1 && NumMmaWarps == 8 &&
                   ChunksPerStage == 1 && Segmented && ProduceOutput && AffinePrepared &&
                   !IdentityState && !ZeroValue && !DualSummary && DirectOutput),
                  "prefix reuse is isolated to full-row C16 segment recurrences");
    static_assert(!Segmented || !IsVarlen, "segment bounds are derived from prepared tile ranges");
    static_assert(!DualSummary || (!HasStateIn && HasStateOut && !StateFP32 && !ProduceOutput &&
                                   !IdentityState && !ZeroValue),
                  "dual summary is B=F(0,V) plus A=F(I,0)");
    if constexpr (PathFlagMode == 1) {
        if (*path_flag == 0)
            return;
    } else if constexpr (PathFlagMode == -1) {
        if (*path_flag != 0)
            return;
    } else if constexpr (PathFlagMode == 2) {
        // Prefer P3 whenever eligible; P2 is its independently guarded fallback.
        if (path_flag[0] != 0 || path_flag[1] == 0)
            return;
    } else if constexpr (PathFlagMode == -2) {
        if (path_flag[0] != 0 || path_flag[1] != 0)
            return;
    }
    static_assert(SerialSegmentIndex < 0 ||
                  ((SegmentSequences == 8 || kH12N4P2Scope || kH12N4P3Scope) &&
                   (SegmentsPerSequence == 2 || SegmentsPerSequence == 3) &&
                   SerialSegmentIndex < SegmentsPerSequence &&
                   (PrefixMode == 2 || PrefixMode == 3 || PrefixMode == 4 || PrefixMode == 6 || PrefixMode == 7)));
    static_assert(!SegmentCarryOut || (SerialSegmentIndex >= 0 &&
                  (PrefixMode == 3 || PrefixMode == 4 || PrefixMode == 6 || PrefixMode == 7) && HasStateOut));
    static_assert(!FirstSegmentTrueB ||
                  (PrefixMode == 1 && !HasStateIn && !StateFP32 && ExternalHasStateIn &&
                   (SegmentSequences == 8 || kH12N4P2Scope || kH12N4P3Scope) &&
                   (SegmentsPerSequence == 2 || SegmentsPerSequence == 3)));
    static_assert(PrefixMode != 4 ||
                  (((StaticHeads == 6 && StaticTokens == 16384 && StaticSequences == 16 &&
                     SegmentSequences == 8 && SegmentsPerSequence == 2) || kH12N4P2Scope) &&
                   SerialSegmentIndex == 1 &&
                   HasStateIn && HasStateOut && !StateFP32 && !OutputStateFP32 &&
                   InputStages == 2 && ChunksPerStage == 1 && PrefixChunks == 3 &&
                   ExternalHasStateIn && DirectZeroCarry && SegmentCarryOut && PublicHasStateOut));
    static_assert(PrefixMode != 5 && PrefixMode != 6 ||
                  (((StaticHeads == 6 && StaticTokens == 16384 && StaticSequences == 24 &&
                     SegmentSequences == 8 && SegmentsPerSequence == 3) || kH12N4P3Scope) &&
                   HasStateIn && HasStateOut && !StateFP32 && !OutputStateFP32 &&
                   PrefixChunks == 3));
    static_assert(PrefixMode != 5 ||
                  (SerialSegmentIndex < 0 && InputStages == 3 && DirectZeroCarry &&
                   ExternalHasStateIn && !SegmentCarryOut && !PublicHasStateOut));
    static_assert(PrefixMode != 6 ||
                  (SerialSegmentIndex == 2 && InputStages == 2 && !DirectZeroCarry &&
                   SegmentCarryOut && PublicHasStateOut));
    static_assert(PrefixMode != 7 ||
                  (((StaticHeads == 6 && StaticTokens == 16384 && StaticSequences == 24 &&
                     SegmentSequences == 8 && SegmentsPerSequence == 3) || kH12N4P3Scope) && SerialSegmentIndex == 0 &&
                   PrefixChunks == 3 && InputStages == 2 && HasStateIn && HasStateOut &&
                   !StateFP32 && !OutputStateFP32 && !DirectZeroCarry &&
                   SegmentCarryOut && PublicHasStateOut && PrepublishFinal));
    static_assert(!(PrefixMode == 1 && FirstSegmentTrueB && PublicHasStateOut) ||
                  (((StaticSequences == 24 && SegmentSequences == 8 && SegmentsPerSequence == 3) ||
                    kH12N4P3Scope) && HasStateOut && !OutputStateFP32));
#ifdef FLASH_KDA_H12_N4_HYBRID_B_P2_MAPPING
    static_assert(kH12N4P2Scope && PrefixMode == 1 && FirstSegmentTrueB && !PublicHasStateOut);
    int const logical_sequence = (int(blockIdx.x) + 12 * int(blockIdx.y)) % 8;
#elif defined(FLASH_KDA_HYBRID_B_P2_MAPPING)
    static_assert(PrefixMode == 1 && FirstSegmentTrueB &&
                  SegmentsPerSequence == 2 && StaticSequences == 16 &&
                  StaticHeads == 6 && !PublicHasStateOut);
    int const logical_sequence = (int(blockIdx.x) + 24 * int(blockIdx.y)) % 16;
#else
    int const logical_sequence = PrefixMode == 7
        ? int(blockIdx.x) * SegmentsPerSequence + repair_segment_index
        : PrefixMode == 5
        ? (int(blockIdx.x) / (SegmentsPerSequence - 1)) * SegmentsPerSequence +
              1 + int(blockIdx.x) % (SegmentsPerSequence - 1)
        : (SerialSegmentIndex < 0 ? int(blockIdx.x) :
           int(blockIdx.x) * SegmentsPerSequence + SerialSegmentIndex);
#endif
    auto publish_segment_result = [&](BF16 const *source) {
        if constexpr (SegmentCarryOut) {
            // The match flag identifies B itself as the true final. A later
            // prefix reads B directly; no internal carry copy is necessary.
            if constexpr (PublicHasStateOut)
                publish_reused_segment_final<D, ExternalStateFP32, SegmentSequences, SegmentsPerSequence>(
                    source, external_final_state, H, N, logical_sequence);
        } else {
            publish_reused_segment_final<D, OutputStateFP32, SegmentSequences, SegmentsPerSequence>(
                source, final_state_ptr, H, N);
        }
    };
    if constexpr (PrefixMode == 3) {
        // A separate comparison kernel proved every resident BF16 state word.
        // No pipeline has started, so this return is uniform for the whole CTA.
        if (summary_nonzero[logical_sequence * H + int(blockIdx.y)] != 0) {
            if constexpr (HasStateOut && !PrepublishFinal)
                publish_segment_result(static_cast<BF16 const *>(auxiliary_final_state_ptr));
            return;
        }
    }
    bool prior_prefix_matched = true;
    if constexpr (PrefixMode == 6) {
        // Prefix2 was hypothetical. It is authorized only if prefix1 matched
        // its complete BF16 checkpoint; otherwise recompute ALL of segment2.
        prior_prefix_matched =
            summary_nonzero[N * H + (logical_sequence - 1) * H + int(blockIdx.y)] != 0;
        bool const current_matched =
            summary_nonzero[N * H + logical_sequence * H + int(blockIdx.y)] != 0;
        if (prior_prefix_matched && current_matched) {
            // The disjoint public final was already provisionally stored by B2.
            // Both complete-state comparisons now authorize that exact result.
            if constexpr (!PrepublishFinal) publish_segment_result(zero_summary_b);
            return;
        }
        if (!prior_prefix_matched && threadIdx.x == 0)
            summary_nonzero[N * H + logical_sequence * H + int(blockIdx.y)] = 0;
    }
    // A CTA owns D / DvSplit rows of the d_v axis; blockIdx.z says which.
    //
    // The recurrence is row-local in d_v at every phase, which is what makes the
    // split free: u^T[d_v,t] and out[t,d_v] read only row d_v of the state,
    // U^T = u^T @ INV mixes over t alone, and dS[d_v,d_k] writes only row d_v. So
    // the CTAs need no communication -- they replicate the kd/qd/kr loads and write
    // disjoint slices of out and of the final state.
    //
    // This exists because the grid was (N, H) and nothing else. Kimi K3 has 96 KDA
    // heads (linear_attn_config.num_heads) sharded by TP, so TP16 leaves 6 per rank
    // and a single-request prefill launched 6 CTAs -- the chip idle and the
    // chunk-to-chunk chain fully exposed. Measured against SGLang's Triton chunk_kda
    // at H=6, FlashKDA ran 0.47x at T=8192 for that reason.
    constexpr int kDvPerCta = D / DvSplit;
    using L2 = K2Layouts<D, CHUNK, kDvPerCta>;
    constexpr int kStageRows = CHUNK * ChunksPerStage;
    using Tsm = TsmOps<3 * kStageRows, L2::kCubeW, L2::kCubesPerTile>;
    using VTsm = TsmOps<kStageRows, L2::kCubeW, L2::kVCubesPerTile>;
    using Loader = AiuTileLoader<BF16, 3 * kStageRows, L2::kCubeW, L2::kCubesPerTile>;
    using VLoader = AiuTileLoader<BF16, kStageRows, L2::kCubeW, L2::kVCubesPerTile>;

    constexpr int kMmaThreads = 32 * NumMmaWarps;
    constexpr int kThreads = kMmaThreads + 32 + (ProduceOutput && !DirectOutput ? 32 : 0);
    static_assert(ChunksPerStage == 1 || ChunksPerStage == 2 || ChunksPerStage == 3 ||
                      ChunksPerStage == 4,
                  "the grouped recurrence supports one to four chunks per stage");
    // Each MMA warp owns kDvPerCta / NumMmaWarps rows, as 16-row blocks.
    constexpr int kDvPerWarp = kDvPerCta / NumMmaWarps;
    constexpr int kMBlocks = kDvPerWarp / 16;
    constexpr int kKBlocks = D / 16;
    static_assert(kDvPerWarp % 16 == 0,
                  "D / (DvSplit * NumMmaWarps) must be a whole number of 16-row blocks");

    using Storage =
        SharedStorageK2<D, CHUNK, InputStages, OutputStages, NumMmaWarps,
                        (HasStateIn || (FirstSegmentTrueB && !ExternalStateFP32)), DvSplit,
                        ChunksPerStage, ProduceOutput, ZeroValue, AffinePrepared, DirectOutput>;
#ifdef FLASH_KDA_EXTERNAL_REPAIR_STORAGE
    Storage *storage = &FLASH_KDA_EXTERNAL_REPAIR_STORAGE.pipeline;
    unsigned *fused_prefix_mismatch = &FLASH_KDA_EXTERNAL_REPAIR_STORAGE.mismatch;
#elif defined(FLASH_KDA_EXTERNAL_B_STORAGE)
    // The private hybrid kernel selects exactly one family per complete CTA.
    // A typed, common root allocation preserves AIU alignment in both branches.
    Storage *storage = &FLASH_KDA_EXTERNAL_B_STORAGE;
    unsigned *fused_prefix_mismatch = nullptr;
#else
    Storage *storage = nullptr;
    unsigned *fused_prefix_mismatch = nullptr;
    if constexpr (PrefixMode == 4 || PrefixMode == 5) {
        // Keep every live shared field in one aggregate. On this SDK, adding a
        // separate scalar symbol beside a storage array can misalign AIU tiles.
        struct FusedStorage { Storage pipeline; unsigned mismatch; };
        __shared__ FusedStorage fused;
        storage = &fused.pipeline;
        fused_prefix_mismatch = &fused.mismatch;
    } else {
        __shared__ Storage local_storage;
        storage = &local_storage;
    }
#endif
    auto &sm = *storage;

    int const seq_idx = logical_sequence;
#ifdef FLASH_KDA_H12_N4_HYBRID_B_P2_MAPPING
    int const head_idx = (int(blockIdx.x) + 12 * int(blockIdx.y)) / 8;
#elif defined(FLASH_KDA_HYBRID_B_P2_MAPPING)
    int const head_idx = (int(blockIdx.x) + 24 * int(blockIdx.y)) / 16;
#else
    int const head_idx = blockIdx.y;
#endif
    int const dv_cta = int(blockIdx.z) * kDvPerCta;
    int const tid = threadIdx.x;
    // This one-dimensional CTA has converged full warps here; communicate
    // the same logical index through the installed SDK's uniform helper.
    int const warp_id = cutlass::canonical_warp_idx_sync();
#ifdef FLASH_KDA_HYBRID_B_LOGICAL_RANGES
    int const lane = int(unsigned(threadIdx.x) & 31u);
#else
    int const lane = tid % 32;
#endif

    WarpRole role = WarpRole::NonParticipant;
    if (warp_id < NumMmaWarps)
        role = WarpRole::MMA;
    else if (warp_id == NumMmaWarps)
        role = WarpRole::LOAD;
    else if constexpr (ProduceOutput && !DirectOutput) {
        if (warp_id == NumMmaWarps + 1)
            role = WarpRole::STORE;
    }

    int64_t bos, eos;
    int tile_base;
    if constexpr (IsVarlen) {
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
        // The prefix table Kernel 1 already built, instead of the CUDA kernel's
        // O(N) linear scan per CTA (fwd_kernel2.cuh:221-234).
        tile_base = tile_prefix[seq_idx];
    } else if constexpr (Segmented) {
        // Divide complete prepare tiles, distributing the remainder instead of
        // dropping tokens when the segment count does not divide the sequence.
        if constexpr (SegmentSequences == 1) {
            int const tiles = (T_total - 1) / CHUNK + 1;
            int const per_segment = tiles / N;
            int const remainder = tiles - per_segment * N;
            int const first = seq_idx * per_segment + min(seq_idx, remainder);
            int const count = per_segment + (seq_idx < remainder);
            tile_base = first;
            bos = int64_t(first) * CHUNK;
            eos = min(int64_t(T_total), int64_t(first + count) * CHUNK);
        } else {
            static_assert(SegmentsPerSequence > 0);
            int const sequence = seq_idx / SegmentsPerSequence;
            int const segment = seq_idx % SegmentsPerSequence;
            int const segments = SegmentsPerSequence;
            int const sequence_tokens = T_total / SegmentSequences;
            int const tiles = (sequence_tokens - 1) / CHUNK + 1;
            int const per_segment = tiles / segments;
            int const remainder = tiles - per_segment * segments;
            int const first = segment * per_segment + min(segment, remainder);
            int const count = per_segment + (segment < remainder);
            tile_base = sequence * tiles + first;
            bos = int64_t(sequence) * sequence_tokens + int64_t(first) * CHUNK;
            eos = int64_t(sequence) * sequence_tokens +
                  min(int64_t(sequence_tokens), int64_t(first + count) * CHUNK);
        }
    } else {
        int const T_seq = T_total / N;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
        tile_base = seq_idx * ((T_seq + CHUNK - 1) / CHUNK);
    }
    bool const prefix_covers_segment = eos - bos <= int64_t(PrefixChunks * CHUNK);
    if constexpr (PrefixMode == 2 || PrefixMode == 5) {
        eos = min(eos, bos + int64_t(PrefixChunks * CHUNK));
    } else if constexpr (PrefixMode == 3) {
        int const skipped = min(PrefixChunks, int((eos - bos + CHUNK - 1) / CHUNK));
        bos += int64_t(skipped) * CHUNK;
        tile_base += skipped;
        if (bos >= eos) {
            if constexpr (HasStateOut)
                publish_segment_result(static_cast<BF16 const *>(initial_state_ptr));
            return;
        }
    }
    if constexpr (PrefixMode == 6) {
        if (prior_prefix_matched) {
            bos += int64_t(PrefixChunks) * CHUNK;
            tile_base += PrefixChunks;
        }
    }
    if constexpr (PrefixMode == 7) {
        if (repair_segment_index == 1 || repair_prior_matched) {
            bos += int64_t(PrefixChunks) * CHUNK;
            tile_base += PrefixChunks;
        }
    }
    int const seq_len = int(eos - bos);
    // An empty sequence is an identity update, including FP32 storage. Do not
    // narrow its initial state through the BF16 MMA staging buffer.
    if (IsVarlen && seq_len == 0) {
        if constexpr (HasStateOut) {
            int64_t const base = (int64_t(seq_idx) * H + head_idx) * D * D + int64_t(dv_cta) * D;
            for (int i = tid; i < kDvPerCta * D; i += kThreads) {
                if constexpr (OutputStateFP32) {
                    if constexpr (StateFP32) {
                        static_cast<float *>(final_state_ptr)[base + i] =
                            HasStateIn ? static_cast<float const *>(initial_state_ptr)[base + i]
                                       : 0.0f;
                    } else {
                        static_cast<float *>(final_state_ptr)[base + i] =
                            HasStateIn
                                ? float(static_cast<BF16 const *>(initial_state_ptr)[base + i])
                                : 0.0f;
                    }
                } else {
                    static_cast<BF16 *>(final_state_ptr)[base + i] =
                        HasStateIn ? static_cast<BF16 const *>(initial_state_ptr)[base + i]
                                   : BF16(0.0f);
                }
            }
        }
        return;
    }
    int const t_tiles = (seq_len + CHUNK - 1) / CHUNK;
    int const stage_groups = (t_tiles + ChunksPerStage - 1) / ChunksPerStage;

    auto mma =
        make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr_mma = mma.get_slice(lane);

    // A 16x16 reference tile, used only to shape fragments.
    Tensor ref = make_tensor(make_smem_ptr(sm.in[0].inv[0]), typename L2::LM::Layout{});
    Tensor ref16 = local_tile(ref, Shape<_16, _16>{}, make_coord(0, 0));
    using AFrag = decltype(thr_mma.partition_fragment_A(ref16));
    using BFrag = decltype(thr_mma.partition_fragment_B(ref16));
    using AccFrag = decltype(thr_mma.make_fragment_C(thr_mma.partition_C(ref16)));

    // The state: [kDvPerWarp, D] bf16 per warp, held as A-shaped fragments.
    // 2 x 8 x 4 registers = 64 per lane at NumMmaWarps == 4.
    AFrag s_frag[kMBlocks][kKBlocks];
    AFrag auxiliary_s_frag[DualSummary ? kMBlocks : 1][kKBlocks];

    int const dv_base = warp_id * kDvPerWarp;

    unsigned *prefix_mismatch = nullptr;
    if constexpr (PrefixMode == 2 || PrefixMode == 4 || PrefixMode == 5) {
        static_assert(HasStateOut && !OutputStateFP32 && DvSplit == 1);
        if constexpr (PrefixMode == 4 || PrefixMode == 5) {
            prefix_mismatch = fused_prefix_mismatch;
        } else {
            __shared__ unsigned mismatch;
            prefix_mismatch = &mismatch;
        }
        // Full prefixes need publication of their actual state even when they
        // differ from F0. A live A matrix never serves as a checkpoint.
        if (tid == 0)
            *prefix_mismatch = prefix_covers_segment || summary_nonzero[seq_idx * H + head_idx] != 0;
    }
    unsigned *homogeneous_live_mask = nullptr;
    if constexpr (IdentityState && ZeroValue) {
        // Local declaration is pruned entirely from every other specialization.
        __shared__ unsigned live_mask;
        homogeneous_live_mask = &live_mask;
        static_assert(NumMmaWarps < 32);
        static_assert(kStageFree + InputStages +
                          cutlass::arch::NamedBarrier::ReservedNamedBarrierCount < 16,
                      "homogeneous rendezvous must use an available user barrier");
        if (tid == 0) live_mask = (1u << NumMmaWarps)-1u;
    }
    auto homogeneous_finished = [&]() {
        if constexpr (IdentityState && ZeroValue) {
            unsigned const mask = lane == 0 ? atomicAdd(homogeneous_live_mask,0u) : 1u;
            return __shfl_sync(0xffffffffu,mask,0) == 0;
        } else {
            return false;
        }
    };

#ifdef FLASH_KDA_EXTERNAL_B_STORAGE
    // Each private B CTA owns this complete sequence/head. Initialize its
    // comparison scratch before any local checkpoint read; later prefix and
    // repair kernels consume it only after this whole launch completes.
    static_assert(FirstSegmentTrueB && PrefixMode == 1 && DvSplit == 1);
    if (tid == 0) {
        summary_nonzero[seq_idx * H + head_idx] = 0;
        // The existing true-first-segment prologue publishes 1 for segment0.
        if (seq_idx % SegmentsPerSequence != 0)
            summary_nonzero[N * H + seq_idx * H + head_idx] = 0;
    }
#endif


    if constexpr (DualSummary || (IdentityState && ZeroValue)) {
        static_assert(DvSplit == 1);
        if (tid == 0)
            summary_nonzero[seq_idx * H + head_idx] = 0;
    }
    // --- prologue: fill the state fragments
    sm.in_pipe.init(tid);
    if constexpr (FirstSegmentTrueB && !ExternalStateFP32) {
        // The pipeline is larger than its prologue state buffer for this exact
        // B geometry. Give state_stage its real34816-byte type, then reuse the
        // existing coalesced stage helpers before the input pipeline starts.
        static_assert(sizeof(Storage) == sizeof(SharedStorageK2<
            D,CHUNK,InputStages,OutputStages,NumMmaWarps,false,DvSplit,
            ChunksPerStage,ProduceOutput,ZeroValue,AffinePrepared,DirectOutput>));
        bool const first = seq_idx % SegmentsPerSequence == 0;
        if (first && tid == 0)
            summary_nonzero[N * H + seq_idx * H + head_idx] = 1;
        if (first) {
            constexpr int kStagePad = D + 8;
            auto *stage = reinterpret_cast<BF16 *>(sm.state_stage);
            int64_t const base = int64_t((seq_idx / SegmentsPerSequence) * H + head_idx) * D * D
                                 + int64_t(dv_cta) * D;
            if constexpr (ExternalStateFP32)
                stage_state_2d_cvt<kDvPerCta,D,kStagePad,kThreads>(
                    stage,static_cast<float const *>(external_initial_state)+base,tid);
            else
                stage_state_2d<BF16,kDvPerCta,D,kStagePad,kThreads>(
                    stage,static_cast<BF16 const *>(external_initial_state)+base,tid);
            __syncthreads();
            if (role == WarpRole::MMA) {
                Tensor ts = make_tensor(make_smem_ptr(stage), typename L2::StateStage::Layout{});
                CUTE_UNROLL
                for (int mb = 0; mb < kMBlocks; ++mb) {
                    CUTE_UNROLL
                    for (int kb = 0; kb < kKBlocks; ++kb) {
                        Tensor blk = local_tile(ts,Shape<_16,_16>{},make_coord(dv_base/16+mb,kb));
                        copy(AutoVectorizingCopy{},thr_mma.partition_A(blk),s_frag[mb][kb]);
                    }
                }
            }
        } else if (role == WarpRole::MMA) {
            CUTE_UNROLL
            for (int mb = 0; mb < kMBlocks; ++mb) {
                CUTE_UNROLL
                for (int kb = 0; kb < kKBlocks; ++kb) clear(s_frag[mb][kb]);
            }
        }
    } else if constexpr (FirstSegmentTrueB) {
        // The first segment starts at the real external state. Its full B result
        // is therefore the true carry; no prefix0/tail0 verification is needed.
        bool const first = seq_idx % SegmentsPerSequence == 0;
        if (first && tid == 0)
            summary_nonzero[N * H + seq_idx * H + head_idx] = 1;
        if (role == WarpRole::MMA) {
            if (first) {
                int64_t const base =
                    int64_t((seq_idx / SegmentsPerSequence) * H + head_idx) * D * D;
                CUTE_UNROLL
                for (int mb = 0; mb < kMBlocks; ++mb) {
                    CUTE_UNROLL
                    for (int kb = 0; kb < kKBlocks; ++kb) {
                        CUTE_UNROLL
                        for (int j = 0; j < decltype(size(s_frag[0][0]))::value / 2; ++j) {
                            FragCoord const c = frag_coord_mn(lane, 2*j);
                            int const dv = dv_cta + dv_base + 16*mb + c.m;
                            int const dk = 16*kb + c.n;
                            int64_t const off = base + int64_t(dv)*D + dk;
                            if constexpr (ExternalStateFP32) {
                                float2 const source = *reinterpret_cast<float2 const *>(
                                    static_cast<float const *>(external_initial_state) + off);
                                BF16 const pair[2] = {BF16(source.x), BF16(source.y)};
                                *reinterpret_cast<uint32_t *>(&s_frag[mb][kb](2*j)) =
                                    *reinterpret_cast<uint32_t const *>(pair);
                            } else {
                                *reinterpret_cast<uint32_t *>(&s_frag[mb][kb](2*j)) =
                                    *reinterpret_cast<uint32_t const *>(
                                        static_cast<BF16 const *>(external_initial_state) + off);
                            }
                        }
                    }
                }
            } else {
                CUTE_UNROLL
                for (int mb = 0; mb < kMBlocks; ++mb) {
                    CUTE_UNROLL
                    for (int kb = 0; kb < kKBlocks; ++kb) clear(s_frag[mb][kb]);
                }
            }
        }
    } else if constexpr (IdentityState) {
        // Segment summaries need F(I) as well as F(0).  Materialising P copies
        // of a 128x128 identity matrix would waste workspace bandwidth, so the
        // MMA warps synthesize their diagonal fragments directly in registers.
        static_assert(!HasStateIn && !StateFP32,
                      "identity initialization does not consume an input state");
        if (role == WarpRole::MMA) {
            CUTE_UNROLL
            for (int mb = 0; mb < kMBlocks; ++mb) {
                CUTE_UNROLL
                for (int kb = 0; kb < kKBlocks; ++kb) {
                    CUTE_UNROLL
                    for (int v = 0; v < decltype(size(s_frag[0][0]))::value; ++v) {
                        FragCoord const c = frag_coord_mn(lane, v);
                        int const dv = dv_cta + dv_base + 16 * mb + c.m;
                        int const dk = 16 * kb + c.n;
                        s_frag[mb][kb](v) = BF16(dv == dk ? 1.0f : 0.0f);
                    }
                }
            }
        }
    } else if constexpr (HasStateIn) {
        // gmem state is [N*H, D, D] row-major over (d_v, d_k). Stage it, then each
        // warp partitions out its own rows. Staging rather than reading gmem per
        // fragment element keeps this coalesced; it happens once per sequence.
        //
        // Always staged as bf16, which is what the fragments hold: an fp32 source is
        // narrowed during the copy instead of after it, rounding at the same point
        // for half the shared memory.
        int64_t const state_off = int64_t(seq_idx * H + head_idx) * D * D;
        constexpr int kStagePad = D + 8;
        auto *stage = reinterpret_cast<BF16 *>(sm.state_stage);
        // Only this CTA's d_v rows: the slice starts dv_cta rows in.
        int64_t const src_off = state_off + int64_t(dv_cta) * D;

        auto stage_carry = [&]() {
            if constexpr (StateFP32) {
                stage_state_2d_cvt<kDvPerCta, D, kStagePad, kThreads>(
                    stage, static_cast<float const *>(initial_state_ptr) + src_off, tid);
            } else {
                stage_state_2d<BF16, kDvPerCta, D, kStagePad, kThreads>(
                    stage, static_cast<BF16 const *>(initial_state_ptr) + src_off, tid);
            }
        };
        if constexpr (PrefixMode == 7) {
            if (repair_segment_index == 1 || repair_prior_matched) {
                stage_carry();
            } else {
                int64_t const prior = (int64_t(seq_idx - 1) * H + head_idx) * D * D +
                                      int64_t(dv_cta) * D;
                stage_state_2d<BF16,kDvPerCta,D,kStagePad,kThreads>(
                    stage, static_cast<BF16 const *>(final_state_ptr) + prior, tid);
                // This phase reads the carry written by the previous phase.
                // Derive both accesses from final_state_ptr: using the distinct
                // restrict-qualified auxiliary pointer here would break its
                // no-alias promise across the whole kernel invocation.
            }
        } else if constexpr (PrefixMode == 5) {
            // A hypothesis, never a claim that this previous B is a true carry.
            int64_t const prior = (int64_t(seq_idx - 1) * H + head_idx) * D * D +
                                  int64_t(dv_cta) * D;
            stage_state_2d<BF16,kDvPerCta,D,kStagePad,kThreads>(
                stage, zero_summary_b + prior, tid);
        } else if constexpr (PrefixMode == 6) {
            if (prior_prefix_matched) {
                stage_carry();
            } else {
                int64_t const prior = (int64_t(seq_idx - 1) * H + head_idx) * D * D +
                                      int64_t(dv_cta) * D;
                stage_state_2d<BF16,kDvPerCta,D,kStagePad,kThreads>(
                    stage, static_cast<BF16 const *>(auxiliary_final_state_ptr) + prior, tid);
            }
        } else if constexpr (SerialSegmentIndex >= 0 && (PrefixMode == 2 || PrefixMode == 4)) {
            static_assert(ExternalHasStateIn && !StateFP32);
            constexpr int kSegments = SegmentsPerSequence;
            int const sequence = seq_idx / kSegments;
            if constexpr (SerialSegmentIndex == 0) {
                int64_t const first = (int64_t(sequence)*H+head_idx)*D*D+int64_t(dv_cta)*D;
                if constexpr (ExternalStateFP32)
                    stage_state_2d_cvt<kDvPerCta,D,kStagePad,kThreads>(
                        stage,static_cast<float const *>(external_initial_state)+first,tid);
                else
                    stage_state_2d<BF16,kDvPerCta,D,kStagePad,kThreads>(
                        stage,static_cast<BF16 const *>(external_initial_state)+first,tid);
            } else {
                int64_t const prior = (int64_t(seq_idx-1)*H+head_idx)*D*D+int64_t(dv_cta)*D;
                if constexpr (kCompactP2Repair) {
                    // This fixed second segment follows B's true first segment.
                    // B and repair are ordered on the caller's stream; both
                    // external state dtypes publish the first B as a true carry.
                    stage_state_2d<BF16,kDvPerCta,D,kStagePad,kThreads>(
                        stage,zero_summary_b+prior,tid);
                } else {
                    bool const prior_matched = summary_nonzero[N*H+(seq_idx-1)*H+head_idx] != 0;
                    BF16 const *source = prior_matched ? zero_summary_b :
                        static_cast<BF16 const *>(initial_state_ptr);
                    stage_state_2d<BF16,kDvPerCta,D,kStagePad,kThreads>(stage,source+prior,tid);
                }
            }
        } else if constexpr (DirectZeroCarry) {
            // Exactly the same all-segment bitwise-zero predicate as the
            // original carry-copy and affine-scan kernels. A live A keeps the
            // complete affine scan and its materialized carry, unchanged.
            constexpr int kSegments = SegmentSequences == 1 ? StaticSequences
                                                            : SegmentsPerSequence;
            int const sequence = SegmentSequences == 1 ? 0 : seq_idx / kSegments;
            int const local_segment = SegmentSequences == 1 ? seq_idx : seq_idx % kSegments;
            bool nonzero = false;
            for (int segment = lane; segment < kSegments; segment += 32)
                nonzero |= summary_nonzero[(sequence * kSegments + segment) * H + head_idx] != 0;
            if (__any_sync(0xffffffffu, nonzero)) {
                stage_carry();
            } else if (local_segment > 0) {
                int64_t const prior = (int64_t(seq_idx - 1) * H + head_idx) * D * D +
                                      int64_t(dv_cta) * D;
                stage_state_2d<BF16, kDvPerCta, D, kStagePad, kThreads>(
                    stage, zero_summary_b + prior, tid);
            } else if constexpr (ExternalHasStateIn) {
                int64_t const first = (int64_t(sequence) * H + head_idx) * D * D +
                                      int64_t(dv_cta) * D;
                if constexpr (ExternalStateFP32) {
                    stage_state_2d_cvt<kDvPerCta, D, kStagePad, kThreads>(
                        stage, static_cast<float const *>(external_initial_state) + first, tid);
                } else {
                    stage_state_2d<BF16, kDvPerCta, D, kStagePad, kThreads>(
                        stage, static_cast<BF16 const *>(external_initial_state) + first, tid);
                }
            } else {
                for (int packet = tid; packet < kDvPerCta * (D / 8); packet += kThreads) {
                    int const row = packet / (D / 8), col = (packet % (D / 8)) * 8;
                    *reinterpret_cast<cute::uint128_t *>(stage + row * kStagePad + col) = {};
                }
            }
        } else {
            stage_carry();
        }
        __syncthreads();
        if (role == WarpRole::MMA) {
            Tensor ts = make_tensor(make_smem_ptr(stage), typename L2::StateStage::Layout{});
            CUTE_UNROLL
            for (int mb = 0; mb < kMBlocks; ++mb) {
                CUTE_UNROLL
                for (int kb = 0; kb < kKBlocks; ++kb) {
                    Tensor blk =
                        local_tile(ts, Shape<_16, _16>{}, make_coord(dv_base / 16 + mb, kb));
                    copy(AutoVectorizingCopy{}, thr_mma.partition_A(blk), s_frag[mb][kb]);
                }
            }
        }
    } else {
        // No shared memory and no memset: the state starts at zero in registers.
        if (role == WarpRole::MMA) {
            CUTE_UNROLL
            for (int mb = 0; mb < kMBlocks; ++mb) {
                CUTE_UNROLL
                for (int kb = 0; kb < kKBlocks; ++kb)
                    clear(s_frag[mb][kb]);
            }
        }
    }

    if constexpr (DualSummary) {
        if (role == WarpRole::MMA) {
            CUTE_UNROLL
            for (int mb = 0; mb < kMBlocks; ++mb) {
                CUTE_UNROLL
                for (int kb = 0; kb < kKBlocks; ++kb) {
                    CUTE_UNROLL
                    for (int v = 0; v < decltype(size(s_frag[0][0]))::value; ++v) {
                        FragCoord const c = frag_coord_mn(lane, v);
                        int const dv = dv_cta + dv_base + 16 * mb + c.m;
                        int const dk = 16 * kb + c.n;
                        auxiliary_s_frag[mb][kb](v) = BF16(dv == dk ? 1.0f : 0.0f);
                    }
                }
            }
        }
    }
    // The staging buffer aliases the pipeline, so nothing may fill a stage until
    // every warp has finished reading it.
    __syncthreads();

    // ------------------------------------------------------------------
    // load warp
    // ------------------------------------------------------------------
    if (role == WarpRole::LOAD) {
        VLoader ld_v;
        Loader ld_transforms;
        // v is (H, T_total, D) : (D, D*H, 1).  AIU loads whole 64-column cubes,
        // so align the CTA's slice down to a cube boundary.  With split4 the two
        // 32-column CTAs in each cube intentionally load the same 64 columns;
        // this is still half the old full-D fallback.  padz zero-fills rows past
        // eos, which makes a short tail tile safe.
        constexpr int kVStageCols = L2::kVCubesPerTile * L2::kCubeW;
        int const v_cube_base = (dv_cta / L2::kCubeW) * L2::kCubeW;
        if constexpr (!ZeroValue) {
            ld_v.init(v_ptr + head_idx * D + v_cube_base, int(eos), kVStageCols,
                      v_stride ? v_stride : H * D);
        }
        // The workspace arrays are [H*total_tiles, CHUNK, D] row-major, read as
        // [H*total_tiles*CHUNK, D].
        int const ws_rows = H * total_tiles * CHUNK;
        ld_transforms.init(ws_kd, 3 * ws_rows, D, D);

        int produced_groups = 0;
        for (int group = 0; group < stage_groups; ++group) {
            if constexpr (PrefixMode == 4) {
                // Stop BEFORE acquiring the first suffix stage. The consumers
                // must be able to complete their checkpoint and release it.
                if (group == PrefixChunks) {
                    // Installed NamedBarrier asm has no memory clobber.
                    asm volatile("" ::: "memory");
                    cutlass::arch::NamedBarrier::arrive_and_wait(kThreads, 10);
                    asm volatile("" ::: "memory");
                    if (*reinterpret_cast<volatile unsigned *>(prefix_mismatch) == 0) break;
                }
            }
            int const stage = group % InputStages;
            sm.in_pipe.producer_acquire(stage, group);
            if constexpr (IdentityState && ZeroValue) {
                // The final consumer release wakes this acquire even if the
                // producer was already waiting when the exact-zero mask formed.
                if (homogeneous_finished()) break;
            }
            auto &s = sm.in[stage];

            int const first_t = group * ChunksPerStage;
            int const ws_idx = head_idx * total_tiles + tile_base + first_t;
            if constexpr (!ZeroValue) {
                ld_v.load(s.v.data(), int(bos) + first_t * CHUNK);
            }
            // One AIU descriptor streams adjacent Kd/Qd/Kr transforms.
            ld_transforms.load(s.transforms.data(), ws_idx * 3 * CHUNK);

            // The small arrays are copied synchronously. producer_commit() arms
            // completion tracking for the async engine only, so a synchronous
            // store issued earlier by this warp is covered by the fence below,
            // whereas a cp.async here would need its own wait whose interaction
            // with the AIU commit counter is unverified.
            //
            // Vectorized to 16 bytes per lane. Element-at-a-time these were 16
            // two-byte global loads plus 4 four-byte ones per stage, and this is
            // the only serial section on the critical path: one load warp fills
            // every stage, so its rate caps the whole CTA no matter how deep the
            // pipeline is. Profiling showed the memory pipe active on 98% of cycles
            // while moving 6% of peak bandwidth and the scheduler finding no
            // eligible warp on 69% of them, which is what a stream of tiny
            // transactions looks like.
            //
            // 256 bf16 elements over 32 lanes is 8 per lane, and 8 lies inside one
            // 16-wide row, so a lane's source and destination runs are both
            // contiguous despite the destination's padded stride. Every workspace
            // array is a whole multiple of 512 B into a torch allocation, so the
            // sources are 16-byte aligned; the destinations are alignas(128) with
            // 8-element offsets. uint128_t rather than uint4 because that is the
            // spelling epilogue.cuh is measured to lower to a single vmem.st.b32x4.
            static_assert(CHUNK == 16 || CHUNK == 32);
            static_assert(D % (32 * 4) == 0, "g_total must split into four-value lane chunks");
            CUTE_UNROLL
            for (int sub = 0; sub < ChunksPerStage; ++sub) {
                int const t = first_t + sub;
                if (t < t_tiles) {
                    using V16 = cute::uint128_t;
                    int const sub_ws_idx = ws_idx + sub;
                    CUTE_UNROLL
                    for (int packet = 0; packet < CHUNK*CHUNK/256; ++packet) {
                    int const flat = packet * 256 + lane * 8;
                    int const dst = (flat / CHUNK) * (CHUNK + 8) + flat % CHUNK;
                    int64_t const src = int64_t(sub_ws_idx) * CHUNK * CHUNK + flat;
#if FLASH_KDA_PPU_ASYNC_SMALL_ARRAYS
                    if constexpr (!AffinePrepared) {
                        CpAsyncVec16::copy(*reinterpret_cast<V16 const *>(&ws_inv[src]),
                                           *reinterpret_cast<V16 *>(&s.inv[sub][dst]));
                    }
                    if constexpr (ProduceOutput) {
                        CpAsyncVec16::copy(*reinterpret_cast<V16 const *>(&ws_mqk[src]),
                                           *reinterpret_cast<V16 *>(&s.mqk[sub][dst]));
                    }
#else
                    if constexpr (!AffinePrepared) {
                        *reinterpret_cast<V16 *>(&s.inv[sub][dst]) =
                            *reinterpret_cast<V16 const *>(&ws_inv[src]);
                    }
                    if constexpr (ProduceOutput) {
                        *reinterpret_cast<V16 *>(&s.mqk[sub][dst]) =
                            *reinterpret_cast<V16 const *>(&ws_mqk[src]);
                    }
#endif
                    }

                    CUTE_UNROLL
                    for (int i = 0; i < D / (32 * 4); ++i) {
                        int const o = (i * 32 + lane) * 4;
#if FLASH_KDA_PPU_ASYNC_SMALL_ARRAYS
                        CpAsyncVec16::copy(
                            *reinterpret_cast<V16 const *>(&ws_gt[int64_t(sub_ws_idx) * D + o]),
                            *reinterpret_cast<V16 *>(&s.g_total[sub][o]));
#else
                        *reinterpret_cast<V16 *>(&s.g_total[sub][o]) =
                            *reinterpret_cast<V16 const *>(&ws_gt[int64_t(sub_ws_idx) * D + o]);
#endif
                    }
                    // K1 already evaluated sigmoid(beta). Copy its four aligned
                    // packets with the other small stage arrays instead of
                    // repeating the SFU and strided gather in every split CTA.
                    if (!AffinePrepared && lane < CHUNK / 4) {
                        constexpr int kBetaStride =
                            WorkspaceSizes<CHUNK, D>::kBeta / int(sizeof(float));
                        int64_t const beta_src = int64_t(sub_ws_idx) * kBetaStride + lane * 4;
                        int const beta_dst = sub * CHUNK + lane * 4;
#if FLASH_KDA_PPU_ASYNC_SMALL_ARRAYS
                        CpAsyncVec16::copy(*reinterpret_cast<V16 const *>(&ws_beta[beta_src]),
                                           *reinterpret_cast<V16 *>(&s.beta[beta_dst]));
#else
                        *reinterpret_cast<V16 *>(&s.beta[beta_dst]) =
                            *reinterpret_cast<V16 const *>(&ws_beta[beta_src]);
#endif
                    }
                }
            }
#if FLASH_KDA_PPU_ASYNC_SMALL_ARRAYS
            // Commit all lane-issued small-array copies.  producer_commit below
            // attaches the stage mbarrier after both these copies and the
            // preceding warp-level AIU tile loads.
            cp_async_fence_all();
#endif

            kda_fence_block();
            sm.in_pipe.producer_commit(stage);
            produced_groups = group+1;
        }
        if constexpr (PrefixMode == 4 || PrefixMode == 7) {
            // Complete the pending final consumer arrivals on both stage-free
            // barriers before the whole CTA joins and may return together.
            CUTE_UNROLL
            for (int stage = 0; stage < InputStages; ++stage) {
                if (produced_groups > stage)
                    cutlass::arch::NamedBarrier::arrive_and_wait(kThreads, kStageFree + stage);
            }
        }
        if constexpr (IdentityState && ZeroValue) {
            // Drain the latest committed phase of every occupied stage. Its
            // mbarrier includes actual AIU/cp.async completion, including any
            // prefetched stage that no consumer will use after termination.
            CUTE_UNROLL
            for (int stage = 0; stage < InputStages; ++stage) {
                if (produced_groups > stage) {
                    int const last = produced_groups-1 -
                        (produced_groups-1-stage)%InputStages;
                    sm.in_pipe.consumer_wait(stage,last);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // MMA warps
    // ------------------------------------------------------------------
    if (role == WarpRole::MMA) {
#ifdef FLASH_KDA_HYBRID_B_LOGICAL_RANGES
        static_assert(NumMmaWarps == 8 && CHUNK == 16 && DvSplit == 1);
        // Only MMA warps enter this scope. Keep the original outer coordinate
        // for prologue and producer roles, and expose this scope's legal range.
        int const dv_base = (warp_id & (NumMmaWarps - 1)) * kDvPerWarp;
#endif
        bool homogeneous_live = true;
        for (int group = 0; group < stage_groups; ++group) {
            int const in_stage = group % InputStages;
            int const out_stage = group % OutputStages;

            if constexpr (ProduceOutput && !DirectOutput) {
                sm.out_pipe.producer_acquire(out_stage, group);
            }
            sm.in_pipe.consumer_wait(in_stage, group);
            auto &s = sm.in[in_stage];
            int const trbase = __ppu_read_firstlane(unsigned(uintptr_t(s.transforms.data()) / 16));
            int const vbase = __ppu_read_firstlane(unsigned(uintptr_t(s.v.data()) / 16));

            bool terminate_summary = false;
            // Dead row warps still participate in the periodic all-row decision.
            if (!IdentityState || !ZeroValue || homogeneous_live) {
            CUTE_UNROLL
            for (int sub = 0; sub < ChunksPerStage; ++sub) {
                int const t = group * ChunksPerStage + sub;
                if (t >= t_tiles)
                    continue;
                int const beta_off = sub * CHUNK;

                if constexpr (CHUNK == 32) {
                    static_assert(AffinePrepared && !DualSummary,
                                  "32-token recurrence consumes affine W/M");
                    AccFrag residual[2][kMBlocks];
                    AccFrag output[2][kMBlocks];
                    AFrag innovation[2][kMBlocks];
                    CUTE_UNROLL
                    for (int cb = 0; cb < 2; ++cb) {
                        CUTE_UNROLL
                        for (int b = 0; b < kMBlocks; ++b) {
                            clear(residual[cb][b]);
                            if constexpr (ProduceOutput) clear(output[cb][b]);
                        }
                    }
                    // Both token blocks read the same incoming state before a
                    // single 32-token state update. Each uses a verified 16x16
                    // atom and the existing AIU swizzle-aware loads.
                    CUTE_UNROLL
                    for (int kb = 0; kb < kKBlocks; ++kb) {
                        CUTE_UNROLL
                        for (int cb = 0; cb < 2; ++cb) {
                            BFrag kd;
                            Tsm::load_b(&kd(0),trbase,kb,3*sub*CHUNK+16*cb);
                            AFrag qd;
                            if constexpr (ProduceOutput)
                                Tsm::load_a(&qd(0),trbase,kb,(3*sub+1)*CHUNK+16*cb);
                            CUTE_UNROLL
                            for (int b = 0; b < kMBlocks; ++b) {
                                cute::gemm(thr_mma,s_frag[b][kb],kd,residual[cb][b]);
                                if constexpr (ProduceOutput)
                                    cute::gemm(thr_mma,qd,as_B(s_frag[b][kb]),output[cb][b]);
                            }
                        }
                    }
                    CUTE_UNROLL
                    for (int cb = 0; cb < 2; ++cb) {
                        CUTE_UNROLL
                        for (int b = 0; b < kMBlocks; ++b) {
                            BFrag vt;
                            if constexpr (ZeroValue) clear(vt);
                            else {
                                int const vb = ((dv_cta%L2::kCubeW)+dv_base)/16+b;
                                VTsm::load_bt(&vt(0),vbase,vb,sub*CHUNK+16*cb);
                            }
                            auto v = as_A(vt);
                            CUTE_UNROLL
                            for (int i = 0; i < decltype(size(v))::value; ++i)
                                residual[cb][b](i) = float(v(i))-residual[cb][b](i);
                            narrow_into(innovation[cb][b],residual[cb][b]);
                        }
                    }
                    if constexpr (ProduceOutput) {
                        auto m = make_tensor(make_smem_ptr(s.mqk[sub]),typename L2::LM::Layout{});
                        CUTE_UNROLL
                        for (int cb = 0; cb < 2; ++cb) {
                            CUTE_UNROLL
                            for (int source = 0; source <= cb; ++source) {
                                auto tile = local_tile(m,Shape<_16,_16>{},make_coord(cb,source));
                                auto ma = thr_mma.partition_fragment_A(tile);
                                copy(AutoVectorizingCopy{},thr_mma.partition_A(tile),ma);
                                CUTE_UNROLL
                                for (int b = 0; b < kMBlocks; ++b)
                                    cute::gemm(thr_mma,ma,as_B(innovation[source][b]),output[cb][b]);
                            }
                            CUTE_UNROLL
                            for (int b = 0; b < kMBlocks; ++b) {
                                CUTE_UNROLL
                                for (int v = 0; v < decltype(size(output[0][0]))::value/2; ++v) {
                                    FragCoord c = frag_coord_mn(lane,2*v);
                                    BF16 pair[2] = {BF16(output[cb][b](2*v)),BF16(output[cb][b](2*v+1))};
                                    if constexpr (DirectOutput) {
                                        int const row = cb*16+c.m;
                                        if (row < seq_len-t*CHUNK) {
                                            BF16 *dst = out_ptr+(bos+int64_t(t)*CHUNK+row)*H*D+
                                                        head_idx*D+dv_cta+dv_base+16*b+c.n;
                                            *reinterpret_cast<uint32_t *>(dst) =
                                                *reinterpret_cast<uint32_t const *>(pair);
                                        }
                                    } else {
                                        BF16 *dst = sm.out[out_stage].out+
                                            (sub*CHUNK+cb*16+c.m)*(kDvPerCta+8)+dv_base+16*b+c.n;
                                        *reinterpret_cast<uint32_t *>(dst) =
                                            *reinterpret_cast<uint32_t const *>(pair);
                                    }
                                }
                            }
                        }
                    }
                    CUTE_UNROLL
                    for (int kb = 0; kb < kKBlocks; ++kb) {
                        CUTE_UNROLL
                        for (int b = 0; b < kMBlocks; ++b) {
                            AccFrag delta; clear(delta);
                            CUTE_UNROLL
                            for (int cb = 0; cb < 2; ++cb) {
                                BFrag w;
                                Tsm::load_bt(&w(0),trbase,kb,(3*sub+2)*CHUNK+16*cb);
                                cute::gemm(thr_mma,innovation[cb][b],w,delta);
                            }
                            CUTE_UNROLL
                            for (int v = 0; v < decltype(size(delta))::value/2; ++v) {
                                FragCoord c = frag_coord_mn(lane,2*v);
                                __ppu_bfloat162 update = __floats2bfloat162_rn(
                                    float(s_frag[b][kb](2*v))*s.g_total[sub][16*kb+c.n]+delta(2*v),
                                    float(s_frag[b][kb](2*v+1))*s.g_total[sub][16*kb+c.n+1]+delta(2*v+1));
                                BF16 pair[2];
                                *reinterpret_cast<uint32_t *>(pair) =
                                    *reinterpret_cast<uint32_t const *>(&update);
                                s_frag[b][kb](2*v)=pair[0]; s_frag[b][kb](2*v+1)=pair[1];
                            }
                        }
                    }
                } else {
                AccFrag u_acc[kMBlocks]; // [d_v, t]
                AccFrag auxiliary_u_acc[DualSummary ? kMBlocks : 1];
                AccFrag out_acc[kMBlocks]; // [t, d_v]
                CUTE_UNROLL
                for (int b = 0; b < kMBlocks; ++b) {
                    clear(u_acc[b]);
                    if constexpr (DualSummary)
                        clear(auxiliary_u_acc[b]);
                    if constexpr (ProduceOutput)
                        clear(out_acc[b]);
                }

                // ---- P1: u^T[d_v,t] = sum_dk S[d_v,d_k] Kd[t,d_k]
                //          out[t,d_v] = sum_dk Qd[t,d_k] S[d_v,d_k]
                // The state serves as A in the first and B in the second, from the
                // same registers. Four independent accumulator chains, which is what
                // saturates the MMA pipe on this part.
                CUTE_UNROLL
                for (int kb = 0; kb < kKBlocks; ++kb) {
                    BFrag kd_b; // B[n=t, k=d_k] from a tile stored [t][d_k]
                    Tsm::load_b(&kd_b(0), trbase, kb, 3 * sub * CHUNK);
                    AFrag qd_a; // A[m=t, k=d_k] from the same storage orientation
                    if constexpr (ProduceOutput || 0) {
                        Tsm::load_a(&qd_a(0), trbase, kb, (3 * sub + 1) * CHUNK);
                    }

                    CUTE_UNROLL
                    for (int b = 0; b < kMBlocks; ++b) {
                        cute::gemm(thr_mma, s_frag[b][kb], kd_b, u_acc[b]);
                        if constexpr (DualSummary) {
                            cute::gemm(thr_mma, auxiliary_s_frag[b][kb], kd_b, auxiliary_u_acc[b]);
                        }
                        if constexpr (ProduceOutput) {
                            cute::gemm(thr_mma, qd_a, as_B(s_frag[b][kb]), out_acc[b]);
                        }
                    }
                }

                // ---- P2/P3: residual V^T - S Kd^T. Raw transforms apply beta
                // here; affine W/M already incorporate it and INV.
                // V is stored [t][d_v]. The transposing load gives B[n=d_v, k=t] = V^T;
                // as_A relabels it into the A/C orientation, which is where u^T lives.
                // The (Swap=false, Trans=true) load that would produce this directly is
                // unverified, so it is reached through the verified pair instead.
                CUTE_UNROLL
                for (int b = 0; b < kMBlocks; ++b) {
                    BFrag vt;
                    if constexpr (!ZeroValue) {
                        // Address within the aligned V cube staged by this CTA.  For
                        // a 32-column split4 slice, dv_cta % 64 selects the lower or
                        // upper half of its shared cube.
                        int const v_block = ((dv_cta % L2::kCubeW) + dv_base) / 16 + b;
                        VTsm::load_bt(&vt(0), vbase, v_block, sub * CHUNK);
                    } else {
                        clear(vt);
                    }
                    auto vt_c = as_A(vt);
                    CUTE_UNROLL
                    for (int v = 0; v < decltype(size(u_acc[0]))::value; ++v) {
                        // n of a [d_v, t] fragment is the token index.
                        FragCoord const c = frag_coord_mn(lane, v);
                        float const beta = AffinePrepared ? 1.f : s.beta[beta_off + c.n];
                        u_acc[b](v) = (float(vt_c(v)) - u_acc[b](v)) * beta;
                    }
                    if constexpr (DualSummary) {
                        CUTE_UNROLL
                        for (int v = 0; v < decltype(size(auxiliary_u_acc[0]))::value; ++v) {
                            FragCoord const c = frag_coord_mn(lane, v);
                            float const beta = AffinePrepared ? 1.f : s.beta[beta_off + c.n];
                            auxiliary_u_acc[b](v) = -auxiliary_u_acc[b](v) * beta;
                        }
                    }
                }

                // ---- P4: raw transforms apply INV; affine transforms use the
                // narrowed residual directly.
                // C -> A is free, so u^T becomes the A operand with no data movement.
                AFrag u_a[kMBlocks];
                AFrag auxiliary_u_a[DualSummary ? kMBlocks : 1];
                if constexpr (AffinePrepared) {
                    CUTE_UNROLL
                    for (int b = 0; b < kMBlocks; ++b) {
                        if constexpr (PrefixMode == 1 && FirstSegmentTrueB &&
                                      ((StaticHeads == 6 && StaticTokens == 16384 &&
                                        StaticSequences == 24 && SegmentSequences == 8 &&
                                        SegmentsPerSequence == 3) || kH12N4P3Scope) && D == 128 && CHUNK == 16 &&
                                      NumMmaWarps == 8 && DvSplit == 1 && ProduceOutput && DirectOutput) {
                            auto words = cute::recast<uint32_t>(u_a[b]);
                            CUTE_UNROLL
                            for (int j=0;j<decltype(size(u_acc[0]))::value/2;++j) {
                                __ppu_bfloat162 const packed = __floats2bfloat162_rn(
                                    u_acc[b](2*j),u_acc[b](2*j+1));
                                words(j)=*reinterpret_cast<uint32_t const *>(&packed);
                            }
                        } else {
                            narrow_into(u_a[b], u_acc[b]);
                        }
                        if constexpr (DualSummary)
                            narrow_into(auxiliary_u_a[b], auxiliary_u_acc[b]);
                    }
                } else {
                    Tensor tinv = make_tensor(make_smem_ptr(s.inv[sub]), typename L2::LM::Layout{});
                    Tensor inv16 = local_tile(tinv, Shape<_16, _16>{}, make_coord(0, 0));
                    auto inv_b = thr_mma.partition_fragment_B(inv16);
                    copy(AutoVectorizingCopy{}, thr_mma.partition_B(inv16), inv_b);

                    CUTE_UNROLL
                    for (int b = 0; b < kMBlocks; ++b) {
                        auto a = c_to_a_bf16(u_acc[b]);
                        clear(u_acc[b]);
                        cute::gemm(thr_mma, a, inv_b, u_acc[b]);
                        narrow_into(u_a[b], u_acc[b]);
                        if constexpr (DualSummary) {
                            auto auxiliary_a = c_to_a_bf16(auxiliary_u_acc[b]);
                            clear(auxiliary_u_acc[b]);
                            cute::gemm(thr_mma, auxiliary_a, inv_b, auxiliary_u_acc[b]);
                            narrow_into(auxiliary_u_a[b], auxiliary_u_acc[b]);
                        }
                    }
                }

                // ---- P5: out[t,d_v] += sum_s Mqk[t,s] U^T[d_v,s]
                // U^T serves as B here via the free relabel, and as A in P7.
                if constexpr (ProduceOutput) {
                    Tensor tmqk = make_tensor(make_smem_ptr(s.mqk[sub]), typename L2::LM::Layout{});
                    Tensor mqk16 = local_tile(tmqk, Shape<_16, _16>{}, make_coord(0, 0));
                    auto mqk_a = thr_mma.partition_fragment_A(mqk16);
                    copy(AutoVectorizingCopy{}, thr_mma.partition_A(mqk16), mqk_a);

                    CUTE_UNROLL
                    for (int b = 0; b < kMBlocks; ++b) {
                        cute::gemm(thr_mma, mqk_a, as_B(u_a[b]), out_acc[b]);
                    }
                }

                // ---- P6: out -> sOut. Natural orientation means value mode 0 maps to
                // adjacent d_v, so this is a packed 4-byte store; transposed it would
                // have been eight scattered 2-byte stores per lane.
                if constexpr (ProduceOutput) {
                    if constexpr (DirectOutput) {
                        int const rows_valid = min(CHUNK, seq_len - t * CHUNK);
                        BF16 *dst =
                            out_ptr + (bos + int64_t(t) * CHUNK) * H * D + head_idx * D + dv_cta;
                        CUTE_UNROLL
                        for (int b = 0; b < kMBlocks; ++b) {
                            CUTE_UNROLL
                            for (int j = 0; j < decltype(size(out_acc[0]))::value / 2; ++j) {
                                FragCoord const c = frag_coord_mn(lane, 2 * j);
                                // Private reuse modes partition T16384 into
                                // complete16-token tiles. The device path flag
                                // accepts the full sequence before reaching here.
                                if (PrefixMode != 0 || c.m < rows_valid) {
                                    BF16 p[2] = {BF16(out_acc[b](2 * j)),
                                                 BF16(out_acc[b](2 * j + 1))};
                                    *reinterpret_cast<uint32_t *>(
                                        &dst[int64_t(c.m) * H * D + dv_base + 16 * b + c.n]) =
                                        *reinterpret_cast<uint32_t const *>(p);
                                }
                            }
                        }
                    } else {
                        BF16 *dst = sm.out[out_stage].out;
                        CUTE_UNROLL
                        for (int b = 0; b < kMBlocks; ++b) {
                            CUTE_UNROLL
                            for (int j = 0; j < decltype(size(out_acc[0]))::value / 2; ++j) {
                                FragCoord const c = frag_coord_mn(lane, 2 * j);
                                BF16 p[2] = {BF16(out_acc[b](2 * j)), BF16(out_acc[b](2 * j + 1))};
                                *reinterpret_cast<uint32_t *>(
                                    &dst[(sub * CHUNK + c.m) * (kDvPerCta + 8) + dv_base + 16 * b +
                                         c.n]) = *reinterpret_cast<uint32_t const *>(p);
                            }
                        }
                    }
                }
                // ---- P7/P8: S = S * g_total + U^T Kr (raw) or residual W (affine).
                // Kr is stored [t][d_k], so the transposing load gives B[n=d_k, k=t].
                // The product's C layout equals the A layout, so the accumulator for
                // (d_v block, d_k block) *is* the state fragment for that block --
                // g_total is indexed by the fragment's n, which is d_k. g_total already
                // holds exp2 of the gate sum (Kernel 1 exponentiates before storing,
                // matching the CUDA backend's workspace contract).
                // Match CUDA: keep decay and the MMA delta in FP32 through
                // the FMA, then round only the updated resident state to BF16.
                // Homogeneous summaries must use the same update sequence.
                AccFrag ds;
                AccFrag auxiliary_ds;
                CUTE_UNROLL
                for (int kb = 0; kb < kKBlocks; ++kb) {
                    BFrag kr_bt;
                    Tsm::load_bt(&kr_bt(0), trbase, kb, (3 * sub + 2) * CHUNK);

                    CUTE_UNROLL
                    for (int b = 0; b < kMBlocks; ++b) {
                        clear(ds);
                        cute::gemm(thr_mma, u_a[b], kr_bt, ds);
                        if constexpr (DualSummary) {
                            clear(auxiliary_ds);
                            cute::gemm(thr_mma, auxiliary_u_a[b], kr_bt, auxiliary_ds);
                        }

                        {
                            // Keep both update operands in FP32. Pack only the
                            // final BF16 stores, never the decay or MMA delta.
                            CUTE_UNROLL
                            for (int j = 0; j < decltype(size(ds))::value / 2; ++j) {
                                int const v = 2 * j;
                                FragCoord const c = frag_coord_mn(lane, v);
                                float const decay0 = s.g_total[sub][16 * kb + c.n];
                                float const decay1 = s.g_total[sub][16 * kb + c.n + 1];
                                __ppu_bfloat162 const updated2 = __floats2bfloat162_rn(
                                    float(s_frag[b][kb](v)) * decay0 + ds(v),
                                    float(s_frag[b][kb](v + 1)) * decay1 + ds(v + 1));
                                BF16 updated_pair[2];
                                *reinterpret_cast<uint32_t *>(updated_pair) =
                                    *reinterpret_cast<uint32_t const *>(&updated2);
                                s_frag[b][kb](v) = updated_pair[0];
                                s_frag[b][kb](v + 1) = updated_pair[1];
                                if constexpr (DualSummary) {
                                    __ppu_bfloat162 const auxiliary_updated2 = __floats2bfloat162_rn(
                                        float(auxiliary_s_frag[b][kb](v)) * decay0 +
                                            auxiliary_ds(v),
                                        float(auxiliary_s_frag[b][kb](v + 1)) * decay1 +
                                            auxiliary_ds(v + 1));
                                    BF16 auxiliary_updated_pair[2];
                                    *reinterpret_cast<uint32_t *>(auxiliary_updated_pair) =
                                        *reinterpret_cast<uint32_t const *>(&auxiliary_updated2);
                                    auxiliary_s_frag[b][kb](v) = auxiliary_updated_pair[0];
                                    auxiliary_s_frag[b][kb](v + 1) = auxiliary_updated_pair[1];
                                }
                            }
                        }
                    }
                }
                }
                if constexpr (PrefixMode == 1) {
                    // A zero homogeneous summary has no live matrix reader.
                    // Reuse that dead matrix for the state at the replay prefix.
                    if (t + 1 == min(PrefixChunks, t_tiles) &&
                        (!FirstSegmentTrueB || seq_idx % (SegmentsPerSequence > 0 ? SegmentsPerSequence : 1) != 0) &&
                        summary_nonzero[seq_idx * H + head_idx] == 0) {
                        auto *checkpoint = static_cast<BF16 *>(auxiliary_final_state_ptr);
                        int64_t const offset = int64_t(seq_idx * H + head_idx) * D * D;
                        CUTE_UNROLL
                        for (int mb = 0; mb < kMBlocks; ++mb) {
                            CUTE_UNROLL
                            for (int kb = 0; kb < kKBlocks; ++kb) {
                                CUTE_UNROLL
                                for (int j = 0; j < decltype(size(s_frag[0][0]))::value / 2; ++j) {
                                    static_assert(kMBlocks == 1 && kKBlocks == 8);
                                    // Private dead-A checkpoint: adjacent lanes
                                    // publish adjacent packed words. The prefix
                                    // compares the same fragment ownership.
                                    int const word = ((warp_id * kKBlocks + kb) * 4 + j) * 32 + lane;
                                    reinterpret_cast<uint32_t *>(checkpoint + offset)[word] =
                                        *reinterpret_cast<uint32_t const *>(&s_frag[mb][kb](2*j));
                                }
                            }
                        }
                    }
                }
            }
            } // arithmetic for live homogeneous rows (or ordinary recurrence)
            if constexpr (IdentityState && ZeroValue) {
                constexpr int kCheckGroups = CHUNK == 16 ? 4 : 8;
                if ((group & (kCheckGroups - 1)) == kCheckGroups - 1) {
                    if (homogeneous_live) {
                        uint32_t nonzero = 0;
                        CUTE_UNROLL
                        for (int b = 0; b < kMBlocks; ++b) {
                            CUTE_UNROLL
                            for (int kb = 0; kb < kKBlocks; ++kb) {
                                CUTE_UNROLL
                                for (int j = 0; j < decltype(size(s_frag[0][0]))::value/2; ++j)
                                    nonzero |= *reinterpret_cast<uint32_t const *>(
                                        &s_frag[b][kb](2*j)) & 0x7fff7fffu;
                            }
                        }
                        homogeneous_live = __any_sync(0xffffffffu,nonzero != 0);
                        if (!homogeneous_live && lane == 0)
                            atomicAnd(homogeneous_live_mask,~(1u << warp_id));
                    }
                    kda_fence_block();
                    asm volatile("" ::: "memory");
                    cutlass::arch::NamedBarrier::arrive_and_wait(
                        kMmaThreads,kStageFree+InputStages);
                    asm volatile("" ::: "memory");
                    terminate_summary = homogeneous_finished();
                }
            }
            if constexpr (ProduceOutput && !DirectOutput) {
                kda_fence_block();
                sm.out_pipe.producer_commit(out_stage);
            }
            sm.in_pipe.consumer_release(in_stage);
            if constexpr (PrefixMode == 4) {
                if (group + 1 == PrefixChunks) {
                    if (summary_nonzero[seq_idx * H + head_idx] == 0) {
                        auto const *checkpoint = static_cast<BF16 const *>(auxiliary_final_state_ptr);
                        int64_t const offset = int64_t(seq_idx * H + head_idx) * D * D;
                        bool different = false;
                        CUTE_UNROLL
                        for (int mb = 0; mb < kMBlocks; ++mb) {
                            CUTE_UNROLL
                            for (int kb = 0; kb < kKBlocks; ++kb) {
                                CUTE_UNROLL
                                for (int j = 0; j < decltype(size(s_frag[0][0]))::value / 2; ++j) {
                                    static_assert(kMBlocks == 1 && kKBlocks == 8);
                                    int const word = ((warp_id * kKBlocks + kb) * 4 + j) * 32 + lane;
                                    uint32_t const reference =
                                        reinterpret_cast<uint32_t const *>(checkpoint + offset)[word];
                                    uint32_t const actual =
                                        *reinterpret_cast<uint32_t const *>(&s_frag[mb][kb](2*j));
                                    different |= actual != reference;
                                }
                            }
                        }
                        if (__any_sync(0xffffffffu, different) && lane == 0)
                            atomicOr(prefix_mismatch, 1u);
                    }
                    kda_fence_block();
                    __syncwarp();
                    asm volatile("" ::: "memory");
                    cutlass::arch::NamedBarrier::arrive_and_wait(kThreads, 10);
                    asm volatile("" ::: "memory");
                    if (*reinterpret_cast<volatile unsigned *>(prefix_mismatch) == 0) break;
                }
            }

            if constexpr (IdentityState && ZeroValue) {
                if (terminate_summary) break;
            }
        }
    }

    // ------------------------------------------------------------------
    // store warp
    // ------------------------------------------------------------------
    if constexpr (ProduceOutput && !DirectOutput) {
        if (role == WarpRole::STORE) {
            for (int group = 0; group < stage_groups; ++group) {
                int const out_stage = group % OutputStages;
                sm.out_pipe.consumer_wait(out_stage, group);

                int const rows_valid = min(kStageRows, seq_len - group * kStageRows);
                BF16 *g =
                    out_ptr + (bos + int64_t(group) * kStageRows) * H * D + head_idx * D + dv_cta;
                store_tile_2d<kStageRows, kDvPerCta, kDvPerCta + 8, 32>(
                    g, H * D, sm.out[out_stage].out, rows_valid, lane);

                sm.out_pipe.consumer_release(out_stage);
            }
        }
    }

    // ------------------------------------------------------------------
    // final state, straight from the MMA warps' registers
    // ------------------------------------------------------------------
    if constexpr (PrefixMode == 4) {
        // Every producer copy and consumer read is finished. A matching complete
        // resident BF16 state authorizes the already-computed deterministic
        // suffix; otherwise the same registers have computed ALL remaining tiles.
        __syncthreads();
        bool const equal = *prefix_mismatch == 0;
        if (tid == 0) summary_nonzero[N * H + seq_idx * H + head_idx] = equal;
        if (equal) {
            publish_reused_segment_final<D, ExternalStateFP32, SegmentSequences, SegmentsPerSequence>(
                zero_summary_b, external_final_state, H, N, logical_sequence);
            return;
        }
    }
    if constexpr (HasStateOut) {
        // The summary/carry pass already supplies every replay input. Only
        // the last segment's final state belongs to the public result.
        if constexpr (Segmented && ProduceOutput && PrefixMode != 1 && PrefixMode != 2 && PrefixMode != 5 && !SegmentCarryOut) {
            if constexpr (SegmentSequences == 1) {
                if (seq_idx + 1 != N)
                    return;
            } else {
                if ((seq_idx + 1) % SegmentsPerSequence != 0)
                    return;
            }
        }
        if constexpr (PrefixMode == 2 || PrefixMode == 5) {
            if (role == WarpRole::MMA && !prefix_covers_segment &&
                summary_nonzero[seq_idx * H + head_idx] == 0) {
                auto const *checkpoint = static_cast<BF16 const *>(auxiliary_final_state_ptr);
                int64_t const offset = int64_t(seq_idx * H + head_idx) * D * D;
                bool different = false;
                CUTE_UNROLL
                for (int mb = 0; mb < kMBlocks; ++mb) {
                    CUTE_UNROLL
                    for (int kb = 0; kb < kKBlocks; ++kb) {
                        CUTE_UNROLL
                        for (int j = 0; j < decltype(size(s_frag[0][0]))::value / 2; ++j) {
                            static_assert(kMBlocks == 1 && kKBlocks == 8);
                            int const word = ((warp_id * kKBlocks + kb) * 4 + j) * 32 + lane;
                            uint32_t const reference =
                                reinterpret_cast<uint32_t const *>(checkpoint + offset)[word];
                            uint32_t const actual = *reinterpret_cast<uint32_t const *>(&s_frag[mb][kb](2*j));
                            different |= actual != reference;
                        }
                    }
                }
                if (__any_sync(0xffffffffu, different) && lane == 0)
                    atomicOr(prefix_mismatch, 1u);
            }
        }
        __syncthreads();
        if constexpr (PrefixMode == 2 || PrefixMode == 5) {
            bool const equal = *prefix_mismatch == 0;
            if (tid == 0) summary_nonzero[N * H + seq_idx * H + head_idx] = equal;
            if (equal) {
                // The host permits this only when public final storage is
                // disjoint from initial state, V, out and the active workspace.
                // Otherwise the original tail publishes after ALL prefix CTAs.
                if constexpr (PrepublishFinal && PublicHasStateOut) {
                    if constexpr (SerialSegmentIndex >= 0)
                        publish_reused_segment_final<D, ExternalStateFP32,
                                                      SegmentSequences, SegmentsPerSequence>(
                            zero_summary_b, external_final_state, H, N, logical_sequence);
                    else
                        publish_reused_segment_final<D, ExternalStateFP32,
                                                      SegmentSequences, SegmentsPerSequence>(
                            zero_summary_b, external_final_state, H, N);
                }
                return;
            }
        }
        if constexpr (IdentityState && ZeroValue) {
            // Metadata was initialized to zero before recurrence. Every A
            // reader guards its load with that metadata, so an exact-zero
            // summary need not materialize an unused internal matrix.
            if (homogeneous_finished()) return;
        }
        if (role == WarpRole::MMA) {
            if constexpr (DualSummary || (IdentityState && ZeroValue)) {
                bool nonzero = false;
                CUTE_UNROLL
                for (int mb = 0; mb < kMBlocks; ++mb) {
                    CUTE_UNROLL
                    for (int kb = 0; kb < kKBlocks; ++kb) {
                        CUTE_UNROLL
                        for (int v = 0; v < decltype(size(s_frag[0][0]))::value; ++v)
                            if constexpr (DualSummary) {
                                nonzero |= (*reinterpret_cast<uint16_t const *>(
                                                &auxiliary_s_frag[mb][kb](v)) &
                                            0x7fffu) != 0;
                            } else {
                                nonzero |=
                                    (*reinterpret_cast<uint16_t const *>(&s_frag[mb][kb](v)) &
                                     0x7fffu) != 0;
                            }
                    }
                }
                // Inspect BF16 bits, including subnormals; no decay threshold.
                if (__any_sync(0xffffffffu, nonzero) && lane == 0)
                    atomicExch(summary_nonzero + seq_idx * H + head_idx, 1);
            }
            int state_seq = seq_idx;
            if constexpr (Segmented && ProduceOutput && PrefixMode != 1 && PrefixMode != 2 && PrefixMode != 5 && !SegmentCarryOut) {
                if constexpr (SegmentSequences == 1)
                    state_seq = 0;
                else
                    state_seq = seq_idx / SegmentsPerSequence;
            }
            int64_t const state_off = int64_t(state_seq * H + head_idx) * D * D;
            // Value slots 2j and 2j+1 differ only in the fragment's v0, which moves
            // n by one and leaves m alone, so they are adjacent d_k of the same d_v
            // and go out as one packed store. That halves what was 128 scattered
            // two-byte global stores per lane.
            CUTE_UNROLL
            for (int mb = 0; mb < kMBlocks; ++mb) {
                CUTE_UNROLL
                for (int kb = 0; kb < kKBlocks; ++kb) {
                    CUTE_UNROLL
                    for (int j = 0; j < decltype(size(s_frag[0][0]))::value / 2; ++j) {
                        FragCoord const c0 = frag_coord_mn(lane, 2 * j);
                        int const dv = dv_cta + dv_base + 16 * mb + c0.m;
                        int const dk = 16 * kb + c0.n;
                        int64_t const o = state_off + int64_t(dv) * D + dk;
                        if constexpr (!kCompactP2Repair) {
                            if constexpr (OutputStateFP32) {
                                float2 p;
                                p.x = float(s_frag[mb][kb](2 * j));
                                p.y = float(s_frag[mb][kb](2 * j + 1));
                                *reinterpret_cast<float2 *>(&static_cast<float *>(final_state_ptr)[o]) =
                                    p;
                            } else {
                                BF16 p[2] = {s_frag[mb][kb](2 * j), s_frag[mb][kb](2 * j + 1)};
                                *reinterpret_cast<uint32_t *>(&static_cast<BF16 *>(
                                    final_state_ptr)[o]) = *reinterpret_cast<uint32_t const *>(p);
                            }
                        }
                        if constexpr (PublicHasStateOut &&
                                      (SegmentCarryOut || (PrefixMode == 1 && FirstSegmentTrueB))) {
                            // B may publish a provisional last-segment state only
                            // under the host's full disjoint-span guard. Any failed
                            // prefix proof repairs it before API completion.
                            if ((seq_idx + 1) % SegmentsPerSequence == 0) {
                                int64_t const public_offset =
                                    int64_t((seq_idx / SegmentsPerSequence) * H + head_idx) * D * D +
                                    int64_t(dv) * D + dk;
                                if constexpr (ExternalStateFP32) {
                                    float2 const value = make_float2(float(s_frag[mb][kb](2*j)),
                                                                   float(s_frag[mb][kb](2*j+1)));
                                    *reinterpret_cast<float2 *>(static_cast<float *>(external_final_state) +
                                                               public_offset) = value;
                                } else {
                                    BF16 const value[2] = {s_frag[mb][kb](2*j),s_frag[mb][kb](2*j+1)};
                                    *reinterpret_cast<uint32_t *>(static_cast<BF16 *>(external_final_state) +
                                                                 public_offset) =
                                        *reinterpret_cast<uint32_t const *>(value);
                                }
                            }
                        }
                        if constexpr (DualSummary) {
                            BF16 auxiliary_p[2] = {auxiliary_s_frag[mb][kb](2 * j),
                                                   auxiliary_s_frag[mb][kb](2 * j + 1)};
                            *reinterpret_cast<uint32_t *>(
                                &static_cast<BF16 *>(auxiliary_final_state_ptr)[o]) =
                                *reinterpret_cast<uint32_t const *>(auxiliary_p);
                        }
                    }
                }
            }
        }
    }
    if constexpr (PrefixMode == 7) {
        // All actual carry words and output stores are complete before another
        // phase reinitializes shared storage or reads the prior true carry.
        kda_fence_block();
        __syncthreads();
    }
}
