// Intentionally unguarded: the ordinary kernel and bounded tile helper share this math.

static_assert(StaticHeads == 0 || StaticHeads == 6 || StaticHeads == 12);
int const H = StaticHeads ? StaticHeads : H_runtime;
int const q_stride = StaticHeads ? 3 * StaticHeads * D : q_stride_runtime;
int const k_stride = StaticHeads ? 3 * StaticHeads * D : k_stride_runtime;
using namespace cute;
using L1 = K1Layouts<D, CHUNK>;
constexpr int kPad = L1::kColsPad;

constexpr bool kNeedF32Work = !(AffinePrepared && Prec == InvPrec::FP32_FORWARD_SUBSTITUTION);
constexpr bool kAliasGram = NumThreads == 128 && !kNeedF32Work;
constexpr bool kOverlapGramStores = kAliasGram && BuildSinglePrefix;
constexpr bool kTransposeInverse =
    NumThreads == 128 && AffinePrepared && Prec == InvPrec::FP32_FORWARD_SUBSTITUTION;
using PreparedStorage = std::conditional_t<kAliasGram, SharedStorageK1Packed<D, CHUNK>,
                                           SharedStorageK1<D, CHUNK, kNeedF32Work>>;
static_assert(!kAliasGram || sizeof(PreparedStorage) ==
                                 4 * L1::Tile::kBytes + 2 * D * sizeof(float) + 32 * sizeof(float));
#ifdef FLASH_KDA_PREPARE_SCRATCH
auto &sm = *reinterpret_cast<PreparedStorage *>(FLASH_KDA_PREPARE_SCRATCH);
#else
__shared__ PreparedStorage sm;
#endif
auto &gram = [&]() -> auto & {
    if constexpr (kAliasGram)
        return sm.gram;
    else
        return sm;
}();

#ifdef FLASH_KDA_PREPARE_TILE
int const global_tile = FLASH_KDA_PREPARE_TILE;
#else
int const global_tile = blockIdx.x;
#endif
int const head_idx = blockIdx.y;
// Every C16 launcher uses a one-dimensional NumThreads-sized block.
// The existing 128/256-thread static constraint makes this an identity,
// while exposing the actual range to copy/norm/address optimization.
int const tid = int(threadIdx.x) & (NumThreads - 1);
// Preserve the converged warp broadcast, then expose the legal 4/8-warp
// index range of this one-dimensional NumThreads-sized C16 block.
int const warp_idx = cutlass::canonical_warp_idx_sync() & (NumThreads / 32 - 1);

int seq_idx, local_t;
int64_t bos, eos;
if constexpr (BuildSinglePrefix) {
    static_assert(IsVarlen && CHUNK == 16);
    // No CTA in this specialization reads the prefix being published.
    // Stream completion makes it visible to all later recurrence launches.
    seq_idx = 0;
    local_t = global_tile;
#ifdef FLASH_KDA_PREPARE_TILE
    // The boundary is immutable throughout this CTA. Keep both 32-bit
    // halves scalar, including offsets beyond the low word's range.
    uint64_t const begin_bits = uint64_t(cu_seqlens[0]);
    uint64_t const end_bits = uint64_t(cu_seqlens[1]);
    bos = int64_t(uint64_t(__ppu_read_firstlane(uint32_t(begin_bits))) |
                  (uint64_t(__ppu_read_firstlane(uint32_t(begin_bits >> 32))) << 32));
    eos = int64_t(uint64_t(__ppu_read_firstlane(uint32_t(end_bits))) |
                  (uint64_t(__ppu_read_firstlane(uint32_t(end_bits >> 32))) << 32));
#else
    bos = cu_seqlens[0];
    eos = cu_seqlens[1];
#endif
    if (global_tile == 0 && head_idx == 0 && tid == 0) {
        tile_prefix[0] = 0;
        tile_prefix[1] = (int(eos - bos) + CHUNK - 1) / CHUNK;
#ifdef FLASH_KDA_PREPARE_TILE
        tile_prefix[N + 2] = eos - bos >= flash_kda_ppu_general_serial_tokens;
#endif
        if (full_sequence_flag)
            *full_sequence_flag = bos == 0 && eos == T_total;
    }
} else if constexpr (BuildSmallPrefix) {
    static_assert(IsVarlen && CHUNK == 16);
    // Each CTA derives its bounds from the current device lengths; it
    // never reads the prefix another CTA is publishing. The next kernel
    // observes the complete table after this prepare launch finishes.
    int const lane = tid % 32;
    int const begin = lane < N ? int(cu_seqlens[lane]) : 0;
    int const end = lane < N ? int(cu_seqlens[lane + 1]) : 0;
    int const tiles = lane < N ? (end - begin + CHUNK - 1) / CHUNK : 0;
    int cumulative = tiles;
#ifdef FLASH_KDA_PREPARE_TILE
    int longest_prefix = end - begin >= flash_kda_ppu_general_serial_tokens ? end - begin : 0;
    int short_prefix =
        end > begin && end - begin < flash_kda_ppu_general_serial_tokens ? end - begin : 0;
#endif
    CUTE_UNROLL
    for (int offset = 1; offset < 32; offset <<= 1) {
        if (N > offset) {
            int const previous = __shfl_up_sync(0xffffffffu, cumulative, offset);
            if (lane >= offset)
                cumulative += previous;
#ifdef FLASH_KDA_PREPARE_TILE
            int const previous_short = __shfl_up_sync(0xffffffffu, short_prefix, offset);
            int const previous_longest = __shfl_up_sync(0xffffffffu, longest_prefix, offset);
            if (lane >= offset) {
                short_prefix = max(short_prefix, previous_short);
                longest_prefix = max(longest_prefix, previous_longest);
            }
#endif
        }
    }
    int const preceding = cumulative - tiles;
    if (global_tile == 0 && head_idx == 0 && tid < 32) {
        if (lane == 0)
            tile_prefix[0] = 0;
        if (lane < N)
            tile_prefix[lane + 1] = cumulative;
#ifdef FLASH_KDA_PREPARE_TILE
        if (lane == N - 1)
            tile_prefix[N + 2] =
                longest_prefix >= flash_kda_ppu_general_serial_tokens + 2 * short_prefix;
#endif
    }
    unsigned const owners = __ballot_sync(0xffffffffu, lane < N && global_tile >= preceding &&
                                                           global_tile < cumulative);
    if (!owners) {
#ifdef FLASH_KDA_PREPARE_TILE
        return false;
#else
        return;
#endif
    }
    int const owner = __ffs(owners) - 1;
    seq_idx = owner;
    local_t = global_tile - __shfl_sync(0xffffffffu, preceding, owner);
    bos = __shfl_sync(0xffffffffu, begin, owner);
    eos = __shfl_sync(0xffffffffu, end, owner);
} else if constexpr (IsVarlen) {
    // O(log N) instead of the CUDA kernel's per-CTA linear scan.
    int lo = 0, hi = N;
    while (lo + 1 < hi) {
        int const mid = (lo + hi) >> 1;
        if (tile_prefix[mid] <= global_tile)
            lo = mid;
        else
            hi = mid;
    }
    seq_idx = lo;
    local_t = global_tile - tile_prefix[lo];
    bos = cu_seqlens[seq_idx];
    eos = cu_seqlens[seq_idx + 1];
} else {
    int const T_seq = T_total / N;
    int const tiles_per_seq = (T_seq + CHUNK - 1) / CHUNK;
    seq_idx = global_tile / tiles_per_seq;
    local_t = global_tile - seq_idx * tiles_per_seq;
    bos = seq_idx * T_seq;
    eos = bos + T_seq;
#ifdef FLASH_KDA_PREPARE_TILE
    if (global_tile == 0 && head_idx == 0 && tid == 0)
        tile_prefix[N + 2] = 1;
#endif
}
int const seq_len = int(eos - bos);
int const t_tiles = (seq_len + CHUNK - 1) / CHUNK;
// total_tiles is an upper bound in varlen mode, so trailing CTAs have no work.
if (local_t >= t_tiles) {
#ifdef FLASH_KDA_PREPARE_TILE
    return false;
#else
    return;
#endif
}

int const row0 = int(bos) + local_t * CHUNK;
int const rows_valid = max(0, min(CHUNK, seq_len - local_t * CHUNK));
int const gmem_row_stride = H * D;
BF16 const *q_tile = q_ptr + int64_t(row0) * q_stride + head_idx * D;
BF16 const *k_tile = k_ptr + int64_t(row0) * k_stride + head_idx * D;
BF16 const *g_tile = g_ptr + int64_t(row0) * gmem_row_stride + head_idx * D;

// --- load. cp.async into plain padded destinations; tail rows zero-filled.
// This device-side condition is uniform within the CTA and is rechecked
// on every invocation/replay. Partial tiles keep the original zero-fill path.
// The256-thread route retains its original normalization packing.
if (NumThreads == 128 && rows_valid == CHUNK) {
    cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads, true>(sm.a.q, q_tile, q_stride,
                                                                rows_valid, tid);
    cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads, true>(sm.a.k, k_tile, k_stride,
                                                                rows_valid, tid);
    cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads, true>(sm.a.g_raw, g_tile, gmem_row_stride,
                                                                rows_valid, tid);
} else {
    cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads>(sm.a.q, q_tile, q_stride, rows_valid,
                                                          tid);
    cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads>(sm.a.k, k_tile, k_stride, rows_valid,
                                                          tid);
    cp_async_c16_prepare_tile<CHUNK, D, kPad, NumThreads>(sm.a.g_raw, g_tile, gmem_row_stride,
                                                          rows_valid, tid);
}
for (int i = tid; i < D; i += NumThreads)
    sm.dt_bias[i] = dt_bias_ptr[head_idx * D + i];
// PPU consumes beta in its original [T_total, H] layout, avoiding the host
// framework's full-tensor transpose/materialisation on every forward.
for (int i = tid; i < CHUNK; i += NumThreads) {
    sm.beta[i] =
        i < rows_valid ? kda_sigmoid(float(beta_ptr[int64_t(row0 + i) * H + head_idx])) : 0.0f;
}

// exp(A_log) overlaps the loads.
float const a_log_exp = expf(A_log_ptr[head_idx]);

cp_async_fence_all();
cp_async_wait_all<0>();
flash_kda_ppu_prepare_sync<PersistentPrepare>();

// --- L2 normalise q and k over D, one row per 16 lanes.
{
    constexpr int kPerThread = 8;
    constexpr int kThreadsPerRow = D / kPerThread; // 16
    int const col = (tid % kThreadsPerRow) * kPerThread;

    for (int row = tid / kThreadsPerRow; row < CHUNK; row += NumThreads / kThreadsPerRow) {
        float qv[kPerThread], kv[kPerThread];
        float q_sq = 0.0f, k_sq = 0.0f;
        CUTE_UNROLL
        for (int i = 0; i < kPerThread; ++i) {
            qv[i] = float(sm.a.q[row * kPad + col + i]);
            kv[i] = float(sm.a.k[row * kPad + col + i]);
            q_sq += qv[i] * qv[i];
            k_sq += kv[i] * kv[i];
        }
        CUTE_UNROLL
        for (int delta = 8; delta >= 1; delta >>= 1) {
            q_sq += __shfl_xor_sync(0xFFFFFFFFu, q_sq, delta);
            k_sq += __shfl_xor_sync(0xFFFFFFFFu, k_sq, delta);
        }
        float const q_inv = rsqrtf(q_sq + 1e-6f);
        float const k_inv = rsqrtf(k_sq + 1e-6f);
        CUTE_UNROLL
        for (int i = 0; i < kPerThread; ++i) {
            sm.a.q[row * kPad + col + i] = BF16(qv[i] * q_inv);
            sm.a.k[row * kPad + col + i] = BF16(kv[i] * k_inv);
        }
    }
}
// Gate reads a disjoint phase-A buffer; the following barrier also completes
// normalization before decay reads Q/K and reuses the phase-A storage.

// Two8-token or four4-token partitions share adjacent gate column pairs.
// Pack all owner-local phase-A reads and phase-B writes into32-bit words.
static_assert(CHUNK == 16 && (NumThreads == 128 || NumThreads == 256) && D == 128);
static_assert(L1::QK::kElems == L1::Tile::kElems);
{
    constexpr int kParts = NumThreads == 128 ? 2 : 4;
    constexpr int kPairLanes = 32 / kParts, kPartRows = CHUNK / kParts;
    int const lane = tid % 32, part = lane / kPairLanes;
    int const pair = (warp_idx)*kPairLanes + lane % kPairLanes, col = pair * 2;
    float const dt[2] = {sm.dt_bias[col], sm.dt_bias[col + 1]};
    float prefix[2][kPartRows], sum[2] = {0.f, 0.f};
    auto read_pair = [](BF16 const *p) {
        uint32_t const bits = *reinterpret_cast<uint32_t const *>(p);
        return __bfloat1622float2(*reinterpret_cast<__ppu_bfloat162 const *>(&bits));
    };
    auto write_pair = [](BF16 *p, float x, float y) {
        __ppu_bfloat162 const packed = __floats2bfloat162_rn(x, y);
        *reinterpret_cast<uint32_t *>(p) = *reinterpret_cast<uint32_t const *>(&packed);
    };
    if constexpr (GuardWeakP3 || BuildSinglePrefix) {
        // Expose the independent SFU work before the ordered accumulation.
        // Reuse the existing prefix slots, storing unscaled sigmoid values
        // so the original scale/multiply/add expression stays in the scan.
        CUTE_UNROLL
        for (int i = 0; i < kPartRows; ++i) {
            int const row = part * kPartRows + i;
            float2 const raw = read_pair(sm.a.g_raw + row * kPad + col);
            float const g[2] = {raw.x, raw.y};
            CUTE_UNROLL
            for (int c = 0; c < 2; ++c)
                // Padded raw rows are already zero-filled in shared memory.
                // Their unused activations are discarded by the original
                // row conditional in the following accumulation stage.
                prefix[c][i] = kda_sigmoid(a_log_exp * (g[c] + dt[c]));
        }
        CUTE_UNROLL
        for (int i = 0; i < kPartRows; ++i) {
            int const row = part * kPartRows + i;
            CUTE_UNROLL
            for (int c = 0; c < 2; ++c) {
                float const value = row < rows_valid ? gate_scale * prefix[c][i] : 0.f;
                sum[c] += value;
                prefix[c][i] = sum[c];
            }
        }
    } else {
        CUTE_UNROLL
        for (int i = 0; i < kPartRows; ++i) {
            int const row = part * kPartRows + i;
            float2 const raw = read_pair(sm.a.g_raw + row * kPad + col);
            float const g[2] = {raw.x, raw.y};
            CUTE_UNROLL
            for (int c = 0; c < 2; ++c) {
                float const value =
                    row < rows_valid ? gate_scale * kda_sigmoid(a_log_exp * (g[c] + dt[c])) : 0.f;
                sum[c] += value;
                prefix[c][i] = sum[c];
            }
        }
    }
    float preceding[2], total_decay[2];
    CUTE_UNROLL
    for (int c = 0; c < 2; ++c) {
        float scan = sum[c];
        CUTE_UNROLL
        for (int delta = 1; delta < kParts; delta *= 2) {
            float const other = __shfl_up_sync(0xffffffffu, scan, delta * kPairLanes);
            if (part >= delta)
                scan += other;
        }
        float const previous = __shfl_up_sync(0xffffffffu, scan, kPairLanes);
        preceding[c] = part ? previous : 0.f;
        float const total = __shfl_sync(0xffffffffu, scan, (lane % kPairLanes) + 32 - kPairLanes);
        total_decay[c] = kda_exp2(total);
        if (part == kParts - 1) {
            sm.g_total[col + c] = total_decay[c];
            if constexpr (ValidateScan) {
                if (!(total_decay[c] >= 0.f && total_decay[c] <= scan_threshold))
                    atomicExch(scan_flag, 0);
            }
        }
    }
    if constexpr (GuardWeakP3) {
        static_assert(IsVarlen && CHUNK == 16 && NumThreads == 128 && !BuildSinglePrefix);
        // Only clear scheduling eligibility. Every enabled family still
        // requires its complete exact checkpoint before reusing a state.
        bool const p3_prefix = (local_t >= 43 && local_t < 46) || (local_t >= 86 && local_t < 89);
        bool const p2_prefix = local_t >= 64 && local_t < 67;
        if (p3_prefix || p2_prefix) {
            bool const weak = !(total_decay[0] >= 0.f && total_decay[0] <= 0.125f) ||
                              !(total_decay[1] >= 0.f && total_decay[1] <= 0.125f);
            if (__any_sync(0xffffffffu, weak) && lane == 0)
                atomicExch(full_sequence_flag + (p2_prefix ? 1 : 0), 0);
        }
    }
    // Complete normalization/raw gate reads before overwriting their
    // aliased words. Every word has exactly one gate/decay owner.
    flash_kda_ppu_prepare_sync<PersistentPrepare>();
    CUTE_UNROLL
    for (int i = 0; i < kPartRows; ++i) {
        int const o = (part * kPartRows + i) * kPad + col;
        float2 const q = read_pair(sm.a.q + o), k = read_pair(sm.a.k + o);
        float const dec0 = kda_exp2(prefix[0][i] + preceding[0]);
        float const dec1 = kda_exp2(prefix[1][i] + preceding[1]);
        float const inv0 = __ppu_rcpf(dec0), inv1 = __ppu_rcpf(dec1);
        write_pair(sm.b.q_decayed + o, q.x * dec0 * scale, q.y * dec1 * scale);
        write_pair(sm.b.k_decayed + o, k.x * dec0, k.y * dec1);
        write_pair(sm.b.k_inv + o, k.x * inv0, k.y * inv1);
        write_pair(sm.b.k_restored + o, k.x * inv0 * total_decay[0], k.y * inv1 * total_decay[1]);
    }
}
flash_kda_ppu_prepare_sync<PersistentPrepare>();

int const ws_idx = head_idx * total_tiles + global_tile;

// --- L = k_decayed @ k_inv^T and Mqk = q_decayed @ k_inv^T, on two warps.
// Both are 16x16 with K = D, so 8 k-steps of the 16x16x16 atom.
{
    auto mma =
        make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    int const warp = warp_idx;
    int const lane = tid % 32;

    if (warp < 2) {
        auto thr = mma.get_slice(lane);
        BF16 const *a_base = (warp == 0) ? sm.b.k_decayed : sm.b.q_decayed;

        Tensor ta = make_tensor(make_smem_ptr(a_base), typename L1::Tile::Layout{});
        Tensor tb = make_tensor(make_smem_ptr(sm.b.k_inv), typename L1::Tile::Layout{});
        Tensor tc = make_tensor(make_smem_ptr(gram.Mqk), typename L1::LMBF16::Layout{});

        auto acc = thr.make_fragment_C(
            thr.partition_C(local_tile(tc, Shape<_16, _16>{}, make_coord(0, 0))));
        clear(acc);

        CUTE_UNROLL
        for (int kb = 0; kb < D / 16; ++kb) {
            Tensor a_k = local_tile(ta, Shape<_16, _16>{}, make_coord(0, kb));
            Tensor b_k = local_tile(tb, Shape<_16, _16>{}, make_coord(0, kb));
            auto ra = thr.partition_fragment_A(a_k);
            auto rb = thr.partition_fragment_B(b_k);
            copy(AutoVectorizingCopy{}, thr.partition_A(a_k), ra);
            copy(AutoVectorizingCopy{}, thr.partition_B(b_k), rb);
            cute::gemm(thr, ra, rb, acc);
        }

        if constexpr (kAliasGram) {
            // Both Gram warps must finish every k_inv read before either
            // publishes L/Mqk into the same bytes. Other warps wait at the
            // original following CTA barrier.
            kda_fence_block();
            asm volatile("" ::: "memory");
            cutlass::arch::NamedBarrier::arrive_and_wait(
                64, PersistentPrepare ? 2 + int(threadIdx.x) / 128 : 0);
            asm volatile("" ::: "memory");
        }

        // Masks, beta scaling and INV = I - L are applied in the C fragment
        // via the decoded (m, n), which removes an smem round-trip and a
        // __syncthreads() that the CUDA kernel needed.
        if (warp == 0) {
            uint32_t nonfinite_lower = 0;
            CUTE_UNROLL
            for (int v = 0; v < decltype(size(acc))::value; ++v) {
                FragCoord const c = frag_coord_mn(lane, v);
                // Strictly lower triangular, scaled by sigmoid(beta[row]).
                float const l = (c.m <= c.n) ? 0.0f : acc(v) * sm.beta[c.m];
                gram.Lf32[c.m * (CHUNK + 4) + c.n] = l;
                if constexpr (NumThreads == 128 && AffinePrepared &&
                              Prec == InvPrec::FP32_FORWARD_SUBSTITUTION)
                    nonfinite_lower |= (__float_as_uint(l) & 0x7f800000u) == 0x7f800000u;
                if constexpr (kNeedF32Work)
                    sm.INVf32[c.m * (CHUNK + 4) + c.n] = (c.m == c.n) ? 1.0f - l : -l;
            }
            if constexpr (NumThreads == 128 && AffinePrepared &&
                          Prec == InvPrec::FP32_FORWARD_SUBSTITUTION) {
                bool const any_nonfinite = __any_sync(0xffffffffu, nonfinite_lower != 0);
                // beta[0:16] remains live; slot16 is unused padding.
                if (lane == 0)
                    reinterpret_cast<int *>(sm.beta)[16] = any_nonfinite;
            }
        } else {
            CUTE_UNROLL
            for (int v = 0; v < decltype(size(acc))::value; ++v) {
                FragCoord const c = frag_coord_mn(lane, v);
                gram.Mqk[c.m * (CHUNK + 8) + c.n] = BF16((c.m < c.n) ? 0.0f : acc(v));
            }
        }
    } else if constexpr (kOverlapGramStores) {
        // These two warps otherwise wait for the Gram pair. The first two
        // tiles and g_total remain read-only and disjoint from its overlaid
        // L/Mqk workspace. The existing CTA barrier joins both tasks before
        // any inverse/affine consumer proceeds.
        constexpr int kCopyThreads = 64;
        int const copy_tid = tid - 64;
        store_tile_2d_full<CHUNK, D, kPad, kCopyThreads>(ws_kd + int64_t(ws_idx) * 3 * CHUNK * D, D,
                                                         sm.b.k_decayed, copy_tid);
        store_tile_2d_full<CHUNK, D, kPad, kCopyThreads>(ws_qd + int64_t(ws_idx) * 3 * CHUNK * D, D,
                                                         sm.b.q_decayed, copy_tid);
        store_contiguous<kCopyThreads>(ws_gt + int64_t(ws_idx) * D, sm.g_total,
                                       D * int(sizeof(float)), copy_tid);
    }
}
flash_kda_ppu_prepare_sync<PersistentPrepare>();

// BF16 affine preparation distributes independent FP32 inverse rows over
// all resident warps. Other inverse algorithms keep their warp-zero inverse,
// overlapped with Kd/Qd/gate writeback. The affine factors are produced only
// after the CTA barrier makes every completed inverse row visible.
if constexpr (AffinePrepared && Prec == InvPrec::FP32_FORWARD_SUBSTITUTION) {
    static_assert((NumThreads == 128 || NumThreads == 256) && CHUNK == 16);
    if constexpr (!kOverlapGramStores) {
        store_tile_2d_full<CHUNK, D, kPad, NumThreads>(ws_kd + int64_t(ws_idx) * 3 * CHUNK * D, D,
                                                       sm.b.k_decayed, tid);
        store_tile_2d_full<CHUNK, D, kPad, NumThreads>(ws_qd + int64_t(ws_idx) * 3 * CHUNK * D, D,
                                                       sm.b.q_decayed, tid);
        store_contiguous<NumThreads>(ws_gt + int64_t(ws_idx) * D, sm.g_total,
                                     D * int(sizeof(float)), tid);
    }

    // Solve each inverse row independently from right to left. A half-warp
    // owns one or two rows and works through 15 FP32 pivots instead
    // of waiting for one warp's 120 dependent dot-product FMAs.
    constexpr int kRowsPerHalfWarp = 256 / NumThreads;
    int const first_row = tid / 16, col = tid % 16, lane = tid % 32;
    float inverse[kRowsPerHalfWarp];
    CUTE_UNROLL
    for (int r = 0; r < kRowsPerHalfWarp; ++r)
        inverse[r] = first_row + r * (NumThreads / 16) == col ? 1.f : 0.f;
    if constexpr (NumThreads == 128) {
        int const nonfinite =
            __shfl_sync(0xffffffffu, reinterpret_cast<int const *>(sm.beta)[16], 0);
        if (nonfinite == 0) {
            // Each arm is uniform across a complete warp. Keep the full
            // shuffle mask while specializing the highest live row pair.
            if (warp_idx == 0)
                solve_parallel_inverse_rows<NumThreads, true, 0>(inverse, gram.Lf32, col, lane);
            else if (warp_idx == 1)
                solve_parallel_inverse_rows<NumThreads, true, 1>(inverse, gram.Lf32, col, lane);
            else if (warp_idx == 2)
                solve_parallel_inverse_rows<NumThreads, true, 2>(inverse, gram.Lf32, col, lane);
            else
                solve_parallel_inverse_rows<NumThreads, true, 3>(inverse, gram.Lf32, col, lane);
        } else
            solve_parallel_inverse_rows<NumThreads, false>(inverse, gram.Lf32, col, lane);
    } else {
        solve_parallel_inverse_rows<NumThreads, false>(inverse, gram.Lf32, col, lane);
    }
    CUTE_UNROLL
    for (int r = 0; r < kRowsPerHalfWarp; ++r) {
        int const row = first_row + r * (NumThreads / 16);
        constexpr int kInvPad = CHUNK + 8;
        int const offset = kTransposeInverse ? col * kInvPad + row : row * kInvPad + col;
        gram.INVbf[offset] = BF16(inverse[r]);
    }
} else {
    if (tid < 32) {
        chunk_inverse_1warp<D, CHUNK, Prec>(sm, tid);
    } else {
        constexpr int kStoreThreads = NumThreads - 32;
        int const stid = tid - 32;
        store_tile_2d_full<CHUNK, D, kPad, kStoreThreads>(ws_kd + int64_t(ws_idx) * 3 * CHUNK * D,
                                                          D, sm.b.k_decayed, stid);
        store_tile_2d_full<CHUNK, D, kPad, kStoreThreads>(ws_qd + int64_t(ws_idx) * 3 * CHUNK * D,
                                                          D, sm.b.q_decayed, stid);
        if constexpr (!AffinePrepared) {
            store_tile_2d_full<CHUNK, D, kPad, kStoreThreads>(
                ws_kr + int64_t(ws_idx) * 3 * CHUNK * D, D, sm.b.k_restored, stid);
        }
        store_contiguous<kStoreThreads>(ws_gt + int64_t(ws_idx) * D, sm.g_total,
                                        D * int(sizeof(float)), stid);
        if constexpr (!AffinePrepared) {
            store_tile_2d_full<CHUNK, CHUNK, CHUNK + 8, kStoreThreads>(
                ws_mqk + int64_t(ws_idx) * CHUNK * CHUNK, CHUNK, gram.Mqk, stid);
            store_contiguous<kStoreThreads>(ws_beta + int64_t(ws_idx) *
                                                          (WorkspaceSizes<CHUNK, D>::kBeta / 4),
                                            sm.beta, CHUNK * int(sizeof(float)), stid);
        }
    }
}
if constexpr (!(AffinePrepared && Prec == InvPrec::FP32_FORWARD_SUBSTITUTION)) {
    flash_kda_ppu_prepare_sync<PersistentPrepare>();
    // Other inverse algorithms publish FP32 rows for this separate narrowing.
    for (int i = tid; i < CHUNK * CHUNK; i += NumThreads) {
        int const r = i / CHUNK, c = i % CHUNK;
        gram.INVbf[r * (CHUNK + 8) + c] = BF16(sm.INVf32[r * (CHUNK + 4) + c]);
    }
}
// Every affine consumer sees all completed BF16 inverse rows.
flash_kda_ppu_prepare_sync<PersistentPrepare>();
if constexpr (AffinePrepared) {
    // Hoist the state-independent inverse/beta products out of the serial
    // recurrence. W = diag(beta) INV^T Kr, M = Mqk INV diag(beta).
    // K2 consumes r = V^T - S Kd^T, then S' = S g + r W and
    // out = Qd S^T + M r^T. Reuse the existing Kr/Mqk workspace slots.
    static_assert(CHUNK == 16 && (NumThreads == 128 || NumThreads == 256) && D == 128);
    auto mma =
        make_tiled_mma(MMA_Atom<MmaBF16>{}, Layout<Shape<_1, _1, _1>>{}, Tile<_16, _16, _16>{});
    int const warp = warp_idx, lane = tid % 32;
    auto thr = mma.get_slice(lane);
    using InvView =
        std::conditional_t<kTransposeInverse, Layout<Shape<_16, _16>, Stride<Int<CHUNK + 8>, _1>>,
                           Layout<Shape<_16, _16>, Stride<_1, Int<CHUNK + 8>>>>;
    Tensor ti = make_tensor(make_smem_ptr(gram.INVbf), InvView{});
    Tensor tr = make_tensor(make_smem_ptr(sm.b.k_restored),
                            Layout<Shape<Int<D>, Int<CHUNK>>, Stride<_1, Int<kPad>>>{});
    Tensor tm =
        make_tensor(make_smem_ptr(gram.Mqk), Layout<Shape<_16, _16>, Stride<Int<CHUNK + 8>, _1>>{});
    auto a = thr.partition_fragment_A(ti);
    auto acc = thr.make_fragment_C(thr.partition_C(tm));
    copy(AutoVectorizingCopy{}, thr.partition_A(ti), a);
    CUTE_UNROLL
    for (int wb = warp; wb < D / 16; wb += NumThreads / 32) {
        auto rb = local_tile(tr, Shape<_16, _16>{}, make_coord(wb, 0));
        auto b = thr.partition_fragment_B(rb);
        copy(AutoVectorizingCopy{}, thr.partition_B(rb), b);
        clear(acc);
        cute::gemm(thr, a, b, acc);
        CUTE_UNROLL
        for (int j = 0; j < decltype(size(acc))::value / 2; ++j) {
            FragCoord const c = frag_coord_mn(lane, 2 * j);
            BF16 pair[2] = {BF16(acc(2 * j) * sm.beta[c.m]), BF16(acc(2 * j + 1) * sm.beta[c.m])};
            *reinterpret_cast<uint32_t *>(ws_kr + int64_t(ws_idx) * 3 * CHUNK * D + c.m * D +
                                          wb * 16 + c.n) =
                *reinterpret_cast<uint32_t const *>(pair);
        }
    }
    if (warp == 0) {
        auto ma = thr.partition_fragment_A(tm);
        auto ib = thr.partition_fragment_B(ti);
        copy(AutoVectorizingCopy{}, thr.partition_A(tm), ma);
        copy(AutoVectorizingCopy{}, thr.partition_B(ti), ib);
        clear(acc);
        cute::gemm(thr, ma, ib, acc);
        CUTE_UNROLL
        for (int j = 0; j < decltype(size(acc))::value / 2; ++j) {
            FragCoord const c = frag_coord_mn(lane, 2 * j);
            BF16 pair[2] = {BF16(acc(2 * j) * sm.beta[c.n]),
                            BF16(acc(2 * j + 1) * sm.beta[c.n + 1])};
            *reinterpret_cast<uint32_t *>(ws_mqk + int64_t(ws_idx) * CHUNK * CHUNK + c.m * CHUNK +
                                          c.n) = *reinterpret_cast<uint32_t const *>(pair);
        }
    }
} else {
    store_tile_2d_full<CHUNK, CHUNK, CHUNK + 8, NumThreads>(
        ws_inv + int64_t(ws_idx) * CHUNK * CHUNK, CHUNK, gram.INVbf, tid);
}
