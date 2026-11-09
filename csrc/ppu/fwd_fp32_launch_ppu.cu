// Explicit FP32 accumulation. Default prefill dispatch remains in fwd_launch_ppu.cu.
#include "fwd_fp32_launch.h"
#include "fwd_kernel1_ppu.cuh"
#include "fwd_kernel2_ppu.cuh"
#include "fwd_paired_ppu.cuh"

namespace {
using namespace flash_kda_ppu;
using WS = WorkspaceSizes<16, 128>;

template <bool HasIn, bool HasOut, bool StateFP32, bool Varlen>
void forward_fp32(BF16 const *q, BF16 const *k, BF16 const *v, BF16 const *g, BF16 const *beta,
                  void const *initial, float scale, void *final, BF16 *out, void *workspace,
                  int tiles, int tokens, int H, int N, int64_t const *cu, float const *A_log,
                  float const *bias, float gate_scale, hggcStream_t stream, int qs, int ks,
                  int vs) {
    int64_t const records = int64_t(H) * tiles;
    char *base = static_cast<char *>(workspace);
    BF16 *tr = reinterpret_cast<BF16 *>(base);
    float *gt = reinterpret_cast<float *>(
        base + records * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored));
    BF16 *inv = reinterpret_cast<BF16 *>(reinterpret_cast<char *>(gt) + records * WS::kGTotal);
    BF16 *mqk = inv + records * 16 * 16;
    float *wb = reinterpret_cast<float *>(base + records * (WS::kPerTile - WS::kBeta));
    int *prefix = reinterpret_cast<int *>(base + records * WS::kPerTile);
    int64_t const prefix_bytes = ((int64_t(N) + 1) * sizeof(int) + 127) / 128 * 128;
    // Pairing halves the serial recurrence once a sequence spans multiple
    // recurrent tiles. The short path avoids a composition launch and workspace.
    bool const paired = tokens / N > 32;
    if (paired) {
        int const pairs = Varlen ? (tokens + 31) / 32 + N : N * ((tokens / N + 31) / 32);
        int *pair_prefix = reinterpret_cast<int *>(reinterpret_cast<char *>(prefix) + prefix_bytes);
        flash_kda_ppu_pair_prefix<<<1, 32, 0, stream>>>(cu, tokens, N, prefix, pair_prefix);
        flash_kda_ppu_fwd_prepare<16, 128, 128, Varlen, InvPrec::FP32_FORWARD_SUBSTITUTION>
            <<<dim3(tiles, H), 128, 0, stream>>>(
                q, k, g, beta, A_log, bias, tr, tr + 16 * 128, tr + 32 * 128, gt, inv, mqk, wb,
                scale, gate_scale, tokens, H, N, cu, tiles, prefix, nullptr, 0.f, qs, ks);
        char *paired_base = reinterpret_cast<char *>(pair_prefix) + prefix_bytes;
        int64_t const pair_records = int64_t(pairs) * H;
        BF16 *pt = reinterpret_cast<BF16 *>(paired_base);
        float *pg = reinterpret_cast<float *>(pt + pair_records * 96 * 128);
        BF16 *pi = reinterpret_cast<BF16 *>(pg + pair_records * 128);
        BF16 *pm = pi + pair_records * 32 * 32;
        float *pb = reinterpret_cast<float *>(pm + pair_records * 32 * 32);
        flash_kda_ppu_compose_pairs<<<dim3(pairs, H), 64, 0, stream>>>(
            tr, gt, inv, mqk, wb, pt, pg, pi, pm, pb, prefix, pair_prefix, tiles, pairs, N);
        flash_kda_ppu_fwd_recurrence_paired<HasIn, HasOut, StateFP32>
            <<<dim3(N, H, 4), 96, 0, stream>>>(pt, pg, pi, pm, pb, v, initial, final, out, tokens,
                                               H, N, vs, cu, pair_prefix, pairs);
    } else {
        if constexpr (Varlen)
            flash_kda_ppu_build_tile_prefix<<<1, 32, 0, stream>>>(cu, N, 16, prefix);
        flash_kda_ppu_fwd_prepare<16, 128, 128, Varlen, InvPrec::FP32_EXACT>
            <<<dim3(tiles, H), 128, 0, stream>>>(
                q, k, g, beta, A_log, bias, tr, tr + 16 * 128, tr + 32 * 128, gt, inv, mqk, wb,
                scale, gate_scale, tokens, H, N, cu, tiles, prefix, nullptr, 0.f, qs, ks);
        // Four row slices keep the additional FP32 fragments within registers.
        flash_kda_ppu_fwd_recurrence<16, 128, 2, 2, 2, HasIn, HasOut, StateFP32, Varlen, 4, 1, true,
                                     false, 0, false, false, false, false, false, StateFP32, 1, 0,
                                     0, 4, 0, 0, 0, false, false, false, false, false, -1, false,
                                     false, true><<<dim3(N, H, 4), 128, 0, stream>>>(
            v, wb, tr, tr + 16 * 128, tr + 32 * 128, gt, inv, mqk, initial, final, out, tokens, H,
            N, cu, tiles, prefix, nullptr, nullptr, vs, nullptr);
    }
}
} // namespace

void launch_fwd_fp32(BF16 const *q, BF16 const *k, BF16 const *v, BF16 const *g, BF16 const *beta,
                     void const *initial, float scale, void *final, BF16 *out, void *workspace,
                     int tiles, int tokens, int H, int N, int64_t const *cu, float const *A_log,
                     float const *bias, float gate_scale, hggcStream_t stream, int qs, int ks,
                     int vs, bool has_in, bool has_out, bool state_fp32) {
#define RUN(HI, HO, FP, VL)                                                                        \
    forward_fp32<HI, HO, FP, VL>(q, k, v, g, beta, initial, scale, final, out, workspace, tiles,   \
                                 tokens, H, N, cu, A_log, bias, gate_scale, stream, qs, ks, vs)
#define STATES(VL)                                                                                 \
    if (!has_in && !has_out) {                                                                     \
        RUN(false, false, false, VL);                                                              \
    } else if (has_in && has_out && state_fp32) {                                                  \
        RUN(true, true, true, VL);                                                                 \
    } else if (has_in && has_out) {                                                                \
        RUN(true, true, false, VL);                                                                \
    } else if (has_in && state_fp32) {                                                             \
        RUN(true, false, true, VL);                                                                \
    } else if (has_in) {                                                                           \
        RUN(true, false, false, VL);                                                               \
    } else if (state_fp32) {                                                                       \
        RUN(false, true, true, VL);                                                                \
    } else {                                                                                       \
        RUN(false, true, false, VL);                                                               \
    }
    if (cu) {
        STATES(true);
    } else {
        STATES(false);
    }
#undef STATES
#undef RUN
}
