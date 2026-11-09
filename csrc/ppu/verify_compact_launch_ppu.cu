#include "verify_compact_kernel_ppu.cuh"

namespace {

template <typename StateT>
void verify_compact_typed(cutlass::bfloat16_t const *q, cutlass::bfloat16_t const *k,
                          cutlass::bfloat16_t const *v, cutlass::bfloat16_t const *gate,
                          cutlass::bfloat16_t const *beta, float const *A_log, float const *dt_bias,
                          CompactStateArgs a, cutlass::bfloat16_t *out, float scale,
                          float lower_bound, int B, int T, int H, int64_t qs, int64_t ks,
                          int64_t vs, int64_t gs, hggcStream_t stream) {
    using namespace flash_kda_ppu_verify;
    constexpr int shared_bytes = VerifyScratch<kCompactRows>::kStrideBytes;
    flash_kda_ppu_verify_compact_kernel<StateT>
        <<<dim3(kD / kCompactRows, H, B), dim3(32, kWarps), shared_bytes, stream>>>(
            q, k, v, gate, beta, A_log, dt_bias, static_cast<StateT const *>(a.state),
            a.state_indices, static_cast<StateT *>(a.end_state), a.end_indices, out, a.ring_k,
            a.ring_decay, a.ring_v, a.ring_beta, scale, lower_bound, T, H, qs, ks, vs, gs,
            a.state_slots, a.end_slots);
}

template <typename StateT>
void commit_compact_typed(CompactStateArgs a, int32_t const *accepted, int B, int T, int H,
                          hggcStream_t stream) {
    using namespace flash_kda_ppu_verify;
    flash_kda_ppu_commit_compact_kernel<StateT>
        <<<dim3(kD / kCompactRows, H, B), dim3(32, kWarps), 0, stream>>>(
            static_cast<StateT const *>(a.state), a.state_indices,
            static_cast<StateT *>(a.end_state), a.end_indices, a.ring_k, a.ring_decay, a.ring_v,
            a.ring_beta, accepted, T, H, a.state_slots, a.end_slots);
}

} // namespace

void launch_verify_compact(cutlass::bfloat16_t const *q, cutlass::bfloat16_t const *k,
                           cutlass::bfloat16_t const *v, cutlass::bfloat16_t const *gate,
                           cutlass::bfloat16_t const *beta, float const *A_log,
                           float const *dt_bias, CompactStateArgs state, cutlass::bfloat16_t *out,
                           float *prepared, float scale, float lower_bound, int B, int T, int H,
                           int64_t qs, int64_t ks, int64_t vs, int64_t gs, bool state_bf16,
                           hggcStream_t stream) {
    using namespace flash_kda_ppu_verify;
    if (prepared) {
        prepare_compact_operands<<<dim3((B * T + 3) / 4, H), dim3(32, kWarps), 0, stream>>>(
            q, k, v, gate, beta, A_log, dt_bias, state, prepared, scale, lower_bound, B, T, H, qs,
            ks, vs, gs);
        if (state_bf16)
            recurrent_compact_operands<BF16, 32>
                <<<dim3(kD / 32, H, B), dim3(32, kWarps), 0, stream>>>(state, prepared, v, out, T,
                                                                       H, vs);
        else
            recurrent_compact_operands<float, 32>
                <<<dim3(kD / 32, H, B), dim3(32, kWarps), 0, stream>>>(state, prepared, v, out, T,
                                                                       H, vs);
    } else if (state_bf16) {
        verify_compact_typed<BF16>(q, k, v, gate, beta, A_log, dt_bias, state, out, scale,
                                   lower_bound, B, T, H, qs, ks, vs, gs, stream);
    } else {
        verify_compact_typed<float>(q, k, v, gate, beta, A_log, dt_bias, state, out, scale,
                                    lower_bound, B, T, H, qs, ks, vs, gs, stream);
    }
}

void launch_commit_compact(CompactStateArgs state, int32_t const *accepted, int B, int T, int H,
                           bool state_bf16, hggcStream_t stream) {
    if (state_bf16)
        commit_compact_typed<flash_kda_ppu_verify::BF16>(state, accepted, B, T, H, stream);
    else
        commit_compact_typed<float>(state, accepted, B, T, H, stream);
}
