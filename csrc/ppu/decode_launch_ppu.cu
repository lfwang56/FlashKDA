#include "decode_kernel_ppu.cuh"

template <typename StateT>
void launch_decode_typed(cutlass::bfloat16_t const *mixed_qkv, cutlass::bfloat16_t const *gate,
                         cutlass::bfloat16_t const *beta, float const *A_log, float const *dt_bias,
                         float scale, float lower_bound, bool use_lower_bound, StateT *state,
                         int32_t const *state_indices, cutlass::bfloat16_t *out, int B, int H,
                         int64_t state_slot_stride, int state_slots, cudaStream_t stream) {
    if ((sizeof(StateT) == sizeof(flash_kda_ppu::BF16) && B * H <= 12) ||
        (sizeof(StateT) == sizeof(float) && B * H <= 96)) {
        constexpr bool kBf16 = sizeof(StateT) == sizeof(flash_kda_ppu::BF16);
        constexpr int kWarps = 4;
        auto launch_rows = [&](auto split_tag) {
            constexpr int kSplits = decltype(split_tag)::value;
            constexpr int kStateRows =
                kBf16 ? flash_kda_ppu::kD / kSplits : flash_kda_ppu::kStages * kWarps;
            int const smem = sizeof(StateT) * kStateRows * flash_kda_ppu::kD +
                             sizeof(float) * (3 * flash_kda_ppu::kD + 1) +
                             sizeof(cutlass::bfloat16_t) * flash_kda_ppu::kD;
            flash_kda_ppu::flash_kda_ppu_decode_kernel<StateT, kBf16, kWarps, kSplits>
                <<<dim3(kSplits, H, B), dim3(32, kWarps), smem, stream>>>(
                    mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound, use_lower_bound,
                    state, state_indices, out, H, state_slot_stride, state_slots);
        };
        if constexpr (kBf16) {
            launch_rows(std::integral_constant<int, 8>{});
        } else {
            launch_rows(std::integral_constant<int, 4>{});
        }
        return;
    }
    // Pack eight FP32 columns per lane and two independent rows per half warp.
    // Each row reduction uses four shuffles instead of five; resident state
    // and all arithmetic stay FP32. Preserve native AIU bulk state transport.
    if constexpr (sizeof(StateT) == sizeof(float)) {
        if (B >= 128) {
            constexpr int kWarps = 16;
            constexpr int kSplits = 2;
            constexpr int kSmem =
                sizeof(float) * (64 * 128 + 3 * 128 + 1) + sizeof(cutlass::bfloat16_t) * 128;
            flash_kda_ppu::flash_kda_ppu_decode_kernel<StateT, true, kWarps, kSplits, true>
                <<<dim3(kSplits, H, B), dim3(32, kWarps), kSmem, stream>>>(
                    mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound, use_lower_bound,
                    state, state_indices, out, H, state_slot_stride, state_slots);
            return;
        }
    }
    if constexpr (sizeof(StateT) == sizeof(flash_kda_ppu::BF16)) {
        if (H == 12 && B >= 16 && B <= 256) {
            constexpr int kWarps = 8, kPitch = 136;
            auto launch_bf16_rows = [&](auto split_tag) {
                constexpr int kSplits = decltype(split_tag)::value;
                constexpr int kSmem = sizeof(StateT) * (128 / kSplits) * kPitch +
                                      sizeof(float) * (3 * 128 + 1) +
                                      sizeof(cutlass::bfloat16_t) * 128;
                auto launch_join_policy = [&](auto merge_tag) {
                    constexpr bool merge = decltype(merge_tag)::value;
                    flash_kda_ppu::flash_kda_ppu_decode_kernel<StateT, true, kWarps, kSplits, false,
                                                               kPitch, merge>
                        <<<dim3(kSplits, H, B), dim3(32, kWarps), kSmem, stream>>>(
                            mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound,
                            use_lower_bound, state, state_indices, out, H, state_slot_stride,
                            state_slots);
                };
                if constexpr (kSplits == 1) {
                    // One full-head owner reuses normalization, gates and cached
                    // parameter packets across all four state row groups.
                    launch_join_policy(std::integral_constant<bool, true>{});
                } else {
                    if (B >= 128)
                        launch_join_policy(std::integral_constant<bool, true>{});
                    else
                        launch_join_policy(std::integral_constant<bool, false>{});
                }
            };
            if (B == 128 || B == 256)
                launch_bf16_rows(std::integral_constant<int, 1>{});
            else
                launch_bf16_rows(std::integral_constant<int, 2>{});
            return;
        }
    }
    // BF16 uses the native B16 full-state descriptor at every batch size;
    // the FP32 path retains the tuned rolling/bulk crossover byte-for-byte.
    bool const use_bulk_aiu = sizeof(StateT) == sizeof(flash_kda_ppu::BF16) || B >= 128;
    bool const use_bulk_split = use_bulk_aiu && B * H <= 3072;
    bool const use_narrow_rolling = !use_bulk_aiu && B * H == 384;
    bool const use_tall_rolling = !use_bulk_aiu && B * H == 768;
    int const rolling_warps = use_narrow_rolling ? flash_kda_ppu::kNarrowRollingWarps
                              : use_tall_rolling ? flash_kda_ppu::kTallRollingWarps
                                                 : flash_kda_ppu::kWideRollingWarps;
    int const state_rows = use_bulk_aiu
                               ? (use_bulk_split ? flash_kda_ppu::kD / flash_kda_ppu::kBulkRowSplits
                                                 : flash_kda_ppu::kD)
                               : flash_kda_ppu::kStages * rolling_warps;
    int const smem_bytes = sizeof(StateT) * state_rows * 128 + sizeof(float) * (3 * 128 + 1) +
                           sizeof(cutlass::bfloat16_t) * 128;
    constexpr int kUnsplitWarps = sizeof(StateT) == sizeof(flash_kda_ppu::BF16)
                                      ? flash_kda_ppu::kBf16UnsplitWarps
                                      : flash_kda_ppu::kBulkUnsplitWarps;
    if (use_bulk_split) {
        flash_kda_ppu::flash_kda_ppu_decode_kernel<StateT, true, flash_kda_ppu::kBulkSplitWarps,
                                                   flash_kda_ppu::kBulkRowSplits>
            <<<dim3(flash_kda_ppu::kBulkRowSplits, H, B), dim3(32, flash_kda_ppu::kBulkSplitWarps),
               smem_bytes, stream>>>(mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound,
                                     use_lower_bound, state, state_indices, out, H,
                                     state_slot_stride, state_slots);
    } else if (use_bulk_aiu) {
        flash_kda_ppu::flash_kda_ppu_decode_kernel<StateT, true, kUnsplitWarps>
            <<<dim3(1, H, B), dim3(32, kUnsplitWarps), smem_bytes, stream>>>(
                mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound, use_lower_bound, state,
                state_indices, out, H, state_slot_stride, state_slots);
    } else if constexpr (sizeof(StateT) == sizeof(float)) {
        if (use_narrow_rolling) {
            flash_kda_ppu::flash_kda_ppu_decode_kernel<StateT, false,
                                                       flash_kda_ppu::kNarrowRollingWarps>
                <<<dim3(1, H, B), dim3(32, flash_kda_ppu::kNarrowRollingWarps), smem_bytes,
                   stream>>>(mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound,
                             use_lower_bound, state, state_indices, out, H, state_slot_stride,
                             state_slots);
        } else if (use_tall_rolling) {
            flash_kda_ppu::flash_kda_ppu_decode_kernel<
                StateT, false, flash_kda_ppu::kTallRollingWarps, flash_kda_ppu::kTallRowSplits>
                <<<dim3(flash_kda_ppu::kTallRowSplits, H, B),
                   dim3(32, flash_kda_ppu::kTallRollingWarps), smem_bytes, stream>>>(
                    mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound, use_lower_bound,
                    state, state_indices, out, H, state_slot_stride, state_slots);
        } else {
            flash_kda_ppu::flash_kda_ppu_decode_kernel<StateT, false,
                                                       flash_kda_ppu::kWideRollingWarps>
                <<<dim3(1, H, B), dim3(32, flash_kda_ppu::kWideRollingWarps), smem_bytes, stream>>>(
                    mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound, use_lower_bound,
                    state, state_indices, out, H, state_slot_stride, state_slots);
        }
    }
}

void launch_decode(cutlass::bfloat16_t const *mixed_qkv, cutlass::bfloat16_t const *gate,
                   cutlass::bfloat16_t const *beta, float const *A_log, float const *dt_bias,
                   float scale, float lower_bound, bool use_lower_bound, void *state,
                   bool state_bf16, int32_t const *state_indices, cutlass::bfloat16_t *out, int B,
                   int H, int64_t state_slot_stride, int state_slots, cudaStream_t stream) {
    if (state_bf16) {
        launch_decode_typed(mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound,
                            use_lower_bound, reinterpret_cast<flash_kda_ppu::BF16 *>(state),
                            state_indices, out, B, H, state_slot_stride, state_slots, stream);
    } else {
        launch_decode_typed(mixed_qkv, gate, beta, A_log, dt_bias, scale, lower_bound,
                            use_lower_bound, reinterpret_cast<float *>(state), state_indices, out,
                            B, H, state_slot_stride, state_slots, stream);
    }
}
