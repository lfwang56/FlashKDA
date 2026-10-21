#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include <algorithm>
#include <cmath>
#include <torch/extension.h>
#include "launch.h"
#include "torch_api.h"
#include "prefill_chunk_ppu.h"

namespace {

template <int CHUNK>
int64_t get_workspace_size_for_chunk(int64_t T_total, int64_t H, int64_t N) {
    constexpr int D = 128;
    // The native launchers and AIU row coordinates use signed 32-bit indices.
    constexpr int64_t index_limit = std::numeric_limits<int>::max();
    TORCH_CHECK(T_total > 0 && H > 0 && N > 0, "T_total, H and N must be positive");
    TORCH_CHECK(T_total <= index_limit - CHUNK && N <= index_limit / 4 && H <= index_limit / D,
                "forward dimensions exceed the native index range");

    // Upper bound: each of N sequences adds at most 1 extra tile vs floor division
    int64_t total_tiles = (T_total + CHUNK - 1) / CHUNK + N;
    TORCH_CHECK(total_tiles <= index_limit / (H * 3 * CHUNK),
                "forward workspace exceeds the native AIU row index range");

    static_assert(CHUNK * D * 2 % 128 == 0,
                  "k_decayed/q_decayed/k_restored size must be 128-byte aligned");
    static_assert(D * 4 % 128 == 0, "g_total size must be 128-byte aligned");
    static_assert(CHUNK * CHUNK * 2 % 128 == 0, "INV/Mqk size must be 128-byte aligned");

    int64_t per_tile_bytes = 3 * CHUNK * D * 2 + D * 4 + 2 * CHUNK * CHUNK * 2;
    // PPU preparation also writes one aligned sigmoid(beta) record per tile.
    per_tile_bytes += 128;

    // Trailing buffer for the tile prefix-sum (N+1 int32), 128-byte aligned.
    int64_t tile_prefix_bytes = ((N + 1) * 4 + 127) / 128 * 128;

    int64_t bytes = H * total_tiles * per_tile_bytes + tile_prefix_bytes;

    // Kimi K3 time-parallel path.  The production full-affine scan needs B, A,
    // carry starts, and replay finals; the optional strong-contraction zero-A
    // build needs only three arrays.  P32 at 4K uses the first half, while 8K
    // through 128K keep P64 and grow segment length, fixing production extra
    // storage at about 48 MiB.
    bool const scan_shape =
        N == 1 && ((H == 6 && (T_total == 1024 || T_total == 2048 || T_total == 4096 ||
                               T_total == 8192 || T_total == 16384 || T_total == 32768 ||
                               T_total == 65536 || T_total == 131072)) ||
                   (H == 12 && (T_total == 4096 || T_total == 8192 || T_total == 16384 ||
                                T_total == 32768 || T_total == 65536 || T_total == 131072)));
    if (scan_shape) {
        constexpr int64_t SEGMENTS = 64;
        constexpr int64_t MATRICES = 4;
        bytes += MATRICES * SEGMENTS * H * D * D * 2 + 128;
    }
    if (H == 6 && N == 4 && T_total == 16384) {
        constexpr int64_t SEQUENCES = 4;
        constexpr int64_t SEGMENTS = 6;
        bytes += 4 * SEQUENCES * SEGMENTS * H * D * D * 2 + 128;
    }

    if (H == 6 && N == 8 && T_total == 16384) {
        // Three temporal segments: B, checkpoint, prefix and true carry.
        bytes += 4 * 8 * 3 * H * D * D * 2 + 128;
    }
    if (H == 12 && N == 4 && T_total == 8192) {
        // Prefix checks finish before B states become mutable true carries.
        // Each mismatch reuses its dead checkpoint slot for prefix state.
        bytes += 2 * 4 * 3 * H * D * D * 2 + 128;
    }
    return bytes;
}

int64_t get_workspace_size_impl(int64_t T_total, int64_t H, int64_t N = 1) {
    if (flash_kda_ppu_reserve_centered32(T_total, H, N)) {
        // The public allocator does not take a layout flag. Reserve an upper
        // bound for both paths; fixed batches retain the measured C16 route.
        return std::max(get_workspace_size_for_chunk<32>(T_total, H, N),
                        get_workspace_size_for_chunk<16>(T_total, H, N));
    }
    return get_workspace_size_for_chunk<16>(T_total, H, N);
}

static void fwd_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
                     torch::Tensor beta, float scale, torch::Tensor out, torch::Tensor workspace,
                     torch::Tensor A_log, torch::Tensor dt_bias, double lower_bound,
                     std::optional<torch::Tensor> initial_state = std::nullopt,
                     std::optional<torch::Tensor> final_state = std::nullopt,
                     std::optional<torch::Tensor> cu_seqlens = std::nullopt) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda() &&
                    out.is_cuda() && workspace.is_cuda(),
                "all tensors must be on CUDA");
    TORCH_CHECK(g.is_contiguous() && beta.is_contiguous() && out.is_contiguous() &&
                    workspace.is_contiguous(),
                "all tensors must be contiguous");

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(g.dtype() == torch::kBFloat16, "g must be bfloat16");
    TORCH_CHECK(beta.dtype() == torch::kBFloat16, "beta must be bfloat16");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bfloat16");

    const c10::cuda::CUDAGuard device_guard(q.device());
    for (auto const &tensor : {k, v, g, beta, out, workspace, A_log, dt_bias}) {
        TORCH_CHECK(tensor.device() == q.device(), "all forward tensors must be on one device");
    }
    TORCH_CHECK(workspace.scalar_type() == at::kByte, "workspace must have uint8 dtype");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
                "output data pointer must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(g.data_ptr()) % 16 == 0,
                "gate data pointer must be 16-byte aligned");

    // Validate state tensors if present
    bool has_state_in = initial_state.has_value();
    bool has_state_out = final_state.has_value();
    bool state_fp32 = false;

    if (has_state_in) {
        auto &is = initial_state.value();
        TORCH_CHECK(is.is_cuda() && is.is_contiguous(),
                    "initial_state must be contiguous CUDA tensor");
        TORCH_CHECK(is.dtype() == torch::kBFloat16 || is.dtype() == torch::kFloat32,
                    "initial_state must be bfloat16 or float32");
        if (is.dtype() == torch::kFloat32)
            state_fp32 = true;
    }
    if (has_state_out) {
        auto &fs = final_state.value();
        TORCH_CHECK(fs.is_cuda() && fs.is_contiguous(),
                    "final_state must be contiguous CUDA tensor");
        TORCH_CHECK(fs.dtype() == torch::kBFloat16 || fs.dtype() == torch::kFloat32,
                    "final_state must be bfloat16 or float32");
        if (fs.dtype() == torch::kFloat32)
            state_fp32 = true;
    }
    for (auto const &state : {initial_state, final_state}) {
        if (state.has_value()) {
            TORCH_CHECK(state->device() == q.device(), "state must be on the input device");
            TORCH_CHECK(reinterpret_cast<uintptr_t>(state->data_ptr()) % 16 == 0,
                        "state data pointer must be 16-byte aligned");
        }
    }
    // If both present, dtypes must match
    if (has_state_in && has_state_out) {
        TORCH_CHECK(initial_state->dtype() == final_state->dtype(),
                    "initial_state and final_state must have the same dtype");
    }

    TORCH_CHECK(A_log.is_cuda() && A_log.is_contiguous(), "A_log must be contiguous CUDA tensor");
    TORCH_CHECK(A_log.dtype() == torch::kFloat32, "A_log must be float32");
    TORCH_CHECK(dt_bias.is_cuda() && dt_bias.is_contiguous(),
                "dt_bias must be contiguous CUDA tensor");
    TORCH_CHECK(dt_bias.dtype() == torch::kFloat32, "dt_bias must be float32");

    // Accept 4D input [B, T, H, D]
    TORCH_CHECK(q.dim() == 4, "q must be [B, T, H, D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B, T, H, D]");
    TORCH_CHECK(v.dim() == 4, "v must be [B, T, H, D]");
    TORCH_CHECK(g.dim() == 4, "g must be [B, T, H, D]");
    TORCH_CHECK(beta.dim() == 3, "beta must be [B, T, H]");
    TORCH_CHECK(out.dim() == 4, "out must be [B, T, H, D]");

    int64_t B = q.size(0);
    int64_t T_seq = q.size(1);
    int64_t H = q.size(2);
    int64_t D = q.size(3);
    int64_t T_total = B * T_seq;
    TORCH_CHECK(B > 0 && T_seq > 0 && H > 0, "batch, token and head counts must be positive");

    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape");
    TORCH_CHECK(g.sizes() == q.sizes(), "g must match q shape");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(beta.size(0) == B && beta.size(1) == T_seq && beta.size(2) == H,
                "beta must be [B, T, H] matching q");

    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == H, "A_log must be [H]");
    TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.size(0) == H && dt_bias.size(1) == D,
                "dt_bias must be [H, D]");

    TORCH_CHECK(D == 128, "currently only supports D == 128");
    TORCH_CHECK(std::isfinite(float(scale)), "scale must be finite");
    TORCH_CHECK(std::isfinite(float(lower_bound)) && lower_bound >= -5.0 && lower_bound <= 0.0,
                "prefill lower_bound must be finite and in [-5, 0]");

    for (auto const &x : {q, k, v}) {
        TORCH_CHECK(x.stride(3) == 1 && x.stride(2) == D, "QKV heads and channels must be dense");
        TORCH_CHECK(x.stride(1) >= H * D && x.stride(1) % 64 == 0,
                    "QKV token strides must be non-overlapping and 64-element aligned");
        TORCH_CHECK(B == 1 || x.is_contiguous(), "strided QKV requires flattened B=1");
        TORCH_CHECK(x.stride(1) <= std::numeric_limits<int>::max(),
                    "QKV token stride exceeds the native index range");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                    "QKV data pointers must be 16-byte aligned");
    }

    // Flatten [B, T, H, D] -> [B*T, H, D] (contiguous, same data pointer)
    auto q_3d = q.reshape({T_total, H, D});
    auto k_3d = k.reshape({T_total, H, D});
    auto v_3d = v.reshape({T_total, H, D});
    auto g_3d = g.reshape({T_total, H, D});
    auto beta_2d = beta.reshape({T_total, H});
    auto out_3d = out.reshape({T_total, H, D});

    auto q_ptr = reinterpret_cast<cutlass::bfloat16_t const *>(q_3d.data_ptr<at::BFloat16>());
    auto k_ptr = reinterpret_cast<cutlass::bfloat16_t const *>(k_3d.data_ptr<at::BFloat16>());
    auto v_ptr = reinterpret_cast<cutlass::bfloat16_t const *>(v_3d.data_ptr<at::BFloat16>());
    auto g_ptr = reinterpret_cast<cutlass::bfloat16_t const *>(g_3d.data_ptr<at::BFloat16>());
    float scale_f = scale;
    auto out_ptr = reinterpret_cast<cutlass::bfloat16_t *>(out_3d.data_ptr<at::BFloat16>());
    auto A_log_ptr = A_log.data_ptr<float>();
    auto dt_bias_ptr = dt_bias.data_ptr<float>();
    float gate_scale = float(lower_bound * 1.4426950408889634);

    // Gather beta directly from the original [T_total, H] layout.
    auto beta_kernel = beta_2d;
    auto beta_kernel_ptr =
        reinterpret_cast<cutlass::bfloat16_t const *>(beta_kernel.data_ptr<at::BFloat16>());

    auto workspace_ptr = workspace.data_ptr();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();


    // Get state pointers (nullptr if not present)
    void const *initial_state_raw = has_state_in ? initial_state->data_ptr() : nullptr;
    void *final_state_raw = has_state_out ? final_state->data_ptr() : nullptr;

    // Determine cu_seqlens and N
    bool is_varlen = cu_seqlens.has_value();
    int64_t N_val;
    int64_t const *cu_seqlens_dev = nullptr;

    if (is_varlen) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        auto &cu_seqlens_t = cu_seqlens.value();
        TORCH_CHECK(cu_seqlens_t.device() == q.device() && cu_seqlens_t.is_contiguous(),
                    "cu_seqlens must be contiguous on the input device");
        TORCH_CHECK(cu_seqlens_t.dtype() == torch::kLong, "cu_seqlens must be int64");
        TORCH_CHECK(cu_seqlens_t.dim() == 1, "cu_seqlens must be 1D");
        N_val = cu_seqlens_t.numel() - 1;
        TORCH_CHECK(N_val > 0, "cu_seqlens must have at least 2 elements");
        cu_seqlens_dev = cu_seqlens_t.data_ptr<int64_t>();
    } else {
        N_val = B;
    }

    // Validate state shapes: always [N, H, D, D]
    if (has_state_in) {
        auto &is = initial_state.value();
        TORCH_CHECK(is.dim() == 4, "initial_state must be [N, H, D, D]");
        TORCH_CHECK(is.size(0) == N_val && is.size(1) == H && is.size(2) == D && is.size(3) == D,
                    "initial_state must be [N, H, D, D]");
    }
    if (has_state_out) {
        auto &fs = final_state.value();
        TORCH_CHECK(fs.dim() == 4, "final_state must be [N, H, D, D]");
        TORCH_CHECK(fs.size(0) == N_val && fs.size(1) == H && fs.size(2) == D && fs.size(3) == D,
                    "final_state must be [N, H, D, D]");
    }

    int const CHUNK = is_varlen && flash_kda_ppu_use_centered32(T_total, H, N_val) ? 32 : 16;
    int total_tiles;
    if (is_varlen) {
        total_tiles = int((T_total + CHUNK - 1) / CHUNK + N_val); // upper bound for varlen
    } else {
        total_tiles = int(N_val * ((T_seq + CHUNK - 1) / CHUNK)); // exact for batched
    }

    TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 128 == 0,
                "workspace must be 128-byte aligned");
    TORCH_CHECK(workspace.numel() >= get_workspace_size_impl(T_total, H, N_val),
                "forward workspace is too small");

// Dispatch based on state configuration and varlen
#define LAUNCH(HI, HO, FP32, VL)                                                                   \
    launch_fwd<128, HI, HO, FP32, VL>(                                                             \
        q_ptr, k_ptr, v_ptr, g_ptr, beta_kernel_ptr, initial_state_raw, scale_f, final_state_raw,  \
        out_ptr, workspace_ptr, total_tiles, int(T_total), int(H), int(N_val), cu_seqlens_dev,     \
        A_log_ptr, dt_bias_ptr, gate_scale, stream, int(q.stride(1)), int(k.stride(1)),            \
        int(v.stride(1)))

#define DISPATCH_STATE(VL)                                                                         \
    if (!has_state_in && !has_state_out) {                                                         \
        LAUNCH(false, false, false, VL);                                                           \
    } else if (has_state_in && has_state_out && state_fp32) {                                      \
        LAUNCH(true, true, true, VL);                                                              \
    } else if (has_state_in && has_state_out && !state_fp32) {                                     \
        LAUNCH(true, true, false, VL);                                                             \
    } else if (!has_state_in && has_state_out && state_fp32) {                                     \
        LAUNCH(false, true, true, VL);                                                             \
    } else if (!has_state_in && has_state_out && !state_fp32) {                                    \
        LAUNCH(false, true, false, VL);                                                            \
    } else if (has_state_in && !has_state_out && state_fp32) {                                     \
        LAUNCH(true, false, true, VL);                                                             \
    } else {                                                                                       \
        LAUNCH(true, false, false, VL);                                                            \
    }

    if (is_varlen) {
        DISPATCH_STATE(true);
    } else {
        DISPATCH_STATE(false);
    }

#undef DISPATCH_STATE
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void decode(torch::Tensor mixed_qkv, torch::Tensor gate, torch::Tensor beta, torch::Tensor A_log,
            torch::Tensor dt_bias, double scale, torch::Tensor state, torch::Tensor state_indices,
            torch::Tensor out, std::optional<double> lower_bound) {
    TORCH_CHECK(mixed_qkv.is_cuda() && gate.is_cuda() && beta.is_cuda() && A_log.is_cuda() &&
                    dt_bias.is_cuda() && state.is_cuda() && state_indices.is_cuda() &&
                    out.is_cuda(),
                "all decode tensors must be on CUDA");
    const c10::cuda::CUDAGuard device_guard(mixed_qkv.device());
    for (auto const &tensor : {gate, beta, A_log, dt_bias, state, state_indices, out}) {
        TORCH_CHECK(tensor.device() == mixed_qkv.device(),
                    "all decode tensors must be on one device");
    }
    TORCH_CHECK(std::isfinite(float(scale)), "scale must be finite");
    for (auto const &tensor : {mixed_qkv, gate, dt_bias, state, out}) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                    "decode vector buffers must be 16-byte aligned");
    }
    TORCH_CHECK(mixed_qkv.is_contiguous() && gate.is_contiguous() && beta.is_contiguous() &&
                    A_log.is_contiguous() && dt_bias.is_contiguous() && state.is_contiguous() &&
                    state_indices.is_contiguous() && out.is_contiguous(),
                "all decode tensors must be contiguous");
    TORCH_CHECK(mixed_qkv.dtype() == torch::kBFloat16 && gate.dtype() == torch::kBFloat16 &&
                    beta.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16,
                "mixed_qkv, gate, beta, and out must be bfloat16");
    TORCH_CHECK(A_log.dtype() == torch::kFloat32 && dt_bias.dtype() == torch::kFloat32,
                "A_log and dt_bias must be float32");
    TORCH_CHECK(state.dtype() == torch::kFloat32 || state.dtype() == torch::kBFloat16,
                "recurrent state must be float32 or bfloat16");
    TORCH_CHECK(state_indices.dtype() == torch::kInt32, "state_indices must be int32");
    TORCH_CHECK(mixed_qkv.dim() == 2 && gate.dim() == 2 && beta.dim() == 2,
                "mixed_qkv, gate, and beta must be 2D");
    TORCH_CHECK(state.dim() == 4, "state must be [slots, H, 128, 128]");
    TORCH_CHECK(state_indices.dim() == 1, "state_indices must be [B]");

    int64_t const B = mixed_qkv.size(0);
    int64_t const H = state.size(1);
    constexpr int64_t D = 128;
    TORCH_CHECK(state.size(0) <= std::numeric_limits<int>::max(),
                "state slot count exceeds the native index range");
    TORCH_CHECK(B > 0 && H > 0 && state.size(0) > 0, "B, H and state slot count must be positive");
    TORCH_CHECK(B <= 65535 && H <= 65535 && B * H * D <= std::numeric_limits<int>::max(),
                "decode dimensions exceed the native launch/index range");
    TORCH_CHECK(state.size(2) == D && state.size(3) == D, "decode currently requires K=V=128");
    TORCH_CHECK(mixed_qkv.size(1) == 3 * H * D, "mixed_qkv must be [B, 3*H*128]");
    TORCH_CHECK(gate.size(0) == B && gate.size(1) == H * D, "gate must be [B, H*128]");
    TORCH_CHECK(beta.size(0) == B && beta.size(1) == H, "beta must be [B, H]");
    TORCH_CHECK(A_log.numel() == H, "A_log must have H elements");
    TORCH_CHECK(dt_bias.numel() == H * D, "dt_bias must have H*128 elements");
    if (lower_bound.has_value()) {
        TORCH_CHECK(std::isfinite(float(lower_bound.value())) && lower_bound.value() <= 0.0,
                    "lower_bound must be finite and non-positive when provided");
    }
    TORCH_CHECK(state_indices.numel() == B, "state_indices must have B elements");
    TORCH_CHECK(out.dim() == 4 && out.size(0) == B && out.size(1) == 1 && out.size(2) == H &&
                    out.size(3) == D,
                "out must be [B, 1, H, 128]");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_decode(reinterpret_cast<cutlass::bfloat16_t const *>(mixed_qkv.data_ptr<at::BFloat16>()),
                  reinterpret_cast<cutlass::bfloat16_t const *>(gate.data_ptr<at::BFloat16>()),
                  reinterpret_cast<cutlass::bfloat16_t const *>(beta.data_ptr<at::BFloat16>()),
                  A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), float(scale),
                  float(lower_bound.value_or(0.0)), lower_bound.has_value(), state.data_ptr(),
                  state.dtype() == torch::kBFloat16, state_indices.data_ptr<int32_t>(),
                  reinterpret_cast<cutlass::bfloat16_t *>(out.data_ptr<at::BFloat16>()), int(B),
                  int(H), state.stride(0), int(state.size(0)), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void verify(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor gate,
            torch::Tensor beta, torch::Tensor A_log, torch::Tensor dt_bias, double scale,
            double lower_bound, torch::Tensor state, torch::Tensor state_indices,
            torch::Tensor intermediate_states, torch::Tensor intermediate_state_indices,
            torch::Tensor out) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && gate.is_cuda() && beta.is_cuda() &&
                    A_log.is_cuda() && dt_bias.is_cuda() && state.is_cuda() &&
                    state_indices.is_cuda() && intermediate_states.is_cuda() &&
                    intermediate_state_indices.is_cuda() && out.is_cuda(),
                "all verify tensors must be on CUDA");
    const c10::cuda::CUDAGuard device_guard(q.device());
    for (auto const &tensor : {k, v, gate, beta, A_log, dt_bias, state, state_indices,
                               intermediate_states, intermediate_state_indices, out}) {
        TORCH_CHECK(tensor.device() == q.device(), "all verify tensors must be on one device");
    }
    TORCH_CHECK(std::isfinite(float(scale)), "scale must be finite");
    TORCH_CHECK(std::isfinite(float(lower_bound)) && lower_bound <= 0.0,
                "lower_bound must be finite and non-positive");
    for (auto const &tensor : {q, k, v, gate, dt_bias, state, intermediate_states}) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                    "verify vector buffers must be 16-byte aligned");
    }
    TORCH_CHECK(state_indices.dim() == 1 && intermediate_state_indices.dim() == 1,
                "state index tensors must be 1D");
    TORCH_CHECK(beta.is_contiguous() && A_log.is_contiguous() && dt_bias.is_contiguous() &&
                    state.is_contiguous() && state_indices.is_contiguous() &&
                    intermediate_states.is_contiguous() &&
                    intermediate_state_indices.is_contiguous() && out.is_contiguous(),
                "beta, parameters, state, indices, checkpoints, and output must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kBFloat16 && k.dtype() == torch::kBFloat16 &&
                    v.dtype() == torch::kBFloat16 && gate.dtype() == torch::kBFloat16 &&
                    beta.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16,
                "q, k, v, gate, beta, and out must be bfloat16");
    TORCH_CHECK(A_log.dtype() == torch::kFloat32 && dt_bias.dtype() == torch::kFloat32,
                "A_log and dt_bias must be float32");
    TORCH_CHECK(state.dtype() == torch::kFloat32 || state.dtype() == torch::kBFloat16,
                "committed state must be float32 or bfloat16");
    TORCH_CHECK(intermediate_states.dtype() == state.dtype(),
                "intermediate state dtype must match committed state dtype");
    TORCH_CHECK(state_indices.dtype() == torch::kInt32 &&
                    intermediate_state_indices.dtype() == torch::kInt32,
                "state indices must be int32");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && gate.dim() == 4 &&
                    beta.dim() == 3 && out.dim() == 4,
                "q, k, v, gate, beta and out must have dense [B, T, ...] layouts");
    TORCH_CHECK(state.dim() == 4, "state must be [slots, H, 128, 128]");
    TORCH_CHECK(intermediate_states.dim() == 5,
                "intermediate_states must be [slots, cache_steps, H, 128, 128]");

    constexpr int64_t D = 128;
    int64_t const B = q.size(0);
    int64_t const T = q.size(1);
    int64_t const H = q.size(2);
    TORCH_CHECK(state.size(0) <= std::numeric_limits<int>::max() &&
                    intermediate_states.size(0) <= std::numeric_limits<int>::max(),
                "state slot counts exceed the native index range");
    TORCH_CHECK(B > 0 && T > 0 && H > 0 && state.size(0) > 0 && intermediate_states.size(0) > 0,
                "B, T, H and state slot counts must be positive");
    TORCH_CHECK(B <= 65535 && H <= 65535 && T <= std::numeric_limits<int>::max() &&
                    B * T <= std::numeric_limits<int>::max() / (H * D),
                "verify dimensions exceed the native launch/index range");
    TORCH_CHECK(q.size(3) == D && k.sizes() == q.sizes() && v.sizes() == q.sizes() &&
                    gate.sizes() == q.sizes(),
                "q, k, v, and gate must be matching [B, T, H, 128] tensors");
    auto token_stride = [B, T](torch::Tensor const &tensor) {
        // Singleton token axes can have arbitrary strides after unflatten().
        return T == 1 && B > 1 ? tensor.stride(0) : tensor.stride(1);
    };
    auto check_vector_layout = [B, T, H, D, &token_stride](torch::Tensor const &tensor,
                                                           char const *name) {
        TORCH_CHECK(tensor.stride(3) == 1 && tensor.stride(2) == D &&
                        token_stride(tensor) >= H * D &&
                        (B == 1 || T == 1 || tensor.stride(0) == T * tensor.stride(1)),
                    name,
                    " must keep each [H, 128] token row contiguous and use one constant "
                    "token-row stride; got strides ",
                    tensor.strides());
    };
    check_vector_layout(q, "q");
    check_vector_layout(k, "k");
    check_vector_layout(v, "v");
    check_vector_layout(gate, "gate");
    TORCH_CHECK(beta.size(0) == B && beta.size(1) == T && beta.size(2) == H,
                "beta must be [B, T, H]");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must be [B, T, H, 128]");
    TORCH_CHECK(state.size(1) == H && state.size(2) == D && state.size(3) == D,
                "state must be [slots, H, 128, 128]");
    TORCH_CHECK(intermediate_states.size(1) >= T && intermediate_states.size(2) == H &&
                    intermediate_states.size(3) == D && intermediate_states.size(4) == D,
                "intermediate_states must have at least T checkpoint steps and [H, 128, 128]");
    TORCH_CHECK(A_log.numel() == H && dt_bias.numel() == H * D, "A_log and dt_bias must match H");
    TORCH_CHECK(state_indices.numel() == B && intermediate_state_indices.numel() == B,
                "state index tensors must have B elements");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_verify(
        reinterpret_cast<cutlass::bfloat16_t const *>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t const *>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t const *>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t const *>(gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t const *>(beta.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(), dt_bias.data_ptr<float>(), float(scale), float(lower_bound),
        state.data_ptr(), state.dtype() == torch::kBFloat16, state_indices.data_ptr<int32_t>(),
        intermediate_states.data_ptr(), intermediate_state_indices.data_ptr<int32_t>(),
        reinterpret_cast<cutlass::bfloat16_t *>(out.data_ptr<at::BFloat16>()), int(B), int(T),
        int(H), token_stride(q), token_stride(k), token_stride(v), token_stride(gate),
        state.stride(0), intermediate_states.stride(0), intermediate_states.stride(1),
        int(state.size(0)), int(intermediate_states.size(0)), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

int64_t get_workspace_size(int64_t T_total, int64_t H, int64_t N) {
    return get_workspace_size_impl(T_total, H, N);
}

void fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g, torch::Tensor beta,
         float scale, torch::Tensor out, torch::Tensor workspace, torch::Tensor A_log,
         torch::Tensor dt_bias, double lower_bound, std::optional<torch::Tensor> initial_state,
         std::optional<torch::Tensor> final_state, std::optional<torch::Tensor> cu_seqlens) {
    fwd_impl(q, k, v, g, beta, scale, out, workspace, A_log, dt_bias, lower_bound, initial_state,
             final_state, cu_seqlens);
}

void bind_ppu(pybind11::module_ &m) {
    m.attr("DECODE_SUPPORTS_LOWER_BOUND") = py::bool_(true);
    m.def("decode", &decode, "FlashKDA packed recurrent decode (PPU)", py::arg("mixed_qkv"),
          py::arg("gate"), py::arg("beta"), py::arg("A_log"), py::arg("dt_bias"), py::arg("scale"),
          py::arg("state"), py::arg("state_indices"), py::arg("out"),
          py::arg("lower_bound") = py::none());
    m.def("verify", &verify, "FlashKDA dense chain target verify (PPU)", py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("gate"), py::arg("beta"), py::arg("A_log"), py::arg("dt_bias"),
          py::arg("scale"), py::arg("lower_bound"), py::arg("state"), py::arg("state_indices"),
          py::arg("intermediate_states"), py::arg("intermediate_state_indices"), py::arg("out"));
    m.def("verify_strided", &verify,
          "FlashKDA dense chain target verify with packed token-row strides (PPU)", py::arg("q"),
          py::arg("k"), py::arg("v"), py::arg("gate"), py::arg("beta"), py::arg("A_log"),
          py::arg("dt_bias"), py::arg("scale"), py::arg("lower_bound"), py::arg("state"),
          py::arg("state_indices"), py::arg("intermediate_states"),
          py::arg("intermediate_state_indices"), py::arg("out"));
}
