#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <limits>
#include "verify_compact_launch.h"
#include <ATen/cuda/CUDAContext.h>
#include "torch_api.h"

namespace {

struct CompactShape {
    int B, T, H;
    CompactStateArgs args;
};

CompactShape check_compact(torch::Tensor state, torch::Tensor state_indices,
                           torch::Tensor end_state, torch::Tensor end_indices, torch::Tensor ring_k,
                           torch::Tensor ring_decay, torch::Tensor ring_v,
                           torch::Tensor ring_beta) {
    for (auto const &x :
         {state, state_indices, end_state, end_indices, ring_k, ring_decay, ring_v, ring_beta}) {
        TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.device() == state.device(),
                    "compact state buffers must be contiguous on one CUDA device");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                    "compact state buffers must be 16-byte aligned");
    }
    TORCH_CHECK(state.dim() == 4 && end_state.dim() == 4 && ring_k.dim() == 4,
                "state/end_state must be [slots,H,128,128]; rings must be [end_slots,T,H,128]");
    TORCH_CHECK(state.scalar_type() == at::kFloat || state.scalar_type() == at::kBFloat16,
                "state must be float32 or bfloat16");
    TORCH_CHECK(end_state.scalar_type() == state.scalar_type(), "state dtypes must match");
    TORCH_CHECK(state_indices.scalar_type() == at::kInt && end_indices.scalar_type() == at::kInt &&
                    state_indices.dim() == 1 && end_indices.sizes() == state_indices.sizes(),
                "state_indices and end_indices must be int32 [B]");
    int64_t const B = state_indices.numel(), T = ring_k.size(1), H = state.size(1);
    int64_t const limit = std::numeric_limits<int>::max();
    TORCH_CHECK(B > 0 && B <= 65535 && H > 0 && H <= 65535 && T > 0 && T <= limit &&
                    B * T <= limit / (H * 128),
                "compact dimensions exceed native index range");
    TORCH_CHECK(state.size(0) > 0 && state.size(0) <= limit && end_state.size(0) > 0 &&
                    end_state.size(0) <= limit,
                "invalid compact state slot count");
    TORCH_CHECK(state.size(2) == 128 && state.size(3) == 128 && end_state.size(1) == H &&
                    end_state.size(2) == 128 && end_state.size(3) == 128,
                "compact states must have matching [H,128,128] dimensions");
    TORCH_CHECK(ring_k.size(0) == end_state.size(0) && ring_k.size(2) == H &&
                    ring_k.size(3) == 128 && ring_decay.sizes() == ring_k.sizes() &&
                    ring_v.sizes() == ring_k.sizes(),
                "compact operand ring shapes must match");
    TORCH_CHECK(ring_beta.dim() == 3 && ring_beta.size(0) == end_state.size(0) &&
                    ring_beta.size(1) == T && ring_beta.size(2) == H,
                "ring_beta must be [end_slots,T,H]");
    TORCH_CHECK(ring_k.scalar_type() == at::kFloat && ring_decay.scalar_type() == at::kFloat &&
                    ring_beta.scalar_type() == at::kFloat && ring_v.scalar_type() == at::kBFloat16,
                "ring_k/ring_decay/ring_beta must be float32; ring_v must be bfloat16");
    CompactStateArgs args{state.data_ptr(),
                          state_indices.data_ptr<int32_t>(),
                          end_state.data_ptr(),
                          end_indices.data_ptr<int32_t>(),
                          ring_k.data_ptr<float>(),
                          ring_decay.data_ptr<float>(),
                          reinterpret_cast<cutlass::bfloat16_t *>(ring_v.data_ptr()),
                          ring_beta.data_ptr<float>(),
                          int(state.size(0)),
                          int(end_state.size(0))};
    return {int(B), int(T), int(H), args};
}

void verify_compact(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor gate,
                    torch::Tensor beta, torch::Tensor A_log, torch::Tensor dt_bias, double scale,
                    double lower_bound, torch::Tensor state, torch::Tensor state_indices,
                    torch::Tensor end_state, torch::Tensor end_indices, torch::Tensor ring_k,
                    torch::Tensor ring_decay, torch::Tensor ring_v, torch::Tensor ring_beta,
                    torch::Tensor out) {
    auto const s = check_compact(state, state_indices, end_state, end_indices, ring_k, ring_decay,
                                 ring_v, ring_beta);
    c10::cuda::CUDAGuard device_guard(state.device());
    TORCH_CHECK(std::isfinite(float(scale)) && std::isfinite(float(lower_bound)) &&
                    lower_bound <= 0,
                "scale must be finite and lower_bound finite/nonpositive");
    auto stride = [&s](torch::Tensor const &x) {
        return s.T == 1 && s.B > 1 ? x.stride(0) : x.stride(1);
    };
    for (auto const &x : {q, k, v, gate, out}) {
        TORCH_CHECK(x.device() == state.device() && x.scalar_type() == at::kBFloat16 &&
                        x.dim() == 4 && x.size(0) == s.B && x.size(1) == s.T && x.size(2) == s.H &&
                        x.size(3) == 128,
                    "q/k/v/gate/out must be bfloat16 [B,T,H,128] on the state device");
        TORCH_CHECK(x.stride(3) == 1 && x.stride(2) == 128 && stride(x) >= s.H * 128 &&
                        (s.B == 1 || s.T == 1 || x.stride(0) == s.T * x.stride(1)) &&
                        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                    "compact inputs require dense [H,128] rows and a constant token stride");
    }
    for (auto const &x : {beta, A_log, dt_bias})
        TORCH_CHECK(x.device() == state.device() && x.is_contiguous(),
                    "beta/A_log/dt_bias must be contiguous on the state device");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16 && beta.dim() == 3 && beta.size(0) == s.B &&
                    beta.size(1) == s.T && beta.size(2) == s.H,
                "beta must be bfloat16 [B,T,H]");
    TORCH_CHECK(A_log.scalar_type() == at::kFloat && A_log.numel() == s.H &&
                    dt_bias.scalar_type() == at::kFloat && dt_bias.numel() == s.H * 128,
                "A_log/dt_bias must be float32 with H/H*128 elements");
    using BF16 = cutlass::bfloat16_t;
    int const compute_units =
        at::cuda::getDeviceProperties(state.get_device())->multiProcessorCount;
    bool const split = s.T > 1 && int64_t(s.B) * s.H >= compute_units;
    auto prepared = split ? torch::empty({int64_t(s.B) * s.T * s.H * kCompactOperandStride},
                                         state.options().dtype(at::kFloat))
                          : torch::Tensor();
    launch_verify_compact(
        static_cast<BF16 const *>(q.data_ptr()), static_cast<BF16 const *>(k.data_ptr()),
        static_cast<BF16 const *>(v.data_ptr()), static_cast<BF16 const *>(gate.data_ptr()),
        static_cast<BF16 const *>(beta.data_ptr()), A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(), s.args, static_cast<BF16 *>(out.data_ptr()),
        split ? prepared.data_ptr<float>() : nullptr, float(scale), float(lower_bound), s.B, s.T,
        s.H, stride(q), stride(k), stride(v), stride(gate), state.scalar_type() == at::kBFloat16,
        at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void commit_compact(torch::Tensor state, torch::Tensor state_indices, torch::Tensor end_state,
                    torch::Tensor end_indices, torch::Tensor ring_k, torch::Tensor ring_decay,
                    torch::Tensor ring_v, torch::Tensor ring_beta, torch::Tensor accepted) {
    auto const s = check_compact(state, state_indices, end_state, end_indices, ring_k, ring_decay,
                                 ring_v, ring_beta);
    c10::cuda::CUDAGuard device_guard(state.device());
    TORCH_CHECK(accepted.device() == state.device() && accepted.is_contiguous() &&
                    accepted.scalar_type() == at::kInt && accepted.dim() == 1 &&
                    accepted.numel() == s.B,
                "accepted must be contiguous int32 [B]");
    launch_commit_compact(s.args, accepted.data_ptr<int32_t>(), s.B, s.T, s.H,
                          state.scalar_type() == at::kBFloat16,
                          at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

void bind_verify_compact(pybind11::module_ &m) {
    m.attr("SUPPORTS_COMPACT_STATE") = py::bool_(true);
    m.def("verify_compact", &verify_compact, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("gate"), py::arg("beta"), py::arg("A_log"), py::arg("dt_bias"), py::arg("scale"),
          py::arg("lower_bound"), py::arg("state"), py::arg("state_indices"), py::arg("end_state"),
          py::arg("end_indices"), py::arg("ring_k"), py::arg("ring_decay"), py::arg("ring_v"),
          py::arg("ring_beta"), py::arg("out"));
    m.def("commit_compact", &commit_compact, py::arg("state"), py::arg("state_indices"),
          py::arg("end_state"), py::arg("end_indices"), py::arg("ring_k"), py::arg("ring_decay"),
          py::arg("ring_v"), py::arg("ring_beta"), py::arg("accepted"));
}
