import torch
import flash_kda_C as _C

_decode_raw = getattr(_C, "decode", None)
_verify_raw = getattr(_C, "verify", None)
_verify_strided_raw = getattr(_C, "verify_strided", None)
_fwd_raw = _C.fwd
get_workspace_size = _C.get_workspace_size
SUPPORTS_FP32_STATE_ACCUMULATION = bool(
    getattr(_C, "SUPPORTS_FP32_STATE_ACCUMULATION", False)
)
# Query the loaded binary, not the Python source revision: a source checkout
# can otherwise advertise a newer ABI while importing an older installed .so.
DECODE_SUPPORTS_LOWER_BOUND = _decode_raw is not None and bool(
    getattr(_C, "DECODE_SUPPORTS_LOWER_BOUND", False)
)
supports_decode_safe_gate = DECODE_SUPPORTS_LOWER_BOUND


def fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    *,
    state_accum_dtype=None,
):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[B, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32. Shape ``[B, H, V, K]`` for batched mode, or ``[N, H, V, K]``
            for varlen mode. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state. Same dtype/shape rules as ``initial_state``.
        state_accum_dtype (torch.dtype, optional): ``torch.float32`` selects
            FP32 resident recurrence on PPU independently of the stored state
            dtype. ``None`` preserves the default recurrence and dispatch.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int64,
            shape ``[N+1]``. When provided, ``B`` must be 1.


    Notes:
        * PPU accepts packed Q/K/V token rows with dense head/channel axes;
          other inputs remain contiguous. CUDA requires contiguous Q/K/V.
        * Currently requires ``K = V = 128``.
        * All input tensors must be on one CUDA device and have the dtypes
          listed above.
    """
    B, T_seq, H = q.shape[0], q.shape[1], q.shape[2]
    T_total = B * T_seq
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B

    if state_accum_dtype not in (None, torch.float32):
        raise ValueError("state_accum_dtype must be None or torch.float32")
    if state_accum_dtype is torch.float32 and not SUPPORTS_FP32_STATE_ACCUMULATION:
        raise RuntimeError(
            "Loaded flash_kda_C has no FP32 resident accumulation support"
        )
    workspace_size = (
        _C.get_workspace_size_fp32(T_total, H, N)
        if state_accum_dtype is torch.float32
        else get_workspace_size(T_total, H, N)
    )
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)

    forward = _C.fwd_fp32 if state_accum_dtype is torch.float32 else _fwd_raw
    forward(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        out,
        workspace,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )


def decode(
    mixed_qkv,
    gate,
    beta,
    A_log,
    dt_bias,
    scale,
    state,
    state_indices,
    out,
    lower_bound=None,
):
    """Packed T=1 KDA decode with an in-place recurrent-state update.

    ``mixed_qkv`` is bf16 ``[B, 3*H*128]``. ``gate`` and ``beta`` are raw
    pre-activation bf16 tensors of shape ``[B, H*128]`` and ``[B, H]``.
    ``A_log`` and ``dt_bias`` stay fp32. ``state`` may be fp32 (the default
    serving contract) or bf16 (SGLang's ``SGLANG_MAMBA_SSM_DTYPE=bfloat16``
    mode). ``state_indices`` selects one pool slot per request and may contain
    ``-1`` for padded rows. Non-negative indices must be unique within
    a decode call. Out-of-range indices produce zero output and leave the pool
    unchanged. ``out`` is bf16 ``[B, 1, H, 128]``. When
    ``lower_bound`` is provided, the kernel uses Kimi K3's bounded safe gate;
    otherwise it preserves the standard unbounded KDA gate.
    """
    if _decode_raw is None:
        raise RuntimeError("FlashKDA packed decode is currently available on PPU only")
    args = (
        mixed_qkv,
        gate,
        beta,
        A_log,
        dt_bias,
        float(scale),
        state,
        state_indices,
        out,
    )
    if lower_bound is None:
        # Preserve the nine-argument ABI for older unbounded decode binaries.
        _decode_raw(*args)
    else:
        if not DECODE_SUPPORTS_LOWER_BOUND:
            raise RuntimeError(
                "Loaded flash_kda_C does not advertise bounded decode support; "
                "rebuild/reinstall FlashKDA from the matching source."
            )
        _decode_raw(*args, float(lower_bound))
    return out


def verify(
    q,
    k,
    v,
    gate,
    beta,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    state,
    state_indices,
    intermediate_states,
    intermediate_state_indices,
    out,
):
    """Dense, topk=1 speculative KDA verification on PPU.

    ``state`` is read only.  The post-state for every draft step is stored in
    ``intermediate_states[intermediate_state_indices, step]`` so SGLang's
    accepted-length logic can commit or roll back without an in-place update.
    Inputs and ``out`` use ``[B, T, H, 128]`` layouts. Checkpoint destinations
    must be unique within a call. A negative source slot pads the request; a
    negative checkpoint slot suppresses checkpoint writes. Out-of-range slots
    produce zero output without reading or writing the pools. Tree and ragged
    verification require a separate implementation.
    """
    if _verify_raw is None:
        raise RuntimeError(
            "FlashKDA dense chain verify is currently available on PPU only"
        )
    _verify_raw(
        q,
        k,
        v,
        gate,
        beta,
        A_log,
        dt_bias,
        float(scale),
        float(lower_bound),
        state,
        state_indices,
        intermediate_states,
        intermediate_state_indices,
        out,
    )
    return out


def verify_strided(
    q,
    k,
    v,
    gate,
    beta,
    A_log,
    dt_bias,
    scale,
    lower_bound,
    state,
    state_indices,
    intermediate_states,
    intermediate_state_indices,
    out,
):
    """Dense chain verify without materializing fused-QKV subviews.

    The last two dimensions of q/k/v/gate must remain contiguous
    ``[H, 128]`` rows. Their token-row strides may be larger than ``H * 128``;
    this is the layout produced when SGLang splits a fused QKV projection.
    """
    if _verify_strided_raw is None:
        raise RuntimeError("this FlashKDA build has no strided verify ABI")
    _verify_strided_raw(
        q,
        k,
        v,
        gate,
        beta,
        A_log,
        dt_bias,
        float(scale),
        float(lower_bound),
        state,
        state_indices,
        intermediate_states,
        intermediate_state_indices,
        out,
    )
    return out


SUPPORTS_COMPACT_STATE = bool(getattr(_C, "SUPPORTS_COMPACT_STATE", False))


def verify_compact(
    q,
    k,
    v,
    gate,
    beta,
    A_log,
    dt_bias,
    state,
    state_indices,
    end_state,
    end_indices,
    out,
    ring_k,
    ring_decay,
    ring_v,
    ring_beta,
    lower_bound,
    *,
    scale=None,
):
    """Verify a linear draft chain using FP32 resident state and a compact cache.

    States use BF16 or FP32 ``[slots,H,128,128]`` storage. The operands use
    ``[end_slots,T,H,128]``: FP32 keys/decays, BF16 values, and FP32 beta
    ``[end_slots,T,H]``. Only the terminal matrix is published.
    Source slots remain read only; live source and destination slots must be
    disjoint, with unique destinations. Negative source slots pad requests.
    Negative destinations suppress cache/state writes while retaining outputs.
    Out-of-range slots pad requests. All buffers are caller owned and graph safe.
    """
    if not SUPPORTS_COMPACT_STATE:
        raise RuntimeError("Loaded flash_kda_C has no compact state support")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    _C.verify_compact(
        q,
        k,
        v,
        gate,
        beta,
        A_log,
        dt_bias,
        float(scale),
        float(lower_bound),
        state,
        state_indices,
        end_state,
        end_indices,
        ring_k,
        ring_decay,
        ring_v,
        ring_beta,
        out,
    )
    return out


def commit_compact(
    state,
    state_indices,
    end_state,
    end_indices,
    ring_k,
    ring_decay,
    ring_v,
    ring_beta,
    num_accept_tokens,
):
    """Publish the accepted prefix in place in ``end_state``.

    ``num_accept_tokens`` is int32 ``[B]``. Zero (or negative) copies the original state;
    partial prefixes replay cached operands with the same FP32 arithmetic as
    verify; counts at least T retain its terminal state. Caller index ownership
    is the same as in :func:`verify_compact`. No host reads or scratch allocation.
    """
    if not SUPPORTS_COMPACT_STATE:
        raise RuntimeError("Loaded flash_kda_C has no compact state support")
    _C.commit_compact(
        state,
        state_indices,
        end_state,
        end_indices,
        ring_k,
        ring_decay,
        ring_v,
        ring_beta,
        num_accept_tokens,
    )
    return end_state
