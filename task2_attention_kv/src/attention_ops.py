"""Part A: exact causal attention three ways -- dense (explicit L x L scores), tiled online-softmax (Python,
autograd backward), and PyTorch SDPA with a recorded / forced backend.

All functions take q, k, v of shape (B, H, L, d) and return (B, H, L, d) in the input dtype.
Optional ``mask``: bool (L, L) or broadcastable, True = may attend (used for edge-case tests).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def dense_attention(q, k, v, causal: bool = True, mask: torch.Tensor | None = None):
    """Explicit scores, causal mask, numerically stable softmax (FP32), multiply by V.
    Rows with no admissible key return 0 (instead of NaN)."""
    L, S, d = q.shape[-2], k.shape[-2], q.shape[-1]
    s = torch.matmul(q * (1.0 / math.sqrt(d)), k.transpose(-1, -2)).float()
    allowed = torch.ones(L, S, dtype=torch.bool, device=q.device)
    if causal:
        allowed = torch.tril(allowed, diagonal=S - L)
    if mask is not None:
        allowed = allowed & mask
    s = s.masked_fill(~allowed, float("-inf"))
    m = s.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(s - m)
    denom = e.sum(-1, keepdim=True)
    p = e / denom.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.matmul(p.to(v.dtype), v)


def tiled_attention(q, k, v, block_k: int = 128, causal: bool = True):
    """Exact blockwise forward with online softmax; never materialises the L x L score matrix.

    Key/value tiles are visited in order; for causal attention, queries before a tile's first key are skipped
    (they cannot see it). Running max m, normaliser l and unnormalised accumulator o are updated per tile:
        m' = max(m, max s);  l' = e^(m-m') l + sum e^(s-m');  o' = e^(m-m') o + sum e^(s-m') V;  A = o'/l'.
    Intermediate storage per step is O(L * block_k). The backward pass is ordinary autograd through this
    Python loop and therefore stores every tile's intermediates (not a fused FlashAttention backward).
    """
    B, H, L, d = q.shape
    S = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    offset = S - L                      # query i (0-based) sits at absolute key position i + offset
    m = torch.full((B, H, L), float("-inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros((B, H, L), device=q.device, dtype=torch.float32)
    o = torch.zeros((B, H, L, v.shape[-1]), device=q.device, dtype=torch.float32)
    for ks in range(0, S, block_k):
        ke = min(ks + block_k, S)
        qs = max(ks - offset, 0) if causal else 0     # earlier queries cannot see this tile
        if qs >= L:
            break
        qi = q[:, :, qs:] * scale
        s = torch.matmul(qi, k[:, :, ks:ke].transpose(-1, -2)).float()          # (B, H, L-qs, Bk)
        if causal:
            qpos = torch.arange(qs, L, device=q.device)[:, None] + offset
            kpos = torch.arange(ks, ke, device=q.device)[None, :]
            s = s.masked_fill(kpos > qpos, float("-inf"))
        m_old, l_old, o_old = m[:, :, qs:], l[:, :, qs:], o[:, :, qs:]
        m_new = torch.maximum(m_old, s.amax(-1))
        m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
        alpha = torch.exp(m_old - m_safe)                                          # 0 when m_old = -inf
        p = torch.exp(s - m_safe[..., None])
        l_new = alpha * l_old + p.sum(-1)
        o_new = alpha[..., None] * o_old + torch.matmul(p.to(v.dtype), v[:, :, ks:ke]).float()
        m = torch.cat([m[:, :, :qs], m_new], 2)
        l = torch.cat([l[:, :, :qs], l_new], 2)
        o = torch.cat([o[:, :, :qs], o_new], 2)
    return (o / l.clamp_min(torch.finfo(torch.float32).tiny)[..., None]).to(q.dtype)


# ----------------------------------------------------------------------------------------------- SDPA
BACKENDS = ("default", "flash", "efficient", "math", "cudnn")
_OPNAMES = {"_scaled_dot_product_flash_attention": "flash", "_scaled_dot_product_efficient_attention": "efficient",
            "_scaled_dot_product_cudnn_attention": "cudnn", "_scaled_dot_product_attention_math": "math",
            "_scaled_dot_product_fused_attention_overrideable": "overrideable", "_flash_attention_forward": "flash",
            "_efficient_attention_forward": "efficient"}


def _backend_enum(name):
    from torch.nn.attention import SDPBackend
    return {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH, "cudnn": SDPBackend.CUDNN_ATTENTION}[name]


def sdpa_attention(q, k, v, backend: str = "default", causal: bool = True, mask=None):
    if backend == "default":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)
    from torch.nn.attention import sdpa_kernel
    with sdpa_kernel(_backend_enum(backend)):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)


def detect_sdpa_backend(q, k, v, backend: str = "default", grad: bool = False) -> str:
    """Name of the SDPA kernel actually dispatched, read from the profiler's aten op names."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = sdpa_attention(q, k, v, backend)
        if grad:
            out.float().sum().backward()
    names = {e.name.replace("aten::", "") for e in prof.events()}
    found = sorted({lab for op, lab in _OPNAMES.items() if op in names})
    return "+".join(found) if found else "unknown"


def get_impl(name: str, block_k: int = 128):
    if name == "dense":
        return dense_attention
    if name == "tiled":
        return lambda q, k, v: tiled_attention(q, k, v, block_k=block_k)
    if name.startswith("sdpa_"):
        b = name.split("_", 1)[1]
        return lambda q, k, v: sdpa_attention(q, k, v, backend=b)
    raise ValueError(name)


def theoretical_intermediate_bytes(impl: str, B, H, L, d, block_k=128, bytes_score=4) -> int:
    """Score-sized temporaries of the forward pass (FP32 scores): dense L*L, tiled L*block_k."""
    if impl == "dense":
        return B * H * L * L * bytes_score
    if impl == "tiled":
        return B * H * L * block_k * bytes_score + 3 * B * H * L * 4 + B * H * L * d * 4
    return 0
