"""Part B: small decoder-only Transformer for MQAR with a pluggable attention operator.

Architecture fixed for every method: learned token embeddings (256 vocab) + sinusoidal positions, 2 pre-LayerNorm
blocks, hidden 128, 4 heads x 32, GELU MLP of width 512, final LayerNorm, linear head over the 64 value IDs
(128..191). Only the attention operator changes:
  dense    softmax attention, causal (SDPA)
  window   softmax, query i attends keys [i-63, i] (explicit boolean mask + SDPA = correctness implementation)
  dilated  heads 0-1 local window 64, heads 2-3 dilated: keys i - 4m, m = 0..63 (explicit mask)
  linear   causal ELU+1 linear attention, chunked recurrence over the (d x d) state S and normaliser z
  delta    B3 intervention: delta-rule (selective, erase-before-write) update of the same d x d state,
           S_t = S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T with beta_t = sigmoid(w^T x_t) per head and
           L2-normalised ELU+1 features; read-out o_t = S_t^T q_t (no normaliser)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES, VALUE_OFFSET = 64, 128


def sinusoidal(T: int, d: int) -> torch.Tensor:
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


def local_mask(T, W=64, device=None):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (j > i - W)


def dilated_mask(T, step=4, n=64, device=None):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    off = i - j
    return (off >= 0) & (off % step == 0) & (off < step * n)


def phi(x):
    return F.elu(x) + 1.0


def causal_linear_attention(q, k, v, chunk: int = 64, eps: float = 1e-6):
    """Exact causal linear attention with feature-mapped q, k (B, H, T, d): chunked state recurrence.
    Persistent state per head: S (d x d_v) and z (d)."""
    B, H, T, d = q.shape
    S = q.new_zeros(B, H, d, v.shape[-1])
    z = q.new_zeros(B, H, d)
    outs = []
    for c in range(0, T, chunk):
        qc, kc, vc = q[:, :, c:c + chunk], k[:, :, c:c + chunk], v[:, :, c:c + chunk]
        C = qc.shape[2]
        intra = torch.tril(qc @ kc.transpose(-1, -2))
        num = qc @ S + intra @ vc
        den = (qc * z[:, :, None, :]).sum(-1) + intra.sum(-1)
        outs.append(num / (den[..., None] + eps))
        S = S + kc.transpose(-1, -2) @ vc
        z = z + kc.sum(2)
    return torch.cat(outs, 2)


def causal_linear_attention_prefix(q, k, v, eps: float = 1e-6):
    """Reference implementation that materialises the linear prefix tensor (B, H, T, d, d_v)."""
    S = torch.cumsum(k[..., :, None] * v[..., None, :], dim=2)
    z = torch.cumsum(k, dim=2)
    num = torch.einsum("bhtd,bhtde->bhte", q, S)
    den = (q * z).sum(-1, keepdim=True)
    return num / (den + eps)


def delta_rule_attention(q, k, v, beta):
    """Sequential delta-rule recurrence. q, k L2-normalised features (B,H,T,d); beta (B,H,T) in (0,1)."""
    B, H, T, d = q.shape
    S = q.new_zeros(B, H, d, v.shape[-1])
    outs = []
    for t in range(T):
        kt, vt, qt, bt = k[:, :, t], v[:, :, t], q[:, :, t], beta[:, :, t, None]
        v_old = torch.einsum("bhd,bhde->bhe", kt, S)
        S = S + torch.einsum("bhd,bhe->bhde", kt, bt * (vt - v_old))
        outs.append(torch.einsum("bhd,bhde->bhe", qt, S))
    return torch.stack(outs, 2)


def banded_attention(q, k, v, offsets: torch.Tensor):
    """Gather-based sparse softmax attention: query i attends keys i - o for o in offsets (o >= 0, i - o >= 0).
    Work and temporaries are O(T * len(offsets)) instead of O(T^2); no dense mask is built."""
    B, H, T, d = q.shape
    idx = torch.arange(T, device=q.device)[:, None] - offsets[None, :].to(q.device)
    valid = idx >= 0
    idx = idx.clamp(min=0)
    kg, vg = k[:, :, idx], v[:, :, idx]                               # (B, H, T, n, d)
    s = torch.einsum("bhtd,bhtnd->bhtn", q, kg) / math.sqrt(d)
    p = torch.softmax(s.float().masked_fill(~valid, float("-inf")), -1).to(v.dtype)
    return torch.einsum("bhtn,bhtnd->bhtd", p, vg)


class Attention(nn.Module):
    def __init__(self, d=128, n_heads=4, kind="dense", window=64, dil_step=4, dil_n=64):
        super().__init__()
        self.h, self.dh, self.kind = n_heads, d // n_heads, kind
        self.window, self.dil_step, self.dil_n = window, dil_step, dil_n
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        if kind == "delta":
            self.beta = nn.Linear(d, n_heads)
        self._mask_cache = {}

    def mask(self, T, device):
        key = (self.kind, T, str(device))
        if key not in self._mask_cache:
            if self.kind == "window":
                m = local_mask(T, self.window, device)[None]
            elif self.kind == "dilated":
                loc = local_mask(T, self.window, device)
                dil = dilated_mask(T, self.dil_step, self.dil_n, device)
                m = torch.stack([loc, loc, dil, dil])            # heads 0-1 local, heads 2-3 dilated
            else:
                m = None
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.kind == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.kind in ("window", "dilated"):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask(T, x.device))
        elif self.kind == "window_banded":
            o = banded_attention(q, k, v, torch.arange(self.window))
        elif self.kind == "dilated_banded":
            loc = banded_attention(q[:, :2], k[:, :2], v[:, :2], torch.arange(self.window))
            dil = banded_attention(q[:, 2:], k[:, 2:], v[:, 2:], torch.arange(self.dil_n) * self.dil_step)
            o = torch.cat([loc, dil], 1)
        elif self.kind == "linear":
            o = causal_linear_attention(phi(q), phi(k), v)
        elif self.kind == "linear_prefix":
            o = causal_linear_attention_prefix(phi(q), phi(k), v)
        elif self.kind == "delta":
            qf, kf = F.normalize(phi(q), dim=-1), F.normalize(phi(k), dim=-1)
            beta = torch.sigmoid(self.beta(x)).transpose(1, 2)    # (B, H, T)
            o = delta_rule_attention(qf, kf, v, beta)
        else:
            raise ValueError(self.kind)
        return self.out(o.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d, n_heads, kind, mlp_mult=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_heads, kind)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MQARModel(nn.Module):
    def __init__(self, kind="dense", vocab=256, d=128, n_heads=4, n_layers=2, max_T=8192):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(vocab, d)
        self.register_buffer("pe", sinusoidal(max_T, d), persistent=False)
        self.blocks = nn.ModuleList([Block(d, n_heads, kind) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_CLASSES)

    def set_attention(self, kind):
        """Swap the operator without touching weights (test-time mask control). Only softmax kinds share weights."""
        for b in self.blocks:
            assert not (kind == "delta" or b.attn.kind == "delta")
            b.attn.kind = kind
            b.attn._mask_cache = {}

    def forward_embeddings(self, h):
        for b in self.blocks:
            h = b(h)
        return self.ln_f(h)

    def forward(self, tokens):
        T = tokens.shape[1]
        h = self.emb(tokens) + self.pe[:T]
        return self.head(self.forward_embeddings(h))          # (B, T, 64) value logits at every position


@torch.no_grad()
def inference_sweep(kinds, Ts=(256, 1024, 4096, 8192), B: int = 4, reps: int = 10, warmup: int = 3,
                    dtype=torch.float16, device="cuda", seed: int = 6304) -> list:
    """Forward-only cost of the 2-block stack on fixed random embeddings (identical B, H, d for all operators).
    Isolates kernel execution from learnability. Implementation notes per kind are returned with the rows."""
    from shared.src.timing import benchmark, peak_memory, reset_peak
    notes = {"dense": "SDPA is_causal (fused kernel)", "window": "SDPA + explicit dense bool mask (T x T)",
             "dilated": "SDPA + explicit dense bool mask (H x T x T)", "window_banded": "gather O(T*64)",
             "dilated_banded": "gather O(T*64) per head group", "linear": "chunked recurrence, O(d^2) state",
             "linear_prefix": "materialises (B,H,T,d,d) prefix tensor", "delta": "sequential Python recurrence"}
    rows = []
    for kind in kinds:
        torch.manual_seed(seed)
        m = MQARModel(kind if kind != "delta" else "delta", max_T=max(Ts)).to(device=device, dtype=dtype).eval()
        for T in Ts:
            g = torch.Generator(device=device).manual_seed(seed)
            h = torch.randn(B, T, 128, device=device, dtype=dtype, generator=g)
            row = dict(kind=kind, T=T, B=B, dtype=str(dtype), impl=notes.get(kind, ""))
            try:
                reset_peak()
                r = benchmark(lambda: m.forward_embeddings(h), warmup=warmup if kind != "delta" else 1,
                              reps=reps if kind != "delta" else 3)
                row.update(ms=r["mean"], std=r["std"], reps=r["reps"], **peak_memory())
            except torch.cuda.OutOfMemoryError as e:
                row.update(ms=None, error="OOM")
            except RuntimeError as e:
                row.update(ms=None, error=str(e)[:200])
            rows.append(row)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del m
    return rows


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------ theoretical resource costs
def attended_pairs(kind: str, T: int, W: int = 64, step: int = 4, n: int = 64, n_heads: int = 4) -> dict:
    """Candidate (query, key) pairs per layer summed over heads, and the maximum per query."""
    i = torch.arange(T)
    dense = (i + 1)
    local = torch.clamp(i + 1, max=W)
    dil = torch.clamp(i // step + 1, max=n)
    if kind in ("dense", "dense@test_window"):
        per_head = [dense] * n_heads if kind == "dense" else [local] * n_heads
    elif kind == "window":
        per_head = [local] * n_heads
    elif kind == "dilated":
        per_head = [local, local, dil, dil]
    else:
        return dict(pairs_per_layer=None, max_positions_per_query=None, note="recurrent state; no explicit pairs")
    return dict(pairs_per_layer=int(sum(int(p.sum()) for p in per_head)),
                max_positions_per_query=int(max(int(p.max()) for p in per_head)))


def state_bytes(kind: str, T: int, d_model=128, n_heads=4, bytes_per=4) -> dict:
    """Persistent per-layer inference memory at sequence length T (FP32 elements)."""
    dh = d_model // n_heads
    if kind == "dense":
        return dict(kv_cache_bytes=2 * T * d_model * bytes_per)
    if kind == "window":
        return dict(kv_cache_bytes=2 * min(T, 64) * d_model * bytes_per)
    if kind == "dilated":
        return dict(kv_cache_bytes=2 * (2 * min(T, 64) * dh + 2 * min(T, 253) * dh) * bytes_per,
                    note="dilated heads must keep keys up to 252 back")
    if kind == "linear":
        return dict(state_bytes=n_heads * (dh * dh + dh) * bytes_per)
    if kind == "delta":
        return dict(state_bytes=n_heads * dh * dh * bytes_per)
    raise ValueError(kind)


def receptive_field(kind: str, n_layers: int = 2, W: int = 64) -> str:
    return {"dense": "all previous tokens", "window": f"{n_layers * (W - 1)} positions",
            "dilated": "<= 252 per dilated hop (sparse coverage)", "linear": "all previous (compressed state)",
            "delta": "all previous (compressed state)"}[kind]
