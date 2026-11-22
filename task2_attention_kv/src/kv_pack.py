"""C5: real packed KV-cache encoding / decoding (INT4 / INT2, group 64, FP16 scale + FP16 offset) and timing.

Layouts: 'token' (token-wise K and V) and 'kivi' (channel-wise K over 64-token groups, token-wise V).
Codes are packed 2 per byte (INT4) or 4 per byte (INT2) into uint8 tensors; serialized bytes are the sum of the
packed tensors' nbytes and match bytes_quant() for complete groups.
"""
from __future__ import annotations

import torch

GROUP = 64


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    per = 8 // bits
    c = codes.reshape(-1, per).to(torch.uint8)
    out = torch.zeros(c.shape[0], dtype=torch.uint8, device=c.device)
    for j in range(per):
        out |= c[:, j] << (bits * j)
    return out


def _unpack(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    per = 8 // bits
    mask = (1 << bits) - 1
    cols = [(packed >> (bits * j)) & mask for j in range(per)]
    return torch.stack(cols, 1).reshape(-1)[:n]


def _q(g: torch.Tensor, dim: int, bits: int):
    mn, mx = g.amin(dim, keepdim=True), g.amax(dim, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((mx - mn) / qmax).clamp_min(1e-5).half()
    off = mn.half()
    q = torch.clamp(torch.round((g - off.float()) / scale.float()), 0, qmax).to(torch.uint8)
    return q, scale, off


def encode(x: torch.Tensor, axis: str, bits: int) -> dict:
    """x: (B, H, L, d) FP16 with L % 64 == 0. axis 'feature' (token-wise groups) or 'token' (channel-wise groups)."""
    B, H, L, d = x.shape
    if axis == "feature":
        g = x.float().view(B, H, L, d // GROUP, GROUP)
        q, s, o = _q(g, -1, bits)
    else:
        g = x.float().view(B, H, L // GROUP, GROUP, d)
        q, s, o = _q(g, 3, bits)
    return dict(packed=_pack(q, bits), scale=s, offset=o, shape=(B, H, L, d), axis=axis, bits=bits)


def decode(e: dict) -> torch.Tensor:
    B, H, L, d = e["shape"]
    n = B * H * L * d
    q = _unpack(e["packed"], e["bits"], n).float()
    if e["axis"] == "feature":
        q = q.view(B, H, L, d // GROUP, GROUP)
    else:
        q = q.view(B, H, L // GROUP, GROUP, d)
    return (q * e["scale"].float() + e["offset"].float()).view(B, H, L, d).half()


def nbytes(e: dict) -> int:
    return e["packed"].numel() + e["scale"].numel() * 2 + e["offset"].numel() * 2


def encode_layer(k, v, layout: str, bits: int):
    return encode(k, "token" if layout == "kivi" else "feature", bits), encode(v, "feature", bits)


def decode_layer(ek, ev):
    return decode(ek), decode(ev)


@torch.no_grad()
def benchmark_pack(model, batch: int, L: int, layout: str = "kivi", bits: int = 4, reps: int = 10, warmup: int = 3,
                   device="cuda", seed: int = 6304) -> dict:
    """Pack/unpack time for a full 28-layer cache, serialized bytes, and one decode step with
    (a) an FP16 cache and (b) unpack-then-decode, at matched shapes. Times in ms."""
    from shared.src.timing import benchmark, peak_memory, reset_peak
    from .kv_cache import build_cache, kv_dims
    dims = kv_dims(model)
    g = torch.Generator(device=device).manual_seed(seed)
    kv = [(torch.randn(batch, dims["n_kv_heads"], L, dims["head_dim"], device=device, generator=g, dtype=torch.float16),
           torch.randn(batch, dims["n_kv_heads"], L, dims["head_dim"], device=device, generator=g, dtype=torch.float16))
          for _ in range(dims["n_layers"])]
    enc = [encode_layer(k, v, layout, bits) for k, v in kv]
    ser = sum(nbytes(a) + nbytes(b) for a, b in enc)
    fp16 = sum(k.numel() * 2 + v.numel() * 2 for k, v in kv)
    t_pack = benchmark(lambda: [encode_layer(k, v, layout, bits) for k, v in kv], warmup, reps)
    t_unpack = benchmark(lambda: [decode_layer(a, b) for a, b in enc], warmup, reps)
    tokens = torch.randint(100, 1000, (batch, 1), device=device, generator=g)
    pos = torch.full((batch, 1), L, device=device)

    def step_fp16():
        c = build_cache([(k.clone(), v.clone()) for k, v in kv])   # clone so appends never alias
        return model(input_ids=tokens, past_key_values=c, position_ids=pos, use_cache=True)

    def step_unpack():
        c = build_cache([decode_layer(a, b) for a, b in enc])
        return model(input_ids=tokens, past_key_values=c, position_ids=pos, use_cache=True)

    def clone_only():
        return build_cache([(k.clone(), v.clone()) for k, v in kv])

    t_clone = benchmark(clone_only, warmup, reps)
    reset_peak()
    t_fp16 = benchmark(step_fp16, warmup, reps)
    mem_fp16 = peak_memory()
    reset_peak()
    t_unp = benchmark(step_unpack, warmup, reps)
    mem_unp = peak_memory()
    return dict(batch=batch, L=L, layout=layout, bits=bits, reps=reps, warmup=warmup, unit="ms",
                serialized_bytes=ser, fp16_bytes=fp16, ratio=ser / fp16,
                pack_ms=t_pack["mean"], pack_std=t_pack["std"], unpack_ms=t_unpack["mean"], unpack_std=t_unpack["std"],
                cache_clone_ms=t_clone["mean"], decode_step_fp16_ms=t_fp16["mean"], decode_step_fp16_std=t_fp16["std"],
                decode_step_unpack_ms=t_unp["mean"], decode_step_unpack_std=t_unp["std"],
                peak_alloc_fp16_mb=mem_fp16["peak_allocated_mb"], peak_alloc_unpack_mb=mem_unp["peak_allocated_mb"])
