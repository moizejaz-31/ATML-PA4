"""Group-wise asymmetric weight-only PTQ shared by Q1 / Q2 / Q3, bit packing and packed-byte accounting.

Fixed conventions (identical for every quantized condition):
  * groups of G=128 consecutive input columns within each output row; d_in is padded to a multiple of G by
    replicating the last column (no padding occurs for Qwen2.5-1.5B: 1536 = 12*128, 8960 = 70*128);
  * asymmetric uniform quantizer, no clipping search: s = max(w_max - w_min, 1e-5) / (2^b - 1) rounded to FP16,
    z = clamp(round(-w_min / s), 0, 2^b - 1), q = clamp(round(w / s) + z, 0, 2^b - 1), w_hat = s (q - z);
    torch.round = round-half-to-even;
  * storage per group: FP16 scale (2 bytes) + b-bit zero-point; codes and zero-points are each packed as one
    little-endian bitstream per matrix (ceil(n*b/8) bytes) -- the layout of GPTQ/AWQ qweight/qzeros generalized to
    3 bits; non-eligible tensors (tied embedding once, norms, biases) stay FP16.
Numerical evaluation reconstructs FP16 weights (fake quantization); no low-bit kernel is used.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .model_utils import ELIGIBLE, eligible_linears, inventory


@dataclass(frozen=True)
class QuantSpec:
    group_size: int = 128
    scale_bytes: int = 2
    zero_bits_equal_weight_bits: bool = True
    rounding: str = "round-half-to-even"
    clipping: str = "none (min/max range)"

    def as_dict(self):
        return asdict(self)


SPEC = QuantSpec()


def _pad_cols(W: torch.Tensor, g: int):
    d_in = W.shape[1]
    pad = (-d_in) % g
    if pad:
        W = torch.cat([W, W[:, -1:].expand(-1, pad)], 1)
    return W, pad


@torch.no_grad()
def quantize_groupwise(W: torch.Tensor, bits: int, group_size: int = SPEC.group_size):
    """Returns codes (uint8, d_out x d_in_padded), scale (fp16, d_out x n_groups), zero (uint8) and pad."""
    Wp, pad = _pad_cols(W.float(), group_size)
    d_out, d_in = Wp.shape
    w = Wp.view(d_out, d_in // group_size, group_size)
    qmax = 2 ** bits - 1
    wmax, wmin = w.amax(-1), w.amin(-1)
    scale = ((wmax - wmin).clamp(min=1e-5) / qmax).half().float()
    zero = torch.round(-wmin / scale).clamp(0, qmax)
    q = (torch.round(w / scale[..., None]) + zero[..., None]).clamp(0, qmax)
    return q.to(torch.uint8).view(d_out, d_in), scale.half(), zero.to(torch.uint8), pad


@torch.no_grad()
def dequantize_groupwise(codes, scale, zero, pad: int, group_size: int = SPEC.group_size, dtype=torch.float16):
    d_out, d_in = codes.shape
    q = codes.float().view(d_out, d_in // group_size, group_size)
    w = (q - zero.float()[..., None]) * scale.float()[..., None]
    w = w.view(d_out, d_in)
    if pad:
        w = w[:, :-pad]
    return w.to(dtype)


@torch.no_grad()
def fake_quant(W: torch.Tensor, bits: int, group_size: int = SPEC.group_size) -> torch.Tensor:
    if bits >= 16:
        return W.clone()
    c, s, z, pad = quantize_groupwise(W, bits, group_size)
    return dequantize_groupwise(c, s, z, pad, group_size, W.dtype)


# ------------------------------------------------------------------------------------------- bit packing
@torch.no_grad()
def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned b-bit integers into a little-endian bitstream of uint8 (ceil(n*b/8) bytes)."""
    flat = codes.reshape(-1).to(torch.int32)
    n = flat.numel()
    shifts = torch.arange(bits, device=flat.device, dtype=torch.int32)
    bitplanes = ((flat[:, None] >> shifts) & 1).reshape(-1).to(torch.uint8)
    pad = (-bitplanes.numel()) % 8
    if pad:
        bitplanes = torch.cat([bitplanes, bitplanes.new_zeros(pad)])
    weights = (2 ** torch.arange(8, device=flat.device, dtype=torch.int32))
    packed = (bitplanes.view(-1, 8).to(torch.int32) * weights).sum(1).to(torch.uint8)
    assert packed.numel() == math.ceil(n * bits / 8)
    return packed


@torch.no_grad()
def unpack_bits(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32)
    bitplanes = ((packed.to(torch.int32)[:, None] >> shifts) & 1).reshape(-1)[: n * bits]
    weights = (2 ** torch.arange(bits, device=packed.device, dtype=torch.int32))
    return (bitplanes.view(n, bits) * weights).sum(1).to(torch.uint8)


def matrix_packed_bytes(d_out: int, d_in: int, bits: int, group_size: int = SPEC.group_size) -> dict:
    """Closed-form packed size of one projection (identical to len() of the serialized pack_bits streams)."""
    if bits >= 16:
        return dict(payload=2 * d_out * d_in, scales=0, zeros=0, total=2 * d_out * d_in)
    d_in_p = d_in + ((-d_in) % group_size)
    n_groups = d_out * d_in_p // group_size
    payload = math.ceil(d_out * d_in_p * bits / 8)
    scales = SPEC.scale_bytes * n_groups
    zeros = math.ceil(n_groups * bits / 8)
    return dict(payload=payload, scales=scales, zeros=zeros, total=payload + scales + zeros)


@torch.no_grad()
def serialize_matrix(W: torch.Tensor, bits: int) -> dict:
    """Actually pack one matrix and return byte counts (used to verify the closed form)."""
    c, s, z, pad = quantize_groupwise(W, bits)
    pw, pz = pack_bits(c, bits), pack_bits(z, bits)
    assert torch.equal(unpack_bits(pw, bits, c.numel()).view_as(c), c)
    return dict(payload=pw.numel(), scales=s.numel() * 2, zeros=pz.numel(), total=pw.numel() + s.numel() * 2 + pz.numel())


def model_packed_bytes(model, bit_assignment: dict, explicit_scale_channels: int = 0) -> dict:
    """Total serialized model bytes: packed eligible projections + FP16 non-eligible params (tied counted once)
    + FP16 compensating AWQ channel scales that could not be folded (``explicit_scale_channels`` elements)."""
    inv = inventory(model)
    elig = {name: m for name, *_, m in eligible_linears(model)}
    payload = scales = zeros = 0
    for name, m in elig.items():
        b = bit_assignment[name]
        r = matrix_packed_bytes(*m.weight.shape, b)
        payload += r["payload"]
        scales += r["scales"]
        zeros += r["zeros"]
    other = 2 * inv["n_other_params_unique"]
    comp = 2 * explicit_scale_channels
    elig_total = payload + scales + zeros
    return dict(eligible_payload=payload, eligible_scales=scales, eligible_zeros=zeros, eligible_total=elig_total,
                awq_explicit_scale_bytes=comp, other_fp16_bytes=other, total_bytes=elig_total + comp + other,
                bits_per_eligible_weight=8 * (elig_total + comp) / inv["n_eligible_weights"],
                nominal_bits=sum(bit_assignment[n] * elig[n].weight.numel() for n in elig) / inv["n_eligible_weights"])


def dense_fp16_bytes(model) -> int:
    inv = inventory(model)
    return 2 * inv["n_total_params_unique"]


@torch.no_grad()
def quantize_model_(model, bit_assignment: dict):
    """In-place fake quantization (dequantized FP16 reconstruction) of every eligible projection."""
    for name, _, _, m in eligible_linears(model):
        m.weight.data.copy_(fake_quant(m.weight.data, bit_assignment[name]))


def uniform_assignment(model, bits: int) -> dict:
    return {name: bits for name, *_ in eligible_linears(model)}


# ------------------------------------------------------------------- explicit (non-foldable) input scales
_HOOKS = "_pa4_input_scale_hook"


def attach_input_scale(mod, s: torch.Tensor):
    """Runtime compensation x -> x / s for a projection whose inverse scale cannot be folded."""
    remove_input_scale(mod)
    mod.register_buffer("pa4_inv_scale", (1.0 / s).to(mod.weight.dtype), persistent=False)
    h = mod.register_forward_pre_hook(lambda m, a: (a[0] * m.pa4_inv_scale,) + tuple(a[1:]))
    setattr(mod, _HOOKS, h)


def remove_input_scale(mod):
    h = getattr(mod, _HOOKS, None)
    if h is not None:
        h.remove()
        delattr(mod, _HOOKS)
    if "pa4_inv_scale" in mod._buffers:
        del mod._buffers["pa4_inv_scale"]


def remove_input_scale_hooks(model):
    for *_, m in eligible_linears(model):
        remove_input_scale(m)
