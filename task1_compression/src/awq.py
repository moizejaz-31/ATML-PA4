"""AWQ-style activation-aware scaling (Q2) and the shared machinery used by Q3.

What is reproduced from Lin et al. (AWQ): per-input-channel scales s = mean|x|^alpha (normalised by
sqrt(max*min)), a grid search over alpha in {0, 1/20, ..., 19/20} that minimises output error of the quantized
projections on calibration activations, and folding of the inverse scale into the preceding RMSNorm (q/k/v and
gate/up) or into up_proj rows (down_proj). Simplifications: (i) the objective is the summed squared output error of
the scaled+quantized linear group computed exactly from the calibration Gram matrix, not the output of the whole
attention / MLP module; (ii) no weight-clipping search (keeps the quantizer identical to Q1); (iii) o_proj cannot be
folded into v_proj because of GQA (256 -> 1536), so it either keeps an explicit FP16 input scale (default, bytes
counted) or is left unscaled like the reference code (o_mode='skip'); (iv) fake-quantized FP16 execution, not the
AWQ inference engine / kernels.
"""
from __future__ import annotations

import time

import torch

from .layerwise import LayerwiseCalibrator
from .model_utils import layers_of, proj_module
from .quant import attach_input_scale, fake_quant

GROUPS = {"qkv": ("q_proj", "k_proj", "v_proj"), "o": ("o_proj",), "gate_up": ("gate_proj", "up_proj"), "down": ("down_proj",)}
PROJ_TO_GROUP = {p: g for g, ps in GROUPS.items() for p in ps}
GRID = [i / 20 for i in range(20)]


def _scales_from_ratio(x_absmean: torch.Tensor, ratio: float) -> torch.Tensor:
    s = x_absmean.float().pow(ratio).clamp(min=1e-4)
    return s / (s.max() * s.min()).sqrt()


def _gram_err(dW: torch.Tensor, G: torch.Tensor) -> float:
    return float(((dW @ G) * dW).sum())


@torch.no_grad()
def search_awq_scales(model, calib_ids, bits_list=(2, 3, 4, 8), grid=GRID, batch_size: int = 8,
                      o_mode: str = "explicit", device: str = "cuda") -> dict:
    """Search AWQ scales for every (block, group) and every bit width in one calibration pass.

    The model is not modified; next-block inputs are the unchanged block outputs (equal, up to FP16 rounding,
    to the outputs of the rescaled-but-unquantized block used by the reference implementation).
    Returns dict(scales={'l.group': {bits: s}}, log={'l.group': {bits: dict(ratio, err_by_ratio, err_identity)}}).
    """
    t0 = time.perf_counter()
    cal = LayerwiseCalibrator(model, calib_ids, batch_size, device)
    inps = cal.block0_inputs()
    scales, log = {}, {}
    for li, layer in enumerate(layers_of(model)):
        outs, stats = cal.block_stats(layer, inps, gram=True)
        for g, projs in GROUPS.items():
            key = f"{li}.{g}"
            st = stats[g]
            x_abs = st["abs"] / st["n"]
            G = st["gram"]
            W = torch.cat([proj_module(layer, p).weight.data for p in projs], 0).float()
            scales[key], log[key] = {}, {}
            for b in bits_list:
                errs = []
                ratios = [0.0] if (g == "o" and o_mode == "skip") else grid
                for r in ratios:
                    s = _scales_from_ratio(x_abs, r)
                    Wq = fake_quant((W * s[None, :]).half(), b).float() / s[None, :]
                    errs.append(_gram_err(Wq - W, G))
                best = min(range(len(errs)), key=errs.__getitem__)
                scales[key][b] = _scales_from_ratio(x_abs, ratios[best]).cpu()
                log[key][b] = dict(ratio=ratios[best], err=errs[best], err_identity=errs[0],
                                   err_by_ratio=errs, n_tokens=st["n"])
        del stats
        inps = outs
    return dict(scales=scales, log=log, grid=list(grid), o_mode=o_mode, bits=list(bits_list),
                search_time_s=time.perf_counter() - t0)


@torch.no_grad()
def apply_awq_(model, scales: dict, unit_bits: dict, o_mode: str = "explicit", quantize: bool = True) -> dict:
    """Fold the scales chosen for each unit's bit width, then (optionally) fake-quantize every projection
    with the Q1 quantizer at its unit's bit width. unit_bits: {'l.group': bits}.
    Returns the number of FP16 compensating scale elements that had to be stored explicitly."""
    explicit = 0
    for li, layer in enumerate(layers_of(model)):
        attn, mlp = layer.self_attn, layer.mlp
        # q/k/v: fold 1/s into input_layernorm
        s = scales[f"{li}.qkv"][unit_bits[f"{li}.qkv"]].to(layer.input_layernorm.weight.device)
        layer.input_layernorm.weight.data.div_(s.to(layer.input_layernorm.weight.dtype))
        for p in GROUPS["qkv"]:
            getattr(attn, p).weight.data.mul_(s[None, :].to(attn.q_proj.weight.dtype))
        # gate/up: fold 1/s into post_attention_layernorm
        s = scales[f"{li}.gate_up"][unit_bits[f"{li}.gate_up"]].to(layer.post_attention_layernorm.weight.device)
        layer.post_attention_layernorm.weight.data.div_(s.to(layer.post_attention_layernorm.weight.dtype))
        for p in GROUPS["gate_up"]:
            getattr(mlp, p).weight.data.mul_(s[None, :].to(mlp.gate_proj.weight.dtype))
        # down: fold 1/s into up_proj output rows (down input = act(gate x) * up x)
        s = scales[f"{li}.down"][unit_bits[f"{li}.down"]].to(mlp.up_proj.weight.device)
        mlp.up_proj.weight.data.div_(s[:, None].to(mlp.up_proj.weight.dtype))
        if mlp.up_proj.bias is not None:
            mlp.up_proj.bias.data.div_(s.to(mlp.up_proj.bias.dtype))
        mlp.down_proj.weight.data.mul_(s[None, :].to(mlp.down_proj.weight.dtype))
        # o: GQA prevents folding into v_proj -> explicit FP16 input scale (or skip)
        if o_mode == "explicit":
            s = scales[f"{li}.o"][unit_bits[f"{li}.o"]].to(attn.o_proj.weight.device)
            attn.o_proj.weight.data.mul_(s[None, :].to(attn.o_proj.weight.dtype))
            attach_input_scale(attn.o_proj, s)
            explicit += s.numel()
    if quantize:
        for li, layer in enumerate(layers_of(model)):
            for g, projs in GROUPS.items():
                b = unit_bits[f"{li}.{g}"]
                for p in projs:
                    m = proj_module(layer, p)
                    m.weight.data.copy_(fake_quant(m.weight.data, b))
    return dict(explicit_scale_channels=explicit)


def _unit_tensors(layer, group: str) -> list:
    attn, mlp = layer.self_attn, layer.mlp
    if group == "qkv":
        return [layer.input_layernorm.weight, attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]
    if group == "o":
        return [attn.o_proj.weight]
    if group == "gate_up":
        return [layer.post_attention_layernorm.weight, mlp.gate_proj.weight, mlp.up_proj.weight]
    if group == "down":
        t = [mlp.up_proj.weight, mlp.down_proj.weight]
        return t + ([mlp.up_proj.bias] if mlp.up_proj.bias is not None else [])
    raise ValueError(group)


def backup_unit(layer, group: str) -> list:
    return [t.detach().clone() for t in _unit_tensors(layer, group)]


@torch.no_grad()
def restore_unit(layer, group: str, backup: list):
    from .quant import remove_input_scale
    for t, b in zip(_unit_tensors(layer, group), backup):
        t.data.copy_(b)
    if group == "o":
        remove_input_scale(layer.self_attn.o_proj)


@torch.no_grad()
def quantize_unit_(layer, group: str, s: torch.Tensor, bits: int, o_mode: str = "explicit"):
    """Isolated-quantization probe: fold this unit's AWQ scale and quantize only its own projections
    (for 'down' the inverse scale is folded into up_proj, which stays FP16 in the probe)."""
    attn, mlp = layer.self_attn, layer.mlp
    dt = attn.q_proj.weight.dtype
    s = s.to(attn.q_proj.weight.device)
    if group in ("qkv", "gate_up"):
        norm = layer.input_layernorm if group == "qkv" else layer.post_attention_layernorm
        norm.weight.data.div_(s.to(norm.weight.dtype))
        for p in GROUPS[group]:
            m = proj_module(layer, p)
            m.weight.data.mul_(s[None, :].to(dt))
    elif group == "down":
        mlp.up_proj.weight.data.div_(s[:, None].to(dt))
        if mlp.up_proj.bias is not None:
            mlp.up_proj.bias.data.div_(s.to(dt))
        mlp.down_proj.weight.data.mul_(s[None, :].to(dt))
    elif group == "o" and o_mode == "explicit":
        attn.o_proj.weight.data.mul_(s[None, :].to(dt))
        attach_input_scale(attn.o_proj, s)
    for p in GROUPS[group]:
        m = proj_module(layer, p)
        m.weight.data.copy_(fake_quant(m.weight.data, bits))


def unit_bits_uniform(n_layers: int, bits: int) -> dict:
    return {f"{li}.{g}": bits for li in range(n_layers) for g in GROUPS}


def projection_bits(unit_bits: dict, n_layers: int) -> dict:
    """Expand {'l.group': bits} to {'layers.l.proj': bits} for storage accounting."""
    return {f"layers.{li}.{p}": unit_bits[f"{li}.{g}"] for li in range(n_layers) for g, ps in GROUPS.items() for p in ps}
