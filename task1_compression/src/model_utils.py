"""Model loading, eligible-projection enumeration and weight snapshots for Task 1."""
from __future__ import annotations

import torch
import torch.nn as nn

from shared.src.paths import model_revision

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
ELIGIBLE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def load_model(name: str = MODEL_NAME, device: str = "cuda", dtype=torch.float16, attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rev = model_revision("model_base") if name == MODEL_NAME else None
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, torch_dtype=dtype, attn_implementation=attn)
    model.to(device).eval()
    model.config.use_cache = False
    model._pa4_revision = rev
    return model, tok


def layers_of(model):
    return model.model.layers


def proj_module(layer, proj: str) -> nn.Linear:
    return getattr(layer.self_attn if proj in ATTN else layer.mlp, proj)


def eligible_linears(model):
    """Ordered list of (name, layer_idx, proj, module) for the 7 x n_layers eligible projections."""
    out = []
    for i, layer in enumerate(layers_of(model)):
        for p in ELIGIBLE:
            out.append((f"layers.{i}.{p}", i, p, proj_module(layer, p)))
    return out


def unique_parameters(model):
    """Parameters deduplicated by storage (tied embeddings / lm_head counted once)."""
    seen, out = set(), []
    for n, p in model.named_parameters(remove_duplicate=False):
        key = p.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        out.append((n, p))
    return out


def inventory(model) -> dict:
    """Counts used by every storage calculation; also the enumeration test of manual step 1."""
    elig = eligible_linears(model)
    elig_ptrs = {m.weight.data_ptr() for *_, m in elig}
    uniq = unique_parameters(model)
    n_elig = sum(m.weight.numel() for *_, m in elig)
    n_other = sum(p.numel() for _, p in uniq if p.data_ptr() not in elig_ptrs)
    tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    shapes = {p: tuple(proj_module(layers_of(model)[0], p).weight.shape) for p in ELIGIBLE}
    n_layers = len(layers_of(model))
    assert len(elig) == 7 * n_layers and len(elig_ptrs) == len(elig), "eligible projections must be distinct"
    assert all(isinstance(m, nn.Linear) for *_, m in elig)
    return dict(n_layers=n_layers, n_eligible_matrices=len(elig), n_eligible_weights=n_elig,
                n_other_params_unique=n_other, n_total_params_unique=n_elig + n_other,
                n_params_with_duplicates=sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False)),
                tied_embeddings=bool(tied), shapes=shapes,
                biases=[p for p in ELIGIBLE if proj_module(layers_of(model)[0], p).bias is not None])


class WeightSnapshot:
    """CPU copy of every (deduplicated) parameter so each condition starts from the unchanged FP16 weights."""

    def __init__(self, model):
        self.state = {n: p.detach().to("cpu", copy=True) for n, p in unique_parameters(model)}

    @torch.no_grad()
    def restore(self, model):
        from .quant import remove_input_scale_hooks
        remove_input_scale_hooks(model)
        params = dict(unique_parameters(model))
        for n, t in self.state.items():
            params[n].data.copy_(t, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


@torch.no_grad()
def count_nonzeros(model) -> dict:
    per = {}
    for name, _, _, m in eligible_linears(model):
        per[name] = int((m.weight != 0).sum().item())
    total = sum(m.weight.numel() for *_, m in eligible_linears(model))
    nnz = sum(per.values())
    return dict(nnz_eligible=nnz, n_eligible=total, nonzero_fraction=nnz / total, sparsity=1 - nnz / total, per_matrix=per)
