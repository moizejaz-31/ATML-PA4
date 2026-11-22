"""P1 magnitude, P2 Wanda, P3 nonuniform Wanda and the 2:4 structured diagnostic.

All methods share one masking routine: for each output row, the k_row lowest-scoring weights are set to
zero. Rows of a projection receive either floor(Z/d_out) or floor(Z/d_out)+1 zeros so that a projection-level
integer target Z is met exactly (only needed for nonuniform allocations).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .layerwise import INPUT_GROUPS, LayerwiseCalibrator, gram_output_error, relative_error
from .model_utils import ELIGIBLE, eligible_linears, layers_of, proj_module


def row_counts(n_zero: int, d_out: int, d_in: int, device=None) -> torch.Tensor:
    assert 0 <= n_zero <= d_out * d_in
    base, extra = divmod(n_zero, d_out)
    k = torch.full((d_out,), base, dtype=torch.long, device=device)
    k[:extra] += 1
    return k


def row_mask(score: torch.Tensor, k_row: torch.Tensor) -> torch.Tensor:
    """Boolean keep-mask: per row, drop the k_row[i] smallest scores (ties broken by index order)."""
    rank = torch.argsort(torch.argsort(score.float(), dim=1, stable=True), dim=1)
    return rank >= k_row[:, None]


def nm_mask(score: torch.Tensor, n_keep: int = 2, m: int = 4) -> torch.Tensor:
    """N:M keep-mask over groups of M contiguous input channels (2:4 -> keep the 2 highest)."""
    d_out, d_in = score.shape
    assert d_in % m == 0
    s = score.float().view(d_out, d_in // m, m)
    idx = torch.topk(s, n_keep, dim=-1).indices
    mask = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, True)
    return mask.view(d_out, d_in)


def score_matrix(W: torch.Tensor, method: str, sq_norm: torch.Tensor | None = None) -> torch.Tensor:
    if method == "magnitude":
        return W.abs().float()
    if method == "wanda":
        return W.abs().float() * sq_norm.float().sqrt()[None, :]
    raise ValueError(method)


def score_summary(W: torch.Tensor, sq_norm: torch.Tensor, keep_mag: torch.Tensor, keep_wanda: torch.Tensor) -> dict:
    """Per-projection diagnostics for research question 1: score distributions and mask overlap."""
    mag = W.abs().float()
    xn = sq_norm.float().sqrt()
    wanda = mag * xn[None, :]
    q = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], device=W.device)
    sub = lambda t: t.flatten()[torch.randperm(t.numel(), device=t.device)[:2_000_000]]
    pruned_mag, pruned_wanda = ~keep_mag, ~keep_wanda
    inter = (pruned_mag & pruned_wanda).sum().item()
    union = (pruned_mag | pruned_wanda).sum().item()
    # Spearman correlation between |W| and Wanda score on a subsample
    s1, s2 = mag.flatten(), wanda.flatten()
    idx = torch.randperm(s1.numel(), device=W.device)[:500_000]
    r1 = torch.argsort(torch.argsort(s1[idx])).float()
    r2 = torch.argsort(torch.argsort(s2[idx])).float()
    spearman = float(torch.corrcoef(torch.stack([r1, r2]))[0, 1])
    return dict(mag_quantiles=torch.quantile(sub(mag), q).tolist(), act_norm_quantiles=torch.quantile(xn, q).tolist(),
                wanda_quantiles=torch.quantile(sub(wanda), q).tolist(), act_norm_max_over_median=float(xn.max() / xn.median()),
                mask_jaccard_pruned=inter / max(union, 1), spearman_mag_vs_wanda=spearman)


@torch.no_grad()
def prune_sequential(model, calib_ids, method: str, sparsity, nm: tuple | None = None, gram: bool = True,
                     batch_size: int = 8, collect_scores: bool = False, device: str = "cuda") -> dict:
    """Prune every eligible projection, walking blocks with calibration activations.

    method: 'magnitude' | 'wanda'. sparsity: float (uniform) or dict layer_idx -> s_l (nonuniform, block unit)
    or dict name -> exact zero count. nm=(2, 4) applies the N:M mask instead of the per-row mask.
    Inputs of block l+1 are the outputs of the already-pruned block l (Wanda reference implementation).
    Returns per-projection and per-block calibration errors plus timing.
    """
    t0 = time.perf_counter()
    cal = LayerwiseCalibrator(model, calib_ids, batch_size, device)
    inps = cal.block0_inputs()
    per_proj, per_block, score_stats = {}, [], {}
    for li, layer in enumerate(layers_of(model)):
        dense_out, stats = cal.block_stats(layer, inps, gram=gram)
        for p in ELIGIBLE:
            mod = proj_module(layer, p)
            W = mod.weight.data
            st = stats[INPUT_GROUPS[p]]
            score = score_matrix(W, method, st["sq"])
            name = f"layers.{li}.{p}"
            if nm is not None:
                keep = nm_mask(score, *nm)
            else:
                if isinstance(sparsity, dict):
                    z = sparsity[name] if name in sparsity else int(round(sparsity[li] * W.numel()))
                else:
                    z = int(round(sparsity * W.numel()))
                keep = row_mask(score, row_counts(z, W.shape[0], W.shape[1], W.device))
            if collect_scores:
                other = score_matrix(W, "magnitude" if method == "wanda" else "wanda", st["sq"])
                keep_other = nm_mask(other, *nm) if nm is not None else row_mask(other, row_counts(int((~keep).sum()), *W.shape, W.device))
                km, kw = (keep, keep_other) if method == "magnitude" else (keep_other, keep)
                score_stats[name] = score_summary(W, st["sq"], km, kw)
            W_new = W * keep
            rec = dict(zeros=int((~keep).sum()), numel=W.numel())
            if gram:
                rec.update({f"out_err_{k}": v for k, v in gram_output_error(W_new - W, W, st["gram"]).items()})
            mod.weight.data.copy_(W_new)
            per_proj[name] = rec
        del stats
        comp_out = cal.run_block(layer, inps)
        per_block.append(dict(layer=li, block_out_rel_err=relative_error(comp_out, dense_out)))
        inps = comp_out
        del dense_out
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    zeros = sum(r["zeros"] for r in per_proj.values())
    numel = sum(r["numel"] for r in per_proj.values())
    return dict(method=method, nm=nm, achieved_sparsity=zeros / numel, zeros=zeros, n_eligible=numel,
                per_projection=per_proj, per_block=per_block, score_stats=score_stats,
                compression_time_s=time.perf_counter() - t0)


@torch.no_grad()
def dense_input_norms(model, calib_ids, batch_size: int = 8, device: str = "cuda", gram: bool = False):
    """One dense pass: per block, per input group sum x^2 (and |x|, optional Gram) -- used by sensitivity probes."""
    cal = LayerwiseCalibrator(model, calib_ids, batch_size, device)
    inps = cal.block0_inputs()
    out = []
    for layer in layers_of(model):
        inps_next, stats = cal.block_stats(layer, inps, gram=gram)
        out.append({g: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in st.items()} for g, st in stats.items()})
        inps = inps_next
    return out


@torch.no_grad()
def prune_block_oneshot(layer, norms_l: dict, s: float, method: str = "wanda"):
    """Prune one block in place at sparsity s with precomputed dense norms (isolated-removal probe)."""
    for p in ELIGIBLE:
        mod = proj_module(layer, p)
        W = mod.weight.data
        sq = norms_l[INPUT_GROUPS[p]]["sq"].to(W.device)
        keep = row_mask(score_matrix(W, method, sq), row_counts(int(round(s * W.numel())), *W.shape, W.device))
        W.mul_(keep)


def exact_zero_targets(model, layer_sparsity: dict, total_sparsity: float) -> dict:
    """Integer zero counts per projection for a block-level allocation whose total equals
    round(total_sparsity * N_eligible) exactly; rounding residue is absorbed by the largest projections."""
    elig = eligible_linears(model)
    n_total = sum(m.weight.numel() for *_, m in elig)
    target = int(round(total_sparsity * n_total))
    z = {name: int(round(layer_sparsity[li] * m.weight.numel())) for name, li, _, m in elig}
    resid = target - sum(z.values())
    order = sorted(elig, key=lambda e: -e[3].weight.numel())
    i = 0
    while resid != 0:
        name, _, _, m = order[i % len(order)]
        step = int(np.sign(resid)) * min(abs(resid), m.weight.shape[0])
        z[name] = int(min(max(z[name] + step, 0), m.weight.numel()))
        resid = target - sum(z.values())
        i += 1
    assert sum(z.values()) == target
    return z


def sparse16_storage_bytes(n_eligible_per_matrix: list, nnz_per_matrix: list, n_other_params: int) -> dict:
    """Specified sparse format: 1 mask bit per eligible weight (byte aligned per matrix) + 16 bits per
    nonzero; non-eligible parameters stay dense FP16 (tied weights counted once)."""
    mask = sum((n + 7) // 8 for n in n_eligible_per_matrix)
    vals = 2 * sum(nnz_per_matrix)
    other = 2 * n_other_params
    return dict(mask_bytes=mask, value_bytes=vals, other_fp16_bytes=other, total_bytes=mask + vals + other)
