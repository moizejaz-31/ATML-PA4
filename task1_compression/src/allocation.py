"""Sensitivity measures and allocation rules for P3 (sparsity) and Q3 (bit widths).

P3 (allocation unit = full Transformer block, all 7 eligible projections share s_l):
  * sensitivity A -- OWL layer outlier ratio (Yin et al. 2024): fraction of Wanda scores above M x block mean.
    rule A: s_l = S - (2 lambda * norm(LOD_l) - mean), clipped to [0.10, 0.85] with mean re-centred.
  * sensitivity B -- isolated-removal probe: Delta NLL on calibration windows when only block l is Wanda-pruned
    at s in a grid (others dense). rule B: exact dynamic programme minimising sum_l DeltaNLL_l(s_l) on a 0.025 grid
    subject to mean_l s_l = S (equal block sizes => equal zero count), then shrink towards uniform by beta.
Q3 (allocation unit = AWQ scale group: qkv / o / gate_up / down of each block):
  * sensitivity C -- isolated-quantization probe: Delta NLL on calibration windows when only that unit is quantized.
  * sensitivity D -- calibration output error of the AWQ-scaled quantized unit (Gram proxy, HAWQ-style).
    rule: multiple-choice knapsack over bits {2,3,4,8} minimising sum cost_u(b_u) subject to packed bytes <= budget,
    solved by Lagrangian relaxation plus greedy fill.
"""
from __future__ import annotations

import numpy as np
import torch

from .layerwise import INPUT_GROUPS
from .model_utils import ELIGIBLE, layers_of, proj_module

S_MIN, S_MAX, S_STEP = 0.10, 0.85, 0.025


# ------------------------------------------------------------------------------------------------ OWL
@torch.no_grad()
def owl_outlier_ratios(model, dense_norms: list, M: float = 5.0) -> np.ndarray:
    lod = []
    for li, layer in enumerate(layers_of(model)):
        cnt = tot = 0
        scores = []
        for p in ELIGIBLE:
            W = proj_module(layer, p).weight.data
            xn = dense_norms[li][INPUT_GROUPS[p]]["sq"].to(W.device).float().sqrt()
            scores.append((W.abs().float() * xn[None, :]).flatten())
        s = torch.cat(scores)
        lod.append(float((s > M * s.mean()).float().mean()))
    return np.array(lod)


def _project_mean(s: np.ndarray, target: float, lo=S_MIN, hi=S_MAX, iters: int = 100) -> np.ndarray:
    """Clip to [lo, hi] while keeping the mean exactly at target (bisection on an additive shift)."""
    a, b = -1.0, 1.0
    for _ in range(iters):
        mid = (a + b) / 2
        if np.clip(s + mid, lo, hi).mean() < target:
            a = mid
        else:
            b = mid
    out = np.clip(s + (a + b) / 2, lo, hi)
    return out


def owl_allocation(lod: np.ndarray, S: float, lam: float) -> np.ndarray:
    norm = (lod - lod.min()) / max(lod.max() - lod.min(), 1e-12)
    keep = norm * 2 * lam
    keep = keep - keep.mean() + (1 - S)
    return _project_mean(1 - keep, S)


# ----------------------------------------------------------------------------------- isolated-block probe
def probe_grid():
    return [0.10, 0.30, 0.50, 0.70, 0.85]


def interp_curve(grid, values, s):
    """Piecewise-linear Delta NLL curve with Delta NLL(0) = 0."""
    g = np.concatenate([[0.0], grid])
    v = np.concatenate([[0.0], values])
    return np.interp(s, g, v)


def dp_allocation(curves: np.ndarray, grid, S: float, lo=S_MIN, hi=S_MAX, step=S_STEP) -> np.ndarray:
    """Exact minimiser of sum_l c_l(s_l) over s_l in {lo, lo+step, ..., hi} with mean s_l = S."""
    levels = np.round(np.arange(lo, hi + 1e-9, step), 6)
    L, K = curves.shape[0], len(levels)
    cost = np.stack([interp_curve(grid, curves[l], levels) for l in range(L)])  # (L, K)
    total_units = int(round((S * L - lo * L) / step))
    assert abs(total_units * step + lo * L - S * L) < 1e-6, "S must lie on the allocation grid"
    INF = 1e18
    dp = np.full((L + 1, total_units + 1), INF)
    arg = np.zeros((L + 1, total_units + 1), dtype=int)
    dp[0, 0] = 0.0
    for l in range(L):
        for u in range(total_units + 1):
            if dp[l, u] >= INF:
                continue
            for k in range(K):
                v = u + k
                if v > total_units:
                    break
                c = dp[l, u] + cost[l, k]
                if c < dp[l + 1, v]:
                    dp[l + 1, v] = c
                    arg[l + 1, v] = k
    alloc, u = np.zeros(L), total_units
    for l in range(L, 0, -1):
        k = arg[l, u]
        alloc[l - 1] = levels[k]
        u -= k
    assert abs(alloc.mean() - S) < 1e-9
    return alloc


def shrink(alloc: np.ndarray, S: float, beta: float) -> np.ndarray:
    return (1 - beta) * S + beta * alloc


# --------------------------------------------------------------------------------------- bit allocation
BITS = (2, 3, 4, 8)


def monotone_costs(costs: dict) -> dict:
    """Enforce cost(b) >= cost(b') for b < b' (removes probe noise inversions)."""
    out, best = {}, -np.inf
    for b in sorted(costs, reverse=True):
        best = max(best, costs[b])
        out[b] = best
    return out


def allocate_bits(units: dict, budget: float, base_bytes: float = 0.0, bits=BITS) -> dict:
    """units: {name: dict(cost={b: c}, bytes={b: n})}. Minimise sum cost s.t. base + sum bytes <= budget."""
    names = list(units)
    cost = {u: monotone_costs({b: units[u]["cost"][b] for b in bits}) for u in names}
    nbytes = {u: units[u]["bytes"] for u in names}

    def choose(lam):
        sel = {u: min(bits, key=lambda b: cost[u][b] + lam * nbytes[u][b]) for u in names}
        return sel, base_bytes + sum(nbytes[u][sel[u]] for u in names)

    lo, hi = 0.0, 1.0
    while choose(hi)[1] > budget:
        hi *= 4
        if hi > 1e30:
            raise ValueError("budget infeasible even at minimum bits")
    for _ in range(200):
        mid = (lo + hi) / 2
        if choose(mid)[1] > budget:
            lo = mid
        else:
            hi = mid
    sel, used = choose(hi)
    # greedy fill: best cost reduction per extra byte that still fits
    improved = True
    while improved:
        improved = False
        best = None
        for u in names:
            for b in bits:
                if b <= sel[u]:
                    continue
                extra = nbytes[u][b] - nbytes[u][sel[u]]
                gain = cost[u][sel[u]] - cost[u][b]
                if gain > 0 and used + extra <= budget:
                    r = gain / max(extra, 1)
                    if best is None or r > best[0]:
                        best = (r, u, b, extra)
        if best:
            _, u, b, extra = best
            sel[u] = b
            used += extra
            improved = True
    return dict(assignment=sel, total_bytes=used, lagrange=hi)
