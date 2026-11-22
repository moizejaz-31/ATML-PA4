"""Block-by-block calibration runner (the pattern used by the Wanda / AWQ reference code).

Captures the inputs of decoder block 0 for the 128 calibration windows once, then walks the
blocks: collect per-projection input statistics, compress the block, recompute its outputs and
feed them to the next block.
"""
from __future__ import annotations

import torch

from .model_utils import ELIGIBLE, layers_of, proj_module

# Projections that read the same input tensor share one statistic.
INPUT_GROUPS = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "o_proj": "o",
                "gate_proj": "gate_up", "up_proj": "gate_up", "down_proj": "down"}
GROUP_READER = {"qkv": "q_proj", "o": "o_proj", "gate_up": "gate_proj", "down": "down_proj"}


class _Stop(Exception):
    pass


class LayerwiseCalibrator:
    def __init__(self, model, calib_ids: torch.Tensor, batch_size: int = 8, device: str = "cuda"):
        assert calib_ids.shape[0] % batch_size == 0
        self.model, self.ids, self.bs, self.device = model, calib_ids, batch_size, device
        self.kwargs = None

    @torch.no_grad()
    def block0_inputs(self) -> torch.Tensor:
        layer0 = layers_of(self.model)[0]
        store = {}

        def hook(module, args, kwargs):
            store["hs"] = args[0] if args else kwargs["hidden_states"]
            store["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
            raise _Stop

        h = layer0.register_forward_pre_hook(hook, with_kwargs=True)
        chunks = []
        try:
            for i in range(0, self.ids.shape[0], self.bs):
                try:
                    self.model(input_ids=self.ids[i:i + self.bs].to(self.device), use_cache=False)
                except _Stop:
                    pass
                chunks.append(store["hs"].detach())
                if self.kwargs is None:
                    kw = store["kw"]
                    kw["use_cache"] = False
                    kw.pop("past_key_values", None)
                    kw.pop("past_key_value", None)
                    self.kwargs = kw
        finally:
            h.remove()
        return torch.cat(chunks, 0)

    @torch.no_grad()
    def run_block(self, layer, inps: torch.Tensor) -> torch.Tensor:
        outs = torch.empty_like(inps)
        for i in range(0, inps.shape[0], self.bs):
            o = layer(inps[i:i + self.bs], **self.kwargs)
            outs[i:i + self.bs] = o[0] if isinstance(o, tuple) else o
        return outs

    @torch.no_grad()
    def block_stats(self, layer, inps: torch.Tensor, gram: bool = False, keep_inputs: dict | None = None,
                    max_keep_tokens: int = 16384, seed: int = 6304):
        """Run the block once, accumulating for each input group the per-channel sum of x^2 (Wanda norms),
        sum |x| (AWQ activation scale) and optionally the full Gram X^T X (exact output-error proxy).

        ``keep_inputs`` (dict) receives a fixed random token subsample of each group's input (AWQ search).
        Returns (outputs, stats) where stats[group] = dict(sq, abs, n, gram?).
        """
        stats, handles = {}, []
        n_tokens_total = inps.shape[0] * inps.shape[1]
        g = torch.Generator().manual_seed(seed)
        keep_idx = torch.randperm(n_tokens_total, generator=g)[:max_keep_tokens].sort().values if keep_inputs is not None else None
        offset = {"v": 0}

        def make_hook(group):
            def hook(mod, args):
                x = args[0].reshape(-1, args[0].shape[-1]).float()
                st = stats.setdefault(group, dict(sq=0.0, abs=0.0, n=0, gram=None))
                st["sq"] = st["sq"] + (x * x).sum(0)
                st["abs"] = st["abs"] + x.abs().sum(0)
                st["n"] += x.shape[0]
                if gram:
                    st["gram"] = (x.T @ x) if st["gram"] is None else st["gram"].addmm_(x.T, x)
                if keep_inputs is not None:
                    lo = offset["v"]
                    sel = keep_idx[(keep_idx >= lo) & (keep_idx < lo + x.shape[0])] - lo
                    keep_inputs.setdefault(group, []).append(args[0].reshape(-1, args[0].shape[-1])[sel.to(x.device)].detach())
            return hook

        for grp, proj in GROUP_READER.items():
            handles.append(proj_module(layer, proj).register_forward_pre_hook(make_hook(grp)))
        outs = torch.empty_like(inps)
        try:
            for i in range(0, inps.shape[0], self.bs):
                o = layer(inps[i:i + self.bs], **self.kwargs)
                outs[i:i + self.bs] = o[0] if isinstance(o, tuple) else o
                offset["v"] += inps[i:i + self.bs].shape[0] * inps.shape[1]
        finally:
            for h in handles:
                h.remove()
        if keep_inputs is not None:
            for k in list(keep_inputs):
                if isinstance(keep_inputs[k], list):
                    keep_inputs[k] = torch.cat(keep_inputs[k], 0)
        return outs, stats


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b||^2 / ||b||^2 computed in float32 chunks."""
    num = den = 0.0
    for i in range(a.shape[0]):
        d = (a[i].float() - b[i].float())
        num += float((d * d).sum())
        den += float((b[i].float() ** 2).sum())
    return num / max(den, 1e-12)


def gram_output_error(dW: torch.Tensor, W: torch.Tensor, gram: torch.Tensor) -> dict:
    """Exact calibration output error of a projection from the input Gram matrix.

    abs = sum_t ||dW x_t||^2 = tr(dW G dW^T); rel = abs / tr(W G W^T).
    """
    dW = dW.float()
    W = W.float()
    num = float(((dW @ gram) * dW).sum())
    den = float(((W @ gram) * W).sum())
    return dict(abs=num, rel=num / max(den, 1e-12))


def proj_names():
    return list(ELIGIBLE)
