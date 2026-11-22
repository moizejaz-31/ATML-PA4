"""MQAR training (fixed protocol) and evaluation by regime, load K and source-query distance."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from shared.src.io import load_json, save_json
from shared.src.seed import set_seed

from .mqar_data import DIST_BINS, batch_to_torch, training_stream
from .mqar_model import VALUE_OFFSET, MQARModel, n_params

PROTOCOL = dict(optimizer="AdamW", lr=8e-4, weight_decay=0.01, batch_size=32, updates=4500, warmup=200,
                final_lr_frac=0.15, grad_clip=1.0, eval_at=[500, 1000, 1500, 2000, 3000, 4500], seeds=[6304, 6305, 6306],
                loss="cross-entropy over 64 value IDs at the K marked query positions only")


def lr_at(step: int, p=PROTOCOL) -> float:
    """Linear warmup to the peak over 200 updates, then linear decay to 15% of the peak at the last update."""
    if step < p["warmup"]:
        return p["lr"] * (step + 1) / p["warmup"]
    frac = (step - p["warmup"]) / max(p["updates"] - 1 - p["warmup"], 1)
    return p["lr"] * (1 - (1 - p["final_lr_frac"]) * frac)


def query_logits(model, tokens, qpos):
    logits = model(tokens)                                   # (B, T, 64)
    return logits.gather(1, qpos[..., None].expand(-1, -1, logits.shape[-1]))  # (B, K, 64)


@torch.no_grad()
def evaluate(model, data: dict, device="cuda", batch_size=256) -> dict:
    """data: {(K, regime): arrays}. Returns per-cell accuracy, loss, distance-bin accuracy and per-query results."""
    model.eval()
    out, per_query = {}, {}
    for (k, reg), arr in data.items():
        n = arr["tokens"].shape[0]
        correct, losses = [], []
        for i in range(0, n, batch_size):
            idx = np.arange(i, min(i + batch_size, n))
            tok, qp, ans = batch_to_torch(arr, idx, device)
            lg = query_logits(model, tok, qp).float()
            tgt = ans - VALUE_OFFSET
            losses.append(F.cross_entropy(lg.flatten(0, 1), tgt.flatten(), reduction="none").view_as(tgt).cpu())
            correct.append((lg.argmax(-1) == tgt).cpu())
        c = torch.cat(correct).numpy()
        L = torch.cat(losses).numpy()
        d = arr["dist"]
        cell = dict(acc=float(c.mean()), loss=float(L.mean()), n_queries=int(c.size), n_seqs=int(n),
                    chance_label=1 / 64, chance_present=1 / k)
        for lo, hi in DIST_BINS:
            sel = (d >= lo) & (d < hi)
            cell[f"acc_d{lo}_{hi}"] = float(c[sel].mean()) if sel.any() else None
            cell[f"n_d{lo}_{hi}"] = int(sel.sum())
        out[f"K{k}_{reg}"] = cell
        per_query[f"K{k}_{reg}"] = dict(correct=c.astype(np.uint8).tolist(), dist=d.tolist())
    model.train()
    return dict(cells=out, per_query=per_query)


def run_name(kind, regime, k, seed):
    return f"{kind}_{regime}_K{k}_s{seed}"


def train_run(kind: str, k: int, regime: str, seed: int, dev: dict, out_dir: Path, cache_dir: Path,
              device: str = "cuda", p=PROTOCOL, log_every: int = 50) -> dict:
    """Train one model; resumable at run granularity (skips if the final checkpoint exists)."""
    name = run_name(kind, regime, k, seed)
    ck = out_dir / "checkpoints" / f"{name}.pt"
    hist_path = out_dir / "histories" / f"{name}.json"
    if ck.exists() and hist_path.exists():
        return load_json(hist_path)
    ck.parent.mkdir(parents=True, exist_ok=True)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = MQARModel(kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    n_total = p["updates"] * p["batch_size"]
    stream = training_stream(k, regime, n_total, cache_dir)
    order = np.random.default_rng(seed).permutation(n_total)
    dev_cells = {key: v for key, v in dev.items() if key[0] == k}     # dev curves at the training load
    hist = dict(name=name, kind=kind, K=k, regime=regime, seed=seed, n_params=n_params(model), protocol=p,
                train_loss=[], dev=[], lr=[])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0, run_loss, run_n = time.perf_counter(), 0.0, 0
    model.train()
    for step in range(p["updates"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, p)
        idx = order[step * p["batch_size"]:(step + 1) * p["batch_size"]]
        tok, qp, ans = batch_to_torch(stream, idx, device)
        lg = query_logits(model, tok, qp)
        loss = F.cross_entropy(lg.flatten(0, 1).float(), (ans - VALUE_OFFSET).flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
        opt.step()
        run_loss += float(loss)
        run_n += 1
        if (step + 1) % log_every == 0:
            hist["train_loss"].append(dict(step=step + 1, loss=run_loss / run_n, grad_norm=float(gn)))
            hist["lr"].append(dict(step=step + 1, lr=lr_at(step, p)))
            run_loss, run_n = 0.0, 0
        if (step + 1) in p["eval_at"]:
            ev = evaluate(model, dev_cells, device)["cells"]
            hist["dev"].append(dict(step=step + 1, **{c: v["acc"] for c, v in ev.items()},
                                    **{f"{c}_loss": v["loss"] for c, v in ev.items()}))
            print(f"  {name} step {step+1}: loss {hist['train_loss'][-1]['loss']:.3f} dev {[round(v['acc'], 3) for v in ev.values()]}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    hist["train_time_s"] = time.perf_counter() - t0
    hist["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    torch.save(model.state_dict(), ck)
    save_json(hist_path, hist)
    return hist


def load_model(out_dir: Path, kind, regime, k, seed, device="cuda"):
    m = MQARModel(kind).to(device)
    m.load_state_dict(torch.load(out_dir / "checkpoints" / f"{run_name(kind, regime, k, seed)}.pt", map_location=device))
    return m.eval()


def test_eval(out_dir: Path, test: dict, kind, regime, k, seed, eval_kind: str | None = None, device="cuda") -> dict:
    """Evaluate a trained model on every (K, regime) test cell; eval_kind swaps the operator (mask control)."""
    tag = run_name(kind, regime, k, seed) + (f"@{eval_kind}" if eval_kind else "")
    f = out_dir / "test" / f"{tag}.json"
    if f.exists():
        return load_json(f)
    m = load_model(out_dir, kind, regime, k, seed, device)
    if eval_kind:
        m.set_attention(eval_kind)
    ev = evaluate(m, test, device)
    f.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_dir / "test" / f"{tag}_queries.json.gz", "wt") as g:
        json.dump(ev["per_query"], g)
    res = dict(tag=tag, kind=kind, eval_kind=eval_kind or kind, train_regime=regime, train_K=k, seed=seed, cells=ev["cells"])
    save_json(f, res)
    return res
