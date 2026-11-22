"""MQAR data: the supplied deterministic generator (training stream) and the frozen dev / test files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from shared.src.io import read_jsonl
from shared.src.paths import DATA_DIR, DATA_REPO

T = 256
DIST_BINS = [(0, 32), (32, 64), (64, 128), (128, 256)]


def generator():
    """Import the course's mqar_generator (make_example / validate) from the data repository."""
    scripts = str(DATA_REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import mqar_generator
    return mqar_generator


def verify_generator(n_per_cell: int = 64) -> dict:
    """Invariant checks of the supplied generator (and regime constraints) before training."""
    g = generator()
    stats = {}
    for regime in ("near", "far", "mixed"):
        for k in (4, 8, 16):
            dists = []
            for i in range(n_per_cell):
                ex = g.make_example("train", k, regime, i)
                g.validate(ex)
                toks = ex["tokens"]
                vals = set(ex["answer_ids"])
                fill = [toks[p] for p in range(T) if p >= 2 * k and p not in ex["query_positions"]]
                assert all(2 <= t < 64 for t in fill) and not vals & set(fill)
                dists += ex["distances"]
            d = np.array(dists)
            stats[f"{regime}_K{k}"] = dict(min=int(d.min()), max=int(d.max()), frac_div4=float((d % 4 == 0).mean()))
            if regime == "near":
                assert d.max() < 64
            if regime == "far":
                assert d.min() >= 128 and d.max() <= 250 and len(set(d.tolist())) > 1
    return stats


def to_arrays(examples, k: int):
    n = len(examples)
    toks = np.zeros((n, T), dtype=np.uint8)
    qpos = np.zeros((n, k), dtype=np.int16)
    ans = np.zeros((n, k), dtype=np.uint8)
    src = np.zeros((n, k), dtype=np.int16)
    for i, ex in enumerate(examples):
        toks[i] = ex["tokens"]
        qpos[i] = ex["query_positions"]
        ans[i] = ex["answer_ids"]
        src[i] = ex["source_positions"]
    return dict(tokens=toks, qpos=qpos, ans=ans, src=src, dist=(qpos - src).astype(np.int16))


def training_stream(k: int, regime: str, n: int, cache_dir: Path) -> dict:
    """First n examples of the supplied 'train' stream for (K, regime), cached as .npz."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"mqar_train_K{k}_{regime}_{n}.npz"
    if f.exists():
        return dict(np.load(f))
    g = generator()
    arr = to_arrays([g.make_example("train", k, regime, i) for i in range(n)], k)
    np.savez_compressed(f, **arr)
    return arr


def load_frozen(split: str) -> dict:
    """Frozen dev (2,304) or test (9,216) records grouped by (K, regime)."""
    rows = read_jsonl(DATA_DIR / "mqar" / f"mqar_{'test' if split == 'test' else 'development'}.jsonl.gz")
    groups = {}
    for r in rows:
        groups.setdefault((r["K"], r["regime"]), []).append(r)
    out = {}
    for (k, reg), exs in groups.items():
        a = to_arrays(exs, k)
        assert np.array_equal(a["dist"], np.array([e["distances"] for e in exs], dtype=np.int16))
        a["ids"] = [e["id"] for e in exs]
        out[(k, reg)] = a
    return out


def batch_to_torch(arr, idx, device):
    return (torch.as_tensor(arr["tokens"][idx].astype(np.int64), device=device),
            torch.as_tensor(arr["qpos"][idx].astype(np.int64), device=device),
            torch.as_tensor(arr["ans"][idx].astype(np.int64), device=device))
