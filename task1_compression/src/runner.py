"""One function runs a compression condition end-to-end with the shared evaluation code and saves:

results/conditions/<name>.json            summary (quality, storage, resources, compression stats)
results/per_example/<name>_ppl.json       per-window NLL sums for both perplexity suites
results/per_example/<name>_hellaswag.jsonl per-example candidate log-likelihoods

A condition whose summary file exists is skipped (re-running a notebook resumes it).
"""
from __future__ import annotations

import json
import time

import torch

from shared.src.io import load_json, save_json, write_jsonl
from shared.src.timing import peak_memory, reset_peak

from .data import HELLASWAG_DIAG, HELLASWAG_FORMAT, HELLASWAG_FULL
from .evaluate import eval_hellaswag, eval_perplexity
from .model_utils import count_nonzeros, eligible_linears, inventory
from .prune import sparse16_storage_bytes
from .quant import dense_fp16_bytes
from .resources import measure_latency

FULL_HELLASWAG = {"fp16", "p1_mag_s50", "p2_wanda_s50", "p3_s50", "q1_int4", "q1_int3", "q2_awq_int4", "q2_awq_int3",
                  "q3_budget_int4", "q3_budget_int3"}


def hellaswag_n(name: str) -> int:
    return HELLASWAG_FULL if name in FULL_HELLASWAG else HELLASWAG_DIAG


class ConditionRunner:
    def __init__(self, ctx, model, tok, eval_sets: dict, snapshot, latency: bool = True):
        self.ctx, self.model, self.tok, self.ev, self.snap, self.latency = ctx, model, tok, eval_sets, snapshot, latency
        self.dir = ctx.results / "conditions"
        self.pe = ctx.results / "per_example"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pe.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        return self.dir / f"{name}.json"

    def done(self, name):
        return self.path(name).exists()

    def load(self, name):
        return load_json(self.path(name))

    def run(self, name: str, apply_fn=None, storage_fn=None, meta: dict | None = None, force: bool = False) -> dict:
        """apply_fn(model) -> dict of compression stats (model already restored to FP16).
        storage_fn(model, stats) -> dict of storage numbers (defaults: dense FP16 + sparse16 estimate)."""
        if self.done(name) and not force:
            print(f"[skip] {name} (results exist)")
            return self.load(name)
        self.snap.restore(self.model)
        t0 = time.perf_counter()
        reset_peak()
        stats = apply_fn(self.model) if apply_fn else {}
        comp_time = time.perf_counter() - t0
        comp_peak = peak_memory()
        inv = inventory(self.model)
        nz = count_nonzeros(self.model)
        mats = eligible_linears(self.model)
        storage = dict(dense_fp16_bytes=dense_fp16_bytes(self.model),
                       sparse16=sparse16_storage_bytes([m.weight.numel() for *_, m in mats],
                                                       [nz["per_matrix"][n] for n, *_ in mats],
                                                       inv["n_other_params_unique"]))
        if storage_fn:
            storage.update(storage_fn(self.model, stats))
        res = dict(name=name, meta=meta or {}, compression=dict(stats=_light(stats), time_s=comp_time, peak_memory=comp_peak),
                   nonzeros={k: v for k, v in nz.items() if k != "per_matrix"}, storage=storage)
        ppl_pe = {}
        for suite in ("wikitext2", "c4"):
            r = eval_perplexity(self.model, self.ev[suite], desc=f"{name}:{suite}")
            ppl_pe[suite] = dict(window_nll_sums=r.pop("window_nll_sums"), window_counts=r.pop("window_counts"))
            res[suite] = r
            print(f"  {name} {suite}: ppl={r['ppl']:.3f} (N={r['n_scored_tokens']})")
        n_hs = hellaswag_n(name)
        hs = eval_hellaswag(self.model, self.tok, self.ev["hellaswag"][:n_hs])
        write_jsonl(self.pe / f"{name}_hellaswag.jsonl", hs.pop("per_example"))
        hs["format"] = HELLASWAG_FORMAT
        res["hellaswag"] = hs
        print(f"  {name} hellaswag(n={n_hs}): acc={hs['acc']:.4f} acc_norm={hs['acc_norm']:.4f}")
        if self.latency:
            res["resources"] = measure_latency(self.model)
        save_json(self.pe / f"{name}_ppl.json", ppl_pe)
        save_json(self.path(name), res)
        return res


def setup_task1(ctx, latency: bool = True):
    """Load the FP16 model once, snapshot its weights and load every locked evaluation set."""
    from types import SimpleNamespace

    from .data import load_calibration, load_development, load_eval_windows, load_hellaswag
    from .model_utils import WeightSnapshot, load_model
    model, tok = load_model()
    snap = WeightSnapshot(model)
    calib, calib_rows = load_calibration()
    dev, dev_rows = load_development()
    ev = dict(wikitext2=load_eval_windows("wikitext2"), c4=load_eval_windows("c4"), hellaswag=load_hellaswag())
    runner = ConditionRunner(ctx, model, tok, ev, snap, latency=latency)
    return SimpleNamespace(model=model, tok=tok, snap=snap, calib=calib, calib_rows=calib_rows, dev=dev,
                           dev_rows=dev_rows, ev=ev, runner=runner)


def _light(stats: dict) -> dict:
    """Keep stats JSON-serialisable (drop tensors)."""
    try:
        json.dumps(stats)
        return stats
    except TypeError:
        return {k: v for k, v in stats.items() if not torch.is_tensor(v)}
