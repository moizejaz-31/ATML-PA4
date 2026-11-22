"""Held-out speculative-decoding benchmark, isolated step costs and the analytical cost model."""
from __future__ import annotations

import time

import numpy as np
import torch

from shared.src.io import append_jsonl, read_jsonl
from shared.src.timing import benchmark, peak_memory, reset_peak, sync

from .data import encode_prompt, eos_ids, test_prompts
from .sampling import sample_generate, speculative_generate, summarize_run


def eval_seed(pid_index: int, rep: int, gamma: int) -> int:
    return 6304 + 1000 * pid_index + 100 * rep + gamma


def run_benchmark(target, drafts: dict, tok, path, gammas=(2, 4), reps=(0, 1), max_new: int = 64, ignore_eos: bool = False,
                  prompt_filter=None, device="cuda") -> None:
    """For every (draft, gamma, rep, prompt): exact speculative generation and an adjacent target-only generation on the
    same prompt and seed. Benchmark order is counterbalanced (spec first on even prompts, target first on odd).
    Appends one row per speculative run (+ its paired target-only timing) to ``path``; resumable."""
    done = {(r["draft"], r["gamma"], r["rep"], r["pid"]) for r in read_jsonl(path)} if path.exists() else set()
    prompts = [p for p in test_prompts() if prompt_filter is None or prompt_filter(p)]
    eos = eos_ids(tok)
    for name, dm in drafts.items():
        for gamma in gammas:
            for rep in reps:
                for i, pr in enumerate(prompts):
                    if (name, gamma, rep, pr["pid"]) in done:
                        continue
                    ids = encode_prompt(tok, pr["prompt"])
                    seed = eval_seed(i, rep, gamma)

                    def spec():
                        reset_peak()
                        g = torch.Generator(device=device).manual_seed(seed)
                        r = speculative_generate(target, dm, ids, gamma, max_new, g, eos, ignore_eos=ignore_eos, device=device)
                        r["peak_allocated_mb"] = peak_memory()["peak_allocated_mb"]
                        return r

                    def base():
                        reset_peak()
                        g = torch.Generator(device=device).manual_seed(seed)
                        r = sample_generate(target, ids, max_new, g, eos, ignore_eos=ignore_eos, device=device)
                        r["peak_allocated_mb"] = peak_memory()["peak_allocated_mb"]
                        return r

                    first_spec = (i + rep) % 2 == 0
                    if first_spec:
                        rs, rb = spec(), base()
                    else:
                        rb, rs = base(), spec()
                    toks = rs.pop("tokens")
                    rb_toks = rb.pop("tokens")
                    row = dict(draft=name, gamma=gamma, rep=rep, pid=pr["pid"], workload=pr["workload"], prompt_len=len(ids),
                               seed=seed, ignore_eos=ignore_eos, max_new=max_new, order="spec_first" if first_spec else "target_first",
                               **rs, **summarize_run(dict(rs, tokens=toks)),
                               base_n_out=rb["n_out"], base_total_ms=rb["total_ms"], base_prefill_ms=rb["prefill_ms"],
                               base_decode_ms=rb["decode_ms"], base_eos=rb["eos"], base_peak_allocated_mb=rb["peak_allocated_mb"],
                               out_tokens=toks)
                    row["base_tokens_per_s"] = rb["n_out"] / (rb["total_ms"] / 1e3)
                    row["speedup_tps"] = row["tokens_per_s"] / row["base_tokens_per_s"]
                    append_jsonl(path, row)
                print(f"{name} gamma={gamma} rep={rep} done", flush=True)


@torch.no_grad()
def isolated_step_costs(target, draft, tok, prompt_lens=(64, 256, 512), gammas=(2, 4), reps: int = 20, device="cuda") -> list:
    """Single decode-step time of draft and target, and a gamma+1-token target verification, at several cached
    context lengths (ms). c = draft step / target step, v_gamma = verification / target step."""
    from transformers import DynamicCache
    rows = []
    g = torch.Generator().manual_seed(6304)
    for L in prompt_lens:
        ctx = torch.randint(1000, 30000, (1, L), generator=g).to(device)

        def step_fn(model, n_new):
            c = DynamicCache()
            model(input_ids=ctx, past_key_values=c, use_cache=True)
            x = torch.randint(1000, 30000, (1, n_new), device=device)

            def f():
                model(input_ids=x, past_key_values=c, use_cache=True)
                c.crop(L)
            return f
        t_t = benchmark(step_fn(target, 1), warmup=3, reps=reps)["mean"]
        t_d = benchmark(step_fn(draft, 1), warmup=3, reps=reps)["mean"]
        row = dict(context=L, target_step_ms=t_t, draft_step_ms=t_d, c=t_d / t_t)
        for gm in gammas:
            t_v = benchmark(step_fn(target, gm + 1), warmup=3, reps=reps)["mean"]
            row[f"verify_{gm}_ms"] = t_v
            row[f"v_{gm}"] = t_v / t_t
        rows.append(row)
    return rows


def predicted_speedup(alpha: float, gamma: int, c: float, v: float) -> float:
    return (1 + sum(alpha ** j for j in range(1, gamma + 1))) / (gamma * c + v)


def break_even_alpha(gamma: int, c: float, v: float):
    """Smallest alpha in [0, 1] with 1 + sum_j alpha^j > gamma c + v; None if no such alpha exists."""
    rhs = gamma * c + v
    f = lambda a: 1 + sum(a ** j for j in range(1, gamma + 1)) - rhs
    if f(1.0) <= 0:
        return None
    if f(0.0) > 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) <= 0 else (lo, mid)
    return hi


@torch.no_grad()
def standalone_accuracy(model, tok, path, name: str, reps=(0, 1), device="cuda", caps=None, greedy: bool = False) -> None:
    """The draft as a model: temperature-1 sampling (or greedy) with answer-appropriate caps (512 math / 32 short)."""
    from .data import EVAL_CAPS, extract_math, score_short
    caps = caps or EVAL_CAPS
    done = {(r["model"], r["rep"], r["pid"]) for r in read_jsonl(path)} if path.exists() else set()
    eos = eos_ids(tok)
    for rep in reps:
        for i, pr in enumerate(test_prompts()):
            if (name, rep, pr["pid"]) in done:
                continue
            ids = encode_prompt(tok, pr["prompt"])
            gen = torch.Generator(device=device).manual_seed(eval_seed(i, rep, 0) + 7)
            if greedy:
                r = _greedy(model, ids, caps[pr["domain"]], eos, device)
            else:
                r = sample_generate(model, ids, caps[pr["domain"]], gen, eos, device=device)
            text = tok.decode(r["tokens"], skip_special_tokens=True)
            row = dict(model=name, rep=rep, pid=pr["pid"], workload=pr["workload"], n_out=r["n_out"], eos=r["eos"],
                       truncated=r["truncated"], text=text)
            if pr["domain"] == "math":
                pred = extract_math(text)
                row.update(pred=pred, correct=int(pred == pr["answer"]), extractable=int(pred is not None))
            else:
                s = score_short(text, pr["answer"])
                row.update(pred=text.strip(), correct=s["exact_format"], contains=s["contains"])
            append_jsonl(path, row)


def _greedy(model, ids, max_new, eos, device):
    from transformers import DynamicCache
    c = DynamicCache()
    out = model(input_ids=torch.tensor([ids], device=device), past_key_values=c, use_cache=True)
    toks = []
    for _ in range(max_new):
        x = int(out.logits[0, -1].argmax())
        toks.append(x)
        if x in eos:
            break
        out = model(input_ids=torch.tensor([[x]], device=device), past_key_values=c, use_cache=True)
    return dict(tokens=toks, n_out=len(toks), eos=bool(toks and toks[-1] in eos), truncated=len(toks) >= max_new)
