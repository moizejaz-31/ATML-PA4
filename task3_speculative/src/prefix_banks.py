"""Fixed prefix banks and common-prefix distribution metrics.

`build_bank` reproduces the course script's protocol (same 120 prompts and prompt ids, per-prompt seed
6304 + 10007 j (+123 for the original draft), the same position-sampling RNG string and record schema), but samples
with our own KV-cache loop because transformers 4.57.1 ``generate`` rejects the ``generator=`` argument used by
``setup_data.py --prefix-banks``. The same function builds the exact-speculative bank and each trained student's
own-rollout bank.
"""
from __future__ import annotations

import random
import zlib

import numpy as np
import torch

from shared.src.io import read_jsonl, write_jsonl

from .data import encode_prompt, eos_ids, test_prompts
from .sampling import sample_generate, speculative_generate

BINS = [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]
POS_BINS = {"1-16": (1, 16), "17-32": (17, 32), "33-64": (33, 64), "65-192": (65, 192)}
SEED_OFFSET = {"teacher": 0, "original_draft": 123}


def bank_seed(j: int, bank: str) -> int:
    return 6304 + 10007 * j + SEED_OFFSET.get(bank, 7919 * (1 + zlib.crc32(bank.encode()) % 1000))


def select_positions(pid: str, bank: str, seq_len: int) -> list:
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{bank}")
    pos = []
    for lo, hi, n in BINS:
        cand = list(range(lo, min(hi, seq_len) + 1))
        pos.extend(sorted(rng.sample(cand, min(n, len(cand)))))
    return pos


@torch.no_grad()
def build_bank(bank: str, tok, model=None, draft=None, gamma: int = 4, seed_of=None, max_new: int = 192,
               producer: str = "", device="cuda", limit: int | None = None) -> list:
    """model only -> ancestral sampling from `model`; model + draft -> exact speculative sampling (target=model)."""
    prompts = test_prompts()[:limit] if limit else test_prompts()
    eos = eos_ids(tok)
    out = []
    for j, pr in enumerate(prompts):
        seed = seed_of(j) if seed_of else bank_seed(j, bank)
        gen = torch.Generator(device=device).manual_seed(seed)
        ids = encode_prompt(tok, pr["prompt"])
        if draft is None:
            seq = sample_generate(model, ids, max_new, gen, eos, device=device)["tokens"]
        else:
            seq = speculative_generate(model, draft, ids, gamma, max_new, gen, eos, device=device)["tokens"]
        for p in select_positions(pr["pid"], bank, len(seq)):
            out.append(dict(prompt_id=pr["pid"], domain=pr["domain"], workload=pr["workload"], bank=bank, position=p,
                            prompt=pr["prompt"], response_prefix_token_ids=seq[:p - 1], source_length=len(seq),
                            source_generation_seed=seed, producer=producer or bank))
        if (j + 1) % 20 == 0:
            print(bank, j + 1, "/", len(prompts), flush=True)
    return out


def workload_of(pid: str) -> str:
    if pid.startswith("GSM8K"):
        return "math"
    return "short_familiar" if "familiar" in pid else "short_shifted"


def pos_bin(p: int) -> str:
    for k, (lo, hi) in POS_BINS.items():
        if lo <= p <= hi:
            return k
    return "other"


@torch.no_grad()
def next_token_dists(model, tok, records: list, device="cuda", temperature: float = 1.0) -> dict:
    """One forward per (prompt, bank) on the longest saved prefix; returns log-probs at each saved position.
    (All saved prefixes of a prompt/bank are prefixes of one sampled response.)"""
    by = {}
    for r in records:
        by.setdefault((r["prompt_id"], r["bank"]), []).append(r)
    out = {}
    for (pid, bank), rs in by.items():
        longest = max(rs, key=lambda r: len(r["response_prefix_token_ids"]))
        ids = encode_prompt(tok, rs[0]["prompt"])
        P = len(ids)
        full = ids + longest["response_prefix_token_ids"]
        logits = model(input_ids=torch.tensor([full], device=device)).logits[0]
        for r in rs:
            assert longest["response_prefix_token_ids"][:len(r["response_prefix_token_ids"])] == r["response_prefix_token_ids"]
            idx = P + len(r["response_prefix_token_ids"]) - 1
            out[(pid, bank, r["position"])] = torch.log_softmax(logits[idx].float() / temperature, -1).cpu()
    return out


def pair_metrics(logp_t: torch.Tensor, logp_q: torch.Tensor) -> dict:
    """Forward KL(p||q), reverse KL(q||p), TV, greedy agreement and expected one-token acceptance 1 - TV."""
    p, q = logp_t.exp(), logp_q.exp()
    tv = 0.5 * (p - q).abs().sum().item()
    return dict(fkl=float((p * (logp_t - logp_q)).sum()), rkl=float((q * (logp_q - logp_t)).sum()), tv=tv,
                greedy_agree=int(int(logp_t.argmax()) == int(logp_q.argmax())), exp_accept=1 - tv)


@torch.no_grad()
def bank_metrics(target, drafts: dict, tok, records: list, device="cuda", chunk_prompts: int = 10) -> list:
    """Metrics for every draft on identical records (same prefix ids, positions and denominators).
    Processes prompts in chunks to bound CPU memory (one 151,936-way distribution per record per model)."""
    rows = []
    pids = sorted({r["prompt_id"] for r in records})
    for c in range(0, len(pids), chunk_prompts):
        chunk = set(pids[c:c + chunk_prompts])
        sub = [r for r in records if r["prompt_id"] in chunk]
        lt = next_token_dists(target, tok, sub, device)
        for name, dm in drafts.items():
            ld = next_token_dists(dm, tok, sub, device)
            for key, lp in lt.items():
                pid, bank, pos = key
                rows.append(dict(draft=name, prompt_id=pid, bank=bank, position=pos, workload=workload_of(pid),
                                 pos_bin=pos_bin(pos), **pair_metrics(lp, ld[key])))
            del ld
        del lt
    return rows


def sampled_token_nll(target, tok, records: list, device="cuda") -> list:
    """Target NLL of the token actually sampled after each prefix (needs the next token from the longest prefix);
    used for the teacher-vs-speculative distributional consistency check."""
    by = {}
    for r in records:
        by.setdefault((r["prompt_id"], r["bank"]), []).append(r)
    rows = []
    for (pid, bank), rs in by.items():
        longest = max(rs, key=lambda r: len(r["response_prefix_token_ids"]))
        seq = longest["response_prefix_token_ids"]
        ids = encode_prompt(tok, rs[0]["prompt"])
        with torch.no_grad():
            lp = torch.log_softmax(target(input_ids=torch.tensor([ids + seq], device=device)).logits[0].float(), -1)
        for r in rs:
            k = len(r["response_prefix_token_ids"])
            if k < len(seq):
                rows.append(dict(prompt_id=pid, bank=bank, position=r["position"], nll=float(-lp[len(ids) + k - 1, seq[k]])))
    return rows
