"""Shared Task 1 quality evaluation: windowed perplexity and zero-shot HellaSwag log-likelihood scoring.

Every condition goes through these exact functions, so contexts and scored tokens are identical.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


@torch.no_grad()
def eval_perplexity(model, windows, batch_size: int = 4, device: str = "cuda", desc: str = "ppl") -> dict:
    """Independent 512-token windows; target t is scored from tokens < t of the same window only.

    The first token of every window is context only (excluded from N).
    Returns pooled N, mean NLL, PPL and the per-window NLL sums needed for window bootstrap.
    """
    nll_sums, counts = [], []
    for i in tqdm(range(0, len(windows), batch_size), desc=desc, leave=False):
        chunk = windows[i:i + batch_size]
        L = max(len(w["token_ids"]) for w in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        att = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, w in enumerate(chunk):
            ids[j, :len(w["token_ids"])] = torch.tensor(w["token_ids"])
            att[j, :len(w["token_ids"])] = 1
        ids, att = ids.to(device), att.to(device)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        for j, w in enumerate(chunk):
            n = w["scored_tokens"]
            lp = F.cross_entropy(logits[j, :n].float(), ids[j, 1:n + 1], reduction="sum")
            nll_sums.append(float(lp))
            counts.append(n)
        del logits
    N = int(sum(counts))
    mean_nll = float(sum(nll_sums) / N)
    return dict(n_scored_tokens=N, n_windows=len(counts), mean_nll=mean_nll, ppl=math.exp(mean_nll),
                window_nll_sums=nll_sums, window_counts=counts)


@torch.no_grad()
def nll_on_sequences(model, ids: torch.Tensor, batch_size: int = 8, device: str = "cuda") -> float:
    """Mean next-token NLL over full fixed-length sequences (used for calibration / dev probes)."""
    tot, n = 0.0, 0
    for i in range(0, ids.shape[0], batch_size):
        x = ids[i:i + batch_size].to(device)
        logits = model(input_ids=x, use_cache=False).logits
        tot += float(F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), x[:, 1:].flatten(), reduction="sum"))
        n += x[:, 1:].numel()
        del logits
    return tot / n


def _encode_pair(tok, context: str, continuation: str):
    """lm-eval ``_encode_pair``: trailing context spaces move to the continuation; no special tokens."""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tok(context + continuation, add_special_tokens=False)["input_ids"]
    ctx = tok(context, add_special_tokens=False)["input_ids"]
    return ctx, whole[len(ctx):]


def build_hellaswag_requests(tok, docs):
    reqs = []
    for d in docs:
        for c, choice in enumerate(d["choices"]):
            ctx, cont = _encode_pair(tok, d["query"], " " + choice)
            reqs.append(dict(order=d["order"], choice=c, ctx=ctx, cont=cont, n_chars=len(choice)))
    return reqs


@torch.no_grad()
def eval_hellaswag(model, tok, docs, batch_size: int = 32, device: str = "cuda", requests=None) -> dict:
    """Zero-shot 4-way scoring by conditional log-likelihood of each ending.

    acc: argmax of summed log-likelihood; acc_norm: argmax of loglik / len(ending characters).
    Returns aggregate scores and per-example candidate scores.
    """
    reqs = requests or build_hellaswag_requests(tok, docs)
    ll = {}
    order = sorted(range(len(reqs)), key=lambda i: -(len(reqs[i]["ctx"]) + len(reqs[i]["cont"])))
    for s in tqdm(range(0, len(order), batch_size), desc="hellaswag", leave=False):
        batch = [reqs[i] for i in order[s:s + batch_size]]
        seqs = [(r["ctx"] + r["cont"])[:-1] for r in batch]
        L = max(len(x) for x in seqs)
        ids = torch.zeros(len(seqs), L, dtype=torch.long)
        att = torch.zeros(len(seqs), L, dtype=torch.long)
        for j, x in enumerate(seqs):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device), use_cache=False).logits
        for j, r in enumerate(batch):
            n_ctx, n_cont = len(r["ctx"]), len(r["cont"])
            lp = torch.log_softmax(logits[j, n_ctx - 1:n_ctx - 1 + n_cont].float(), -1)
            tgt = torch.tensor(r["cont"], device=lp.device)
            ll[(r["order"], r["choice"])] = float(lp.gather(1, tgt[:, None]).sum())
        del logits
    per_example = []
    for d in docs:
        lls = np.array([ll[(d["order"], c)] for c in range(len(d["choices"]))])
        lens = np.array([float(len(ch)) for ch in d["choices"]])
        pred, pred_norm = int(lls.argmax()), int((lls / lens).argmax())
        per_example.append(dict(order=d["order"], gold=d["gold"], loglik=lls.tolist(), n_chars=lens.tolist(),
                                pred=pred, pred_norm=pred_norm, correct=int(pred == d["gold"]),
                                correct_norm=int(pred_norm == d["gold"])))
    acc = float(np.mean([e["correct"] for e in per_example]))
    acc_norm = float(np.mean([e["correct_norm"] for e in per_example]))
    return dict(n=len(docs), acc=acc, acc_norm=acc_norm, per_example=per_example)
