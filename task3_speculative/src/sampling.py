"""Exact speculative sampling (Leviathan et al. 2023) written from scratch, plus plain ancestral sampling.

Core round (`verify_round`): proposals d_1..d_g ~ q; accept d_i with probability min(1, p_i(d_i)/q_i(d_i)); at the first
rejection sample from the normalised positive residual [p_i - q_i]_+; if every proposal is accepted sample a bonus
token from p_{g+1}. The same function drives the toy Markov test and the real Qwen models.

Real-model cache invariant: after every round each model's DynamicCache holds exactly seq[:-1] (or fewer tokens);
the next round feeds the missing suffix (1 or 2 tokens), so rejection rolls back with ``crop`` and full acceptance
advances the draft by the not-yet-cached last proposal.
"""
from __future__ import annotations

import time

import torch

from shared.src.timing import sync


# ------------------------------------------------------------------------------------------------- core
def verify_round(p: torch.Tensor, q: torch.Tensor, drafts: list, gen: torch.Generator, residual: str = "exact"):
    """p: (g+1, V) target probabilities at the g draft positions and the bonus position; q: (g, V) draft probabilities.
    Returns (n_accepted, next_token, rejected: bool). residual='p' is the deliberately wrong negative control."""
    g = len(drafts)
    for i in range(g):
        x = drafts[i]
        a = torch.clamp(p[i, x] / q[i, x], max=1.0)
        u = torch.rand((), generator=gen, device=p.device)
        if u >= a:
            if residual == "exact":
                r = torch.clamp(p[i] - q[i], min=0.0)
                z = r.sum()
                r = r / z if z > 0 else p[i]
            else:
                r = p[i]
            return i, int(torch.multinomial(r, 1, generator=gen)), True
    return g, int(torch.multinomial(p[g], 1, generator=gen)), False


# ------------------------------------------------------------------------------------- toy Markov models
class MarkovLM:
    """Categorical first symbol + first-order transitions over a small alphabet."""

    def __init__(self, p0, P):
        self.p0 = torch.as_tensor(p0, dtype=torch.float64)
        self.P = torch.as_tensor(P, dtype=torch.float64)

    def next_dist(self, prefix):
        return self.p0 if not prefix else self.P[prefix[-1]]

    def joint(self, length):
        import itertools
        V = len(self.p0)
        out = {}
        for seq in itertools.product(range(V), repeat=length):
            pr = self.p0[seq[0]]
            for a, b in zip(seq, seq[1:]):
                pr = pr * self.P[a, b]
            out[seq] = float(pr)
        return out


def toy_speculative_sample(target: MarkovLM, draft: MarkovLM, length: int, gamma: int, gen, residual="exact",
                           clip_gamma: bool = True):
    """One sequence of ``length`` symbols. clip_gamma=False always proposes gamma tokens and relies on final-token
    truncation; both must reproduce the target joint law."""
    seq = []
    stats = dict(rounds=0, accepted=0, examined=0, bonus=0)
    while len(seq) < length:
        g = min(gamma, length - len(seq) - 1) if clip_gamma else gamma
        drafts, qs = [], []
        for _ in range(g):
            qd = draft.next_dist(seq + drafts)
            drafts.append(int(torch.multinomial(qd, 1, generator=gen)))
            qs.append(qd)
        ps = torch.stack([target.next_dist(seq + drafts[:i]) for i in range(g + 1)])
        q = torch.stack(qs) if qs else ps[:0]
        n, tok, rej = verify_round(ps, q, drafts, gen, residual)
        seq += drafts[:n] + [tok]
        stats["rounds"] += 1
        stats["accepted"] += n
        stats["examined"] += n + int(rej)
        stats["bonus"] += int(not rej)
    return tuple(seq[:length]), stats


def chi2_test(counts: dict, probs: dict, n: int) -> dict:
    """Pearson chi-square goodness of fit; p-value from the regularised upper incomplete gamma."""
    keys = list(probs)
    exp = torch.tensor([probs[k] * n for k in keys], dtype=torch.float64)
    obs = torch.tensor([counts.get(k, 0) for k in keys], dtype=torch.float64)
    stat = float(((obs - exp) ** 2 / exp).sum())
    dof = len(keys) - 1
    p = float(torch.special.gammaincc(torch.tensor(dof / 2, dtype=torch.float64), torch.tensor(stat / 2, dtype=torch.float64)))
    return dict(chi2=stat, dof=dof, p_value=p, min_expected=float(exp.min()), n=n)


# ------------------------------------------------------------------------------------------ real models
def _forward(model, cache, tokens, device):
    out = model(input_ids=torch.tensor([tokens], device=device), past_key_values=cache, use_cache=True)
    return out.logits[0], out.past_key_values


def _probs(logits, temperature=1.0):
    return torch.softmax(logits.float() / temperature, -1)


@torch.no_grad()
def sample_generate(model, prompt_ids: list, max_new: int, gen: torch.Generator, eos: set, temperature: float = 1.0,
                    ignore_eos: bool = False, device="cuda") -> dict:
    """Ancestral sampling (temperature 1, top-p 1, no top-k) with a KV cache; the target-only reference."""
    from transformers import DynamicCache
    cache = DynamicCache()
    sync(); t0 = time.perf_counter()
    logits, cache = _forward(model, cache, prompt_ids, device)
    sync(); t_prefill = time.perf_counter() - t0
    out, last = [], logits[-1]
    t1 = time.perf_counter()
    for _ in range(max_new):
        x = int(torch.multinomial(_probs(last, temperature), 1, generator=gen))
        out.append(x)
        if x in eos and not ignore_eos:
            break
        if len(out) == max_new:
            break
        logits, cache = _forward(model, cache, [x], device)
        last = logits[-1]
    sync()
    t_dec = time.perf_counter() - t1
    return dict(tokens=out, n_out=len(out), eos=bool(out and out[-1] in eos), truncated=len(out) >= max_new and not (out and out[-1] in eos),
                target_passes=len(out), prefill_ms=t_prefill * 1e3, decode_ms=t_dec * 1e3, total_ms=(t_prefill + t_dec) * 1e3)


@torch.no_grad()
def speculative_generate(target, draft, prompt_ids: list, gamma: int, max_new: int, gen: torch.Generator, eos: set,
                         temperature: float = 1.0, ignore_eos: bool = False, residual: str = "exact", device="cuda",
                         record_rounds: bool = False) -> dict:
    """Exact speculative sampling with KV caches for both models (batch 1, no padding)."""
    from transformers import DynamicCache
    T = dict(draft_ms=0.0, verify_ms=0.0, accept_ms=0.0, rollback_ms=0.0)
    tc, dc = DynamicCache(), DynamicCache()
    seq, P = list(prompt_ids), len(prompt_ids)
    sync(); t0 = time.perf_counter()
    if P > 1:
        _forward(target, tc, seq[:-1], device)
        _forward(draft, dc, seq[:-1], device)
    sync(); prefill_ms = (time.perf_counter() - t0) * 1e3
    t_len = d_len = P - 1
    rounds, n_draft_fwd, finished = [], 0, False
    st = dict(proposed=0, examined=0, accepted=0, rounds=0, bonus=0)
    t_start = time.perf_counter()
    while len(seq) - P < max_new and not finished:
        remaining = max_new - (len(seq) - P)
        g_max = min(gamma, remaining - 1)
        # ---- draft proposals
        sync(); t0 = time.perf_counter()
        drafts, qs, feed = [], [], seq[d_len:]
        for _ in range(g_max):
            logits, dc = _forward(draft, dc, feed, device)
            n_draft_fwd += 1
            d_len += len(feed)
            q = _probs(logits[-1], temperature)
            x = int(torch.multinomial(q, 1, generator=gen))
            drafts.append(x)
            qs.append(q)
            feed = [x]
            if x in eos and not ignore_eos:
                break
        sync(); T["draft_ms"] += (time.perf_counter() - t0) * 1e3
        # ---- one target verification pass over pending token(s) + proposals
        t0 = time.perf_counter()
        t_feed = seq[t_len:] + drafts
        logits, tc = _forward(target, tc, t_feed, device)
        t_len += len(t_feed)
        p = _probs(logits[-(len(drafts) + 1):], temperature)
        sync(); T["verify_ms"] += (time.perf_counter() - t0) * 1e3
        # ---- accept / residual / bonus
        t0 = time.perf_counter()
        q = torch.stack(qs) if qs else p[:0]
        n, tok, rej = verify_round(p, q, drafts, gen, residual)
        new = drafts[:n] + [tok]
        for j, x in enumerate(new):
            if x in eos and not ignore_eos:
                new, finished = new[:j + 1], True
                break
        new = new[:remaining]
        seq += new
        sync(); T["accept_ms"] += (time.perf_counter() - t0) * 1e3
        # ---- cache rollback / advancement: each cache must hold at most seq[:-1]
        t0 = time.perf_counter()
        keep = len(seq) - 1
        if t_len > keep:
            tc.crop(keep)
            t_len = keep
        if d_len > keep:
            dc.crop(keep)
            d_len = keep
        sync(); T["rollback_ms"] += (time.perf_counter() - t0) * 1e3
        st["rounds"] += 1
        st["proposed"] += len(drafts)
        st["accepted"] += n
        st["examined"] += n + int(rej)
        st["bonus"] += int(not rej)
        if record_rounds:
            rounds.append(dict(g=len(drafts), n=n, rejected=rej, emitted=len(new)))
    sync()
    total_ms = (time.perf_counter() - t_start) * 1e3
    out = seq[P:]
    eos_hit = bool(out and out[-1] in eos and not ignore_eos)
    res = dict(tokens=out, n_out=len(out), eos=eos_hit, truncated=len(out) >= max_new and not eos_hit,
               target_passes=st["rounds"], draft_forward_passes=n_draft_fwd, prefill_ms=prefill_ms,
               decode_ms=total_ms, total_ms=total_ms + prefill_ms, host_overhead_ms=total_ms - sum(T.values()), **T, **st)
    if record_rounds:
        res["rounds_detail"] = rounds
    return res


def summarize_run(r: dict) -> dict:
    """Per-generation speculative metrics used in every table."""
    n = max(r["n_out"], 1)
    return dict(acc_per_examined=r["accepted"] / max(r["examined"], 1), acc_per_proposed=r["accepted"] / max(r["proposed"], 1),
                mean_accepted_per_verification=r["accepted"] / max(r["rounds"], 1), target_passes_per_token=r["target_passes"] / n,
                tokens_per_s=n / (r["total_ms"] / 1e3), ms_per_token=r["total_ms"] / n)
