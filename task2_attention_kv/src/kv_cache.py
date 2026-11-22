"""Part C: KV-cache editing for Qwen2.5-1.5B-Instruct (transformers 4.57.1, SDPA, FP16, batch 1).

Pipeline per prompt and policy
  1. prefill the pre-question context (chat-template prefix + user text up to 'QUESTION_BEGIN') -> FP16 cache;
  2. (question-aware policies) run the question suffix over the full cache, capture each layer's query
     projections, score earlier positions with masked FP32 attention against the RoPE'd cached keys, crop back;
  3. compress the context cache (evict / quantize / hybrid) and rebuild a DynamicCache;
  4. process the identical question suffix and decode greedily. Retained tokens keep their original absolute
     RoPE positions (keys are cached post-RoPE; new tokens get position_ids = original absolute positions) while
     cache_position follows the physical (compacted) cache length, so masks stay consistent.
Quantized caches are dequantized to FP16 before ordinary SDPA (fake-quant execution; packed bytes accounted
analytically and benchmarked separately in kv_pack.py).
"""
from __future__ import annotations

import math
import re
import time

import torch
import torch.nn.functional as F

from shared.src.paths import model_revision

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GROUP = 64
SINK, RECENT = 8, 64
INDEX_BYTES = 2           # int16 position index (all contexts < 32768 tokens)


def load_kv_model(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rev = model_revision("model_target")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=rev, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").to(device).eval()
    model._pa4_revision = rev
    return model, tok


def kv_dims(model) -> dict:
    c = model.config
    n_l, h_kv = c.num_hidden_layers, c.num_key_value_heads
    d_h = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return dict(n_layers=n_l, n_kv_heads=h_kv, head_dim=d_h, n_q_heads=c.num_attention_heads, C=n_l * h_kv * d_h)


# ------------------------------------------------------------------------------------------ byte accounting
def bytes_quant(C: int, L: int, bits: int) -> int:
    """M_b(L) of the manual: 2 C [floor(L/64) (64 b/8 + 4) + 2 (L mod 64)], C = n_l H_kv d_h channels per K or V."""
    return 2 * C * ((L // GROUP) * (GROUP * bits // 8 + 4) + 2 * (L % GROUP))


def bytes_fp16(C: int, L: int) -> int:
    return 4 * C * L


def eviction_keep(C: int, L: int, budget: int, index_bytes_per_pos: int, fixed_index_bytes: int = 0) -> int:
    """Largest number of FP16 positions whose bytes (+ indices) fit the budget."""
    n = (budget - fixed_index_bytes) // (4 * C + index_bytes_per_pos)
    return int(min(max(n, 0), L))


def hybrid_bytes(C: int, L: int, n_fp: int, low_bits: int = 2) -> int:
    m = L - n_fp
    return 2 * C * ((m // GROUP) * (GROUP * low_bits // 8 + 4) + 2 * (m % GROUP) + 2 * n_fp) + INDEX_BYTES * n_fp


def hybrid_keep(C: int, L: int, budget: int, min_fp: int, low_bits: int = 2) -> int:
    best = None
    for n in range(min_fp, L + 1):
        if hybrid_bytes(C, L, n, low_bits) <= budget:
            best = n
    if best is None:
        raise ValueError("hybrid infeasible under budget")
    return best


# ------------------------------------------------------------------------------------------ prompt prep
CODE_RE = re.compile(r"\d{5}")


def extract_code(text: str):
    m = CODE_RE.search(text)
    return m.group(0) if m else None


def prepare_case(tok, case: dict) -> dict:
    """Tokenise with the chat template; split context / question suffix; map evidence chars to tokens."""
    msgs = [{"role": "user", "content": case["user_text"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    u0 = text.index(case["user_text"])
    q_char = u0 + case["user_text"].index("QUESTION_BEGIN")
    split = next(i for i, (a, b) in enumerate(offs) if b > q_char)
    e0, e1 = u0 + case["evidence_char_start"], u0 + case["evidence_char_end"]
    ev = [i for i, (a, b) in enumerate(offs) if a < e1 and b > e0]
    assert ev and max(ev) < split, "evidence must lie in the context"
    ans = tok(case["code"], add_special_tokens=False)["input_ids"]
    return dict(id=case["id"], ids=ids, ctx_len=split, n_tokens=len(ids), suffix_len=len(ids) - split,
                evidence_tokens=ev, answer_ids=ans, code=case["code"], nominal_len=case["target_tokens"],
                position=case["position"], distractors=case["distractors"])


# ------------------------------------------------------------------------------------------ cache helpers
def get_kv(cache, l):
    if hasattr(cache, "layers"):
        return cache.layers[l].keys, cache.layers[l].values
    return cache.key_cache[l], cache.value_cache[l]


def n_cache_layers(cache):
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def build_cache(kv_list):
    from transformers import DynamicCache
    c = DynamicCache()
    for l, (k, v) in enumerate(kv_list):
        c.update(k.contiguous(), v.contiguous(), l)
    return c


def cache_to_list(cache):
    return [get_kv(cache, l) for l in range(n_cache_layers(cache))]


@torch.no_grad()
def prefill(model, ids, device="cuda"):
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x, use_cache=True)
    return out.past_key_values, out.logits[:, -1].float()


@torch.no_grad()
def forward_step(model, cache, tokens, abs_pos: int, device="cuda"):
    """Feed tokens whose first absolute RoPE position is abs_pos; physical positions follow the cache length."""
    phys = cache.get_seq_length()
    n = len(tokens)
    out = model(input_ids=torch.tensor([tokens], device=device), past_key_values=cache, use_cache=True,
                position_ids=torch.arange(abs_pos, abs_pos + n, device=device)[None],
                cache_position=torch.arange(phys, phys + n, device=device))
    return out.logits[0].float(), out.past_key_values


# ------------------------------------------------------------------------------- question-aware scoring
@torch.no_grad()
def question_scores(model, cache, suffix_ids, ctx_len: int, pool: int = 7, device="cuda") -> torch.Tensor:
    """SnapKV-inspired fixed policy: attention mass that the question-suffix queries (all heads, all layers)
    place on each context position, computed in FP32 from captured query projections and cached RoPE'd keys,
    then 1-D average pooling (kernel 7). Uses the working SDPA model; the cache is cropped back afterwards."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    caps = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        def hook(mod, args, kwargs, li=li):
            caps[li] = (kwargs.get("hidden_states", args[0] if args else None), kwargs["position_embeddings"])
        hooks.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        forward_step(model, cache, suffix_ids, ctx_len, device)
    finally:
        for h in hooks:
            h.remove()
    dims = kv_dims(model)
    S, d = len(suffix_ids), dims["head_dim"]
    rep = dims["n_q_heads"] // dims["n_kv_heads"]
    total = torch.zeros(ctx_len, device=device, dtype=torch.float32)
    for li, layer in enumerate(model.model.layers):
        h, (cos, sin) = caps[li]
        q = layer.self_attn.q_proj(h).view(1, S, dims["n_q_heads"], d).transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        K, _ = get_kv(cache, li)                                    # (1, H_kv, ctx+S, d)
        K = K.repeat_interleave(rep, dim=1).float()
        s = (q.float() @ K.transpose(-1, -2)) / math.sqrt(d)      # (1, Hq, S, ctx+S)
        allowed = torch.ones(S, ctx_len + S, dtype=torch.bool, device=device)
        allowed[:, ctx_len:] = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
        p = torch.softmax(s.masked_fill(~allowed, float("-inf")), -1)
        total += p[..., :ctx_len].sum((0, 1, 2))
    cache.crop(ctx_len)
    if pool > 1:
        total = F.avg_pool1d(total[None, None], pool, stride=1, padding=pool // 2, count_include_pad=False)[0, 0, :ctx_len]
    return total


# ---------------------------------------------------------------------------------------- quantization
@torch.no_grad()
def fq_tokenwise(x, bits: int):
    """Groups of 64 along the feature axis, per token and head; FP16 scale + FP16 offset (min)."""
    B, H, L, d = x.shape
    g = x.float().view(B, H, L, d // GROUP, GROUP)
    mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((mx - mn) / qmax).clamp_min(1e-5).half().float()
    off = mn.half().float()
    q = torch.clamp(torch.round((g - off) / scale), 0, qmax)
    return (q * scale + off).view(B, H, L, d).to(x.dtype)


@torch.no_grad()
def fq_channelwise(x, bits: int):
    """Groups of 64 consecutive tokens per (head, channel); L must be a multiple of 64."""
    B, H, L, d = x.shape
    assert L % GROUP == 0
    g = x.float().view(B, H, L // GROUP, GROUP, d)
    mn, mx = g.amin(3, keepdim=True), g.amax(3, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((mx - mn) / qmax).clamp_min(1e-5).half().float()
    off = mn.half().float()
    q = torch.clamp(torch.round((g - off) / scale), 0, qmax)
    return (q * scale + off).view(B, H, L, d).to(x.dtype)


def quantize_kv_block(k, v, layout: str, bits: int):
    """Quantize complete 64-token blocks. layout 'token' = token-wise K and V; 'kivi' = channel-wise K, token-wise V."""
    kq = fq_channelwise(k, bits) if layout == "kivi" else fq_tokenwise(k, bits)
    return kq, fq_tokenwise(v, bits)


def rel_err(a, b):
    return float(((a.float() - b.float()) ** 2).sum() / (b.float() ** 2).sum().clamp_min(1e-12))


# ------------------------------------------------------------------------------------------------ policies
class Policy:
    name = "fp16"
    needs_scores = False

    def compress(self, cache, ctx_len, C, budget, scores=None):
        return cache, dict(bytes=bytes_fp16(C, ctx_len), kept_fp16=list(range(ctx_len)), n_fp16=ctx_len)

    def on_append(self, cache):
        pass


class SinkRecent(Policy):
    name = "evict_sink_recent"

    def compress(self, cache, ctx_len, C, budget, scores=None):
        n = eviction_keep(C, ctx_len, budget, 0, fixed_index_bytes=4 * INDEX_BYTES)
        keep = sorted(set(range(min(SINK, n))) | set(range(ctx_len - max(n - SINK, 0), ctx_len)))
        return _evict(cache, keep), dict(bytes=4 * C * len(keep) + 4 * INDEX_BYTES, kept_fp16=keep, n_fp16=len(keep),
                                         index_bytes=4 * INDEX_BYTES)


class QuestionAware(Policy):
    name = "evict_question"
    needs_scores = True

    def compress(self, cache, ctx_len, C, budget, scores=None):
        n = eviction_keep(C, ctx_len, budget, INDEX_BYTES)
        keep = _select(scores, ctx_len, n)
        return _evict(cache, keep), dict(bytes=4 * C * len(keep) + INDEX_BYTES * len(keep), kept_fp16=keep,
                                         n_fp16=len(keep), index_bytes=INDEX_BYTES * len(keep))


class QuantKV(Policy):
    """INT-b quantization of complete 64-token groups; trailing L mod 64 tokens (and new tokens until they
    complete a group) stay FP16 for both K and V."""
    needs_scores = False

    def __init__(self, layout="kivi", bits=4):
        self.layout, self.bits = layout, bits
        self.name = f"quant_{layout}_int{bits}"

    def compress(self, cache, ctx_len, C, budget, scores=None):
        kv = cache_to_list(cache)
        nq = (ctx_len // GROUP) * GROUP
        out, ek, evv = [], [], []
        for k, v in kv:
            k2, v2 = k.clone(), v.clone()
            if nq:
                k2[:, :, :nq], v2[:, :, :nq] = quantize_kv_block(k[:, :, :nq], v[:, :, :nq], self.layout, self.bits)
            ek.append(rel_err(k2, k))
            evv.append(rel_err(v2, v))
            out.append((k2, v2))
        self.q_end = nq
        return build_cache(out), dict(bytes=bytes_quant(C, ctx_len, self.bits), n_fp16=ctx_len - nq,
                                      kept_fp16=list(range(nq, ctx_len)), k_rel_err=sum(ek) / len(ek), v_rel_err=sum(evv) / len(evv))

    @torch.no_grad()
    def on_append(self, cache):
        L = cache.get_seq_length()
        while L - self.q_end >= GROUP:
            a, b = self.q_end, self.q_end + GROUP
            for l in range(n_cache_layers(cache)):
                k, v = get_kv(cache, l)
                k[:, :, a:b], v[:, :, a:b] = quantize_kv_block(k[:, :, a:b], v[:, :, a:b], self.layout, self.bits)
            self.q_end = b


class Hybrid(Policy):
    """8 sink + 64 recent + question-ranked positions in FP16; all other positions channel-wise-K /
    token-wise-V INT2 in complete 64-position groups (in token order); the incomplete remainder stays FP16."""
    needs_scores = True

    def __init__(self, low_bits=2):
        self.low_bits = low_bits
        self.name = f"hybrid_fp16_int{low_bits}"

    def compress(self, cache, ctx_len, C, budget, scores=None):
        n_fp = hybrid_keep(C, ctx_len, budget, SINK + RECENT, self.low_bits)
        fp = _select(scores, ctx_len, n_fp)
        fp_set = set(fp)
        low = [i for i in range(ctx_len) if i not in fp_set]
        nq = (len(low) // GROUP) * GROUP
        qidx = torch.tensor(low[:nq], device=get_kv(cache, 0)[0].device, dtype=torch.long)
        out, ek, evv = [], [], []
        for k, v in cache_to_list(cache):
            k2, v2 = k.clone(), v.clone()
            if nq:
                kq, vq = quantize_kv_block(k[:, :, qidx], v[:, :, qidx], "kivi", self.low_bits)
                k2[:, :, qidx], v2[:, :, qidx] = kq, vq
            ek.append(rel_err(k2, k))
            evv.append(rel_err(v2, v))
            out.append((k2, v2))
        fp16_all = sorted(fp_set | set(low[nq:]))
        return build_cache(out), dict(bytes=hybrid_bytes(C, ctx_len, n_fp, self.low_bits), n_fp16=len(fp16_all),
                                      kept_fp16=fp, selected_fp16=fp, residual_fp16=low[nq:], n_low=nq,
                                      k_rel_err=sum(ek) / len(ek), v_rel_err=sum(evv) / len(evv), index_bytes=INDEX_BYTES * n_fp)


def _select(scores, ctx_len, n):
    """8 sink + 64 recent safeguards, remaining budget filled by the highest question-aware scores."""
    n = min(n, ctx_len)
    keep = set(range(min(SINK, ctx_len))) | set(range(max(ctx_len - RECENT, 0), ctx_len))
    if n > len(keep):
        s = scores.clone()
        s[list(keep)] = float("-inf")
        extra = torch.topk(s, n - len(keep)).indices.tolist()
        keep |= set(extra)
    return sorted(keep)


def _evict(cache, keep):
    idx = torch.tensor(keep, device=get_kv(cache, 0)[0].device, dtype=torch.long)
    return build_cache([(k[:, :, idx], v[:, :, idx]) for k, v in cache_to_list(cache)])


class KeepAll(Policy):
    """100%-retention control through the eviction code path (rebuilds the cache from an index list)."""
    name = "keep_all_rebuilt"

    def compress(self, cache, ctx_len, C, budget, scores=None):
        keep = list(range(ctx_len))
        return _evict(cache, keep), dict(bytes=bytes_fp16(C, ctx_len), kept_fp16=keep, n_fp16=ctx_len)


def make_policies():
    return [Policy(), SinkRecent(), QuestionAware(), QuantKV("token", 4), QuantKV("kivi", 4), Hybrid(2)]


# ------------------------------------------------------------------------------------------- one prompt
@torch.no_grad()
def run_policy(model, tok, case: dict, policy: Policy, ref: dict | None = None, max_new: int = 16, device="cuda") -> dict:
    """Prefill -> (score) -> compress -> suffix -> teacher-forced answer NLL/KL -> greedy generation."""
    from shared.src.timing import peak_memory, reset_peak, sync
    dims = kv_dims(model)
    C = dims["C"]
    ids, Lc = case["ids"], case["ctx_len"]
    suffix, ans = ids[Lc:], case["answer_ids"]
    budget = bytes_quant(C, Lc, 4)
    t = {}
    reset_peak()
    sync(); t0 = time.perf_counter()
    cache, _ = prefill(model, ids[:Lc], device)
    sync(); t["prefill_ms"] = (time.perf_counter() - t0) * 1e3
    scores = None
    if policy.needs_scores:
        t0 = time.perf_counter()
        scores = question_scores(model, cache, suffix, Lc, device=device)
        sync(); t["scoring_ms"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    cache, info = policy.compress(cache, Lc, C, budget, scores)
    sync(); t["compress_ms"] = (time.perf_counter() - t0) * 1e3
    phys_ctx = cache.get_seq_length()
    # question suffix (identical tokens, original absolute positions)
    t0 = time.perf_counter()
    logits_s, cache = forward_step(model, cache, suffix, Lc, device)
    policy.on_append(cache)
    sync(); t["suffix_ms"] = (time.perf_counter() - t0) * 1e3
    # teacher-forced reference answer (then crop back)
    L_before = cache.get_seq_length()
    lg_tf, _ = forward_step(model, cache, ans[:-1], Lc + len(suffix), device)
    tf_logits = torch.cat([logits_s[-1:], lg_tf], 0) if len(ans) > 1 else logits_s[-1:]
    logp = torch.log_softmax(tf_logits, -1)
    nll = float(-logp[torch.arange(len(ans)), torch.tensor(ans, device=logp.device)].sum())
    cache.crop(L_before)
    # greedy generation
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    gen, nxt, pos = [], int(logits_s[-1].argmax()), Lc + len(suffix)
    t0 = time.perf_counter()
    for _ in range(max_new):
        gen.append(nxt)
        if nxt in eos:
            break
        lg, cache = forward_step(model, cache, [nxt], pos, device)
        policy.on_append(cache)
        pos += 1
        nxt = int(lg[-1].argmax())
    sync(); t["decode_ms"] = (time.perf_counter() - t0) * 1e3
    text = tok.decode(gen, skip_special_tokens=True)
    pred = extract_code(text)
    mem = peak_memory()
    kept = set(info.get("kept_fp16", []))
    ev = case["evidence_tokens"]
    res = dict(id=case["id"], policy=policy.name, ctx_len=Lc, n_tokens=case["n_tokens"], suffix_len=len(suffix),
               nominal_len=case["nominal_len"], position=case["position"], distractors=case["distractors"],
               code=case["code"], pred=pred, correct=int(pred == case["code"]), text=text, n_gen=len(gen),
               answer_nll=nll, bytes=info["bytes"], budget_bytes=budget, fp16_bytes=bytes_fp16(C, Lc),
               frac_fp16_bytes=info["bytes"] / bytes_fp16(C, Lc), within_budget=bool(info["bytes"] <= budget),
               n_fp16_positions=info.get("n_fp16"), phys_ctx_len=phys_ctx,
               evidence_retained=float(sum(i in kept for i in ev) / len(ev)),
               shared_decode_bytes=4 * C * (len(suffix) + len(gen)),
               k_rel_err=info.get("k_rel_err"), v_rel_err=info.get("v_rel_err"), **t, **mem)
    res["kept_fp16"] = info.get("kept_fp16") if policy.name != "fp16" else None
    if ref is not None:
        p_ref = torch.log_softmax(ref["tf_logits"], -1)
        kl = (p_ref.exp() * (p_ref - logp)).sum(-1)
        res["kl_first"] = float(kl[0])
        res["kl_tf_mean"] = float(kl.mean())
        res["ref_correct"] = ref["correct"]
    res["_tf_logits"] = tf_logits
    return res
