"""Task 3 prompts: 500 GSM8K train + 300 short-form train (distillation pool), 60 GSM8K test + 60 short-form test
(held-out). One common non-thinking chat format: [{"role": "user", "content": prompt}] with the model's default
template and add_generation_prompt=True -- identical to the course prefix-bank script."""
from __future__ import annotations

import random
import re

from shared.src.io import read_jsonl
from shared.src.paths import DATA_DIR, PUBLIC_DIR

TARGET = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
CAPS = {"math": 192, "short": 32}               # training response caps
EVAL_CAPS = {"math": 512, "short": 32}          # standalone answer-quality caps
QUOTA = {"math": 30_000, "short": 10_000}       # teacher-scored response tokens per method per seed


def _gsm(split):
    rows = sorted(read_jsonl(PUBLIC_DIR / f"gsm8k_{split}_locked.jsonl.gz"), key=lambda r: r["order"])
    return rows


def train_prompts():
    math = [dict(pid=f"GSM8K-train-{r['order']:04d}", prompt=r["example"]["question"], domain="math",
                 answer=gsm_gold(r["example"]["answer"]), source_row=r["source_row_index"]) for r in _gsm("train")]
    short = [dict(pid=x["id"], prompt=x["input"], domain="short", answer=x["answer"], family=x["family"])
             for x in read_jsonl(DATA_DIR / "short_form" / "short_train.jsonl")]
    assert len(math) == 500 and len(short) == 300
    return math, short


def test_prompts():
    """The 120 held-out prompts, ids identical to the course prefix banks (GSM8K-0000.., SHORT-familiar/shifted-..)."""
    out = [dict(pid=f"GSM8K-{r['order']:04d}", prompt=r["example"]["question"], domain="math",
                answer=gsm_gold(r["example"]["answer"]), workload="math") for r in _gsm("test")]
    for x in read_jsonl(DATA_DIR / "short_form" / "short_test.jsonl"):
        out.append(dict(pid=x["id"], prompt=x["input"], domain="short", answer=x["answer"], family=x["family"],
                        workload=f"short_{x['split']}"))
    assert len(out) == 120
    return out


def check_disjoint():
    tr_m, tr_s = train_prompts()
    te = test_prompts()
    te_text = {x["prompt"] for x in te}
    overlap = [x["pid"] for x in tr_m + tr_s if x["prompt"] in te_text]
    return dict(n_train=len(tr_m) + len(tr_s), n_test=len(te), exact_prompt_overlap=overlap)


def encode_prompt(tok, prompt: str) -> list:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)


def eos_ids(tok) -> set:
    return {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"), tok.eos_token_id}


class PromptSchedule:
    """Seeded round-robin over each category's fixed permutation (prompts reused after exhaustion); the next
    category is the one furthest behind its token quota (deterministic, identical for both methods)."""

    def __init__(self, math, short, seed=6304):
        self.pools = {"math": list(math), "short": list(short)}
        for i, k in enumerate(("math", "short")):
            random.Random(seed + 17 * i).shuffle(self.pools[k])
        self.ptr = {"math": 0, "short": 0}
        self.visits = {}

    def next(self, scored: dict):
        live = [c for c in ("math", "short") if scored[c] < QUOTA[c]]
        if not live:
            return None
        cat = min(live, key=lambda c: scored[c] / QUOTA[c])
        pool = self.pools[cat]
        item = pool[self.ptr[cat] % len(pool)]
        self.ptr[cat] += 1
        rep = self.visits.get(item["pid"], 0)
        self.visits[item["pid"]] = rep + 1
        return dict(item, repetition=rep)


# ------------------------------------------------------------------------------------------ answer scoring
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm_gold(answer: str) -> str:
    return _norm_num(answer.split("####")[-1])


def _norm_num(s: str):
    s = s.strip().replace(",", "").replace("$", "")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_math(text: str):
    """Final answer: number after '####' or 'answer is', else the last number in the response."""
    for pat in (r"####\s*(-?[\d,]*\.?\d+)", r"answer is[:\s\$]*(-?[\d,]*\.?\d+)", r"\\boxed\{(-?[\d,]*\.?\d+)\}"):
        m = re.findall(pat, text)
        if m:
            return _norm_num(m[-1])
    nums = NUM_RE.findall(text)
    return _norm_num(nums[-1]) if nums else None


def score_short(text: str, answer: str) -> dict:
    t = text.strip().strip(".").strip()
    return dict(exact_format=int(t == answer), contains=int(answer.lower() in text.lower()))
