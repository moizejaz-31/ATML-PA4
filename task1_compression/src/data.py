"""Task 1 data: locked C4 calibration/development windows, perplexity windows and HellaSwag."""
from __future__ import annotations

import re

import torch

from shared.src.io import read_jsonl
from shared.src.paths import PUBLIC_DIR

N_CALIB, N_DEV, SEQ_LEN, EVAL_TOKENS = 128, 32, 512, 50_000
HELLASWAG_FULL, HELLASWAG_DIAG = 2000, 512


def _windows(fname, expected_n, expected_len=SEQ_LEN):
    rows = read_jsonl(PUBLIC_DIR / fname)
    assert len(rows) == expected_n, f"{fname}: {len(rows)} rows"
    assert all(len(r["token_ids"]) == expected_len for r in rows)
    return torch.tensor([r["token_ids"] for r in rows], dtype=torch.long), [r["source_row"] for r in rows]


def load_calibration():
    """128 x 512 C4-train token windows (seed 6304 manifest) used by Wanda, AWQ and every probe."""
    return _windows("c4_calibration_locked.jsonl.gz", N_CALIB)


def load_development():
    """32 x 512 disjoint C4-train windows used only to choose allocation hyperparameters."""
    return _windows("c4_development_locked.jsonl.gz", N_DEV)


def load_eval_windows(suite: str):
    """Perplexity windows. suite in {'wikitext2', 'c4'}; each record scores token_ids[1:n+1]."""
    fname = {"wikitext2": "wikitext2_test_windows_locked.jsonl.gz", "c4": "c4_validation_windows_locked.jsonl.gz"}[suite]
    rows = read_jsonl(PUBLIC_DIR / fname)
    total = sum(r["scored_tokens"] for r in rows)
    assert total == EVAL_TOKENS, f"{suite}: {total} scored tokens"
    for r in rows:
        lo, hi = r["score_target_offsets"]
        assert lo == 1 and hi - lo == r["scored_tokens"] and len(r["token_ids"]) == hi
    return rows


# ---------------------------------------------------------------- HellaSwag (lm-evaluation-harness convention)
def hs_preprocess(text: str) -> str:
    """Identical to lm-eval's hellaswag ``preprocess`` (task version 1.0)."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def hs_doc(ex: dict) -> dict:
    ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
    return dict(query=hs_preprocess(ex["activity_label"] + ": " + ctx),
                choices=[hs_preprocess(e) for e in ex["endings"]], gold=int(ex["label"]))


HELLASWAG_FORMAT = ("lm-eval hellaswag v1.0: context = preprocess(activity_label + ': ' + ctx_a + ' ' + ctx_b.capitalize()); "
                    "continuation = ' ' + preprocess(ending); score = sum log p(continuation | context); "
                    "acc_norm divides by len(preprocess(ending)) in characters")


def load_hellaswag(n: int | None = None):
    rows = sorted(read_jsonl(PUBLIC_DIR / "hellaswag_val_locked.jsonl.gz"), key=lambda r: r["order"])
    assert len(rows) == HELLASWAG_FULL
    rows = rows[:n] if n else rows
    out = []
    for r in rows:
        d = hs_doc(r["example"])
        d.update(order=r["order"], source_row_index=r["source_row_index"], ind=r["example"].get("ind"))
        out.append(d)
    return out
