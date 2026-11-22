"""Read saved Task 1 results and build summary tables / paired bootstrap comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.src.io import load_json, read_jsonl
from shared.src.stats import paired_bootstrap_diff, paired_bootstrap_ppl

ORDER = ["fp16", "p1_mag_s25", "p1_mag_s50", "p1_mag_s75", "p2_wanda_s25", "p2_wanda_s50", "p2_wanda_s75", "p3_s50",
         "p3_s75", "p2_wanda_24", "q1_int8", "q1_int4", "q1_int3", "q2_awq_int8", "q2_awq_int4", "q2_awq_int3",
         "q3_budget_int4", "q3_budget_int3"]


def load_condition(results_dir, name):
    return load_json(results_dir / "conditions" / f"{name}.json")


def available(results_dir):
    return [n for n in ORDER if (results_dir / "conditions" / f"{n}.json").exists()]


def summary_table(results_dir) -> pd.DataFrame:
    rows = []
    for n in available(results_dir):
        r = load_condition(results_dir, n)
        st = r["storage"]
        packed = st.get("packed", {}).get("total_bytes")
        rows.append(dict(condition=n, sparsity=r["nonzeros"]["sparsity"], wikitext2_ppl=r["wikitext2"]["ppl"],
                         wikitext2_nll=r["wikitext2"]["mean_nll"], c4_ppl=r["c4"]["ppl"], c4_nll=r["c4"]["mean_nll"],
                         hs_n=r["hellaswag"]["n"], hs_acc=r["hellaswag"]["acc"], hs_acc_norm=r["hellaswag"]["acc_norm"],
                         dense_fp16_MB=st["dense_fp16_bytes"] / 2**20, sparse16_MB=st["sparse16"]["total_bytes"] / 2**20,
                         packed_MB=packed / 2**20 if packed else None, compression_s=r["compression"]["time_s"],
                         prefill512_ms=r.get("resources", {}).get("prefill_512", {}).get("mean_ms"),
                         prefill2048_ms=r.get("resources", {}).get("prefill_2048", {}).get("mean_ms"),
                         decode64_ms=r.get("resources", {}).get("decode_64", {}).get("mean_ms"),
                         peak_alloc_MB=r.get("resources", {}).get("prefill_2048", {}).get("peak_allocated_mb")))
    return pd.DataFrame(rows).set_index("condition")


def per_window(results_dir, name, suite):
    d = load_json(results_dir / "per_example" / f"{name}_ppl.json")[suite]
    return np.array(d["window_nll_sums"]), np.array(d["window_counts"])


def per_example_hs(results_dir, name):
    return {r["order"]: r for r in read_jsonl(results_dir / "per_example" / f"{name}_hellaswag.jsonl")}


def paired_compare(results_dir, a: str, b: str, n_boot: int = 2000) -> dict:
    """A - B: perplexity differences (window bootstrap, pooled NLL) and HellaSwag accuracy differences
    (example bootstrap) on the examples both conditions scored."""
    out = dict(a=a, b=b)
    for suite in ("wikitext2", "c4"):
        na, ca = per_window(results_dir, a, suite)
        nb, cb = per_window(results_dir, b, suite)
        assert np.array_equal(ca, cb), "conditions must share the same windows"
        out[suite] = paired_bootstrap_ppl(na, nb, ca, n_boot)
    ha, hb = per_example_hs(results_dir, a), per_example_hs(results_dir, b)
    common = sorted(set(ha) & set(hb))
    for key in ("correct", "correct_norm"):
        m, lo, hi = paired_bootstrap_diff([ha[o][key] for o in common], [hb[o][key] for o in common], n_boot)
        out[f"hs_{key}"] = dict(diff=m, lo=lo, hi=hi, n=len(common))
    return out


def compare_table(results_dir, pairs) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        c = paired_compare(results_dir, a, b)
        rows.append({"A": a, "B": b,
                     "dPPL wiki [95% CI]": f"{c['wikitext2']['diff']:+.3f} [{c['wikitext2']['diff_lo']:+.3f}, {c['wikitext2']['diff_hi']:+.3f}]",
                     "dPPL c4 [95% CI]": f"{c['c4']['diff']:+.3f} [{c['c4']['diff_lo']:+.3f}, {c['c4']['diff_hi']:+.3f}]",
                     "dAcc pts [95% CI]": f"{100*c['hs_correct']['diff']:+.2f} [{100*c['hs_correct']['lo']:+.2f}, {100*c['hs_correct']['hi']:+.2f}]",
                     "dAccNorm pts [95% CI]": f"{100*c['hs_correct_norm']['diff']:+.2f} [{100*c['hs_correct_norm']['lo']:+.2f}, {100*c['hs_correct_norm']['hi']:+.2f}]",
                     "HS n": c["hs_correct"]["n"]})
    return pd.DataFrame(rows)
