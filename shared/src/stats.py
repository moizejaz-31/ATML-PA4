"""Bootstrap utilities: paired intervals for accuracies and for pooled-token perplexity."""
from __future__ import annotations

import numpy as np

DEFAULT_RESAMPLES = 2000


def bootstrap_mean_ci(x, n_boot: int = DEFAULT_RESAMPLES, ci: float = 0.95, seed: int = 6304):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats = x[idx].mean(1)
    a = (1 - ci) / 2
    return float(x.mean()), float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def paired_bootstrap_diff(a, b, n_boot: int = DEFAULT_RESAMPLES, ci: float = 0.95, seed: int = 6304):
    """Mean of (a - b) over paired examples with a percentile CI (resampling examples)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, "paired arrays must align"
    return bootstrap_mean_ci(a - b, n_boot, ci, seed)


def paired_bootstrap_ppl(nll_a, nll_b, counts, n_boot: int = DEFAULT_RESAMPLES, ci: float = 0.95, seed: int = 6304):
    """Paired PPL difference (A - B) resampling whole evaluation windows.

    ``nll_*`` are per-window summed token NLLs, ``counts`` the scored tokens per window.
    Each resample pools token NLL over the drawn windows before exponentiating.
    Returns dict with observed ppl_a, ppl_b, diff and CI of the diff (and of the log-ratio).
    """
    nll_a, nll_b, counts = (np.asarray(v, dtype=np.float64) for v in (nll_a, nll_b, counts))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    n = counts[idx].sum(1)
    pa = np.exp(nll_a[idx].sum(1) / n)
    pb = np.exp(nll_b[idx].sum(1) / n)
    d = pa - pb
    lo = (1 - ci) / 2
    ppl_a = float(np.exp(nll_a.sum() / counts.sum()))
    ppl_b = float(np.exp(nll_b.sum() / counts.sum()))
    return dict(ppl_a=ppl_a, ppl_b=ppl_b, diff=ppl_a - ppl_b,
                diff_lo=float(np.quantile(d, lo)), diff_hi=float(np.quantile(d, 1 - lo)),
                mean_nll_diff=float((nll_a.sum() - nll_b.sum()) / counts.sum()),
                n_windows=int(len(counts)), n_tokens=int(counts.sum()), n_boot=n_boot)


def paired_bootstrap_ratio(num_a, num_b, n_boot: int = DEFAULT_RESAMPLES, ci: float = 0.95, seed: int = 6304):
    """CI for mean(a)/mean(b) over paired units (e.g. throughput ratios)."""
    a = np.asarray(num_a, dtype=np.float64)
    b = np.asarray(num_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    lo = (1 - ci) / 2
    return float(a.mean() / b.mean()), float(np.quantile(r, lo)), float(np.quantile(r, 1 - lo))
