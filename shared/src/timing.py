"""CUDA-synchronized timing and memory helpers (all latencies reported in milliseconds)."""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Accumulating wall-clock timer with CUDA synchronisation on both ends."""

    def __init__(self):
        self.totals = {}

    @contextmanager
    def __call__(self, name: str):
        sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync()
            self.totals[name] = self.totals.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def as_dict(self):
        return dict(self.totals)


@contextmanager
def timed(store: dict, key: str):
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    store[key] = store.get(key, 0.0) + (time.perf_counter() - t0) * 1e3


def benchmark(fn, warmup: int = 3, reps: int = 10) -> dict:
    """Run ``fn`` ``warmup`` times, then time ``reps`` synchronized repetitions (ms)."""
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    t = np.array(times)
    return dict(unit="ms", reps=reps, warmup=warmup, mean=float(t.mean()), std=float(t.std(ddof=1) if reps > 1 else 0.0),
                median=float(np.median(t)), min=float(t.min()), max=float(t.max()), all=t.tolist())


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory() -> dict:
    if not torch.cuda.is_available():
        return dict(peak_allocated_mb=None, peak_reserved_mb=None)
    return dict(peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20)


def measure_peak(fn) -> tuple:
    """Run ``fn`` once and return (result, peak-memory dict measured from a fresh baseline)."""
    reset_peak()
    base = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None
    out = fn()
    sync()
    pm = peak_memory()
    pm["baseline_allocated_mb"] = base
    if base is not None:
        pm["peak_above_baseline_mb"] = pm["peak_allocated_mb"] - base
    return out, pm
