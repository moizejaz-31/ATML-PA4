"""Dense FP16 execution cost: prefill latency (512 / 2048 tokens), 64-token decode latency, peak memory.

Pruned (zeros) and fake-quantized (dequantized FP16) models run the same dense kernels as the baseline, so
these numbers measure the executed implementation, not a hypothetical sparse / low-bit kernel.
"""
from __future__ import annotations

import torch

from shared.src.timing import benchmark, peak_memory, reset_peak, sync


@torch.no_grad()
def measure_latency(model, prompt_lens=(512, 2048), decode_tokens: int = 64, reps: int = 5, warmup: int = 2,
                    device: str = "cuda", seed: int = 6304) -> dict:
    g = torch.Generator().manual_seed(seed)
    vocab = model.config.vocab_size
    out = dict(batch_size=1, reps=reps, warmup=warmup, unit="ms")
    for L in prompt_lens:
        ids = torch.randint(100, min(vocab, 150_000), (1, L), generator=g).to(device)
        reset_peak()
        r = benchmark(lambda: model(input_ids=ids, use_cache=True), warmup=warmup, reps=reps)
        out[f"prefill_{L}"] = dict(mean_ms=r["mean"], std_ms=r["std"], all_ms=r["all"], **peak_memory())
    # decode: prefill 512 tokens, then 64 greedy single-token steps reusing the KV cache
    ids = torch.randint(100, min(vocab, 150_000), (1, prompt_lens[0]), generator=g).to(device)

    def decode():
        o = model(input_ids=ids, use_cache=True)
        pkv, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        sync()
        import time
        t0 = time.perf_counter()
        for _ in range(decode_tokens):
            o = model(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        sync()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        decode()
    reset_peak()
    times = [decode() for _ in range(reps)]
    import numpy as np
    out[f"decode_{decode_tokens}"] = dict(mean_ms=float(np.mean(times)), std_ms=float(np.std(times, ddof=1)),
                                          per_token_ms=float(np.mean(times)) / decode_tokens, all_ms=times,
                                          context=prompt_lens[0], **peak_memory())
    out["weights_allocated_mb"] = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None
    return out
