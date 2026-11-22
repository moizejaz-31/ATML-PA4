# Task 3: Draft-Model Distillation for Speculative Decoding

Target/teacher `Qwen/Qwen2.5-1.5B-Instruct`, draft/student `Qwen/Qwen2.5-0.5B-Instruct` (same tokenizer and
151,936-token vocabulary, checked). All prompts use `[{"role": "user", "content": prompt}]` with the default chat
template. EOS is `<|im_end|>` or `<|endoftext|>`.

| Notebook | Content |
|---|---|
| `1_sampler_validation` | 100k-sample Markov chi-square (gamma 1/2/4) + rejected negative control, joint-law table, real-model cache / split-prefill / identical-draft checks, EOS and truncation mechanics |
| `2_prefix_banks_and_pretrained_baseline` | teacher, original-draft and exact-speculative banks, teacher-vs-speculative consistency test, isolated step costs, baseline benchmark (EOS-aware 64 cap + fixed-64 ignore-EOS on 24 math prompts) |
| `3_offpolicy_distillation` | frozen teacher pool, off-policy LoRA runs (seeds 6304, 6305) |
| `4_onpolicy_distillation` | on-policy LoRA runs, curves, cost breakdown |
| `5_merge_and_distribution_alignment` | merging + merged-vs-unmerged error, student own-rollout banks, KL/TV/acceptance on every bank |
| `6_speculative_after_distillation` | benchmark of the 4 merged drafts, unmerged wrapper timing, cost model vs measured speedup, time decomposition |
| `7_workload_and_cost_analysis` | workload / length breakdowns, bootstrap CIs, expected vs empirical acceptance, standalone accuracy, deployment break-even, continuations |

* **Sampler** (`src/sampling.py`): `verify_round` implements acceptance, residual and bonus sampling. After every
  round, each KV cache holds at most `seq[:-1]`, so a rejection becomes a `crop` and full acceptance feeds the pending
  proposal on the next round. The same code drives the toy test and the real models.
* **Distillation** (`src/distill.py`): uses the forward KL over the full vocabulary at every response position.
  Updates happen every >= 256 teacher-scored tokens (whole responses), with gradients divided by the update's token
  count after unscaling. Quotas are 30,000 math + 10,000 short tokens. Every presentation is logged with its prompt
  ID, seed, response IDs and scored-token count.
* **Prefix banks:** built locally (see top-level README), with the same protocol as the course script.

Outputs: `results/sampler/`, `results/distill/`, `results/alignment/`, `results/spec/`, figures in
`results/figures/`. Adapters, merged drafts and banks live in `checkpoints/` (not committed).
