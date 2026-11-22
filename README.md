# Model Compression, Efficient Attention & KV Memory, and Speculative Decoding

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![transformers 4.57.1](https://img.shields.io/badge/transformers-4.57.1-orange.svg)](https://github.com/huggingface/transformers)
[![Random Seed: 6304](https://img.shields.io/badge/Seed-6304-green.svg)](shared/src/seed.py)
![Course: EE--5102/CS--6304](https://img.shields.io/badge/Course-ATML--Fall2026-purple.svg)

> **Course:** EE-5102 / CS-6304: Advanced Topics in Machine Learning
> **Institution:** Lahore University of Management Sciences (LUMS)
> **Assignment:** Programming Assignment 4

---

## Executive Summary

1. **Task 1: Model Compression (`Qwen/Qwen2.5-1.5B` base, FP16).**
   * P1 magnitude and P2 Wanda unstructured per-row pruning at 25/50/75% eligible sparsity; P3 nonuniform Wanda with a
     block-level allocation (OWL outlier ratio vs isolated-removal Delta-NLL + exact DP), selected on 32 dev windows.
   * Q1 uniform group-128 asymmetric PTQ (INT8/4/3), Q2 AWQ-style activation-aware scaling (grid search, folding),
     Q3 mixed precision over {2,3,4,8} bits under the Q2 INT4/INT3 packed-byte budgets (knapsack on AWQ scale groups).
   * 2:4 Wanda diagnostic; WikiText-2 / C4 perplexity (50k scored tokens each), zero-shot HellaSwag (lm-eval
     `acc` / `acc_norm`), packed bytes with metadata, latency (prefill 512/2048, 64-token decode) and peak memory.
2. **Task 2: Attention and Context Memory.**
   * A: dense vs online-softmax tiled vs SDPA (dispatched backend recorded) — error, latency, memory up to L = 8192.
   * B: 2-layer MQAR Transformers (dense, W=64 window, ELU+1 linear, local/dilated) x 3 seeds + a delta-rule
     intervention; near/far/mixed accuracy, distance-by-load heatmaps, train- vs test-time mask control.
   * C: `Qwen2.5-1.5B-Instruct` KV cache on 180 retrieval prompts: sink+recent and question-aware FP16 eviction,
     token-wise vs KIVI-style INT4, and an FP16/INT2 hybrid, all at the same per-prompt INT4 byte budget.
3. **Task 3: Draft Distillation for Speculative Decoding (`1.5B-Instruct` target, `0.5B-Instruct` draft).**
   * Exact speculative sampler written from scratch (chi-square validated, negative control rejected).
   * Off-policy vs on-policy forward-KL LoRA distillation (2 seeds each, 40k teacher-scored tokens), merged drafts,
     common-prefix KL/TV/acceptance on fixed prefix banks, measured vs predicted speedup at gamma in {2, 4}.

## Repository Structure

```
.
├── README.md / requirements.txt / setup.py / .gitignore
├── shared/                      # seed, paths, IO, bootstrap stats, CUDA timers, environment record, notebook bootstrap
├── task1_compression/
│   ├── src/                     # data, evaluate (PPL/HellaSwag), layerwise calibrator, prune, quant, awq, allocation,
│   │                            # resources, runner (resumable conditions), analysis
│   ├── notebooks/               # 1 baseline -> 2 P1/P2 -> 3 P3 -> 4 Q1/Q2 -> 5 Q3 -> 6 2:4 + trade-offs
│   ├── tests/                   # CPU tests on a tiny random Qwen2
│   ├── splits/ results/ checkpoints/
├── task2_attention_kv/
│   ├── src/                     # attention_ops, mqar_{model,data,train}, kv_cache, kv_pack
│   ├── notebooks/               # 1 Part A -> 2 B1 -> 3 B2 -> 4 B3 + costs -> 5 C1 -> 6 C2-C4 -> 7 C5
│   ├── tests/  splits/ results/ checkpoints/
└── task3_speculative/
    ├── src/                     # data, sampling (exact speculative sampler), prefix_banks, distill, spec_eval
    ├── notebooks/               # 1 sampler validation -> 2 banks + baseline -> 3 off-policy -> 4 on-policy
    │                            # -> 5 merge + alignment -> 6 speculative after distillation -> 7 workload/cost
    ├── tests/  splits/ results/ checkpoints/
```

Each task folder has its own README with the protocol, conditions and outputs.

## Setup (Kaggle / Colab, one T4 GPU)

Every notebook starts with the same setup cell. It finds this repo, or clones it if it's missing. It installs the pinned
`transformers==4.57.1` / `huggingface_hub==0.36.2` if needed, and then **asks you to restart the kernel once**. It also
clones the course data repository [`atml-pa4-datasets`](https://github.com/AbDu11aHHH/atml-pa4-datasets) into
`./atml-pa4-datasets`, and runs `scripts/setup_data.py --public` on first use (C4 streaming is slow, so do this early).
Keep `atml-pa4-datasets/data/` and `data/local_source_revisions.json` between sessions, for example as a Kaggle
dataset, and set `PA4_DATA_REPO` to its path.

```bash
git clone https://github.com/moizejaz-31/ATML-PA4.git && cd ATML-PA4
pip install -r requirements.txt && pip install -e .
python -m pytest task1_compression/tests task2_attention_kv/tests task3_speculative/tests -q   # CPU, ~2 min
```

**Kaggle runners:** `kaggle_runners/run_task{1,2,3}_kaggle.ipynb` each run one task's notebooks in order with
papermill, starting a fresh kernel for every notebook. To use one, import it into Kaggle (File → Import Notebook →
GitHub), set GPU T4 x1 and Internet On, then choose Save Version → Save & Run All. To resume from an earlier run's
output, attach that output as an input and set `RESUME_FROM` in the runner.

Run each task's notebooks in numeric order. Every expensive step writes its result as soon as it finishes and skips work
that is already saved, so a stopped notebook resumes where it left off when re-run.

## Deviations and Implementation Notes

* **Prefix banks:** `setup_data.py --prefix-banks` passes `generator=` to `model.generate`, which transformers 4.57.1
  rejects. `task3_speculative/src/prefix_banks.py` reproduces that script's protocol (prompts, ids, seeds,
  position-sampling RNG strings, schema) with its own seeded KV-cache sampling loop.
* **Fake quantization / sparsity:** compressed weights run on dense FP16 kernels; packed and sparse storage sizes are
  computed exactly and verified by real bit-packing. Measured latency and memory are therefore labelled as dense execution.
* **AWQ:** reproduces activation-derived scales, the grid search and folding; the objective is the linear-group output
  error from the calibration Gram matrix, there is no clipping search, and o_proj keeps an explicit FP16 input scale
  (GQA prevents folding). The reference code's behaviour is available via `o_mode="skip"`.
* **Question-aware KV retention** is a simplified, fixed SnapKV-inspired policy shared across layers and heads.

## Attribution

All method code (pruning, quantization, AWQ-style scaling, allocation rules, attention operators, MQAR models, KV-cache
policies and packing, the speculative sampler and distillation loops) is our own implementation of the cited methods.
We build on PyTorch, Hugging Face Transformers / PEFT / Datasets, and the course data repository's generator and
data-preparation scripts. The HellaSwag preprocessing reproduces lm-evaluation-harness's `hellaswag` task. Models and
datasets are downloaded at runtime and not committed. Code was written with assistance from Claude (Anthropic).

**Author:** Moiz Ejaz
