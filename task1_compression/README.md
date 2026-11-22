# Task 1: Pruning and Quantization of Qwen2.5-1.5B

Eligible weights: q/k/v/o/gate/up/down of all 28 blocks (196 matrices, 1,310,195,712 weights). Embeddings (tied with
the LM head, counted once), norms and biases stay FP16. Every condition starts from a restored copy of the unchanged FP16 weights.

| Notebook | Conditions / content |
|---|---|
| `1_setup_and_fp16_baseline` | data prep, environment, enumeration test, saved IDs, `fp16` |
| `2_pruning_P1_magnitude_P2_wanda` | `p1_mag_s{25,50,75}`, `p2_wanda_s{25,50,75}`, dense input norms |
| `3_nonuniform_pruning_P3` | OWL + isolated-removal probes, dev selection, `p3_s50`, `p3_s75` |
| `4_quantization_Q1_uniform_Q2_awq` | packing check, `q1_int{8,4,3}`, AWQ search, rescaling check, `q2_awq_int{8,4,3}` |
| `5_mixed_precision_Q3` | unit probes, knapsack allocations, dev selection, `q3_budget_int4`, `q3_budget_int3` |
| `6_structured_2to4_and_tradeoffs` | `p2_wanda_24`, summary tables, trade-off plots, transfer check, qualitative cases |

**Evaluation (identical code for all conditions, `src/evaluate.py`):** independent 512-token windows (first token of each
window is unscored), 50,000 scored tokens each on WikiText-2 test and C4 validation, reporting N, mean NLL and PPL.
HellaSwag uses the lm-eval v1.0 preprocessing, and `acc_norm` divides by the ending's character length. All 2,000 IDs
are scored for `fp16`, the three 50% pruning methods and every INT4/INT3 condition; the first 512 IDs are scored for the rest.

**Selection hygiene:** the probes use the first 32 of the 128 calibration windows. Allocation hyperparameters are chosen
only on the 32 disjoint C4-train development windows (`p3_candidates.json`, `q3_candidates.json`), and every
candidate is kept.

**Outputs:** `results/conditions/<name>.json`, `results/per_example/<name>_{ppl.json,hellaswag.jsonl}`,
`results/summary_all_conditions.csv`, `results/paired_vs_fp16.csv`, figures in `results/figures/`.
