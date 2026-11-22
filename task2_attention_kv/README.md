# Task 2: Exact Attention, MQAR Recall, and KV-Cache Memory

| Notebook | Content |
|---|---|
| `1_exact_attention_A1_A3` | dense / tiled / SDPA: backend table, output + gradient error vs FP32 reference, causality and edge cases, tile sweep, latency + memory to L = 8192 |
| `2_mqar_B1_dense_and_local` | dense (K = 4/8/16 mixed, K = 16 near) and W=64 window (K = 4/8/16 near, K = 16 mixed), 3 seeds, test-time mask control |
| `3_mqar_B2_linear_and_dilated` | ELU+1 linear (K = 4/8/16) and local/dilated heads (K = 4/16), mixed regime |
| `4_mqar_B3_intervention_and_costs` | delta-rule state update at K = 16 (prediction stated first), distance x load heatmaps, theoretical vs measured costs, inference sweep T <= 8192 |
| `5_kv_C1_reference_validation` | split prefill, 100%-retention, RoPE-position tests, full-cache accuracy per cell |
| `6_kv_C2_C4_compressed_caches` | FP16 + 5 compressed caches on all 180 prompts at the per-prompt INT4 KIVI budget |
| `7_kv_C5_retention_and_systems` | main table, paired bootstrap, breakdowns, systems costs, real pack/unpack benchmark, selection masks |

* **MQAR:** the course generator (`atml-pa4-datasets/scripts/mqar_generator.py`) produces the training stream; the
  frozen dev and test files are used unchanged. The loss is computed only at query positions over the 64 value IDs.
  Windowed and dilated attention use explicit dense boolean masks for training (a correctness implementation). A
  gather-based version is benchmarked separately.
* **KV budget:** `M_4(L) = 2 n_l H_kv d_h [floor(L/64)(64*4/8+4) + 2(L mod 64)]` on the pre-question context. Eviction
  keeps the largest FP16 position count that fits, including int16 indices. The hybrid's FP16 count is the largest that
  fits once INT2 payload, metadata, indices and the FP16 remainder are counted. `within_budget` is asserted for every prompt.

Outputs: `results/attention/`, `results/mqar/` (histories, per-query test results, tables), `results/kv/`
(per-prompt JSONL, selections, C5 tables, pack benchmark), figures in `results/figures/`.
