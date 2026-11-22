"""Fast CPU checks of the Task 1 machinery on a tiny random Qwen2 model:  python -m pytest task1_compression/tests -q"""
import numpy as np
import pytest
import torch

from task1_compression.src import allocation, awq, prune, quant
from task1_compression.src.evaluate import eval_perplexity, nll_on_sequences
from task1_compression.src.model_utils import WeightSnapshot, count_nonzeros, eligible_linears, inventory


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=384, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, tie_word_embeddings=True, max_position_embeddings=256)
    m = Qwen2ForCausalLM(cfg).eval()
    m.config.use_cache = False
    ids = torch.randint(0, 512, (8, 64))
    return m, ids


def test_inventory(tiny):
    m, _ = tiny
    inv = inventory(m)
    assert inv["n_eligible_matrices"] == 21 and inv["tied_embeddings"]
    assert inv["n_total_params_unique"] < inv["n_params_with_duplicates"]


def test_masks_and_exact_targets(tiny):
    m, _ = tiny
    s = torch.randn(10, 16)
    keep = prune.row_mask(s, prune.row_counts(37, 10, 16))
    assert (~keep).sum() == 37
    k24 = prune.nm_mask(s, 2, 4)
    assert k24.view(10, 4, 4).sum(-1).eq(2).all()
    z = prune.exact_zero_targets(m, {0: 0.3, 1: 0.5, 2: 0.7}, 0.5)
    assert sum(z.values()) == round(0.5 * inventory(m)["n_eligible_weights"])


@pytest.mark.parametrize("method", ["magnitude", "wanda"])
def test_prune_sequential(tiny, method):
    m, ids = tiny
    snap = WeightSnapshot(m)
    r = prune.prune_sequential(m, ids, method, 0.5, batch_size=4, device="cpu", collect_scores=True)
    assert abs(r["achieved_sparsity"] - 0.5) < 1e-9
    assert abs(count_nonzeros(m)["sparsity"] - 0.5) < 1e-3
    assert all("out_err_rel" in v for v in r["per_projection"].values())
    snap.restore(m)
    r = prune.prune_sequential(m, ids, "wanda", None, nm=(2, 4), batch_size=4, device="cpu")
    assert abs(r["achieved_sparsity"] - 0.5) < 1e-9
    z = prune.exact_zero_targets(m, {0: 0.2, 1: 0.6, 2: 0.7}, 0.5)
    snap.restore(m)
    r = prune.prune_sequential(m, ids, "wanda", z, batch_size=4, device="cpu")
    assert r["zeros"] == sum(z.values())
    snap.restore(m)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_quantizer_and_packing(bits):
    W = torch.randn(64, 256)
    c, s, z, pad = quant.quantize_groupwise(W, bits)
    assert c.max() <= 2 ** bits - 1 and pad == 0
    Wh = quant.dequantize_groupwise(c, s, z, pad, dtype=torch.float32)
    step = (W.view(64, 2, 128).amax(-1) - W.view(64, 2, 128).amin(-1)) / (2 ** bits - 1)
    assert ((W - Wh).abs().view(64, 2, 128) <= step[..., None] * 0.51 + 1e-3).all()
    packed = quant.pack_bits(c, bits)
    assert torch.equal(quant.unpack_bits(packed, bits, c.numel()).view_as(c), c)
    ser = quant.serialize_matrix(W, bits)
    assert ser == quant.matrix_packed_bytes(64, 256, bits)


def test_awq_rescaling_preserves_function(tiny):
    m, ids = tiny
    snap = WeightSnapshot(m)
    with torch.no_grad():
        ref = m(ids[:2]).logits
    res = awq.search_awq_scales(m, ids, bits_list=(3, 4), batch_size=4, device="cpu")
    assert all(res["log"][k][4]["err"] <= res["log"][k][4]["err_identity"] + 1e-6 for k in res["log"])
    ub = awq.unit_bits_uniform(3, 4)
    awq.apply_awq_(m, res["scales"], ub, quantize=False)
    with torch.no_grad():
        out = m(ids[:2]).logits
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3), (out - ref).abs().max()
    snap.restore(m)
    with torch.no_grad():
        assert torch.allclose(m(ids[:2]).logits, ref)
    awq.apply_awq_(m, res["scales"], ub, quantize=True)
    st = quant.model_packed_bytes(m, awq.projection_bits(ub, 3), explicit_scale_channels=3 * 128)
    assert st["total_bytes"] < quant.dense_fp16_bytes(m)
    snap.restore(m)


def test_allocation_rules():
    rng = np.random.default_rng(0)
    grid = allocation.probe_grid()
    curves = np.cumsum(rng.random((6, len(grid))), 1) * rng.random((6, 1))
    a = allocation.dp_allocation(curves, grid, 0.5)
    assert abs(a.mean() - 0.5) < 1e-9 and a.min() >= 0.1 - 1e-9 and a.max() <= 0.85 + 1e-9
    o = allocation.owl_allocation(rng.random(6), 0.75, 0.08)
    assert abs(o.mean() - 0.75) < 1e-6 and o.max() <= 0.85 + 1e-9
    units = {f"u{i}": dict(cost={2: 4.0 / (i + 1), 3: 1.0 / (i + 1), 4: 0.3 / (i + 1), 8: 0.0},
                           bytes={b: 100 * b * (i + 1) for b in (2, 3, 4, 8)}) for i in range(5)}
    budget = sum(u["bytes"][4] for u in units.values())
    r = allocation.allocate_bits(units, budget)
    assert r["total_bytes"] <= budget


def test_perplexity_windows(tiny):
    m, ids = tiny
    wins = [dict(token_ids=ids[i, :40].tolist(), scored_tokens=39, score_target_offsets=[1, 40]) for i in range(5)]
    r = eval_perplexity(m, wins, batch_size=2, device="cpu")
    assert r["n_scored_tokens"] == 5 * 39
    single = np.mean([eval_perplexity(m, [w], device="cpu")["mean_nll"] for w in wins])
    assert abs(single - r["mean_nll"]) < 1e-4
    assert np.isfinite(nll_on_sequences(m, ids, batch_size=4, device="cpu"))
