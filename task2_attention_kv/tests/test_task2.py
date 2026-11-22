"""CPU checks for Task 2:  python -m pytest task2_attention_kv/tests -q"""
import numpy as np
import pytest
import torch

from task2_attention_kv.src import attention_ops as A
from task2_attention_kv.src import kv_cache as KV
from task2_attention_kv.src import kv_pack as KP
from task2_attention_kv.src import mqar_model as M


def _qkv(B=2, H=4, L=96, d=16, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, L, d, generator=g, dtype=dtype) for _ in range(3)]


@pytest.mark.parametrize("bk", [16, 32, 50, 128])
def test_tiled_matches_dense_and_sdpa(bk):
    q, k, v = _qkv()
    ref = A.dense_attention(q.double(), k.double(), v.double())
    assert torch.allclose(A.tiled_attention(q, k, v, block_k=bk).double(), ref, atol=1e-5)
    assert torch.allclose(A.sdpa_attention(q, k, v, "math").double(), ref, atol=1e-5)


def test_tiled_gradients():
    q, k, v = [t.requires_grad_() for t in _qkv(L=64)]
    g = torch.randn(2, 4, 64, 16)
    (A.dense_attention(q, k, v) * g).sum().backward()
    gd = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    (A.tiled_attention(q, k, v, 16) * g).sum().backward()
    for a, b in zip(gd, (q.grad, k.grad, v.grad)):
        assert torch.allclose(a, b, atol=1e-5)


def test_causality_and_edge_cases():
    q, k, v = _qkv(L=64)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 40:] += 5
    v2[:, :, 40:] -= 3
    for f in (A.dense_attention, lambda a, b, c: A.tiled_attention(a, b, c, 16)):
        assert torch.equal(f(q, k, v)[:, :, :40], f(q, k2, v2)[:, :, :40])
    big = A.dense_attention(q * 1e4, k, v)
    assert torch.isfinite(big).all() and torch.isfinite(A.tiled_attention(q * 1e4, k, v, 16)).all()
    mask = torch.ones(64, 64, dtype=torch.bool)
    mask[5] = False                                       # an all-masked row -> defined as 0
    out = A.dense_attention(q, k, v, mask=mask)
    assert torch.isfinite(out).all() and out[:, :, 5].abs().max() == 0


def test_linear_attention_chunked_equals_prefix():
    q, k, v = _qkv(L=100)
    a = M.causal_linear_attention(M.phi(q), M.phi(k), v, chunk=32)
    b = M.causal_linear_attention_prefix(M.phi(q), M.phi(k), v)
    assert torch.allclose(a, b, atol=1e-4)


def test_masks_and_pairs():
    m = M.dilated_mask(256)
    assert m[255].sum() == 64 and m[255, 3].item() and not m[255, 2].item()
    assert M.local_mask(256)[200].sum() == 64
    assert M.attended_pairs("window", 256)["pairs_per_layer"] == 4 * (64 * 65 // 2 + 192 * 64)


@pytest.mark.parametrize("kind", ["dense", "window", "dilated", "linear", "delta"])
def test_mqar_model_trains(kind):
    torch.manual_seed(0)
    m = M.MQARModel(kind, max_T=256)
    tok = torch.randint(2, 256, (4, 256))
    qp = torch.randint(0, 256, (4, 8))
    from task2_attention_kv.src.mqar_train import query_logits
    lg = query_logits(m, tok, qp)
    assert lg.shape == (4, 8, 64)
    lg.sum().backward()
    base = M.n_params(M.MQARModel("dense"))
    assert abs(M.n_params(m) - base) / base < 0.05


def test_lr_schedule():
    from task2_attention_kv.src.mqar_train import lr_at
    assert abs(lr_at(199) - 8e-4) < 1e-12 and abs(lr_at(4499) - 0.15 * 8e-4) < 1e-12


def test_kv_bytes_and_quant():
    C = 28 * 2 * 128
    assert KV.bytes_quant(C, 128, 4) == 2 * C * 2 * 36
    assert KV.bytes_quant(C, 130, 4) == 2 * C * (2 * 36 + 4)
    b = KV.bytes_quant(C, 2100, 4)
    n = KV.eviction_keep(C, 2100, b, 2)
    assert 4 * C * n + 2 * n <= b < 4 * C * (n + 1) + 2 * (n + 1)
    h = KV.hybrid_keep(C, 2100, b, 72)
    assert KV.hybrid_bytes(C, 2100, h) <= b
    x = torch.randn(1, 2, 128, 128)
    for f in (KV.fq_tokenwise, KV.fq_channelwise):
        assert KV.rel_err(f(x, 4), x) < 0.02 and KV.rel_err(f(x, 2), x) < 0.3
    for axis in ("feature", "token"):
        e = KP.encode(x.half(), axis, 4)
        assert KP.nbytes(e) * 2 == KV.bytes_quant(2 * 128, 128, 4)   # one K-or-V tensor = 2 heads x 128 channels
        ref = KV.fq_tokenwise(x.half(), 4) if axis == "feature" else KV.fq_channelwise(x.half(), 4)
        assert torch.allclose(KP.decode(e).float(), ref.float(), atol=2e-2)


class _Tok:
    eos_token_id = 0

    def convert_tokens_to_ids(self, t):
        return 1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


@pytest.fixture(scope="module")
def tiny_lm():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=300, hidden_size=256, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=2048)  # head_dim 128 like Qwen2.5
    return Qwen2ForCausalLM(cfg).eval()


def _case(L=200, S=12):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(5, 300, (L + S,), generator=g).tolist()
    return dict(id="t", ids=ids, ctx_len=L, n_tokens=L + S, suffix_len=S, evidence_tokens=[50, 51, 52],
                answer_ids=[7, 8, 9, 10, 11], code="12345", nominal_len=2048, position="middle", distractors=0)


def test_cache_split_and_keepall(tiny_lm):
    m, case = tiny_lm, _case()
    with torch.no_grad():
        full = m(torch.tensor([case["ids"]])).logits[0].float()
        cache, _ = KV.prefill(m, case["ids"][:200], "cpu")
        lg, cache = KV.forward_step(m, cache, case["ids"][200:], 200, "cpu")
    assert torch.allclose(lg, full[200:], atol=1e-4)
    a = KV.run_policy(m, _Tok(), case, KV.Policy(), device="cpu")
    b = KV.run_policy(m, _Tok(), case, KV.KeepAll(), ref=dict(tf_logits=a["_tf_logits"], correct=0), device="cpu")
    assert a["text"] == b["text"] and b["kl_tf_mean"] < 1e-6


def test_positions_preserved_under_eviction(tiny_lm):
    """Evicting nothing but the first token must equal attending with that key masked at the original positions."""
    m, case = tiny_lm, _case()
    with torch.no_grad():
        cache, _ = KV.prefill(m, case["ids"][:200], "cpu")
        keep = list(range(1, 200))
        c2 = KV._evict(cache, keep)
        lg, _ = KV.forward_step(m, c2, case["ids"][200:201], 200, "cpu")
        # wrong: positions from physical length (199) instead of absolute 200
        c3 = KV._evict(KV.prefill(m, case["ids"][:200], "cpu")[0], keep)
        lg_wrong, _ = KV.forward_step(m, c3, case["ids"][200:201], 199, "cpu")
    assert not torch.allclose(lg, lg_wrong, atol=1e-4)


@pytest.mark.parametrize("policy", [KV.SinkRecent(), KV.QuestionAware(), KV.QuantKV("token", 4), KV.QuantKV("kivi", 4), KV.Hybrid(2)])
def test_policies_respect_budget(tiny_lm, policy):
    m, case = tiny_lm, _case(L=1000)
    r = KV.run_policy(m, _Tok(), case, policy, device="cpu")
    assert r["within_budget"] and r["bytes"] <= r["budget_bytes"]
    assert np.isfinite(r["answer_nll"])


def test_banded_equals_masked():
    torch.manual_seed(0)
    x = torch.randn(2, 300, 128)
    for masked, banded in (("window", "window_banded"), ("dilated", "dilated_banded"), ("linear", "linear_prefix")):
        m = M.MQARModel(masked, max_T=512).eval()
        with torch.no_grad():
            a = m.forward_embeddings(x)
            m.set_attention(banded)
            b = m.forward_embeddings(x)
        assert torch.allclose(a, b, atol=1e-4), (masked, (a - b).abs().max())
