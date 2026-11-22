"""CPU checks for Task 3:  python -m pytest task3_speculative/tests -q"""
import itertools

import pytest
import torch

from task3_speculative.src import sampling as S
from task3_speculative.src.spec_eval import break_even_alpha, predicted_speedup

TARGET = S.MarkovLM([0.5, 0.3, 0.2], [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]])
DRAFT = S.MarkovLM([0.2, 0.3, 0.5], [[0.2, 0.2, 0.6], [0.5, 0.2, 0.3], [0.1, 0.6, 0.3]])


def _sample(n, gamma, residual="exact", clip=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    counts = {}
    for _ in range(n):
        s, _ = S.toy_speculative_sample(TARGET, DRAFT, 3, gamma, g, residual, clip)
        counts[s] = counts.get(s, 0) + 1
    return counts


@pytest.mark.parametrize("gamma,clip", [(1, True), (2, True), (4, False)])
def test_toy_exact(gamma, clip):
    n = 20000
    r = S.chi2_test(_sample(n, gamma, clip=clip), TARGET.joint(3), n)
    assert r["min_expected"] >= 10 and r["p_value"] > 0.01, r


def test_toy_negative_control_rejected():
    n = 20000
    r = S.chi2_test(_sample(n, 2, residual="p"), TARGET.joint(3), n)
    assert r["p_value"] < 0.01, r


def test_verify_round_identical_models_accept_all():
    p = torch.softmax(torch.randn(5, 10), -1)
    g = torch.Generator().manual_seed(0)
    n, tok, rej = S.verify_round(p, p[:4], [1, 2, 3, 4], g)
    assert n == 4 and not rej


def _tiny(seed, vocab=4):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=64)
    m = Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():                      # sharpen so the two models disagree noticeably
        m.lm_head.weight.mul_(30)
    return m


def _exact_joint(model, prompt, length, vocab):
    probs = {}
    with torch.no_grad():
        for seq in itertools.product(range(vocab), repeat=length):
            lp = torch.log_softmax(model(torch.tensor([prompt + list(seq)])).logits[0].float(), -1)
            probs[seq] = float(torch.exp(sum(lp[len(prompt) - 1 + i, seq[i]] for i in range(length))))
    return probs


@pytest.mark.parametrize("gamma", [2, 3])
def test_real_model_speculative_matches_target_joint(gamma):
    tgt, drf = _tiny(0), _tiny(1)
    prompt = [1, 2, 3]
    joint = _exact_joint(tgt, prompt, 3, 4)
    g = torch.Generator().manual_seed(gamma)
    n, counts, acc = 3000, {}, []
    for _ in range(n):
        r = S.speculative_generate(tgt, drf, prompt, gamma, 3, g, eos={-1}, device="cpu")
        counts[tuple(r["tokens"])] = counts.get(tuple(r["tokens"]), 0) + 1
        acc.append(r["accepted"] / max(r["proposed"], 1))
        assert r["n_out"] == 3 and r["truncated"]
    keep = {k: v for k, v in joint.items() if v * n >= 5}
    other = 1 - sum(keep.values())
    obs = {k: counts.get(k, 0) for k in keep}
    obs["rest"] = n - sum(obs.values())
    keep["rest"] = other
    res = S.chi2_test(obs, keep, n)
    assert res["p_value"] > 0.001, res
    assert 0 < sum(acc) / n < 1


def test_eos_and_one_token_rounds():
    tgt = _tiny(0, vocab=4)
    g = torch.Generator().manual_seed(0)
    r = S.speculative_generate(tgt, tgt, [1, 2], 4, 1, g, eos={-1}, device="cpu")
    assert r["n_out"] == 1 and r["proposed"] == 0 and r["rounds"] == 1          # one-token round (gamma clipped to 0)
    r = S.speculative_generate(tgt, tgt, [1, 2], 4, 9, g, eos={-1}, device="cpu")
    assert r["n_out"] == 9 and r["accepted"] == r["proposed"]                     # identical draft: full acceptance + bonus
    r = S.speculative_generate(tgt, tgt, [1, 2], 2, 50, g, eos={0, 1, 2, 3}, device="cpu")
    assert r["n_out"] == 1 and r["eos"]                                           # every token is EOS -> stop after first


def test_cost_model():
    assert abs(predicted_speedup(1.0, 4, 0.0, 1.0) - 5.0) < 1e-12
    a = break_even_alpha(4, 0.3, 1.1)
    assert a is not None and abs(1 + sum(a ** j for j in range(1, 5)) - (4 * 0.3 + 1.1)) < 1e-6
    assert break_even_alpha(2, 2.0, 1.5) is None
