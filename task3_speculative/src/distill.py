"""Draft distillation: forward token-level KL(p_teacher || q_student) over the full common vocabulary on response
prefixes that come either from a frozen teacher-rollout pool (off-policy) or from the current student (on-policy).

Fixed settings: LoRA r=8, alpha=16, dropout 0 on q/k/v/o; AdamW 3e-4, wd 0.01; FP16 frozen backbone, FP32 adapters
and optimiser states, FP16 autocast with dynamic gradient scaling (unscale before clipping); microbatch 1; an optimiser
update once >= 256 teacher-scored response tokens have accumulated (whole responses; short final update allowed);
gradients divided by the update's supervised-token count; global-norm clip 1.0. Quota 30,000 math + 10,000 short-form
teacher-scored tokens per run; a response that crosses the quota is scored only up to the quota (recorded).
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from shared.src.io import append_jsonl, load_json, read_jsonl, save_json
from shared.src.paths import model_revision
from shared.src.seed import set_seed
from shared.src.timing import sync

from .data import CAPS, DRAFT, QUOTA, TARGET, PromptSchedule, encode_prompt, eos_ids, train_prompts
from .sampling import sample_generate

LORA = dict(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
OPT = dict(lr=3e-4, weight_decay=0.01, tokens_per_update=256, grad_clip=1.0)


def load_models(device="cuda", load_target=True, load_draft=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET, revision=model_revision("model_target"))
    out = dict(tok=tok)
    if load_target:
        out["target"] = AutoModelForCausalLM.from_pretrained(TARGET, revision=model_revision("model_target"), torch_dtype=torch.float16,
                                                             attn_implementation="sdpa").to(device).eval()
    if load_draft:
        out["draft"] = load_draft_fp16(device)
    return out


def load_draft_fp16(device="cuda"):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(DRAFT, revision=model_revision("model_draft"), torch_dtype=torch.float16,
                                                attn_implementation="sdpa").to(device).eval()


def check_tokenizers(prompts) -> dict:
    """Token IDs and chat-template outputs must agree between target and draft tokenizers."""
    from transformers import AutoTokenizer
    tt = AutoTokenizer.from_pretrained(TARGET, revision=model_revision("model_target"))
    td = AutoTokenizer.from_pretrained(DRAFT, revision=model_revision("model_draft"))
    mism = [p["pid"] for p in prompts if encode_prompt(tt, p["prompt"]) != encode_prompt(td, p["prompt"])]
    return dict(n_checked=len(prompts), mismatches=mism, eos_target=sorted(eos_ids(tt)), eos_draft=sorted(eos_ids(td)),
                vocab_target=len(tt), vocab_draft=len(td))


def kl_per_token(t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
    lt = torch.log_softmax(t_logits.float(), -1)
    ls = torch.log_softmax(s_logits.float(), -1)
    return (lt.exp() * (lt - ls)).sum(-1)


@torch.no_grad()
def build_teacher_pool(target, tok, path: Path, seed: int = 6304, device="cuda") -> dict:
    """Frozen teacher-response collection: walk the shared prompt schedule until both quotas are covered;
    every response is sampled once (temperature 1, top-p 1, caps 192 / 32) and saved with its seed."""
    if path.exists():
        rows = read_jsonl(path)
        return dict(rows=rows, time_s=sum(r["gen_ms"] for r in rows) / 1e3)
    math, short = train_prompts()
    sched = PromptSchedule(math, short, seed)
    eos = eos_ids(tok)
    scored = {"math": 0, "short": 0}
    rows = []
    i = 0
    while True:
        item = sched.next(scored)
        if item is None:
            break
        s = seed * 1000 + i
        gen = torch.Generator(device=device).manual_seed(s)
        ids = encode_prompt(tok, item["prompt"])
        r = sample_generate(target, ids, CAPS[item["domain"]], gen, eos, device=device)
        rows.append(dict(index=i, pid=item["pid"], domain=item["domain"], repetition=item["repetition"], rollout_seed=s,
                         response=r["tokens"], n_tokens=r["n_out"], eos=r["eos"], truncated=r["truncated"], gen_ms=r["total_ms"]))
        scored[item["domain"]] += r["n_out"]
        i += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    from shared.src.io import write_jsonl
    write_jsonl(path, rows)
    return dict(rows=rows, time_s=sum(r["gen_ms"] for r in rows) / 1e3)


def make_student(seed: int, device="cuda"):
    from peft import LoraConfig, get_peft_model
    set_seed(seed)
    base = load_draft_fp16(device)
    for p in base.parameters():
        p.requires_grad_(False)
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", bias="none", **LORA))
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()                     # FP32 adapters (and therefore FP32 optimiser states)
    return model


def train(method: str, seed: int, target, tok, out_dir: Path, pool: list | None = None, device="cuda") -> dict:
    """method 'off' (teacher pool) or 'on' (fresh student rollouts at the current checkpoint)."""
    run = f"{method}_policy_s{seed}"
    rdir = out_dir / run
    if (rdir / "adapter" / "adapter_config.json").exists() and (rdir / "summary.json").exists():
        return load_json(rdir / "summary.json")
    rdir.mkdir(parents=True, exist_ok=True)
    log_path, pres_path = rdir / "updates.jsonl", rdir / "presentations.jsonl"
    for f in (log_path, pres_path):
        if f.exists():
            f.unlink()
    student = make_student(seed, device)
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=OPT["lr"], weight_decay=OPT["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    math, short = train_prompts()
    sched = PromptSchedule(math, short, 6304)            # identical prompt schedule for both methods / seeds
    pool_by = {"math": [r for r in (pool or []) if r["domain"] == "math"], "short": [r for r in (pool or []) if r["domain"] == "short"]}
    pool_ptr = {"math": 0, "short": 0}
    eos = eos_ids(tok)
    scored = {"math": 0, "short": 0}
    cost = dict(student_rollout_ms=0.0, teacher_scoring_ms=0.0, student_fwd_bwd_ms=0.0, optimizer_ms=0.0)
    acc_tok, acc_kl, n_updates, n_skipped, pres_i = 0, 0.0, 0, 0, 0
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    while True:
        item = sched.next(scored)
        if item is None:
            break
        cat = item["domain"]
        ids = encode_prompt(tok, item["prompt"])
        if method == "off":
            rec = pool_by[cat][pool_ptr[cat] % len(pool_by[cat])]
            pool_ptr[cat] += 1
            resp, rseed, src = rec["response"], rec["rollout_seed"], f"teacher_pool[{rec['index']}]"
            item = dict(item, pid=rec["pid"])
            ids = encode_prompt(tok, next(p for p in (math if cat == "math" else short) if p["pid"] == rec["pid"])["prompt"])
        else:
            rseed = seed * 100_000 + pres_i
            gen = torch.Generator(device=device).manual_seed(rseed)
            student.eval()
            sync(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                r = sample_generate(student, ids, CAPS[cat], gen, eos, device=device)
            sync(); cost["student_rollout_ms"] += (time.perf_counter() - t0) * 1e3
            student.train()
            resp, src = r["tokens"], "student_current"
        n = min(len(resp), QUOTA[cat] - scored[cat])
        truncated_by_quota = n < len(resp)
        resp = resp[:n]
        if n == 0:
            continue
        full = torch.tensor([ids + resp], device=device)
        P = len(ids)
        sync(); t0 = time.perf_counter()
        with torch.no_grad():
            t_logits = target(input_ids=full).logits[0, P - 1:P - 1 + n]
        sync(); cost["teacher_scoring_ms"] += (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            s_logits = student(input_ids=full).logits[0, P - 1:P - 1 + n]
        kl = kl_per_token(t_logits, s_logits)
        scaler.scale(kl.sum()).backward()
        sync(); cost["student_fwd_bwd_ms"] += (time.perf_counter() - t0) * 1e3
        del t_logits, s_logits
        acc_tok += n
        acc_kl += float(kl.detach().sum())
        scored[cat] += n
        append_jsonl(pres_path, dict(i=pres_i, pid=item["pid"], domain=cat, repetition=item.get("repetition"), source=src,
                                     rollout_seed=rseed, response_ids=resp, n_scored=n, truncated_by_quota=truncated_by_quota,
                                     checkpoint_update=n_updates))
        pres_i += 1
        done = all(scored[c] >= QUOTA[c] for c in QUOTA)
        if acc_tok >= OPT["tokens_per_update"] or done:
            t0 = time.perf_counter()
            scaler.unscale_(opt)
            for p in params:
                if p.grad is not None:
                    p.grad.div_(acc_tok)
            gn = torch.nn.utils.clip_grad_norm_(params, OPT["grad_clip"])
            s_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped = scaler.get_scale() < s_before
            opt.zero_grad(set_to_none=True)
            sync(); cost["optimizer_ms"] += (time.perf_counter() - t0) * 1e3
            n_updates += 1
            n_skipped += int(skipped)
            append_jsonl(log_path, dict(update=n_updates, tokens=acc_tok, kl=acc_kl / acc_tok, grad_norm=float(gn), lr=OPT["lr"],
                                        skipped=bool(skipped), scale=scaler.get_scale(), scored_math=scored["math"],
                                        scored_short=scored["short"], elapsed_s=time.perf_counter() - t_start))
            acc_tok, acc_kl = 0, 0.0
        if done:
            break
    student.save_pretrained(rdir / "adapter")
    summary = dict(run=run, method=method, seed=seed, scored=scored, n_updates=n_updates, n_skipped_updates=n_skipped,
                   n_presentations=pres_i, trainable_params=n_train, total_params=n_total, wall_time_s=time.perf_counter() - t_start,
                   peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20, cost_ms=cost, lora=LORA, opt=OPT, quota=QUOTA, caps=CAPS)
    save_json(rdir / "summary.json", summary)
    del student, opt
    torch.cuda.empty_cache()
    return summary


@torch.no_grad()
def merge_adapter(adapter_dir: Path, out_dir: Path, tok, check_prompts: list, device="cuda") -> dict:
    """Merge a LoRA adapter into a fresh FP16 0.5B backbone, save it, and compare merged vs unmerged logits."""
    from peft import PeftModel
    t0 = time.perf_counter()
    base = load_draft_fp16(device)
    peft_model = PeftModel.from_pretrained(base, adapter_dir).eval()
    errs = []
    unmerged_logits = []
    for p in check_prompts:
        ids = torch.tensor([encode_prompt(tok, p["prompt"])], device=device)
        unmerged_logits.append(peft_model(input_ids=ids).logits.float())
    merged = peft_model.merge_and_unload().to(torch.float16).eval()
    sync()
    merge_s = time.perf_counter() - t0
    for p, ul in zip(check_prompts, unmerged_logits):
        ids = torch.tensor([encode_prompt(tok, p["prompt"])], device=device)
        ml = merged(input_ids=ids).logits.float()
        errs.append(dict(max_abs=float((ml - ul).abs().max()), argmax_agree=float((ml.argmax(-1) == ul.argmax(-1)).float().mean())))
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    del merged, peft_model, base
    torch.cuda.empty_cache()
    return dict(merge_time_s=merge_s, max_abs_logit_diff=max(e["max_abs"] for e in errs),
                min_argmax_agree=min(e["argmax_agree"] for e in errs), per_prompt=errs)


def load_merged(path: Path, device="cuda"):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, attn_implementation="sdpa").to(device).eval()


def load_unmerged(adapter_dir: Path, device="cuda"):
    from peft import PeftModel
    return PeftModel.from_pretrained(load_draft_fp16(device), adapter_dir).eval()
