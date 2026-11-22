"""Notebook bootstrap shared by all PA4 notebooks."""
from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from .env import environment_record
from .paths import DATA_DIR, DATA_REPO, PUBLIC_DIR, REPO_ROOT, ensure_data_repo, task_dir
from .plotting import setup_style
from .seed import DEFAULT_SEED, set_seed


def run_setup_data(flag: str) -> None:
    """Run the course data script (``--public`` or ``--prefix-banks``) from the data repo."""
    cmd = [sys.executable, str(DATA_REPO / "scripts" / "setup_data.py"), flag]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(DATA_REPO))


def init_notebook(task: str, need_public: bool = False, need_prefix_banks: bool = False, seed: int = DEFAULT_SEED):
    setup_style()
    set_seed(seed)
    ensure_data_repo()
    marker = {"task3": "gsm8k_test_locked.jsonl.gz"}.get(task, "wikitext2_test_windows_locked.jsonl.gz")
    if need_public and not (PUBLIC_DIR / marker).exists():
        run_setup_data("--public")      # builds only the locked files that are still missing
    if need_prefix_banks and not (DATA_DIR / "prefix_banks" / "prefix_banks_locked.jsonl.gz").exists():
        run_setup_data("--prefix-banks")
    tdir = task_dir(task)
    ctx = SimpleNamespace(root=REPO_ROOT, task_dir=tdir, results=tdir / "results", figures=tdir / "results" / "figures",
                          checkpoints=tdir / "checkpoints", splits=tdir / "splits", configs=tdir / "configs",
                          data=DATA_DIR, data_repo=DATA_REPO, seed=seed)
    for p in (ctx.results, ctx.figures, ctx.checkpoints, ctx.splits):
        p.mkdir(parents=True, exist_ok=True)
    env = environment_record()
    print({k: env[k] for k in ("torch", "transformers", "cuda_available") if k in env}, env.get("gpu"))
    ctx.env = env
    return ctx
