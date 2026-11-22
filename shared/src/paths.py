"""Repository and data paths.

The course data repository (atml-pa4-datasets) is expected next to the code as
``<repo>/atml-pa4-datasets``. Override with the environment variable ``PA4_DATA_REPO``
(for example a Kaggle input dataset path).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_REPO_URL = "https://github.com/AbDu11aHHH/atml-pa4-datasets.git"
DATA_REPO = Path(os.environ.get("PA4_DATA_REPO", REPO_ROOT / "atml-pa4-datasets"))
DATA_DIR = DATA_REPO / "data"
PUBLIC_DIR = DATA_DIR / "public_locked"
REVISION_LOCK = DATA_DIR / "local_source_revisions.json"


def task_dir(task: str) -> Path:
    names = {"task1": "task1_compression", "task2": "task2_attention_kv", "task3": "task3_speculative"}
    return REPO_ROOT / names.get(task, task)


def ensure_data_repo(clone: bool = True) -> Path:
    """Clone the course data repository if it is missing (needs internet)."""
    if not DATA_REPO.exists():
        if not clone:
            raise FileNotFoundError(f"Data repo not found at {DATA_REPO}")
        subprocess.run(["git", "clone", "--depth", "1", DATA_REPO_URL, str(DATA_REPO)], check=True)
    return DATA_REPO


def model_revision(key: str) -> str | None:
    """Return the commit SHA locked by ``setup_data.py`` (keys: model_base, model_target, model_draft,
    c4, wikitext, hellaswag, gsm8k). ``None`` if the lock file does not exist yet."""
    if not REVISION_LOCK.exists():
        return None
    return json.loads(REVISION_LOCK.read_text())["revisions"].get(key)
