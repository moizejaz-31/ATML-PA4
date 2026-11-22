"""JSON / JSONL helpers (gzip-aware) used for every saved result."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator


def _open(path, mode):
    path = Path(path)
    return gzip.open(path, mode + "t", encoding="utf-8") if path.suffix == ".gz" else open(path, mode, encoding="utf-8")


def read_jsonl(path) -> list:
    return list(iter_jsonl(path))


def iter_jsonl(path) -> Iterator[dict]:
    with _open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_default) + "\n")


def append_jsonl(path, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_default) + "\n")


def save_json(path, obj, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_default)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    try:
        import numpy as np
        import torch
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")
