"""Record the software / hardware environment next to every result file."""
from __future__ import annotations

import platform
import sys


def environment_record(extra: dict | None = None) -> dict:
    import torch
    rec = dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
               cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
               cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    try:
        import transformers
        rec["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import huggingface_hub
        rec["huggingface_hub"] = huggingface_hub.__version__
    except ImportError:
        pass
    try:
        import peft
        rec["peft"] = peft.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec.update(gpu=p.name, gpu_count=torch.cuda.device_count(), gpu_mem_gb=round(p.total_memory / 2**30, 2),
                   compute_capability=f"{p.major}.{p.minor}")
    if extra:
        rec.update(extra)
    return rec
