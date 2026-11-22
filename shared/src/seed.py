import os
import random

import numpy as np
import torch

DEFAULT_SEED = 6304


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Seed python, numpy and torch (CPU + CUDA). ``deterministic`` also pins cuDNN."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
