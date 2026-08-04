from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy and (when installed) PyTorch.

    PYTHONHASHSEED only affects interpreters started after it is set, so every
    launcher also records the value rather than pretending it retroactively
    controls the current interpreter.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    report: dict[str, Any] = {
        "seed": seed,
        "python": True,
        "numpy": True,
        "torch": False,
        "deterministic_requested": deterministic,
    }
    if deterministic:
        # This variable must be present before the first CUDA context is
        # initialized; setting it after CUDA work has started is ineffective.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError:
        return report

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    report["torch"] = True
    report["cuda_available"] = bool(torch.cuda.is_available())
    return report
