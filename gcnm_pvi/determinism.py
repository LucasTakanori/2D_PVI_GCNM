"""Reproducible training controls for storage-parity experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_deterministic_training(seed: int) -> torch.Generator:
    """Seed every RNG and return an independent DataLoader shuffle generator."""

    seed = int(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CRT contains MaxPool3D. PyTorch has no deterministic CUDA backward
    # implementation for that operation, so strict mode would abort before the
    # first batch. Keep every available deterministic control and warn for that
    # single unsupported kernel; paired multi-seed runs quantify any residue.
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch.Generator().manual_seed(seed)
