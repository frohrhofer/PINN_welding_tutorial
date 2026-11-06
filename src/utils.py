# SPDX-FileCopyrightText: 2026 Franz Rohrhofer <frohrhofer@know-center.at>
# SPDX-License-Identifier: MIT
"""Reproducibility and device helpers."""

import os
import random

import numpy as np
import torch


def set_seed(
        seed: int,
    ) -> None:
    """Seeds Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(
        n_threads: int | None = None,
        verbose: bool = True,
    ) -> torch.device:
    """CUDA if available, otherwise CPU with a moderate number of threads.

    Small networks do not profit from many CPU threads; too many slow them down.

    Args:
        n_threads (int, None): CPU threads, default: half the cores, at most 8.
        verbose (bool): Whether to print the device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        info = torch.cuda.get_device_name(0)
    else:
        device = torch.device('cpu')
        torch.set_num_threads(n_threads or max(1, min(8, (os.cpu_count() or 2) // 2)))
        info = f"{torch.get_num_threads()} threads"
    if verbose:
        print(f"PyTorch {torch.__version__} on {device} ({info})")
    return device
