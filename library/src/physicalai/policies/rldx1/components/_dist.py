# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

import builtins
import contextlib
import os

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _env_rank(default: int = 0) -> int:
    return int(os.environ.get("RANK", str(default)))


def is_dist_initialized() -> bool:
    return _HAS_TORCH and torch.distributed.is_available() and torch.distributed.is_initialized()


def get_global_rank() -> int:
    if is_dist_initialized():
        with contextlib.suppress(Exception):
            return torch.distributed.get_rank()
    return _env_rank(0)


def is_global_zero() -> bool:
    return get_global_rank() == 0


def rank_zero_print(*args: object, force: bool = False, **kwargs: object) -> None:
    """Print only on global rank 0 (or if force=True)."""
    if force or is_global_zero():
        builtins.print(*args, **kwargs)  # type: ignore[call-overload]  # noqa: T201
