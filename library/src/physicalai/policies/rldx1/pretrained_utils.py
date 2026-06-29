# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading helpers for the RLDX-1 policy.

The RLDX-1 weights ship as sharded ``safetensors`` files alongside a
``config.json``. These helpers resolve a HuggingFace repo (or local directory)
to a local snapshot and load the merged state dict using the ``safetensors``
backend only — never ``torch.load`` / pickle (lib.security rule 8).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from safetensors.torch import load_file as safe_load_file

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Only these files are pulled from a remote checkpoint repo (lib.security rule 8:
# safetensors allowlist; no pickled ``*.bin`` / ``*.pt`` weights).
ALLOW_PATTERNS = ["*.safetensors", "*.safetensors.index.json", "config.json"]


def resolve_checkpoint_dir(
    base_model_path: str,
    *,
    revision: str | None = None,
) -> Path:
    """Resolve a repo id or local path to a local directory of weight files.

    Args:
        base_model_path: HuggingFace repo id or a local directory path.
        revision: Pinned git commit SHA for remote repos (lib.security rule 9).
            A concrete SHA, never a mutable branch/tag.

    Returns:
        Path to a local directory containing the ``safetensors`` shards and
        ``config.json``.
    """
    local = Path(base_model_path)
    if local.is_dir():
        return local

    if revision is None:
        logger.warning(
            "Loading %s without a pinned revision; pass a commit SHA to "
            "guarantee reproducible, tamper-evident weights (lib.security rule 9).",
            base_model_path,
        )

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    snapshot = snapshot_download(
        base_model_path,
        revision=revision,
        allow_patterns=ALLOW_PATTERNS,
    )
    return Path(snapshot)


def load_rldx_state_dict(
    base_model_path: str,
    *,
    revision: str | None = None,
) -> dict[str, torch.Tensor]:
    """Load and merge the RLDX-1 ``safetensors`` shards into one state dict.

    Args:
        base_model_path: HuggingFace repo id or local directory path.
        revision: Pinned git commit SHA for remote repos (lib.security rule 9).

    Returns:
        The merged ``{parameter_name: tensor}`` state dict.

    Raises:
        FileNotFoundError: If no ``safetensors`` shards are found.
    """
    ckpt_dir = resolve_checkpoint_dir(base_model_path, revision=revision)
    shards = sorted(ckpt_dir.glob("*.safetensors"))
    if not shards:
        msg = f"No *.safetensors weight shards found under {ckpt_dir}"
        raise FileNotFoundError(msg)

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        state_dict.update(safe_load_file(str(shard)))
    logger.info("Loaded %d tensors from %d shard(s)", len(state_dict), len(shards))
    return state_dict
