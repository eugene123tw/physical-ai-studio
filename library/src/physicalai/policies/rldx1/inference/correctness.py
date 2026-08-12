# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Numerical-parity helpers for comparing optimized inference paths against eager."""

from __future__ import annotations

import torch


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors, flattened to 1-D.

    Returns:
        The scalar cosine similarity in ``[-1, 1]``.
    """
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def report_parity(
    label: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 1e-2,
    min_cos_sim: float = 0.999,
) -> bool:
    """Print a cos_sim/max_diff/allclose comparison and return whether it passed.

    ``cos_sim`` is the pass/fail signal (matching upstream RLDX-1's own
    benchmark methodology, e.g. cos_sim >= 0.99997 there), not ``allclose``:
    a handful of elements routinely exceed a tight absolute tolerance from
    ordinary bf16 rounding even when the two tensors are functionally
    equivalent, so a strict ``allclose`` alone is a misleading verdict here.
    ``allclose`` is still printed for extra diagnostic context.

    Args:
        label: Human-readable name for the comparison being reported.
        reference: The eager/vanilla baseline tensor.
        candidate: The optimized-path tensor to compare against ``reference``.
        atol: Absolute tolerance passed to ``torch.allclose`` (diagnostic only).
        min_cos_sim: Minimum cosine similarity to consider this a pass.

    Returns:
        ``True`` when shapes match and ``cos_sim >= min_cos_sim``.
    """
    if reference.shape != candidate.shape:
        print(f"  {label}: SHAPE MISMATCH reference={list(reference.shape)} vs candidate={list(candidate.shape)}")  # ruff: ignore[print]
        return False
    # torch.allclose requires matching dtypes; bf16 vs fp32 outputs are
    # common when comparing an autocast vanilla path against a hardcoded-bf16
    # graph-safe path, so compare in a common (float32) dtype.
    reference = reference.float()
    candidate = candidate.float()
    diff = (reference - candidate).abs()
    cs = cos_sim(reference, candidate)
    allclose = torch.allclose(reference, candidate, atol=atol)
    print(  # ruff: ignore[print]
        f"  {label}: max_diff={diff.max().item():.6f}  cos_sim={cs:.8f}  allclose(atol={atol})={allclose}",
    )
    return cs >= min_cos_sim
