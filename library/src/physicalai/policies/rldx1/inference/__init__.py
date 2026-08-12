# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Graph-safe, kernel-optimized inference path for the RLDX-1 policy.

Ported from ``rldx/inference`` upstream (RLWRLD/RLDX-1). Ships ``GraphSafeVLA``
(Path C): static-shape wrapping of the vanilla backbone + action model, plus
:func:`setup_vla_cuda_graph` to capture/replay it as a ``torch.cuda.CUDAGraph``.

Preserves the vanilla model's outputs (see upstream correctness numbers:
cos_sim >= 0.99997 vs eager). Use :func:`physicalai.policies.rldx1.inference.loader.build_graph_safe_vla`
to build a ``GraphSafeVLA`` from a loaded :class:`~physicalai.policies.rldx1.policy.Rldx1`.
"""

from physicalai.policies.rldx1.inference.cuda_graph import setup_vla_cuda_graph
from physicalai.policies.rldx1.inference.graph_safe_vla import GraphSafeVLA
from physicalai.policies.rldx1.inference.loader import build_graph_safe_vla

__all__ = [
    "GraphSafeVLA",
    "build_graph_safe_vla",
    "setup_vla_cuda_graph",
]
