# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: file-ignore[print]

"""Graph-safe wrapper for the Double-Stream (DS) phase of MSAT.

Manages a list of ``DoubleStreamBlock`` (or ``ExpandedDoubleStreamBlock``)
instances with pre-computed static RoPE position ids.

When ``n_physics > 0``, builds a 3-way RoPE layout ``[VL | SA | P]`` for
``ExpandedDoubleStreamBlock``. When ``n_physics == 0``, builds the standard
2-way layout ``[VL | SA]`` for ``DoubleStreamBlock``.
"""

from __future__ import annotations

import torch
from torch import nn

from physicalai.policies.rldx1.inference.action_model.rope import GraphSafeRoPEEmbedder1D


class GraphSafeDoubleStreamBlock(nn.Module):
    """Graph-safe DS phase: static RoPE + block iteration."""

    def __init__(
        self,
        double_blocks: nn.ModuleList,
        rope_embedder: nn.Module,
        n_vl: int,
        n_sa: int,
        n_physics: int = 0,
        device: torch.device | None = None,
    ) -> None:
        """Pre-compute static DS-phase RoPE ids.

        Args:
            double_blocks: ``nn.ModuleList`` of ``DoubleStreamBlock`` or
                ``ExpandedDoubleStreamBlock``.
            rope_embedder: ``RoPEEmbedder1D`` from the original MSAT.
            n_vl: Number of VL (encoder) tokens.
            n_sa: Number of SA tokens **including** the time token
                (``n_sa_pure + num_temb_tokens``).
            n_physics: Number of physics tokens (0 = no physics).
            device: Target device.
        """
        super().__init__()
        self._blocks = double_blocks
        self.n_vl = n_vl
        self.n_sa = n_sa
        self.n_physics = n_physics

        if n_physics > 0:
            # 3-way layout: [VL(n_vl) | SA(n_sa) | P(n_physics)]
            total = n_vl + n_sa + n_physics
            ids = torch.zeros(1, total, 2, dtype=torch.long, device=device)

            sa_start = n_vl
            ids[:, sa_start : sa_start + n_sa, 1] = torch.arange(n_sa, device=device)

            p_start = n_vl + n_sa
            ids[:, p_start:, 0] = 1
            ids[:, p_start:, 1] = torch.arange(n_physics, device=device)
        else:
            # 2-way layout: [VL(n_vl) | SA(n_sa)]
            total = n_vl + n_sa
            ids = torch.zeros(1, total, 2, dtype=torch.long, device=device)
            ids[:, n_vl:, 1] = torch.arange(n_sa, device=device)

        self.register_buffer("static_ids", ids)
        self.rope = GraphSafeRoPEEmbedder1D(rope_embedder, ids)

        print(f"  [GraphSafeDS] ids={list(ids.shape)}, n_vl={n_vl}, n_sa={n_sa}, n_physics={n_physics}")

    def forward(
        self,
        sa: torch.Tensor,
        vl: torch.Tensor,
        temb: torch.Tensor,
        p: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run all DS blocks with static RoPE.

        Args:
            sa: ``(B, n_sa, sa_dim)`` SA tokens with the time token already
                prepended.
            vl: ``(B, n_vl, vl_dim)`` VL tokens.
            temb: ``(B, temb_dim)`` timestep embedding.
            p: ``(B, n_physics, sa_dim)`` physics tokens, or ``None``.

        Returns:
            ``(sa, vl)`` when ``p`` is ``None``, else ``(sa, vl, p)``.
        """
        pe = self.rope()

        if p is not None:
            for blk in self._blocks:
                sa, vl, p = blk(sa, vl, temb, pe=pe, has_time_token=True, p_tokens=p)
            return sa, vl, p
        for blk in self._blocks:
            sa, vl = blk(sa, vl, temb, pe=pe, has_time_token=True)
        return sa, vl
