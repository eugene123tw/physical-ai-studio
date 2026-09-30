# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Normalized goal modulation (NGM).

The goal displacement ``Delta = (q - s) / sigma_Delta`` and a gripper embedding are mapped
by a zero-initialized MLP to a residual ``r`` that is added (behind a flow-time phase
gate) to the AdaRMS condition of every action-expert layer:

    adarms_cond = time_emb + g(t) * r

With Pi0.5's convention ``x_t = t * noise + (1 - t) * actions`` the gate is open during
the coarse (noisy) phase ``t >= gate_hi`` and closed in the refinement phase ``t <= gate_lo``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def phase_gate(t: torch.Tensor, lo: float = 0.2, hi: float = 0.5) -> torch.Tensor:
    """Linear ramp from 0 at ``t <= lo`` to 1 at ``t >= hi``.

    Returns:
        Gate values with the shape of ``t``.
    """
    return ((t - lo) / (hi - lo)).clamp(0.0, 1.0)


class GoalModulation(nn.Module):
    """Deep goal route producing an AdaRMS residual.

    Args:
        num_cfg: Configuration dims.
        num_arms: Gripper slots.
        width: Action-expert hidden width (AdaRMS condition dim).
        gated: Apply the flow-time phase gate (``False`` = naive goal injection).
        gate_lo: Gate fully closed at or below this flow time.
        gate_hi: Gate fully open at or above this flow time.
        gripper_dim: Gripper embedding size per arm.
    """

    def __init__(
        self,
        num_cfg: int,
        num_arms: int,
        width: int,
        *,
        gated: bool = True,
        gate_lo: float = 0.2,
        gate_hi: float = 0.5,
        gripper_dim: int = 16,
    ) -> None:
        """Initialize the goal route with a zero-initialized output projection."""
        super().__init__()
        self.gated = gated
        self.gate_lo = gate_lo
        self.gate_hi = gate_hi
        self.gripper_emb = nn.Embedding(2, gripper_dim)
        in_dim = num_cfg + num_arms * gripper_dim
        self.null_input = nn.Parameter(torch.zeros(in_dim))
        self.mlp_in = nn.Linear(in_dim, width)
        self.mlp_out = nn.Linear(width, width)
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

    def forward(
        self,
        delta_hat: torch.Tensor,
        gripper: torch.Tensor,
        timestep: torch.Tensor,
        drop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the gated residual added to the AdaRMS condition.

        Args:
            delta_hat: ``(B, num_cfg)`` normalized displacement (already noised in training).
            gripper: ``(B, num_arms)`` gripper command.
            timestep: ``(B,)`` flow time.
            drop: Optional ``(B,)`` bool; ``True`` replaces the input with the learned null input.

        Returns:
            ``(B, width)`` residual.
        """
        z = torch.cat([delta_hat.float(), self.gripper_emb(gripper.long()).flatten(1)], dim=-1)
        if drop is not None:
            z = torch.where(drop[:, None], self.null_input.expand_as(z), z)
        r = self.mlp_out(F.silu(self.mlp_in(z)))
        if self.gated:
            r *= phase_gate(timestep.float(), self.gate_lo, self.gate_hi)[:, None]
        return r
