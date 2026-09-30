# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Pi0.5 waypoint hierarchy (Fast Plans, Faithful Actions)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from physicalai.policies.pi05.config import Pi05Config


@dataclass(frozen=True)
class Pi05WaypointConfig(Pi05Config):
    """Pi0.5 config plus waypoint planner / goal-conditioning settings.

    ``chunk_size`` is the expert horizon ``D`` (waypoint durations are in ``[1, D]``).
    The four paper variants are ``(planner_mode, goal_conditioning)`` pairs:
    ``(token_ar, suffix)``, ``(block_ar, suffix)``, ``(token_ar, naive)``, ``(block_ar, ngm)``.

    Attributes:
        planner_mode: ``"token_ar"`` (one pass per value token) or ``"block_ar"`` (one pass per waypoint).
        goal_conditioning: ``"suffix"``, ``"naive"`` (ungated deep route) or ``"ngm"``.
        max_blocks: Maximum decoded blocks ``M`` (terminal block included).
        num_bins: Value bins for configuration tokens.
        planner_loss_weight: Weight of the planner cross-entropy.
        action_loss_weight: Weight ``lambda`` of the flow-matching loss.
        goal_noise_std: Condition noise on the normalized displacement (``None``: 0.7 for ngm, else 0).
        goal_dropout: Null-goal probability (``None``: 0.15 for ngm, else 0).
        gate_lo: NGM gate closed at or below this flow time.
        gate_hi: NGM gate open at or above this flow time.
        replan_mode: ``"full_plan"`` replans after the whole plan executed (LIBERO protocol);
            ``"receding"`` replans after every segment (real robot).
    """

    chunk_size: int = 32
    n_action_steps: int = 32

    planner_mode: Literal["token_ar", "block_ar"] = "block_ar"
    goal_conditioning: Literal["suffix", "naive", "ngm"] = "ngm"
    max_blocks: int = 7
    num_bins: int = 300
    planner_loss_weight: float = 1.0
    action_loss_weight: float = 1.7
    goal_noise_std: float | None = None
    goal_dropout: float | None = None
    gate_lo: float = 0.2
    gate_hi: float = 0.5
    replan_mode: Literal["full_plan", "receding"] = "full_plan"

    def __post_init__(self) -> None:
        """Validate waypoint settings.

        Raises:
            ValueError: If a setting is invalid.
        """
        if self.planner_mode not in {"token_ar", "block_ar"}:
            msg = f"Invalid planner_mode: {self.planner_mode}"
            raise ValueError(msg)
        if self.goal_conditioning not in {"suffix", "naive", "ngm"}:
            msg = f"Invalid goal_conditioning: {self.goal_conditioning}"
            raise ValueError(msg)
        if self.max_blocks < 1:
            msg = f"max_blocks must be >= 1, got {self.max_blocks}"
            raise ValueError(msg)
        if not 0.0 <= self.gate_lo < self.gate_hi <= 1.0:
            msg = f"Need 0 <= gate_lo < gate_hi <= 1, got {self.gate_lo}, {self.gate_hi}"
            raise ValueError(msg)
        if self.train_expert_only:
            msg = "train_expert_only=True freezes the planner backbone; the waypoint planner needs it trainable"
            raise ValueError(msg)
        if self.snapflow_enabled:
            msg = "SnapFlow is not supported by the waypoint hierarchy"
            raise ValueError(msg)
        super().__post_init__()
