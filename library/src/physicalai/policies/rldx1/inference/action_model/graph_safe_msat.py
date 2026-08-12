# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: file-ignore[print]

"""Graph-safe wrapper for MSAT (RLDX-1 action head).

Pre-determines data-dependent variables (position ids, conditional branches)
in ``__init__``, then uses them as static values in ``forward``. Replaces
the original ``MSAT._forward_inner()`` for CUDA Graph / ``torch.compile`` /
TensorRT / custom-op capture.

Does NOT bake tensor values (RoPE cos/sin etc.) -- only makes the *inputs*
to expensive operations static. Actual value pre-computation is an engine
concern.

Architecture::

    GraphSafeMSAT (orchestrator)
      +-- GraphSafeDoubleStreamBlock  (DS phase with static RoPE)
      +-- GraphSafeSingleStreamBlock  (SS phase with static RoPE)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]
from torch import nn

from physicalai.policies.rldx1.inference.action_model.graph_safe_double_stream import (
    GraphSafeDoubleStreamBlock,
)
from physicalai.policies.rldx1.inference.action_model.graph_safe_single_stream import (
    GraphSafeSingleStreamBlock,
)
from physicalai.policies.rldx1.inference.action_model.rope import GraphSafeRoPEEmbedder1D

__all__ = ["GraphSafeMSAT", "GraphSafeRoPEEmbedder1D"]


class GraphSafeMSAT(nn.Module):
    """Graph-safe MSAT wrapper -- orchestrates the DS and SS block phases.

    Delegates RoPE computation and block iteration to
    :class:`GraphSafeDoubleStreamBlock` (DS phase) and
    :class:`GraphSafeSingleStreamBlock` (SS phase).

    Handles:
      - Timestep encoding + time-token construction
      - Time-token prepend/strip between DS and SS phases
      - VL projection to the SA dimension
      - Output projection (action + optional physics)
    """

    def __init__(
        self,
        msat: nn.Module,
        n_vl: int,
        n_sa_pure: int,
        device: torch.device,
        n_physics: int = 0,
    ) -> None:
        """Wire up the DS/SS block-level graph-safe wrappers.

        Args:
            msat: The original MSAT module.
            n_vl: Number of VL (encoder) tokens.
            n_sa_pure: Number of pure SA tokens (excluding the time token).
            device: Target device.
            n_physics: Number of physics tokens.
        """
        super().__init__()
        self._msat = msat
        self.n_vl = n_vl
        self.n_sa_pure = n_sa_pure
        self.n_physics = n_physics
        self.num_temb_tokens = msat.num_temb_tokens  # 1

        n_sa = n_sa_pure + self.num_temb_tokens  # e.g. 18 = 17 + 1

        self.gs_ds = GraphSafeDoubleStreamBlock(
            msat.double_blocks,
            msat.rope_embedder,
            n_vl=n_vl,
            n_sa=n_sa,
            n_physics=n_physics,
            device=device,
        )
        self.gs_ss = GraphSafeSingleStreamBlock(
            msat.single_blocks,
            msat.rope_embedder,
            n_vl=n_vl,
            n_sa_pure=n_sa_pure,
            num_temb_tokens=self.num_temb_tokens,
            n_physics=n_physics,
            device=device,
        )

        print(
            f"  [GraphSafeMSAT] n_vl={n_vl}, n_sa_pure={n_sa_pure}, "
            f"n_physics={n_physics}, num_temb={self.num_temb_tokens}",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        physics_embs: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Graph-safe MSAT forward.

        Args:
            hidden_states: ``(B, n_sa_pure, sa_dim)`` SA tokens (state +
                action).
            encoder_hidden_states: ``(B, n_vl, vl_dim)`` VL tokens.
            timestep: ``(B,)`` timestep values.
            physics_embs: ``(B, n_physics, sa_dim)`` physics tokens, or
                ``None``.

        Returns:
            Action-output tensor when ``physics_embs`` is ``None``, else
            ``{"action": ..., "physics": ...}``.
        """
        msat = self._msat
        sa = hidden_states
        vl = encoder_hidden_states
        p = physics_embs

        temb = msat.timestep_encoder(timestep)

        time_token = msat.time_token_proj(temb).unsqueeze(1)
        time_token = time_token.repeat(1, self.num_temb_tokens, 1)
        sa = torch.cat([time_token, sa], dim=1)

        if p is not None:
            sa, vl, p = self.gs_ds(sa, vl, temb, p=p)
        else:
            sa, vl = self.gs_ds(sa, vl, temb)

        time_token = sa[:, : self.num_temb_tokens]
        sa = sa[:, self.num_temb_tokens :]

        vl_projected = msat.vl_proj_to_sa(vl)
        x = torch.cat([vl_projected, time_token, sa], dim=1)

        if p is not None:
            x, p = self.gs_ss(x, temb, time_token, p=p)
        else:
            x = self.gs_ss(x, temb, time_token)

        sa = x[:, -self.n_sa_pure :]

        shift, scale = msat.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        sa = msat.norm_out(sa) * (1 + scale[:, None]) + shift[:, None]
        action_out = msat.proj_out_2(sa)

        if p is not None:
            p_shift, p_scale = msat.proj_out_physics_1(F.silu(temb)).chunk(2, dim=1)
            p = msat.norm_out_physics(p) * (1 + p_scale[:, None]) + p_shift[:, None]
            physics_out = msat.proj_out_physics_2(p)
            return {"action": action_out, "physics": physics_out}

        return action_out

    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to the original MSAT model.

        Returns:
            The resolved attribute from the wrapped MSAT module.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._msat, name)
