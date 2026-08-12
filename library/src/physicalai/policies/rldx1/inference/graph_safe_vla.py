# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Graph-safe unified VLA: backbone + (optional Memory) + action head.

Composes :class:`~physicalai.policies.rldx1.inference.backbone.graph_safe_backbone.GraphSafeQwen3VLBackbone`
and :class:`~physicalai.policies.rldx1.inference.action_model.graph_safe_action_model.GraphSafeActionModel`
into a single ``nn.Module`` for the full VLA pipeline.

Without memory::

    vl_input -> backbone -> vl_embs -> ActionModel -> action

With memory (``concat_memory=True``)::

    vl_input -> backbone -> cog_all -> Memory (sliding window + Transformer)
             -> [cog_all | cog_augmented] = vl_embs
             -> ActionModel -> action

Memory support is ported for parity with upstream RLDX-1 midtrain
checkpoints; Physical AI Studio's LIBERO checkpoint does not use memory, so
``gs_memory`` stays ``None`` for that checkpoint.

Memory cache is managed with pre-allocated static buffers (in-place
``.copy_()``) so CUDA graph capture and ``torch.compile`` stay compatible.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphSafeVLA(nn.Module):
    """Graph-safe full VLA pipeline with optional memory.

    Forward flow::

        vl_input -> backbone -> vl_embs
        (optional) vl_embs -> Memory -> augmented vl_embs
        (vl_embs, state, embodiment_id) -> ActionModel -> action
    """

    def __init__(
        self,
        gs_backbone: nn.Module,
        gs_action_model: nn.Module,
        gs_memory: nn.Module | None = None,
        memory_config: dict | None = None,
    ) -> None:
        """Compose the graph-safe backbone, action model, and optional memory.

        Args:
            gs_backbone: A ``GraphSafeQwen3VLBackbone`` instance.
            gs_action_model: A ``GraphSafeActionModel`` instance.
            gs_memory: An optional ``GraphSafeMemory`` instance.
            memory_config: Memory sizing config (required when ``gs_memory``
                is provided): ``n_cog_tokens``, ``memory_n_cog_tokens``,
                ``memory_length``, ``concat_memory``, ``hidden_size``.
        """
        super().__init__()
        self.gs_backbone = gs_backbone
        self.gs_action_model = gs_action_model
        self.gs_memory = gs_memory

        if gs_memory is not None and memory_config is not None:
            self.n_q = memory_config["n_cog_tokens"]
            self.n_cog_mem = memory_config["memory_n_cog_tokens"]
            self.n_cog_pass = self.n_q - self.n_cog_mem
            self.memory_length = memory_config["memory_length"]
            self.concat_memory = memory_config["concat_memory"]

            k = self.memory_length
            n = self.n_cog_mem
            d = memory_config["hidden_size"]
            device = next(gs_memory.parameters()).device
            dtype = torch.bfloat16

            self.register_buffer("_cached_cog", torch.zeros(1, k * n, d, device=device, dtype=dtype))
            self.register_buffer("_cache_tmp", torch.zeros(1, k * n, d, device=device, dtype=dtype))

    def reset_memory(self) -> None:
        """Reset recurrent memory state (call at the start of a new episode)."""
        if self.gs_memory is not None:
            self._cached_cog.zero_()

    def forward(
        self,
        vl_input: dict,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        init_noise: torch.Tensor | None = None,
        physics_init_noise: torch.Tensor | None = None,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full VLA forward.

        Args:
            vl_input: dict with ``'pixel_values'`` (other keys use static
                buffers baked into ``gs_backbone``).
            state: ``(B, 1, state_dim)``.
            embodiment_id: ``(B,)``.
            init_noise: ``(B, action_horizon, action_dim)`` or ``None``.
            physics_init_noise: ``(B, fut_len, physics_dim)`` or ``None``.
            prefix_actions: ``(B, prefix_len, action_dim)`` for RTC trained
                mode; ignored when the action head was built with
                ``prefix_len=0``.

        Returns:
            ``(B, action_horizon, action_dim)`` predicted actions.
        """
        vl_embs = self.gs_backbone(vl_input)

        if self.gs_memory is not None:
            vl_embs = self._process_memory(vl_embs)

        return self.gs_action_model(
            vl_embs,
            state,
            embodiment_id,
            init_noise=init_noise,
            physics_init_noise=physics_init_noise,
            prefix_actions=prefix_actions,
        )

    def _process_memory(self, vl_embs: torch.Tensor) -> torch.Tensor:
        """Process backbone output through memory (sliding window + Transformer).

        Uses in-place ``.copy_()`` on pre-allocated static buffers for CUDA
        graph / ``torch.compile`` compatibility -- no tensor allocation in
        the forward path.

        Args:
            vl_embs: ``(B, n_q, d)`` backbone cog tokens.

        Returns:
            ``(B, n_q + n_cog_mem, d)`` when ``concat_memory=True``, else
            ``(B, n_q, d)``.
        """
        cog_all = vl_embs[:, -self.n_q :, :]
        cog_current = cog_all[:, self.n_cog_pass :, :]

        n = self.n_cog_mem

        self._cache_tmp[:, :-n, :].copy_(self._cached_cog[:, n:, :])
        self._cache_tmp[:, -n:, :].copy_(cog_current)
        self._cached_cog.copy_(self._cache_tmp)

        memory_out = self.gs_memory(self._cached_cog)
        cog_augmented = memory_out[:, -n:, :]

        if self.concat_memory:
            return torch.cat([cog_all, cog_augmented], dim=1)
        if self.n_cog_pass > 0:
            return torch.cat([cog_all[:, : self.n_cog_pass, :], cog_augmented], dim=1)
        return cog_augmented
