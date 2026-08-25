# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Export-only graph-safe view over a trained :class:`Rldx1Model`.

RLDX-1's eager forward runs data-dependent ops that ``torch.export`` /
``torch.onnx.export`` cannot trace (``fast_pos_embed_interpolate`` with a
symbolic step count, ``rot_pos_emb``, ``cu_seqlens``, the VTC
``LayerWrapper`` token compression, and ``get_rope_index`` M-RoPE indices).

Upstream (``RLWRLD/RLDX-1`` ``rldx/inference``) solves this with a family of
``graph_safe_*`` wrappers that **share the trained parameters by reference**
(no state-dict copy) and precompute every data-dependent tensor once, eagerly,
in ``__init__`` from a fixed example input -- turning ``forward`` into a static
graph. This module ports that pattern for Studio export.

``GraphSafeRldx1Model`` keeps the exact ``forward(batch)`` contract of
:class:`~physicalai.policies.rldx1.model.Rldx1Model` so the export mixin can
trace it unchanged; the policy swaps it in for tracing backends via
``Rldx1._graph_safe_export_model``.

Port status (v1 = OpenVINO / ONNX, CPU, no motion / memory / RTC):
    [x] vision encoder  -> GraphSafeQwen3VLVisionModel  (pos_embeds, rotary,
        cu_seqlens, static attention splits)
    [x] LLM / VTC       -> GraphSafeQwen3VLTextModel     (static begin/end
        compression indices via _find_compress_info)
    [x] backbone glue   -> GraphSafeQwen3VLBackbone      (embed/scatter/cog
        tokens, get_rope_index -> static position ids)
    [x] action head     -> reused eagerly. The MSAT denoising loop is a fixed
        ``range`` with static shapes and no data-dependent guards; the only
        eager guard (``encoder_attention_mask.all()``) is skipped by passing a
        mask-free ``backbone_output`` (an all-ones mask is set to ``None``
        anyway, so this is numerically identical). Note: the MSAT RoPE uses
        complex64 ops (``view_as_complex`` / ``view_as_real``); those are an
        ONNX/OpenVINO op-support concern that a graph-safe port would not fix
        either -- if tracing fails there, a real-valued RoPE rewrite is the
        targeted follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import BatchFeature

from physicalai.policies.rldx1.components.backbone.graph_safe_backbone import GraphSafeQwen3VLBackbone

if TYPE_CHECKING:
    from physicalai.policies.rldx1.config import Rldx1Config
    from physicalai.policies.rldx1.model import Rldx1Model

INPUT_IDS = "input_ids"
IMAGE_GRID_THW = "image_grid_thw"
BACKBONE_FEATURES = "backbone_features"
ACTION_PRED = "action_pred"
PIXEL_VALUES = "pixel_values"
STATE = "state"
EMBODIMENT_ID = "embodiment_id"


class GraphSafeRldx1Model(nn.Module):
    """Static, trace-safe view over a trained ``Rldx1Model``.

    Shares parameters with the wrapped model by reference and precomputes all
    data-dependent buffers from ``input_sample`` in the constructor. Call
    :meth:`restore` after export to undo the in-place attention swaps and leave
    the trained model untouched.
    """

    # Runtime-varying inputs the static graph consumes. Every other preprocessed
    # tensor (input_ids, image_grid_thw, num_views, ...) is baked into a buffer at
    # build time, so the export sample must be trimmed to these to keep the tracer
    # inputs aligned with the converted graph.
    input_keys: tuple[str, ...] = (PIXEL_VALUES, STATE, EMBODIMENT_ID)

    def __init__(
        self,
        model: Rldx1Model,
        input_sample: dict[str, torch.Tensor],
        config: Rldx1Config,
    ) -> None:
        """Build the graph-safe view.

        Args:
            model: The trained ``Rldx1Model`` whose parameters are reused by
                reference (never copied).
            input_sample: The eagerly preprocessed export sample (tensor
                entries only). Used to precompute static buffers.
            config: The policy config, source of the fixed ``num_views`` /
                ``video_length`` constants baked into the graph.

        Raises:
            KeyError: If ``input_sample`` lacks the tensors required to
                precompute the vision / compression buffers.
        """
        super().__init__()
        self._config = config

        for required in (IMAGE_GRID_THW, INPUT_IDS):
            if required not in input_sample:
                msg = f"input_sample is missing '{required}', required to build the graph-safe backbone."
                raise KeyError(msg)

        # Wrap-by-reference: reuse the trained submodules, precompute static buffers.
        self.gs_backbone = GraphSafeQwen3VLBackbone(model.backbone, input_sample, config)
        self._action_model = model.action_model

    def restore(self) -> None:
        """Undo in-place submodule swaps so the trained model is left intact."""
        self.gs_backbone.restore()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict an action chunk from a preprocessed batch.

        Mirrors :meth:`Rldx1Model.get_action` so the export mixin traces this
        module with the same single-dict input contract: graph-safe backbone ->
        eager action head.

        Returns:
            Predicted action tensor of shape ``(B, action_horizon, action_dim)``.
        """
        backbone_features = self.gs_backbone(batch)  # (B, n_cog, D)

        # Match Rldx1Model.prepare_input: cast floating inputs (e.g. state) to the
        # model dtype so they agree with the bf16 weights (get_action skips that cast).
        dtype = next(self._action_model.parameters()).dtype
        cast_batch = {
            key: value.to(dtype) if torch.is_tensor(value) and torch.is_floating_point(value) else value
            for key, value in batch.items()
        }

        # A mask-free backbone_output skips the eager get_action's
        # ``encoder_attention_mask.all()`` guard (None short-circuits); an
        # all-ones mask would be dropped to None anyway, so results match.
        backbone_output = BatchFeature(data={BACKBONE_FEATURES: backbone_features})
        action_input = self._action_model.prepare_input(cast_batch)
        result = self._action_model.get_action(backbone_output, action_input)
        return result[ACTION_PRED]
