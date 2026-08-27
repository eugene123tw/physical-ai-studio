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
POSITION_IDS = "position_ids"
ATTENTION_MASK = "attention_mask"

# Placeholder token appended for the cog-token M-RoPE positions (matches the
# eager VTCQwen3VLBackbone._forward_qwen_with_cog_tokens).
_PLACEHOLDER_TOKEN_ID = 248068
# Pad token value is irrelevant numerically: pads are masked as attention keys
# and their own outputs are discarded (only cog tokens are read), so 0 is fine.
_PAD_TOKEN_ID = 0


class GraphSafeRldx1Model(nn.Module):
    """Static, trace-safe view over a trained ``Rldx1Model``.

    Shares parameters with the wrapped model by reference and precomputes all
    data-dependent buffers from ``input_sample`` in the constructor. Call
    :meth:`restore` after export to undo the in-place attention swaps and leave
    the trained model untouched.
    """

    def __init__(
        self,
        model: Rldx1Model,
        input_sample: dict[str, torch.Tensor],
        config: Rldx1Config,
        output_action_dim: int,
    ) -> None:
        """Build the graph-safe view.

        Args:
            model: The trained ``Rldx1Model`` whose parameters are reused by
                reference (never copied).
            input_sample: The eagerly preprocessed export sample (tensor
                entries only). Used to precompute static buffers.
            config: The policy config. ``export_dynamic_prompt`` selects between
                the single-prompt bake and the multi-prompt dynamic export.
            output_action_dim: Real exported action width after trimming padded
                ``max_action_dim`` down to the environment action dimension.

        Raises:
            KeyError: If ``input_sample`` lacks the tensors required to
                precompute the vision / compression buffers.
        """
        super().__init__()
        self._config = config
        self._dynamic = bool(config.export_dynamic_prompt)
        self._output_action_dim = output_action_dim

        for required in (IMAGE_GRID_THW, INPUT_IDS):
            if required not in input_sample:
                msg = f"input_sample is missing '{required}', required to build the graph-safe backbone."
                raise KeyError(msg)

        if self._dynamic:
            # Pad to the fixed export length; the backbone bakes its compression
            # / image-mask positions from this padded layout, and the tracer is
            # fed the same padded input_ids/position_ids/attention_mask.
            build_sample = self._build_padded_sample(model, input_sample, config)
            self.export_sample: dict[str, torch.Tensor] | None = build_sample
            self.input_keys: tuple[str, ...] = (PIXEL_VALUES, INPUT_IDS, POSITION_IDS, ATTENTION_MASK, STATE)
        else:
            build_sample = input_sample
            self.export_sample = None
            self.input_keys = (PIXEL_VALUES, STATE)

        # Wrap-by-reference: reuse the trained submodules, precompute static buffers.
        self.gs_backbone = GraphSafeQwen3VLBackbone(
            model.backbone,
            build_sample,
            config,
            dynamic_prompt=self._dynamic,
        )
        self._action_model = model.action_model

        # embodiment_id is fixed by config, so bake it as a buffer instead of a graph
        # input; the category-specific MLP gathers then constant-fold to one slot.
        embodiment_id = input_sample.get(EMBODIMENT_ID)
        if embodiment_id is None:
            embodiment_id = torch.tensor([int(config.embodiment_id)], dtype=torch.long)
        self.register_buffer("embodiment_id", embodiment_id.clone())


    @staticmethod
    def _build_padded_sample(
        model: Rldx1Model,
        input_sample: dict[str, torch.Tensor],
        config: Rldx1Config,
    ) -> dict[str, torch.Tensor]:
        """Left-pad ``input_ids`` to the fixed length and add host layout tensors.

        Produces the fixed-shape ``input_ids`` / ``position_ids`` /
        ``attention_mask`` (over ``[padded ids | cog placeholders]``) the dynamic
        graph consumes. Left-padding keeps the image block right-aligned so the
        backbone's compression / image-mask positions stay constant.

        Returns:
            A copy of ``input_sample`` with padded ``input_ids`` plus
            ``position_ids`` and ``attention_mask``.

        Raises:
            ValueError: If the prompt is longer than ``tokenizer_max_length``.
        """
        input_ids = input_sample[INPUT_IDS]
        grid_thw = input_sample[IMAGE_GRID_THW]
        device = input_ids.device
        length = config.tokenizer_max_length
        actual = input_ids.shape[1]
        if actual > length:
            msg = f"Prompt length {actual} exceeds tokenizer_max_length {length}; increase it or shorten the task."
            raise ValueError(msg)

        pad_len = length - actual
        pads = torch.full((1, pad_len), _PAD_TOKEN_ID, dtype=input_ids.dtype, device=device)
        padded_input_ids = torch.cat([pads, input_ids], dim=1)
        attention_mask = torch.cat(
            [
                torch.zeros(1, pad_len, dtype=torch.long, device=device),
                torch.ones(1, actual, dtype=torch.long, device=device),
            ],
            dim=1,
        )

        n_cog = config.n_cog_tokens
        cog_ids = torch.full((1, n_cog), _PLACEHOLDER_TOKEN_ID, dtype=input_ids.dtype, device=device)
        extended_input_ids = torch.cat([padded_input_ids, cog_ids], dim=1)
        extended_mask = torch.cat([attention_mask, torch.ones(1, n_cog, dtype=torch.long, device=device)], dim=1)

        inner = model.backbone.qwen_model.model
        with torch.no_grad():
            position_ids, _ = inner.get_rope_index(extended_input_ids, grid_thw, extended_mask)

        padded = dict(input_sample)
        padded[INPUT_IDS] = padded_input_ids
        padded[POSITION_IDS] = position_ids
        padded[ATTENTION_MASK] = extended_mask
        return padded

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
        # Inject the baked embodiment_id (dropped from the graph inputs).
        cast_batch[EMBODIMENT_ID] = self.embodiment_id

        # A mask-free backbone_output skips the eager get_action's
        # ``encoder_attention_mask.all()`` guard (None short-circuits); an
        # all-ones mask would be dropped to None anyway, so results match.
        backbone_output = BatchFeature(data={BACKBONE_FEATURES: backbone_features})
        action_input = self._action_model.prepare_input(cast_batch)
        result = self._action_model.get_action(backbone_output, action_input)
        return result[ACTION_PRED][..., : self._config.action_horizon, : self._output_action_dim]
