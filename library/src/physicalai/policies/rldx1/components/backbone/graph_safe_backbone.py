# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Graph-safe unified Qwen3-VL backbone (vision + glue + LLM) for export.

Ports ``rldx/inference/backbone/model/graph_safe_qwen3vl_backbone_model.py``
from RLWRLD/RLDX-1 for the v1 export path (no motion / memory / RTC).

Composes :class:`GraphSafeQwen3VLVisionModel` and
:class:`GraphSafeQwen3VLTextModel` with the embedding / image-scatter /
cog-token glue. The data-dependent glue ops -- ``get_rope_index`` (M-RoPE),
the image-token placeholder mask, and the cog-token append -- are precomputed
in ``__init__`` from the fixed export sample, so ``forward`` is a static graph
that consumes only ``pixel_values`` and its own static buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from physicalai.policies.rldx1.components.backbone.graph_safe_text import GraphSafeQwen3VLTextModel
from physicalai.policies.rldx1.components.backbone.graph_safe_vision import GraphSafeQwen3VLVisionModel

if TYPE_CHECKING:
    from physicalai.policies.rldx1.config import Rldx1Config

INPUT_IDS = "input_ids"
PIXEL_VALUES = "pixel_values"
IMAGE_GRID_THW = "image_grid_thw"
POSITION_IDS = "position_ids"
ATTENTION_MASK = "attention_mask"

# Placeholder token id appended for cog-token M-RoPE positions (matches the
# eager VTCQwen3VLBackbone._forward_qwen_with_cog_tokens).
_PLACEHOLDER_TOKEN_ID = 248068


class GraphSafeQwen3VLBackbone(nn.Module):
    """Static, trace-safe unified VLM backbone (vision + glue + LLM)."""

    def __init__(
        self,
        backbone: nn.Module,
        vl_input: dict[str, torch.Tensor],
        config: Rldx1Config,
        *,
        dynamic_prompt: bool = False,
    ) -> None:
        """Build the graph-safe backbone.

        Args:
            backbone: The trained ``VTCQwen3VLBackbone`` (params reused by
                reference).
            vl_input: The eager export sample; needs ``input_ids`` and
                ``image_grid_thw`` to precompute static buffers. For dynamic
                prompt this must already be padded to the fixed export length.
            config: Policy config (``attn_implementation`` / ``num_views``).
            dynamic_prompt: When ``True`` the forward reads ``input_ids`` /
                ``position_ids`` / ``attention_mask`` from the input dict (one
                model, many prompts); when ``False`` the prompt is baked.
        """
        super().__init__()
        self.dynamic_prompt = dynamic_prompt
        inner = backbone.qwen_model.model
        input_ids = vl_input[INPUT_IDS]
        grid_thw = vl_input[IMAGE_GRID_THW]
        device = input_ids.device
        self.qwen_config = backbone.qwen_model.model.config

        self.n_cog_tokens = backbone.n_cog_tokens if getattr(backbone, "use_cog_tokens", False) else 0
        self.cog_mode = backbone.cog_mode

        self.gs_visual = GraphSafeQwen3VLVisionModel(inner.visual, grid_thw)
        self.gs_text = GraphSafeQwen3VLTextModel(
            inner.language_model,
            input_ids,
            n_cog_tokens=self.n_cog_tokens,
            attn_impl=config.attn_implementation,
            num_views=vl_input["num_views"],
        )

        # Shared modules (references, not owned).
        self.embed_tokens = self.gs_text._text_model.embed_tokens  # noqa: SLF001
        self.qwen_linear = backbone.qwen_linear
        self.image_token_id = inner.config.image_token_id

        batch, length_ids = input_ids.shape
        dim = self.embed_tokens.embedding_dim

        self.register_buffer("static_input_ids", input_ids.clone())
        image_mask = input_ids == self.image_token_id
        self.register_buffer("image_mask_3d", image_mask.unsqueeze(-1).expand(batch, length_ids, dim))

        # 3D M-RoPE position ids over [image ids | cog placeholders].
        if self.n_cog_tokens > 0:
            meta_ids = torch.full(
                (batch, self.n_cog_tokens),
                _PLACEHOLDER_TOKEN_ID,
                dtype=input_ids.dtype,
                device=device,
            )
            extended_input_ids = torch.cat([input_ids, meta_ids], dim=1)
        else:
            extended_input_ids = input_ids
        with torch.no_grad():
            position_ids, _ = inner.get_rope_index(extended_input_ids, grid_thw, None)
        self.register_buffer("static_position_ids", position_ids)

        if self.n_cog_tokens > 0 and hasattr(backbone, "cog_emb"):
            self.register_buffer("static_cog_emb", backbone.cog_emb.data.clone())
        else:
            self.static_cog_emb = None

    def restore(self) -> None:
        """Undo the vision encoder's in-place attention swap."""
        self.gs_visual.restore()

    def forward(self, vl_input: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode the VLM inputs into backbone features.

        In static mode only ``pixel_values`` is read (the prompt is baked). In
        dynamic mode ``input_ids`` / ``position_ids`` / ``attention_mask`` are
        read too, so one exported model serves many prompts.

        Returns:
            Backbone features ``(B, M_out, D)`` -- the cog tokens in
            ``cog_only`` mode.
        """
        if self.dynamic_prompt:
            input_ids = vl_input[INPUT_IDS]
            position_ids = vl_input[POSITION_IDS]
            attention_mask = vl_input[ATTENTION_MASK]
            image_mask_2d = input_ids == self.image_token_id
        else:
            input_ids = self.static_input_ids
            position_ids = self.static_position_ids
            attention_mask = None
            image_mask_2d = self.image_mask_3d[:, :, 0]

        pixel_values = vl_input[PIXEL_VALUES]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        pixel_values = pixel_values.type(self.gs_visual.dtype)

        image_emb, deepstack_features = self.gs_visual(pixel_values)

        dtype = self.embed_tokens.weight.dtype
        image_emb = image_emb.to(dtype=dtype)

        token_emb = self.embed_tokens(input_ids)
        image_mask_3d = image_mask_2d.unsqueeze(-1).expand_as(token_emb)
        token_emb = token_emb.masked_scatter(image_mask_3d, image_emb)

        if self.n_cog_tokens > 0 and self.static_cog_emb is not None:
            meta = self.static_cog_emb.to(dtype).unsqueeze(0).expand(token_emb.size(0), -1, -1)
            full_emb = torch.cat([token_emb, meta], dim=1)
        else:
            full_emb = token_emb

        deepstack_add = None
        if len(deepstack_features) > 0:
            batch, length_full, dim = full_emb.shape
            vis_mask_full = torch.cat(
                [
                    image_mask_2d,
                    torch.zeros(batch, self.n_cog_tokens, dtype=torch.bool, device=full_emb.device),
                ],
                dim=1,
            )
            vis_mask_full_3d = vis_mask_full.unsqueeze(-1).expand(batch, length_full, dim)
            ds_list = []
            for ds_feat in deepstack_features:
                ds_full = torch.zeros_like(full_emb).masked_scatter(vis_mask_full_3d, ds_feat.to(dtype))
                ds_list.append(ds_full)
            deepstack_add = torch.stack(ds_list, dim=0)  # (N_ds, B, L_full, D)

        lm_out = self.gs_text(
            inputs_embeds=full_emb,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deepstack_add=deepstack_add,
        )
        hidden_states = lm_out.last_hidden_state

        if self.n_cog_tokens > 0 and self.cog_mode == "cog_only":
            hidden_states = hidden_states[:, -self.n_cog_tokens :, :]

        return self.qwen_linear(hidden_states)
