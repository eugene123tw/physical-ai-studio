# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: file-ignore[print]

"""Unified graph-safe backbone: vision encoder + glue + LLM decoder + projection.

Composes :class:`GraphSafeQwen3VLVisionModel` and
:class:`GraphSafeQwen3VLTextModel` with the embedding/scatter/cog-token glue
into a single ``nn.Module``. All data-dependent operations are pre-computed
in ``__init__``; ``forward()`` is a pure computation graph.

The cog-token slot is appended after the language ids and routed through the
LLM as additional tokens whose embeddings come from a learned
``backbone.cog_emb`` parameter.
"""

from __future__ import annotations

import types

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from physicalai.policies.rldx1.inference.backbone.graph_safe_text import GraphSafeQwen3VLTextModel
from physicalai.policies.rldx1.inference.backbone.graph_safe_vision import GraphSafeQwen3VLVisionModel


def patch_backbone(backbone: nn.Module, gs_backbone: GraphSafeQwen3VLBackbone) -> None:
    """Patch ``backbone.forward`` to delegate to the unified graph-safe backbone.

    Args:
        backbone: The vanilla backbone adapter (e.g.
            :class:`~physicalai.policies.rldx1.components.backbone.adapter.VTCQwen3VLBackbone`),
            already moved to its target device.
        gs_backbone: A :class:`GraphSafeQwen3VLBackbone` built from ``backbone``.
    """
    image_token_id = backbone.qwen_model.model.config.image_token_id

    def _forward(self: nn.Module, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        features = gs_backbone(vl_input)
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": vl_input.get("attention_mask"),
                "image_mask": (vl_input["input_ids"] == image_token_id),
            },
        )

    backbone.forward = types.MethodType(_forward, backbone)


class GraphSafeQwen3VLBackbone(nn.Module):
    """Unified graph-safe backbone model.

    Forward flow:
      ``pixel_values`` -> Vision Encoder -> Embed + Image Scatter ->
      (cog-token append) -> LLM Decoder (with compression) ->
      (cog-token extract) -> Projection -> ``backbone_features``

    Public attributes for engine builders:
      gs_visual:    :class:`GraphSafeQwen3VLVisionModel` (vision static buffers)
      gs_text:      :class:`GraphSafeQwen3VLTextModel` (LLM static buffers)
      embed_tokens: ``nn.Embedding``
      qwen_linear:  Projection module
    """

    def __init__(  # ruff: ignore[too-many-locals, too-many-statements]
        self,
        backbone: nn.Module,
        vl_input: BatchFeature,
        num_frames: int = 1,
        num_views: int = 1,
    ) -> None:
        """Pre-compute static backbone buffers for a fixed ``vl_input`` shape.

        Args:
            backbone: The vanilla backbone adapter (e.g. ``VTCQwen3VLBackbone``).
            vl_input: A representative preprocessed batch (fixes the static
                shapes -- token count, image grid, cog-token count -- this
                instance is specialized for).
            num_frames: Number of temporal frames per view (motion module).
            num_views: Number of camera views per frame (motion module /
                context compression).
        """
        super().__init__()

        inner_model = backbone.qwen_model.model  # Qwen3VLModel
        visual = inner_model.visual
        language_model = inner_model.language_model

        input_ids = vl_input["input_ids"]
        grid_thw = vl_input["image_grid_thw"]
        device = input_ids.device

        self.n_cog_tokens = getattr(backbone, "n_cog_tokens", 0) if getattr(backbone, "use_cog_tokens", False) else 0
        self.cog_mode = getattr(backbone, "cog_mode", "cog_only")

        print("\nSetting up GraphSafeQwen3VLBackbone...")
        self.gs_visual = GraphSafeQwen3VLVisionModel(visual, grid_thw, num_frames=num_frames, num_views=num_views)

        # Determine num_views for compression (matches the vanilla
        # LayerWrapper logic).
        iwe = vl_input.get("image_wise_encoding")
        if iwe is not None:
            iwe_val = bool(iwe.flatten()[0].item()) if isinstance(iwe, torch.Tensor) else bool(iwe)
        else:
            iwe_val = False
        compress_num_views = vl_input.get("num_views") if iwe_val else None

        self.gs_text = GraphSafeQwen3VLTextModel(
            language_model,
            input_ids,
            self.n_cog_tokens,
            num_views=compress_num_views,
        )

        # Install into the backbone so any code path still reading through
        # it finds the graph-safe submodules. This mutates `backbone` in
        # place -- its vanilla forward (get_image_features etc.) is no
        # longer usable afterward; capture any vanilla-path comparison
        # BEFORE constructing this wrapper.
        inner_model.visual = self.gs_visual
        inner_model.language_model = self.gs_text

        self.embed_tokens = self.gs_text._text_model.embed_tokens  # ruff: ignore[private-member-access]
        self.qwen_linear = backbone.qwen_linear
        self.image_token_id = inner_model.config.image_token_id

        b, l_ids = input_ids.shape
        d = self.embed_tokens.embedding_dim

        self.register_buffer("static_input_ids", input_ids.clone())

        image_mask = input_ids == self.image_token_id
        self.register_buffer("image_mask_3d", image_mask.unsqueeze(-1).expand(b, l_ids, d))

        # 3D MROPE position ids (matches the cog-token append in the
        # vanilla backbone forward). Qwen3-VL uses different position ids
        # per axis (temporal/height/width) for image tokens based on their
        # spatial grid layout.
        attention_mask = vl_input.get("attention_mask")
        if self.n_cog_tokens > 0:
            placeholder_token_id = 248068
            meta_ids = torch.full(
                (b, self.n_cog_tokens),
                placeholder_token_id,
                dtype=input_ids.dtype,
                device=device,
            )
            extended_input_ids = torch.cat([input_ids, meta_ids], dim=1)
            if attention_mask is not None:
                meta_ones = torch.ones(b, self.n_cog_tokens, dtype=attention_mask.dtype, device=device)
                attention_mask = torch.cat([attention_mask, meta_ones], dim=1)
        else:
            extended_input_ids = input_ids

        # get_rope_index needs mm_token_type_ids (0=text, 1=image, 2=video) as
        # of the transformers version this backbone is built against --
        # matches VTCQwen3VLBackbone.forward_qwen's own call.
        video_token_id = inner_model.config.video_token_id
        video_grid_thw = vl_input.get("video_grid_thw")
        mm_token_type_ids = torch.zeros_like(extended_input_ids)
        mm_token_type_ids[extended_input_ids == self.image_token_id] = 1
        mm_token_type_ids[extended_input_ids == video_token_id] = 2

        with torch.no_grad():
            position_ids, _ = inner_model.get_rope_index(
                extended_input_ids,
                mm_token_type_ids,
                grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )
        self.register_buffer("static_position_ids", position_ids)

        if self.n_cog_tokens > 0 and hasattr(backbone, "cog_emb"):
            self.register_buffer("static_cog_emb", backbone.cog_emb.data.clone())
        else:
            self.static_cog_emb = None

        n_img = image_mask.sum().item()
        print(
            f"  Unified backbone: B={b}, L_ids={l_ids}, D={d}, n_cog_tokens={self.n_cog_tokens}, "
            f"image_tokens={n_img}",
        )

    def forward(self, vl_input: BatchFeature) -> torch.Tensor:  # ruff: ignore[too-many-locals]
        """Full backbone forward: ``vl_input`` -> ``backbone_features``.

        Args:
            vl_input: dict with at least a ``pixel_values`` key. Other keys
                (``input_ids``, ``attention_mask``, etc.) are ignored -- the
                model uses its own pre-computed static buffers.

        Returns:
            ``backbone_features`` tensor of shape ``(B, M_out, D_proj)``.
        """
        pixel_values = vl_input["pixel_values"]
        if pixel_values.ndim == 3:  # ruff: ignore[magic-value-comparison]
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        pixel_values = pixel_values.type(self.gs_visual.dtype)

        image_emb, deepstack_features = self.gs_visual(hidden_states=pixel_values, grid_thw=None)

        dtype = self.embed_tokens.weight.dtype
        image_emb = image_emb.to(dtype=dtype)

        token_emb = self.embed_tokens(self.static_input_ids)
        token_emb = token_emb.masked_scatter(self.image_mask_3d, image_emb)

        if self.n_cog_tokens > 0 and self.static_cog_emb is not None:
            meta = self.static_cog_emb.to(dtype).unsqueeze(0).expand(token_emb.size(0), -1, -1)
            full_emb = torch.cat([token_emb, meta], dim=1)
        else:
            full_emb = token_emb

        deepstack_add = None
        if len(deepstack_features) > 0:
            b_ds = full_emb.shape[0]
            l_full_ds = full_emb.shape[1]
            d_ds = full_emb.shape[2]
            vis_mask_2d = self.image_mask_3d[:, :, 0]
            vis_mask_full = torch.cat(
                [
                    vis_mask_2d,
                    torch.zeros(b_ds, self.n_cog_tokens, dtype=torch.bool, device=full_emb.device),
                ],
                dim=1,
            )
            vis_mask_full_3d = vis_mask_full.unsqueeze(-1).expand(b_ds, l_full_ds, d_ds)

            ds_list = []
            for ds_feat in deepstack_features:
                ds_full = torch.zeros_like(full_emb)
                ds_full = ds_full.masked_scatter(vis_mask_full_3d, ds_feat.to(dtype))
                ds_list.append(ds_full)
            deepstack_add = torch.stack(ds_list, dim=0)

        lm_out = self.gs_text(
            inputs_embeds=full_emb,
            position_ids=self.static_position_ids,
            deepstack_add=deepstack_add,
        )
        hidden_states = lm_out.last_hidden_state

        # cog-token extract -- must match the vanilla backbone slice
        # (``cog_mode='cog_only'``) or Path C/D silently disagree with the
        # vanilla model on token count.
        if self.n_cog_tokens > 0 and self.cog_mode == "cog_only":
            hidden_states = hidden_states[:, -self.n_cog_tokens :, :]

        return self.qwen_linear(hidden_states)
