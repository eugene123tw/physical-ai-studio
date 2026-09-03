# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Graph-safe Qwen3-VL vision encoder for RLDX-1 export.

Ports ``rldx/inference/backbone/vision_encoder/model/
graph_safe_qwen3vl_vision_model.py`` from RLWRLD/RLDX-1 for the v1 export path
(no motion module).

The eager :class:`Qwen3VLVisionModel` runs ops ``torch.export`` /
``torch.onnx.export`` cannot trace: ``fast_pos_embed_interpolate`` builds
``torch.linspace(0, N-1, h)`` with a data-dependent step count ``h``, and the
attention splits the sequence via ``cu_seqlens.tolist()`` / ``.max()``. Because
image resolution is fixed for a given export, ``grid_thw`` is constant, so all
of these are compile-time constants. This wrapper precomputes them once in
``__init__`` (``pos_embeds``, rotary ``cos`` / ``sin``, ``cu_seqlens``, per-window
split lengths) and reduces ``forward`` to a static graph.

Parameters are shared with the wrapped module by reference. Each block's
attention is swapped in place for a static variant; :meth:`restore` puts the
originals back so the trained model is left untouched after export.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch import nn


class GraphSafeQwen3VLVisionAttention(nn.Module):
    """Vision attention with pre-computed static window lengths.

    Replaces the eager ``cu_seqlens.tolist()`` / ``.max()`` (data-dependent) with
    Python ints fixed at construction, so the split is a graph constant.
    """

    def __init__(self, attn: nn.Module, static_lengths: list[int], static_max_seqlen: int) -> None:
        super().__init__()
        self.qkv = attn.qkv
        self.proj = attn.proj
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scaling = attn.scaling
        self.config = attn.config
        self.attention_dropout = attn.attention_dropout
        self.is_causal = attn.is_causal
        self.num_key_value_groups = attn.num_key_value_groups
        self.static_lengths = static_lengths
        self.static_max_seqlen = static_max_seqlen

        # Resolve attention helpers from the module that defines the wrapped attn
        # (the vendored modeling_qwen3_vl), then select at runtime via config.
        _vis_mod = sys.modules[type(attn).__module__]
        self._apply_rope_vision = _vis_mod.apply_rotary_pos_emb_vision
        _all_attn_fns = _vis_mod.ALL_ATTENTION_FUNCTIONS
        self._fa2_fn = _all_attn_fns.get("flash_attention_2")
        self._sdpa_fn = _all_attn_fns.get("sdpa")
        self._eager_fn = _vis_mod.eager_attention_forward

    @property
    def _use_fa2(self) -> bool:
        return self.config._attn_implementation == "flash_attention_2"  # noqa: SLF001

    @property
    def _attn_fn(self):  # noqa: ANN202
        impl = self.config._attn_implementation  # noqa: SLF001
        if impl == "flash_attention_2":
            return self._fa2_fn
        if impl == "sdpa":
            return self._sdpa_fn
        return self._eager_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run vision self-attention using the static window split."""
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = self._apply_rope_vision(q, k, cos, sin)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # Call SDPA directly instead of transformers' sdpa_attention_forward:
        # on CPU with attention_mask=None that wrapper sets enable_gqa=True even
        # though vision is not GQA (num_key_value_groups=1, q_heads==kv_heads),
        # and onnxscript's GQA lowering asserts q_heads > kv_heads. Direct SDPA
        # is numerically identical here. transpose(1, 2) matches the wrapper's
        # (B, N, H, Dh) output layout expected by the reshape below.
        splits = [torch.split(t, self.static_lengths, dim=2) for t in (q, k, v)]
        attn_output = torch.cat(
            [
                F.scaled_dot_product_attention(
                    qi,
                    ki,
                    vi,
                    attn_mask=None,
                    dropout_p=0.0,
                    scale=self.scaling,
                    is_causal=False,
                ).transpose(1, 2)
                for qi, ki, vi in zip(*splits, strict=True)
            ],
            dim=1,
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class GraphSafeQwen3VLVisionModel(nn.Module):
    """Qwen3-VL vision model with pre-computed static buffers.

    Data-dependent ops replaced (all constant for a fixed ``grid_thw``):
      - ``fast_pos_embed_interpolate(grid_thw)`` -> ``self.pos_embeds``
      - ``rot_pos_emb(grid_thw)`` -> ``self.pos_cos`` / ``self.pos_sin``
      - ``cu_seqlens`` cumsum -> ``self.cu_seqlens``
      - per-window attention split -> :class:`GraphSafeQwen3VLVisionAttention`
    """

    def __init__(self, visual: nn.Module, grid_thw: torch.Tensor) -> None:
        super().__init__()
        self._visual = visual
        grid_thw = grid_thw.reshape(-1, 3) if grid_thw.ndim == 3 else grid_thw

        with torch.no_grad():
            pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)

            rotary = visual.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary, rotary), dim=-1)

            cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0,
                dtype=torch.int32,
            )
            cu_seqlens = F.pad(cu, (1, 0), value=0)

            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        # Register precomputed tensors as buffers so torch.export treats them as
        # buffers, not lifted constants (non-persistent buffers are lifted as
        # constants, which trips a fake-tensor check).
        self.register_buffer("pos_embeds", pos_embeds)
        self.register_buffer("pos_cos", emb.cos())
        self.register_buffer("pos_sin", emb.sin())
        self.register_buffer("cu_seqlens", cu_seqlens)
        self.max_seqlen = max(lengths)
        self.split_sizes = (grid_thw.prod(-1) // visual.spatial_merge_size**2).tolist()

        # Swap each block's attention in place; keep originals for restore().
        self._orig_attns = [blk.attn for blk in visual.blocks]
        for blk in visual.blocks:
            blk.attn = GraphSafeQwen3VLVisionAttention(blk.attn, lengths, self.max_seqlen)

    def restore(self) -> None:
        """Reinstate the original attention modules (undo the in-place swap)."""
        for blk, attn in zip(self._visual.blocks, self._orig_attns, strict=True):
            blk.attn = attn

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the wrapped vision parameters."""
        return next(iter(self._visual.parameters())).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor | None = None,  # noqa: ARG002 - static; buffers used instead
        **kwargs: object,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode patches into image features using the pre-computed buffers.

        Returns:
            ``(image_features, deepstack_feature_lists)`` matching the eager
            :meth:`Qwen3VLVisionModel.forward` output.
        """
        hidden_states = self._visual.patch_embed(hidden_states)
        hidden_states = hidden_states + self.pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        deepstack_feature_lists: list[torch.Tensor] = []
        for layer_num, blk in enumerate(self._visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=(self.pos_cos, self.pos_sin),
                **kwargs,
            )
            if layer_num in self._visual.deepstack_visual_indexes:
                idx = self._visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(self._visual.deepstack_merger_list[idx](hidden_states))

        hidden_states = self._visual.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attributes to the wrapped vision model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._visual, name)
