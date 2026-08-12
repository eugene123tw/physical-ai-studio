# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: file-ignore[print]

"""Graph-safe wrapper for the Qwen3-VL text model (LLM decoder).

Pre-computes flash-attention kwargs and handles the VTC ``LayerWrapper``
token-compression step with static indices, eliminating the data-dependent
graph breaks (``.item()``, ``torch.nonzero``) that the vanilla
``LayerWrapper.forward`` performs at every call.

Supports two attention modes:
  - ``"flash_attention_2"``: passes ``cu_seq_lens``/``max_length`` FA kwargs
    (default, eager).
  - ``"sdpa"``: omits FA kwargs, relies on SDPA's ``is_causal`` (for
    ONNX/TensorRT export).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast


def _compute_fa_kwargs(seq_len: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """Compute static flash-attention varlen kwargs for one contiguous sequence.

    Returns:
        A ``(cu_seqlens, max_seqlen)`` pair for a single contiguous sequence.
    """
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    return cu, int(seq_len)


def _find_compress_info(
    language_model: nn.Module,
    input_ids: torch.Tensor,
    n_cog_tokens: int,
    num_views: int | None = None,
) -> dict[str, int] | None:
    """Find the compression layer and compute static begin/end indices.

    Args:
        language_model: The wrapped ``Qwen3VLTextModel`` whose ``layers``
            may include a :class:`~physicalai.policies.rldx1.components.backbone.layer_wrapper.LayerWrapper`.
        input_ids: Token-id tensor used to locate image-token spans.
        n_cog_tokens: Number of cog tokens appended after ``input_ids``.
        num_views: Number of camera views. When ``num_views >= 2`` the
            vanilla ``LayerWrapper`` keeps the last ``(num_views - 1)`` image
            sets uncompressed. If ``begin == end``, compression is skipped.

    Returns:
        A dict with static compression layer/index info, or ``None`` when no
        wrapped layer is found or compression is disabled for this input.
    """
    for idx, layer in enumerate(language_model.layers):
        if (
            hasattr(layer, "layer")
            and hasattr(layer, "internal_projection")
            and layer.layer_idx == layer.internal_projection
        ):
            with torch.no_grad():
                dummy = torch.zeros(1, input_ids.shape[1], 1, device=input_ids.device)
                begin_idx, end_idx = layer.get_removing_indices(dummy, input_ids, num_views=num_views)
            b = begin_idx[0, 0].item()
            e = end_idx[0, 0].item()
            if b >= e:
                return None
            l_llm = input_ids.shape[1] + n_cog_tokens
            l_out = b + 1 + (l_llm - e)
            return {
                "compress_layer_idx": idx,
                "static_begin": b,
                "static_end": e,
                "static_out_len": l_out,
            }
    return None


class GraphSafeQwen3VLTextModel(nn.Module):
    """``Qwen3VLTextModel`` with graph-safe forward.

    Data-dependent operations replaced:
      - ``prepare_fa_kwargs_from_position_ids`` (``.item()``) -> pre-computed
        FA kwargs
      - ``LayerWrapper`` compression (``torch.nonzero``) -> static
        begin/end slice
      - ``create_causal_mask`` -> ``attention_mask=None`` (FA varlen uses
        ``cu_seqlens``)
    """

    def __init__(
        self,
        text_model: nn.Module,
        input_ids: torch.Tensor,
        n_cog_tokens: int = 0,
        attn_impl: str = "flash_attention_2",
        num_views: int | None = None,
    ) -> None:
        """Pre-compute static text-model buffers for a fixed sequence length.

        Args:
            text_model: The wrapped ``Qwen3VLTextModel``.
            input_ids: Token-id tensor for the fixed sequence length this
                instance is specialized for.
            n_cog_tokens: Number of cog tokens appended after ``input_ids``.
            attn_impl: ``"flash_attention_2"`` or ``"sdpa"``.
            num_views: Number of camera views (affects which image tokens
                get compressed).
        """
        super().__init__()
        self._text_model = text_model
        self.attn_impl = attn_impl

        l_ids = input_ids.shape[1]
        device = input_ids.device
        l_pre = l_ids + n_cog_tokens

        self.compress_info = _find_compress_info(text_model, input_ids, n_cog_tokens, num_views=num_views)

        if self.compress_info is not None:
            ci = self.compress_info
            l_post = ci["static_out_len"]
            print(
                f"  Static buffers (compression): layer_idx={ci['compress_layer_idx']}, "
                f"begin={ci['static_begin']}, end={ci['static_end']}, "
                f"L_llm={l_pre} -> {l_post} (L_ids={l_ids}, cog_tokens={n_cog_tokens})",
            )
        else:
            l_post = l_pre

        self.pre_cu_seqlens, self.pre_max_seqlen = _compute_fa_kwargs(l_pre, device)
        self.post_cu_seqlens, self.post_max_seqlen = _compute_fa_kwargs(l_post, device)

    def forward(  # ruff: ignore[too-many-locals]
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        deepstack_add: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,  # ruff: ignore[unused-method-argument]
        use_cache: bool | None = None,  # ruff: ignore[unused-method-argument, boolean-type-hint-positional-argument]
        cache_position: torch.Tensor | None = None,  # ruff: ignore[unused-method-argument]
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> BaseModelOutputWithPast:
        """Graph-safe forward.

        Args:
            input_ids: Token ids (used only when ``inputs_embeds`` is None).
            inputs_embeds: Pre-computed token embeddings.
            position_ids: 2-D or 3-D (MROPE) position ids.
            position_embeddings: Optional pre-computed ``(cos, sin)``. When
                provided, skips RoPE computation (for ONNX export).
            deepstack_add: Optional ``[num_ds, B, L, D]`` additive DeepStack
                features (added after the first ``num_ds`` layers).
            attention_mask: Unused (kept for interface parity); FA varlen
                uses the pre-computed ``cu_seqlens`` instead.
            use_cache: Unused (kept for interface parity).
            cache_position: Unused (kept for interface parity).
            **kwargs: Forwarded to each decoder layer.

        Returns:
            ``BaseModelOutputWithPast`` with the final normalized hidden
            states.
        """
        tm = self._text_model

        if inputs_embeds is None:
            inputs_embeds = tm.embed_tokens(input_ids)

        # 2D -> 3D MROPE expansion (matches the original Qwen3VLTextModel).
        if position_ids is not None and position_ids.ndim == 2:  # ruff: ignore[magic-value-comparison]
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        hidden_states = inputs_embeds
        if position_embeddings is None:
            position_embeddings = tm.rotary_emb(hidden_states, position_ids)

        ci = self.compress_info
        cu_seqlens = self.pre_cu_seqlens
        max_seqlen = self.pre_max_seqlen
        use_fa = self.attn_impl == "flash_attention_2"

        for idx, layer in enumerate(tm.layers):
            inner = layer.layer if hasattr(layer, "layer") else layer

            if ci is not None and idx == ci["compress_layer_idx"]:
                b, e = ci["static_begin"], ci["static_end"]
                n_drop = e - b
                drop_mask = torch.zeros(
                    1,
                    hidden_states.shape[1],
                    1,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                drop_mask[:, b:e, :] = 1.0
                motion = (hidden_states * drop_mask).sum(dim=1, keepdim=True) / n_drop
                front = hidden_states[:, :b, :]
                back = hidden_states[:, e:, :]
                hidden_states = torch.cat([front, motion, back], dim=1)

                cos, sin = position_embeddings
                cos = torch.cat([cos[:, :b], cos[:, b : b + 1], cos[:, e:]], dim=1)
                sin = torch.cat([sin[:, :b], sin[:, b : b + 1], sin[:, e:]], dim=1)
                position_embeddings = (cos, sin)

                cu_seqlens = self.post_cu_seqlens
                max_seqlen = self.post_max_seqlen

            if use_fa:
                hidden_states = inner(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    cu_seq_lens_q=cu_seqlens,
                    cu_seq_lens_k=cu_seqlens,
                    max_length_q=max_seqlen,
                    max_length_k=max_seqlen,
                )
            else:
                hidden_states = inner(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            if deepstack_add is not None and idx < deepstack_add.shape[0]:
                hidden_states += deepstack_add[idx]

        hidden_states = tm.norm(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to the original text model.

        Returns:
            The resolved attribute from the wrapped text model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._text_model, name)
