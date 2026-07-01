# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Patched ``Qwen3VLTextModel.forward`` for the VTC token-compression path.

The stock ``transformers`` ``Qwen3VLTextModel.forward``:

1. Enforces ``(input_ids is None) ^ (inputs_embeds is not None)`` — so the
   adapter cannot pass ``input_ids`` alongside ``inputs_embeds``.
2. Never threads ``input_ids`` into the decoder layers.
3. Assigns ``hidden_states = layer_outputs`` (expects a bare tensor) and reuses
   the ``attention_mask`` / ``position_embeddings`` computed once before the
   loop.

:class:`LayerWrapper` needs ``input_ids`` to locate image-token spans, returns
a ``(hidden_states, kwargs)`` tuple, and rewrites the sequence length (plus the
matching masks and position embeddings) when it compresses tokens at the
``internal_projection`` layer. This patched forward passes ``input_ids`` to
each layer, unpacks the tuple, and threads the updated kwargs to subsequent
layers — matching the RLDX-1 upstream vendored modeling.
"""

from __future__ import annotations

import types

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast


def _vtc_qwen3vl_text_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
    **kwargs: object,
) -> BaseModelOutputWithPast:
    """Run the Qwen3-VL text stack while threading ``input_ids`` to each layer.

    Mirrors the installed ``Qwen3VLTextModel.forward`` but drops the
    input/embeds XOR check, forwards ``input_ids`` positionally to every
    (wrapped) decoder layer, and propagates the ``(hidden_states, kwargs)``
    tuple that :class:`LayerWrapper` returns.
    """
    # XOR check intentionally omitted: the adapter passes both `input_ids`
    # (for LayerWrapper image-token detection) and `inputs_embeds`.
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    # The hard-coded `4` is for text, temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    attention_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # Position embeddings shared across the decoder layers (recomputed by
    # LayerWrapper when it compresses the sequence).
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            input_ids,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs[0]
        updated = layer_outputs[1]
        if "attention_mask" in updated:
            attention_mask = updated["attention_mask"]
        if "position_ids" in updated:
            text_position_ids = updated["position_ids"]
        if "past_key_values" in updated:
            past_key_values = updated["past_key_values"]
        if "cache_position" in updated:
            cache_position = updated["cache_position"]
        if "position_embeddings" in updated:
            position_embeddings = updated["position_embeddings"]

        # Add visual features to the hidden states of the first several layers.
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def install_vtc_text_forward(text_model: torch.nn.Module) -> None:
    """Bind the VTC-patched forward onto a ``Qwen3VLTextModel`` instance.

    Must be called after the decoder layers are wrapped with
    :class:`LayerWrapper`, so the loop threads ``input_ids`` and unpacks the
    wrapper's ``(hidden_states, kwargs)`` tuple.

    Args:
        text_model: The ``language_model`` (``Qwen3VLTextModel``) instance.
    """
    text_model.forward = types.MethodType(_vtc_qwen3vl_text_forward, text_model)
