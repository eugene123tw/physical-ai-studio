# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Self-attention transformer blocks (extracted from msat.py)."""

from typing import Optional

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
import torch
from torch import nn

from physicalai.policies.rldx1.components._dist import rank_zero_print as _print


class BasicTransformerBlock(nn.Module):
    """Single transformer block used by the MSAT self-attention stack.

    The block applies layer norm, self-/cross-attention, residual connection,
    then feed-forward + residual connection.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        """Initialize the transformer block.

        Args:
            dim: Hidden dimension of the input sequence.
            num_attention_heads: Number of attention heads.
            attention_head_dim: Per-head feature dimension.
            dropout: Dropout probability used in attention/MLP modules.
            cross_attention_dim: Optional encoder dimension for cross-attention.
            activation_fn: Feed-forward activation function name.
            attention_bias: Whether attention projections use bias.
            upcast_attention: Whether to upcast attention computation for stability.
            norm_elementwise_affine: Whether LayerNorm uses learnable affine params.
            norm_type: Normalization type (kept for API compatibility).
            norm_eps: Epsilon for normalization layers.
            final_dropout: Whether to apply dropout after attention output.
            attention_type: Attention variant name (kept for API compatibility).
            positional_embeddings: Positional embedding mode. Supported: "sinusoidal" or None.
            max_seq_length: Maximum sequence length used by positional embeddings.
            ff_inner_dim: Optional feed-forward hidden dimension.
            ff_bias: Whether feed-forward linear layers use bias.
            attention_out_bias: Whether attention output projection uses bias.

        Returns:
            None.
        """
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.max_seq_length = max_seq_length
        self.norm_type = norm_type

        if positional_embeddings and (max_seq_length is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `max_seq_length` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=max_seq_length)
        elif positional_embeddings is None:
            self.pos_embed = None
        else:
            raise ValueError(
                "Invalid positional embedding type: `positional_embeddings` must be 'sinusoidal' or None."
            )

        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Run one transformer block pass.

        Args:
            hidden_states: Input tensor of shape ``(B, T, D)`` (or temporary
                4-D shape produced by upstream ops).
            attention_mask: Optional mask broadcastable to attention scores.
            encoder_hidden_states: Optional cross-attention context.
            temb: Optional timestep embedding (unused, kept for compatibility).

        Returns:
            Updated hidden states with the same semantic shape as input,
            typically ``(B, T, D)``.
        """

        # 0. Self-Attention
        norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states
