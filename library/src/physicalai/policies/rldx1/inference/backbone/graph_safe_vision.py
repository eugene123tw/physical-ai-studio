# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: file-ignore[print]

"""Graph-safe wrappers for the Qwen3-VL vision tower.

Pre-computes data-dependent values (position embeddings, rotary tables,
``cu_seqlens``) in ``__init__``, then reuses them as static values in
``forward``. Replaces the original ``Qwen3VLVisionModel`` /
``Qwen3VLVisionAttention`` submodules for CUDA Graph / ``torch.compile`` /
TensorRT capture, where any data-dependent Python control flow (``.item()``,
``.tolist()``, ``.max()``) would otherwise force a host/device sync and break
the capture.

Motion-module support (RLDX-1 midtrain variants): when the wrapped vision
model exposes a ``motion_block`` submodule, it is inserted after
``motion_insert_layer`` with a pre-computed static grid-sizes buffer. Only
the ``"vision_encoder"`` injection mode (residual add) is supported. Physical
AI Studio's stock-``transformers``-based backbone does not currently
construct ``motion_block`` on the vision tower, so this path is inert
(``motion_block is None``) until that wiring exists on the Studio side.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]
from torch import nn


class GraphSafeQwen3VLVisionAttention(nn.Module):
    """``Qwen3VLVisionAttention`` with pre-computed static lengths/``max_seqlen``.

    Eliminates ``.tolist()`` and ``.max()`` calls that cause graph breaks.
    """

    def __init__(self, attn: nn.Module, static_lengths: list[int], static_max_seqlen: int) -> None:
        """Wrap an existing vision attention module with static shape metadata.

        Args:
            attn: The original ``Qwen3VLVisionAttention`` instance.
            static_lengths: Per-image token-count split sizes (Python ints).
            static_max_seqlen: Max sequence length across the batch (Python int).
        """
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

        # Resolve attention functions at init from the module that defines
        # ``attn``'s class (stock transformers or a vendored variant);
        # selected at runtime via the shared ``config._attn_implementation``.
        vis_mod = sys.modules[type(attn).__module__]
        self._apply_rope_vision = vis_mod.apply_rotary_pos_emb_vision
        all_attn_fns = vis_mod.ALL_ATTENTION_FUNCTIONS
        self._fa2_fn = all_attn_fns.get("flash_attention_2")
        self._sdpa_fn = all_attn_fns.get("sdpa")
        self._eager_fn = vis_mod.eager_attention_forward

    @property
    def _use_fa2(self) -> bool:
        return self.config._attn_implementation == "flash_attention_2"  # ruff: ignore[private-member-access]

    @property
    def _attn_fn(self):  # ruff: ignore[missing-return-type-private-function]
        impl = self.config._attn_implementation  # ruff: ignore[private-member-access]
        if impl == "flash_attention_2":
            return self._fa2_fn
        if impl == "sdpa":
            return self._sdpa_fn
        return self._eager_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,  # ruff: ignore[unused-method-argument]
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run graph-safe vision attention with pre-split static lengths.

        Returns:
            The vision attention output, shape ``(seq_len, hidden_dim)``.
        """
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        q, k = self._apply_rope_vision(q, k, cos, sin)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        if self._use_fa2:
            attn_output, _ = self._attn_fn(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=self.static_max_seqlen,
                max_length_k=self.static_max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            splits = [torch.split(t, self.static_lengths, dim=2) for t in (q, k, v)]
            attn_output = torch.cat(
                [
                    self._attn_fn(
                        self,
                        qi,
                        ki,
                        vi,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for qi, ki, vi in zip(*splits, strict=True)
                ],
                dim=1,
            )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class GraphSafeQwen3VLVisionModel(nn.Module):
    """``Qwen3VLVisionModel`` with pre-computed static buffers.

    Data-dependent operations replaced:
      - ``fast_pos_embed_interpolate(grid_thw)`` -> ``self.pos_embeds``
      - ``rot_pos_emb(grid_thw)`` -> ``self.pos_cos`` / ``self.pos_sin``
      - ``cu_seqlens`` computation -> ``self.cu_seqlens``
      - ``VisionAttention`` splits -> :class:`GraphSafeQwen3VLVisionAttention`
      - ``get_image_features`` split sizes -> ``self.split_sizes``
    """

    def __init__(  # ruff: ignore[too-many-locals, too-many-statements]
        self,
        visual: nn.Module,
        grid_thw: torch.Tensor,
        num_frames: int = 1,
        num_views: int = 1,
    ) -> None:
        """Pre-compute static vision buffers from a fixed ``grid_thw`` shape.

        Args:
            visual: The wrapped vision tower (e.g. ``model.model.visual``).
            grid_thw: ``(num_images, 3)`` patch-grid shape tensor for the
                fixed input shape this instance is specialized for.
            num_frames: Number of temporal frames per view (motion module).
            num_views: Number of camera views per frame (motion module).
        """
        super().__init__()
        self._visual = visual
        grid_thw = grid_thw.reshape(-1, 3) if grid_thw.ndim == 3 else grid_thw  # ruff: ignore[magic-value-comparison]

        with torch.no_grad():
            self.pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)

            rotary = visual.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary, rotary), dim=-1)
            self.pos_cos = emb.cos()
            self.pos_sin = emb.sin()

            cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0,
                dtype=torch.int32,
            )
            self.cu_seqlens = F.pad(cu, (1, 0), value=0)

            lengths = (self.cu_seqlens[1:] - self.cu_seqlens[:-1]).tolist()
            self.max_seqlen = max(lengths)
            self.split_sizes = (grid_thw.prod(-1) // visual.spatial_merge_size**2).tolist()

        for blk in visual.blocks:
            blk.attn = GraphSafeQwen3VLVisionAttention(blk.attn, lengths, self.max_seqlen)

        # Motion-module setup. ``motion_block`` is only present when the
        # wrapped vision tower was built with a motion module wired in;
        # Studio's stock-transformers backbone leaves this ``None`` today.
        self.motion_block = getattr(visual, "motion_block", None)
        self.motion_insert_layer = getattr(visual, "motion_insert_layer", None)

        if self.motion_block is not None:
            self.motion_block.eval()

            num_images = grid_thw.shape[0]
            true_batch = num_images // (num_frames * num_views)
            h = grid_thw[0, 1].item()
            w = grid_thw[0, 2].item()
            num_patches = h * w
            static_grid = (num_frames, h, w)

            # Replace all data-dependent `grid_sizes[0]` GPU reads inside the
            # motion module with the closure-captured static Python ints
            # above (a GPU->CPU sync there would break CUDA graph capture).
            for enc in self.motion_block.stss_encoders:
                orig_stss_fwd = enc.stss_transformation.forward

                def _make_stss_trans_fwd(orig, grid=static_grid):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
                    def fwd(x, grid_sizes):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function, unused-function-argument]
                        return orig(x, [grid])

                    return fwd

                enc.stss_transformation.forward = _make_stss_trans_fwd(orig_stss_fwd)

                orig_enc_fwd = enc.forward

                def _make_enc_fwd(orig_enc, grid=static_grid):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
                    def fwd(x, grid_sizes=None):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function, unused-function-argument]
                        return orig_enc(x, grid_sizes=[grid])

                    return fwd

                enc.forward = _make_enc_fwd(orig_enc_fwd)

            orig_encoders = self.motion_block.stss_encoders
            orig_use_ls = self.motion_block.use_layerscale
            orig_ls = self.motion_block.layerscale if orig_use_ls else None
            orig_out_proj = None if orig_use_ls else self.motion_block.out_proj

            def _motion_forward_static(x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
                out = x
                encoder_outputs = []
                for enc in orig_encoders:
                    out = enc(out, grid_sizes=grid_sizes)
                    encoder_outputs.append(out)
                out = torch.stack(encoder_outputs, dim=0).sum(dim=0)
                if orig_use_ls:
                    return out * orig_ls
                return orig_out_proj(out)

            self.motion_block.forward = _motion_forward_static

            self._motion_true_batch = true_batch
            self._motion_num_frames = num_frames
            self._motion_num_views = num_views
            self._motion_h = h
            self._motion_w = w
            self._motion_num_patches = num_patches

            motion_grid_sizes = torch.tensor(
                [[num_frames, h, w]] * (true_batch * num_views),
                dtype=torch.long,
                device=grid_thw.device,
            )
            self.register_buffer("motion_grid_sizes", motion_grid_sizes)

            injection_point = getattr(visual, "motion_injection_point", "vision_encoder")
            print(
                f"  [Motion] insert_layer={self.motion_insert_layer}, "
                f"injection={injection_point}, "
                f"batch={true_batch}, T={num_frames}, V={num_views}, "
                f"H={h}, W={w}",
            )
        else:
            self.motion_grid_sizes = None

        print(
            f"  Static buffers (vision): pos_embeds={list(self.pos_embeds.shape)}, "
            f"cos={list(self.pos_cos.shape)}, cu_seqlens={list(self.cu_seqlens.shape)}, "
            f"lengths={lengths}, max_seqlen={self.max_seqlen}, "
            f"split_sizes={self.split_sizes}",
        )

    def _apply_motion_static(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the motion module with pre-computed static grid sizes (vision_encoder mode).

        Reshapes flat tokens to ``(B, V, T, P, D)`` for the motion module,
        then adds the residual. Includes raster <-> interleaved patch-order
        conversion to match Qwen3-VL's block-interleaved patch layout.

        Returns:
            The vision hidden states with the motion-module residual added.
        """
        b = self._motion_true_batch
        t = self._motion_num_frames
        v = self._motion_num_views
        p = self._motion_num_patches
        d = hidden_states.shape[-1]
        h = self._motion_h
        w = self._motion_w
        merge_size = self._visual.spatial_merge_size
        merged_h, merged_w = h // merge_size, w // merge_size

        hidden_3d = hidden_states.reshape(b * t * v, p, d)
        hidden_5d = hidden_3d.reshape(b, t, v, p, d)

        # Undo Qwen's block-interleaved patch ordering before the motion
        # module: tokens are in (merged_h, merged_w, merge_size, merge_size)
        # order; the motion module expects raster (h, w) order.
        hidden_5d = hidden_5d.reshape(b, t, v, merged_h, merged_w, merge_size, merge_size, d)
        hidden_5d = hidden_5d.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
        hidden_5d = hidden_5d.reshape(b, t, v, p, d)

        hidden_bvtpd = hidden_5d.permute(0, 2, 1, 3, 4).contiguous()
        motion_input = hidden_bvtpd.reshape(b * v * t * p, d)

        motion_out = self.motion_block(motion_input, self.motion_grid_sizes)

        motion_out = motion_out.reshape(b, v, t, p, d)
        motion_out = motion_out.permute(0, 2, 1, 3, 4).contiguous()

        # Convert motion-module output back to block-interleaved order for
        # the residual add.
        motion_out = motion_out.reshape(b, t, v, merged_h, merge_size, merged_w, merge_size, d)
        motion_out = motion_out.permute(0, 1, 2, 3, 5, 4, 6, 7).contiguous()
        motion_out = motion_out.reshape(b, t, v, p, d)

        return hidden_states + motion_out.reshape(-1, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor | None = None,  # ruff: ignore[unused-method-argument]
        **kwargs: object,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the vision tower against the pre-computed static buffers.

        Returns:
            A ``(hidden_states, deepstack_feature_lists)`` pair.
        """
        hidden_states = self._visual.patch_embed(hidden_states)
        hidden_states += self.pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self._visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self.cu_seqlens,
                position_embeddings=(self.pos_cos, self.pos_sin),
                **kwargs,
            )

            if self.motion_block is not None and layer_num == self.motion_insert_layer:
                hidden_states = self._apply_motion_static(hidden_states)

            if layer_num in self._visual.deepstack_visual_indexes:
                idx = self._visual.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(self._visual.deepstack_merger_list[idx](hidden_states))

        hidden_states = self._visual.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to the original vision tower.

        Returns:
            The resolved attribute from the wrapped vision tower.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._visual, name)
