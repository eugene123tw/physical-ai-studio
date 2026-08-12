# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Graph-safe rotary position embedding shared by the DS/SS block wrappers.

Split into its own module (upstream inlined it in ``graph_safe_msat.py``) so
that :mod:`graph_safe_double_stream` and :mod:`graph_safe_single_stream` can
import it without a circular dependency on :mod:`graph_safe_msat` (which in
turn imports the DS/SS block wrappers).
"""

from __future__ import annotations

import torch
from torch import nn


class GraphSafeRoPEEmbedder1D(nn.Module):
    """``RoPEEmbedder1D`` with a pre-computed static PE tensor.

    Pre-computes the complex64 PE at construction time (before any bf16
    conversion can corrupt the imaginary/sin component), then caches it as a
    non-persistent buffer so ``.to(dtype)`` won't touch it.
    """

    def __init__(self, rope_embedder: nn.Module, static_ids: torch.Tensor) -> None:
        """Pre-compute the static positional-encoding tensor.

        Args:
            rope_embedder: The original ``RoPEEmbedder1D`` (source of the
                per-axis ``freqs_cis_i`` buffers).
            static_ids: ``(B, N, n_axes)`` static position ids for this
                block phase's fixed token layout.
        """
        super().__init__()
        # Use the SAME freqs_cis buffers as the original rope_embedder.
        # These may have been converted to bf16 (losing the imaginary part)
        # when the model was loaded with torch_dtype=bfloat16; we must use
        # the identical (potentially lossy) values to match vanilla exactly.
        device = static_ids.device

        freqs_list = []
        for i in range(rope_embedder.n_axes):
            freqs_cis = getattr(rope_embedder, f"freqs_cis_{i}").to(device)
            pos_ids = static_ids[..., i]
            freqs_list.append(freqs_cis[pos_ids])
        pe = torch.cat(freqs_list, dim=-1)  # (B, N, D//2)

        # persistent=False: .to(dtype) won't convert this buffer.
        self.register_buffer("static_pe", pe, persistent=False)

    def forward(self) -> torch.Tensor:
        """Return the pre-computed complex64 PE tensor."""
        return self.static_pe
