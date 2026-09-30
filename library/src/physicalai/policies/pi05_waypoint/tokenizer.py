# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Waypoint tokenizer: maps ``(q, g, d)`` waypoints to reserved PaliGemma token ids.

A waypoint block is ``<wp> q_1 .. q_n g_1 .. g_a <dur> d`` (LIBERO: 10 tokens, 8 value
slots; bimanual 14-joint robot: 19 tokens). Token ids are reserved at the tail of the
PaliGemma vocabulary (below its trailing special tokens):

- ``num_bins`` value bins shared by all configuration dims (inputs normalized to [-1, 1]),
- 2 gripper tokens (open / closed),
- ``max_duration + 2`` duration codes: ``0`` = end of plan, ``1..D`` durations,
  ``D + 1`` = current-state block / jitter overflow (clamped to ``D`` on decode),
- 2 structural tokens ``<wp>``, ``<dur>``.

With the defaults (300 bins, D = 32) this is 338 ids, matching the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

PALIGEMMA_VOCAB_SIZE = 257_152
# PaliGemma's last 128 ids are <seg###> tokens; Pi0-FAST also reserves ids just below them.
PALIGEMMA_SKIP_TAIL = 128


class SlotKind(IntEnum):
    """Role of each position inside a waypoint block."""

    WP = 0
    Q = 1
    G = 2
    DUR = 3
    D = 4


@dataclass(frozen=True)
class WaypointTokenizer:
    """Encode / decode waypoint blocks to reserved token ids.

    Attributes:
        num_cfg: Configuration dims per waypoint (LIBERO: 6).
        num_arms: Gripper slots per waypoint (LIBERO: 1).
        num_bins: Value bins shared by all configuration dims.
        max_duration: Expert horizon ``D``.
        vocab_size: Backbone vocabulary size.
        skip_tail: Number of trailing vocabulary ids left untouched.
    """

    num_cfg: int = 6
    num_arms: int = 1
    num_bins: int = 300
    max_duration: int = 32
    vocab_size: int = PALIGEMMA_VOCAB_SIZE
    skip_tail: int = PALIGEMMA_SKIP_TAIL

    @property
    def num_duration_codes(self) -> int:
        """End code, ``D`` durations and the current-state code."""
        return self.max_duration + 2

    @property
    def state_code(self) -> int:
        """Duration code marking the current-state block."""
        return self.max_duration + 1

    @property
    def num_reserved(self) -> int:
        """Total reserved ids (338 with defaults)."""
        return self.num_bins + 2 + self.num_duration_codes + 2

    @property
    def base_id(self) -> int:
        """First reserved token id."""
        return self.vocab_size - self.skip_tail - self.num_reserved

    @property
    def bin_offset(self) -> int:
        """Id of value bin 0."""
        return self.base_id

    @property
    def gripper_offset(self) -> int:
        """Id of the "open" gripper token."""
        return self.base_id + self.num_bins

    @property
    def duration_offset(self) -> int:
        """Id of duration code 0."""
        return self.gripper_offset + 2

    @property
    def wp_id(self) -> int:
        """Id of the ``<wp>`` structural token."""
        return self.duration_offset + self.num_duration_codes

    @property
    def dur_id(self) -> int:
        """Id of the ``<dur>`` structural token."""
        return self.wp_id + 1

    @property
    def block_len(self) -> int:
        """Tokens per block."""
        return 1 + self.num_cfg + self.num_arms + 2

    @property
    def num_values(self) -> int:
        """Value slots per block."""
        return self.num_cfg + self.num_arms + 1

    def slot_kinds(self) -> list[SlotKind]:
        """Per-position role inside one block.

        Returns:
            List of length :attr:`block_len`.
        """
        return [SlotKind.WP] + [SlotKind.Q] * self.num_cfg + [SlotKind.G] * self.num_arms + [SlotKind.DUR, SlotKind.D]

    def value_positions(self) -> torch.Tensor:
        """Indices of value slots (q, g, d) inside a block.

        Returns:
            ``(num_values,)`` long tensor.
        """
        kinds = self.slot_kinds()
        return torch.tensor([i for i, k in enumerate(kinds) if k in {SlotKind.Q, SlotKind.G, SlotKind.D}])

    def family_ranges(self) -> dict[SlotKind, tuple[int, int]]:
        """Reserved-vocabulary sub-ranges ``[lo, hi)`` (relative to :attr:`base_id`) per value family.

        Returns:
            Dict for ``Q``, ``G`` and ``D``.
        """
        g0 = self.num_bins
        d0 = g0 + 2
        return {
            SlotKind.Q: (0, self.num_bins),
            SlotKind.G: (g0, g0 + 2),
            SlotKind.D: (d0, d0 + self.num_duration_codes),
        }

    def value_family_mask(self) -> torch.Tensor:
        """Boolean mask ``(num_values, num_reserved)`` of allowed reserved ids per value slot.

        Returns:
            Mask used to constrain decoding to the right token family per slot.
        """
        ranges = self.family_ranges()
        kinds = [SlotKind.Q] * self.num_cfg + [SlotKind.G] * self.num_arms + [SlotKind.D]
        mask = torch.zeros(len(kinds), self.num_reserved, dtype=torch.bool)
        for i, kind in enumerate(kinds):
            lo, hi = ranges[kind]
            mask[i, lo:hi] = True
        return mask

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        """Map normalized values in ``[-1, 1]`` to bin indices (clamped).

        Returns:
            Long tensor of bin indices in ``[0, num_bins)``.
        """
        idx = torch.floor((values.float().clamp(-1.0, 1.0) + 1.0) * 0.5 * self.num_bins).long()
        return idx.clamp(0, self.num_bins - 1)

    def dequantize(self, bins: torch.Tensor) -> torch.Tensor:
        """Map bin indices to bin-center values in ``[-1, 1]``.

        Returns:
            Float tensor.
        """
        return (bins.float() + 0.5) / self.num_bins * 2.0 - 1.0

    def encode(self, q: torch.Tensor, g: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Encode waypoint blocks to token ids.

        Args:
            q: ``(..., num_cfg)`` normalized configuration in ``[-1, 1]``.
            g: ``(..., num_arms)`` gripper state (0 open, 1 closed).
            d: ``(...)`` duration code in ``[0, D + 1]``.

        Returns:
            ``(..., block_len)`` long tensor of token ids.
        """
        lead = q.shape[:-1]
        wp = torch.full((*lead, 1), self.wp_id, dtype=torch.long, device=q.device)
        dur = torch.full((*lead, 1), self.dur_id, dtype=torch.long, device=q.device)
        q_ids = self.quantize(q) + self.bin_offset
        g_ids = g.long().clamp(0, 1) + self.gripper_offset
        d_ids = (d.long().clamp(0, self.state_code) + self.duration_offset).unsqueeze(-1)
        return torch.cat([wp, q_ids, g_ids, dur, d_ids], dim=-1)

    def values_from_reserved(self, reserved_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode per-slot reserved-vocabulary indices (value slots only) to ``(q, g, d)``.

        Args:
            reserved_idx: ``(..., num_values)`` indices relative to :attr:`base_id`.

        Returns:
            ``q`` in ``[-1, 1]``, ``g`` in ``{0, 1}``, ``d`` in ``[0, D]`` (state code clamped to ``D``).
        """
        ranges = self.family_ranges()
        q = self.dequantize(reserved_idx[..., : self.num_cfg] - ranges[SlotKind.Q][0])
        g = reserved_idx[..., self.num_cfg : self.num_cfg + self.num_arms] - ranges[SlotKind.G][0]
        d = (reserved_idx[..., -1] - ranges[SlotKind.D][0]).clamp(0, self.max_duration)
        return q, g, d

    def reserved_targets(self, q: torch.Tensor, g: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Per value slot target index relative to :attr:`base_id` (for cross-entropy).

        Returns:
            ``(..., num_values)`` long tensor.
        """
        ids = self.encode(q, g, d)
        return ids[..., self.value_positions().to(ids.device)] - self.base_id
