# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing for the waypoint hierarchy: Pi0.5 inputs plus normalized waypoint targets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from physicalai.policies.pi05.preprocessor import Pi05Postprocessor, Pi05Preprocessor, make_pi05_preprocessors

from .model import KEY_SEG_Q, KEY_WP_Q

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class ConfigNormalizer(nn.Module):
    """Normalize / denormalize waypoint configurations with the state normalization (config dims only).

    Mirrors ``FeatureNormalizeTransform`` for ``QUANTILES`` and ``MEAN_STD``.
    """

    def __init__(self, state_stats: Mapping[str, Any], config_dims: Sequence[int], mode: str) -> None:
        """Select the normalization constants for ``config_dims``.

        Raises:
            ValueError: If ``mode`` is unsupported.
        """
        super().__init__()
        dims = list(config_dims)
        self.mode = mode.upper()
        if self.mode == "QUANTILES":
            lo = torch.tensor(state_stats["q01"], dtype=torch.float32)[dims]
            hi = torch.tensor(state_stats["q99"], dtype=torch.float32)[dims]
            scale = hi - lo
            scale = torch.where(scale == 0, torch.full_like(scale, 1e-8), scale)
            self.register_buffer("offset", lo)
            self.register_buffer("scale", scale)
        elif self.mode == "MEAN_STD":
            self.register_buffer("offset", torch.tensor(state_stats["mean"], dtype=torch.float32)[dims])
            self.register_buffer("scale", torch.tensor(state_stats["std"], dtype=torch.float32)[dims] + 1e-8)
        else:
            msg = f"Unsupported normalization mode for waypoints: {mode}"
            raise ValueError(msg)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize raw configurations.

        Returns:
            Normalized values.
        """
        offset, scale = self.offset.to(values.device), self.scale.to(values.device)
        if self.mode == "QUANTILES":
            return 2.0 * (values.float() - offset) / scale - 1.0
        return (values.float() - offset) / scale

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        """Denormalize to raw configurations.

        Returns:
            Raw values.
        """
        offset, scale = self.offset.to(values.device), self.scale.to(values.device)
        if self.mode == "QUANTILES":
            return (values.float() + 1.0) * scale / 2.0 + offset
        return values.float() * scale + offset


class Pi05WaypointPreprocessor(Pi05Preprocessor):
    """Pi0.5 preprocessing followed by normalization of ``extra.wp_q`` / ``extra.seg_q``.

    Args:
        config_normalizer: Normalizer for waypoint configurations.
        **kwargs: Forwarded to :class:`Pi05Preprocessor`.
    """

    def __init__(self, config_normalizer: ConfigNormalizer, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the Pi0.5 preprocessor and the configuration normalizer."""
        super().__init__(**kwargs)
        self.config_normalizer = config_normalizer

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run Pi0.5 preprocessing and normalize waypoint configurations when present.

        Returns:
            The processed batch.
        """
        batch = super().forward(batch)
        for key in (KEY_WP_Q, KEY_SEG_Q):
            if batch.get(key) is not None:
                batch[key] = self.config_normalizer(batch[key])
        return batch


def make_pi05_waypoint_preprocessors(
    stats: dict[str, Any],
    config_dims: Sequence[int],
    **kwargs: Any,  # noqa: ANN401
) -> tuple[Pi05WaypointPreprocessor, Pi05Postprocessor, ConfigNormalizer]:
    """Create the waypoint preprocessor, action postprocessor and configuration normalizer.

    Returns:
        Tuple of (preprocessor, postprocessor, config normalizer).
    """
    base, post = make_pi05_preprocessors(stats=stats, **kwargs)
    mode = kwargs.get("normalization_mode", "QUANTILES")
    normalizer = ConfigNormalizer(stats["observation.state"], config_dims, mode)
    pre = Pi05WaypointPreprocessor(
        normalizer,
        max_action_dim=base.max_action_dim,
        image_resolution=base.image_resolution,
        max_token_len=base.max_token_len,
        tokenizer_name=base.tokenizer_name,
        empty_cameras=base.empty_cameras,
        normalization_mode=base.normalization_mode,
    )
    # Reuse the state/action normalizer built from the stats by the Pi0.5 factory.
    pre._state_action_normalizer = base._state_action_normalizer  # noqa: SLF001
    return pre, post, normalizer
