# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Dataset-stats and schema helpers for the RLDX-1 policy."""

from __future__ import annotations

from typing import Any

from physicalai.data import Feature, FeatureType
from physicalai.data.observation import ACTION, IMAGES, STATE
from physicalai.policies.utils.features import infer_shape_from_stats


def merge_explicit_features(
    dataset_stats: dict[str, dict[str, Any]] | None,
    input_features: dict[str, Feature] | None,
    output_features: dict[str, Feature] | None,
) -> dict[str, dict[str, Any]] | None:
    """Merge explicit ``Feature`` overrides into a ``dataset_stats``-shaped dict.

    User-supplied features take precedence over anything already in
    ``dataset_stats`` (e.g. auto-fetched state/action stats) -- required for
    RLWRLD checkpoints, whose ``statistics.json`` never records camera shapes.

    Returns:
        The merged dict, or ``None`` if there is nothing to merge.
    """
    merged = dict(dataset_stats or {})
    for name, feature in {**(input_features or {}), **(output_features or {})}.items():
        if feature.shape is None:
            continue
        if feature.ftype == FeatureType.VISUAL:
            key = f"observation.{IMAGES}.{name}"
        elif feature.ftype == FeatureType.STATE:
            key = f"observation.{STATE}"
        elif feature.ftype == FeatureType.ACTION:
            key = ACTION
        else:
            key = name
        merged[key] = {"name": feature.name or name, "shape": feature.shape, "type": str(feature.ftype)}
    return merged or None


def resolve_feature_shape(feature: dict[str, Any]) -> tuple[int, ...]:
    """Return a feature's shape, raising if it can't be inferred from stats.

    RLDX1's own ``extract_dataset_stats`` (used when loading a raw HF release
    checkpoint via ``_from_hf``) returns bare ``min``/``max``/``mean``/``std``/
    ``q01``/``q99`` vectors with no ``"shape"`` key at all -- unlike the
    LeRobot-style enriched stats (e.g. a Studio-trained checkpoint's full
    ``train_dataset.stats``, or an explicit ``input_features``/``output_features``
    override) which carry an explicit ``"shape"``. Both are valid dataset_stats
    entries for this policy; :func:`infer_shape_from_stats` handles both.

    Returns:
        The feature's shape as a tuple.

    Raises:
        ValueError: If neither ``"shape"`` nor a stat vector is present.
    """
    shape = infer_shape_from_stats(feature)
    if shape is None:
        msg = f"Cannot resolve a shape for feature {feature!r}: no 'shape' key and no stat vector to infer it from."
        raise ValueError(msg)
    return shape


def get_dataset_stats_entry(dataset_stats: dict[str, dict[str, Any]], *keys: str) -> dict[str, Any]:
    """Return the first present entry among candidate ``dataset_stats`` keys.

    ``extract_dataset_stats`` (raw HF release checkpoints) uses bare keys like
    ``"state"``; a Studio-trained checkpoint's full LeRobot-style stats use
    ``"observation.state"``. Callers pass both spellings as candidates.

    Returns:
        The matching stats dict.

    Raises:
        KeyError: If none of ``keys`` is present in ``dataset_stats``.
    """
    for key in keys:
        if key in dataset_stats:
            return dataset_stats[key]
    msg = f"None of {keys!r} found in dataset_stats (keys present: {sorted(dataset_stats)!r})"
    raise KeyError(msg)
