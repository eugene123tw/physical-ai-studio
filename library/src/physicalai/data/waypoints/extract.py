# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline waypoint extraction over whole datasets.

Produces a :class:`WaypointTable` (per-episode AWE keyframes plus the configuration and
gripper state at each keyframe) and the waypoint statistics consumed by the
``pi05_waypoint`` policy (``sigma_delta`` for normalized goal modulation).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .awe import extract_waypoints

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

WAYPOINT_STATS_KEY = "waypoint"
_MAD_TO_STD = 1.4826


@dataclass(frozen=True)
class WaypointSpec:
    """How to read configuration / gripper signals from a dataset and run AWE.

    Attributes:
        config_dims: Indices into ``observation.state`` that form the waypoint configuration
            (LIBERO: end-effector position + axis-angle, dims 0-5).
        gripper_source: Read the binary gripper signal from ``"action"`` or ``"state"``.
        gripper_indices: One index per arm into the gripper source vector.
        gripper_threshold: Values strictly above this are "closed" (1), else "open" (0).
        eta: AWE segment error threshold, in the units of the raw configuration.
        max_gap: Maximum segment length in control steps (the expert horizon ``D``).
        metric: AWE segment error metric (``"geometric"`` matches the reference implementation).
        reference_dp: Replicate the reference DP recursion (see ``extract_keyframes``). With it,
            ``eta=0.008`` reproduces the paper's LIBERO-Long statistics on ``lerobot/libero``
            (13.3k segments, median 7, p95 13, max 29 vs. 13.8k / 7 / 13 / 29).
    """

    config_dims: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    gripper_source: Literal["action", "state"] = "action"
    gripper_indices: tuple[int, ...] = (-1,)
    gripper_threshold: float = 0.0
    eta: float = 0.008
    max_gap: int = 32
    metric: Literal["geometric", "temporal"] = "geometric"
    reference_dp: bool = True


@dataclass
class EpisodeWaypoints:
    """AWE result for a single episode.

    Attributes:
        episode_index: Dataset episode index.
        length: Number of frames in the episode.
        keyframes: ``(K + 1,)`` keyframe frame indices, including ``0`` and ``length - 1``.
        config: ``(K + 1, n_cfg)`` raw configuration at each keyframe.
        gripper: ``(K + 1, n_arms)`` binary gripper state at each keyframe.
    """

    episode_index: int
    length: int
    keyframes: np.ndarray
    config: np.ndarray
    gripper: np.ndarray

    @property
    def durations(self) -> np.ndarray:
        """Segment lengths ``keyframes[m] - keyframes[m - 1]``."""
        return np.diff(self.keyframes)


@dataclass
class WaypointTable:
    """Waypoints for every episode of a dataset."""

    spec: WaypointSpec
    episodes: dict[int, EpisodeWaypoints] = field(default_factory=dict)

    def segment_durations(self, episode_indices: Sequence[int] | None = None) -> np.ndarray:
        """Concatenate segment durations over the given (default: all) episodes.

        Returns:
            1-D ``int64`` array of segment durations.
        """
        keys = self.episodes.keys() if episode_indices is None else episode_indices
        parts = [self.episodes[k].durations for k in keys if k in self.episodes]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def statistics(self, episode_indices: Sequence[int] | None = None) -> dict[str, float]:
        """Summary statistics comparable to Table 3 of the paper.

        Returns:
            Dict with segment count, median / p95 / max duration and fraction at the cap.
        """
        durations = self.segment_durations(episode_indices)
        if durations.size == 0:
            return {"num_segments": 0}
        return {
            "num_segments": int(durations.size),
            "median": float(np.median(durations)),
            "p95": float(np.percentile(durations, 95)),
            "max": int(durations.max()),
            "frac_at_cap": float(np.mean(durations == self.spec.max_gap)),
        }

    def save(self, path: str | Path) -> None:
        """Save to a single ``.npz`` file (no pickling)."""
        arrays: dict[str, Any] = {}
        index = sorted(self.episodes)
        arrays["episode_index"] = np.asarray(index, dtype=np.int64)
        arrays["length"] = np.asarray([self.episodes[k].length for k in index], dtype=np.int64)
        arrays["num_keyframes"] = np.asarray([len(self.episodes[k].keyframes) for k in index], dtype=np.int64)
        if index:
            arrays["keyframes"] = np.concatenate([self.episodes[k].keyframes for k in index])
            arrays["config"] = np.concatenate([self.episodes[k].config for k in index]).astype(np.float32)
            arrays["gripper"] = np.concatenate([self.episodes[k].gripper for k in index]).astype(np.int64)
        spec = asdict(self.spec)
        arrays["spec_config_dims"] = np.asarray(spec["config_dims"], dtype=np.int64)
        arrays["spec_gripper_indices"] = np.asarray(spec["gripper_indices"], dtype=np.int64)
        arrays["spec_scalars"] = np.asarray(
            [spec["gripper_threshold"], spec["eta"], spec["max_gap"]],
            dtype=np.float64,
        )
        arrays["spec_gripper_source"] = np.asarray([spec["gripper_source"]])
        arrays["spec_metric"] = np.asarray([spec["metric"]])
        arrays["spec_reference_dp"] = np.asarray([spec["reference_dp"]])
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> WaypointTable:
        """Load a table saved by :meth:`save`.

        Returns:
            The loaded table.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            threshold, eta, max_gap = data["spec_scalars"].tolist()
            spec = WaypointSpec(
                config_dims=tuple(int(x) for x in data["spec_config_dims"]),
                gripper_source=str(data["spec_gripper_source"][0]),  # type: ignore[arg-type]
                gripper_indices=tuple(int(x) for x in data["spec_gripper_indices"]),
                gripper_threshold=float(threshold),
                eta=float(eta),
                max_gap=int(max_gap),
                metric=str(data["spec_metric"][0]) if "spec_metric" in data else "geometric",  # type: ignore[arg-type]
                reference_dp=bool(data["spec_reference_dp"][0]) if "spec_reference_dp" in data else True,
            )
            table = cls(spec=spec)
            offsets = np.concatenate([[0], np.cumsum(data["num_keyframes"])])
            for n, ep in enumerate(data["episode_index"].tolist()):
                sl = slice(int(offsets[n]), int(offsets[n + 1]))
                table.episodes[int(ep)] = EpisodeWaypoints(
                    episode_index=int(ep),
                    length=int(data["length"][n]),
                    keyframes=data["keyframes"][sl].astype(np.int64),
                    config=data["config"][sl].astype(np.float32),
                    gripper=data["gripper"][sl].astype(np.int64),
                )
        return table


def binarize_gripper(values: np.ndarray, spec: WaypointSpec) -> np.ndarray:
    """Convert a gripper source vector to ``(T, n_arms)`` binary states.

    Returns:
        ``int64`` array with ``1`` = closed, ``0`` = open.
    """
    return (values[:, list(spec.gripper_indices)] > spec.gripper_threshold).astype(np.int64)


def extract_table(
    episode_index: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    spec: WaypointSpec,
) -> WaypointTable:
    """Run AWE on every episode given flat per-frame arrays.

    Args:
        episode_index: ``(N,)`` episode id per frame, frames ordered by time within an episode.
        state: ``(N, state_dim)`` observation state per frame.
        action: ``(N, action_dim)`` action per frame.
        spec: Extraction settings.

    Returns:
        A populated :class:`WaypointTable`.
    """
    table = WaypointTable(spec=spec)
    episode_index = np.asarray(episode_index).reshape(-1)
    boundaries = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(episode_index)]])
    gripper_all = binarize_gripper(action if spec.gripper_source == "action" else state, spec)
    config_all = state[:, list(spec.config_dims)].astype(np.float64)

    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        config = config_all[start:end]
        gripper = gripper_all[start:end]
        keyframes = extract_waypoints(
            config,
            gripper,
            eta=spec.eta,
            max_gap=spec.max_gap,
            metric=spec.metric,
            reference_dp=spec.reference_dp,
        )
        ep = int(episode_index[start])
        table.episodes[ep] = EpisodeWaypoints(
            episode_index=ep,
            length=end - start,
            keyframes=keyframes,
            config=config[keyframes].astype(np.float32),
            gripper=gripper[keyframes],
        )
    return table


def extract_table_from_lerobot(lerobot_dataset: Any, spec: WaypointSpec) -> WaypointTable:  # noqa: ANN401
    """Run AWE over a ``LeRobotDataset`` using only its tabular columns (no video decoding).

    Returns:
        A populated :class:`WaypointTable`.
    """
    columns = lerobot_dataset.hf_dataset.select_columns(["episode_index", "observation.state", "action"])
    columns = columns.with_format("numpy")
    episode_index = np.asarray(columns["episode_index"]).reshape(-1)
    state = np.stack(columns["observation.state"]).astype(np.float64)
    action = np.stack(columns["action"]).astype(np.float64)
    logger.info("Extracting waypoints from %d frames", len(episode_index))
    return extract_table(episode_index, state, action, spec)


def normalize_config(
    values: np.ndarray,
    state_stats: Mapping[str, Any],
    config_dims: Sequence[int],
    mode: str,
) -> np.ndarray:
    """Normalize configuration values with the policy's state normalization.

    Mirrors ``FeatureNormalizeTransform``: QUANTILES maps ``[q01, q99]`` to ``[-1, 1]``,
    MEAN_STD standardizes.

    Returns:
        Normalized values, same shape as ``values``.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    dims = list(config_dims)
    if mode.upper() == "QUANTILES":
        lo = np.asarray(state_stats["q01"], dtype=np.float64)[dims]
        hi = np.asarray(state_stats["q99"], dtype=np.float64)[dims]
        return 2.0 * (values - lo) / np.maximum(hi - lo, 1e-8) - 1.0
    if mode.upper() == "MEAN_STD":
        mean = np.asarray(state_stats["mean"], dtype=np.float64)[dims]
        std = np.asarray(state_stats["std"], dtype=np.float64)[dims]
        return (values - mean) / np.maximum(std, 1e-8)
    msg = f"Unknown normalization mode: {mode}"
    raise ValueError(msg)


def compute_sigma_delta(
    table: WaypointTable,
    state_stats: Mapping[str, Any],
    mode: str,
    floor: float = 1e-3,
) -> np.ndarray:
    """Per-dimension scale of normalized segment displacements.

    ``sigma_j = max(1.4826 * MAD_j, 0.25 * std_j, floor)`` over all ``q_i - s_i`` where
    ``s_i`` is the segment start keyframe and ``q_i`` its target.

    Returns:
        ``(n_cfg,)`` float array.
    """
    deltas = []
    for ep in table.episodes.values():
        normed = normalize_config(ep.config.astype(np.float64), state_stats, table.spec.config_dims, mode)
        deltas.append(np.diff(normed, axis=0))
    delta = np.concatenate(deltas) if deltas else np.zeros((1, len(table.spec.config_dims)))
    mad = np.median(np.abs(delta - np.median(delta, axis=0)), axis=0)
    std = delta.std(axis=0)
    return np.maximum(np.maximum(_MAD_TO_STD * mad, 0.25 * std), floor)


def waypoint_stats(
    table: WaypointTable,
    state_stats: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Build the ``dataset_stats["waypoint"]`` entry consumed by the policy.

    Returns:
        Dict with ``sigma_delta``, ``config_dims``, ``num_arms`` and ``max_gap``.
    """
    return {
        "name": WAYPOINT_STATS_KEY,
        "sigma_delta": compute_sigma_delta(table, state_stats, mode).tolist(),
        "config_dims": list(table.spec.config_dims),
        "num_arms": len(table.spec.gripper_indices),
        "max_gap": table.spec.max_gap,
        "normalization_mode": mode.upper(),
    }
