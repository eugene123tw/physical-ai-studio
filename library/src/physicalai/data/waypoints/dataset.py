# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Waypoint-annotated datasets and datamodule.

Each sample is one AWE segment start. The wrapped observation carries, in
``Observation.extra``, both supervision targets used by the ``pi05_waypoint`` policy:

- **planner targets** — the next ``max_blocks`` waypoints (including a terminal
  ``d = 0`` block when the episode ends within the window), and
- **expert segment** — the first waypoint ``(q, g, d)`` the action chunk should reach.

Waypoint configurations are stored raw; normalization happens in the policy preprocessor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from physicalai.data.dataset import Dataset
from physicalai.data.lerobot.datamodule import LeRobotDataModule
from physicalai.data.lerobot.dataset import _LeRobotDatasetAdapter  # noqa: PLC2701

from .extract import WAYPOINT_STATS_KEY, WaypointSpec, WaypointTable, extract_table_from_lerobot, waypoint_stats

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.data import Feature, Observation

logger = logging.getLogger(__name__)

WP_Q = "wp_q"
WP_G = "wp_g"
WP_D = "wp_d"
WP_VALID = "wp_valid"
WP_CUR_G = "wp_cur_g"
SEG_Q = "seg_q"
SEG_G = "seg_g"
SEG_D = "seg_d"
WAYPOINT_EXTRA_KEYS = (WP_Q, WP_G, WP_D, WP_VALID, WP_CUR_G, SEG_Q, SEG_G, SEG_D)


def episode_start_rows(adapter: _LeRobotDatasetAdapter) -> dict[int, int]:
    """Map each episode index to the dataset row of its first frame.

    Returns:
        Dict ``episode_index -> row``.
    """
    column = adapter._lerobot_dataset.hf_dataset.select_columns(["episode_index"]).with_format("numpy")  # noqa: SLF001
    episodes = np.asarray(column["episode_index"]).reshape(-1)
    uniq, first = np.unique(episodes, return_index=True)
    return {int(e): int(r) for e, r in zip(uniq, first, strict=True)}


def build_planner_targets(
    table_episode: Any,  # noqa: ANN401
    segment: int,
    start_frame: int,
    max_blocks: int,
) -> dict[str, np.ndarray]:
    """Build padded planner and expert targets for one segment start.

    Args:
        table_episode: :class:`~physicalai.data.waypoints.extract.EpisodeWaypoints`.
        segment: Index ``m`` of the segment whose start keyframe is ``keyframes[m]``.
        start_frame: Observation frame (``keyframes[m]`` plus any jitter).
        max_blocks: Maximum number of plan blocks ``M`` (terminal block included).

    Returns:
        Dict of numpy arrays keyed by the ``WP_*`` / ``SEG_*`` constants.
    """
    kf = table_episode.keyframes
    config = table_episode.config
    gripper = table_episode.gripper
    n_cfg = config.shape[1]
    n_arms = gripper.shape[1]

    target_ids = list(range(segment + 1, len(kf)))[:max_blocks]
    q = np.zeros((max_blocks, n_cfg), dtype=np.float32)
    g = np.zeros((max_blocks, n_arms), dtype=np.int64)
    d = np.zeros(max_blocks, dtype=np.int64)
    valid = np.zeros(max_blocks, dtype=bool)

    prev_frame = start_frame
    for b, k in enumerate(target_ids):
        q[b] = config[k]
        g[b] = gripper[k]
        d[b] = kf[k] - prev_frame
        valid[b] = True
        prev_frame = kf[k]

    n = len(target_ids)
    if target_ids[-1] == len(kf) - 1 and n < max_blocks:
        q[n] = config[-1]
        g[n] = gripper[-1]
        d[n] = 0
        valid[n] = True

    return {
        WP_Q: q,
        WP_G: g,
        WP_D: d,
        WP_VALID: valid,
        WP_CUR_G: gripper[segment].astype(np.int64),
        SEG_Q: q[0].copy(),
        SEG_G: g[0].copy(),
        SEG_D: np.asarray(d[0], dtype=np.int64),
    }


class WaypointDataset(Dataset):
    """Wrap a frame dataset so that each item is an AWE segment start with waypoint targets.

    Args:
        base_dataset: Frame-level dataset (e.g. ``_LeRobotDatasetAdapter``).
        table: Waypoints for the episodes contained in ``base_dataset``.
        start_rows: Map ``episode_index -> row`` of the first frame in ``base_dataset``.
        waypoint_stats: The ``dataset_stats["waypoint"]`` entry (see :func:`waypoint_stats`).
        max_blocks: Maximum number of planner blocks ``M``.
        jitter: Random start-frame shift in ``[-jitter, jitter]`` (0 disables).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        table: WaypointTable,
        start_rows: Mapping[int, int],
        waypoint_stats: dict[str, Any],
        *,
        max_blocks: int = 7,
        jitter: int = 1,
    ) -> None:
        """Initialize the dataset and index all segment starts."""
        super().__init__()
        self.base_dataset = base_dataset
        self.table = table
        self.start_rows = dict(start_rows)
        self.waypoint_stats = waypoint_stats
        self.max_blocks = max_blocks
        self.jitter = jitter

        samples = [
            (ep, m)
            for ep in sorted(self.start_rows)
            if ep in table.episodes
            for m in range(len(table.episodes[ep].keyframes) - 1)
        ]
        self._samples = np.asarray(samples, dtype=np.int64).reshape(-1, 2)
        missing = set(self.start_rows) - set(table.episodes)
        if missing:
            logger.warning("%d episodes have no waypoints and are skipped", len(missing))

    def __len__(self) -> int:
        """Number of segment starts."""
        return len(self._samples)

    def _start_frame(self, ep: int, m: int) -> int:
        kf = self.table.episodes[ep].keyframes
        frame = int(kf[m])
        if self.jitter <= 0:
            return frame
        shifted = frame + int(torch.randint(-self.jitter, self.jitter + 1, ()).item())
        # Keep the first duration positive and the frame inside the episode.
        if 0 <= shifted < int(kf[m + 1]):
            return shifted
        return frame

    def __getitem__(self, idx: int) -> Observation:
        """Return the observation at a segment start with waypoint targets in ``extra``.

        Returns:
            The observation with ``extra`` extended by the ``WP_*`` / ``SEG_*`` targets.
        """
        ep, m = (int(x) for x in self._samples[idx])
        frame = self._start_frame(ep, m)
        obs = self.base_dataset[self.start_rows[ep] + frame]
        targets = build_planner_targets(self.table.episodes[ep], m, frame, self.max_blocks)
        extra = dict(obs.extra or {})
        extra.update({k: torch.from_numpy(np.asarray(v)) for k, v in targets.items()})
        obs.extra = extra
        return obs

    @property
    def raw_features(self) -> dict:
        """Raw features of the wrapped dataset."""
        return self.base_dataset.raw_features

    @property
    def observation_features(self) -> dict[str, Feature]:
        """Observation features of the wrapped dataset."""
        return self.base_dataset.observation_features

    @property
    def action_features(self) -> dict[str, Feature]:
        """Action features of the wrapped dataset."""
        return self.base_dataset.action_features

    @property
    def fps(self) -> int:
        """Frames per second of the wrapped dataset."""
        return self.base_dataset.fps

    @property
    def tolerance_s(self) -> float:
        """Timestamp tolerance of the wrapped dataset."""
        return self.base_dataset.tolerance_s

    @property
    def delta_indices(self) -> dict[str, list[int]]:
        """Delta indices of the wrapped dataset."""
        return self.base_dataset.delta_indices

    @delta_indices.setter
    def delta_indices(self, indices: dict[str, list[int]]) -> None:
        self.base_dataset.delta_indices = indices

    @property
    def stats(self) -> dict[str, dict[str, list[float] | tuple | str]]:
        """Base dataset stats plus the ``waypoint`` entry."""
        stats = dict(self.base_dataset.stats)
        stats[WAYPOINT_STATS_KEY] = self.waypoint_stats
        return stats


class WaypointLeRobotDataModule(LeRobotDataModule):
    """``LeRobotDataModule`` whose train / eval datasets are :class:`WaypointDataset` wrappers.

    Waypoints are extracted once with AWE over the tabular columns (no video decoding)
    and cached to ``waypoint_cache`` when given.

    Args:
        waypoint_spec: AWE / signal settings. Defaults to LIBERO (EEF pose dims 0-5,
            gripper from the last action dim).
        waypoint_cache: Optional ``.npz`` path to load from / save to.
        max_blocks: Maximum planner blocks ``M``.
        jitter: Start-frame jitter for training samples.
        normalization_mode: Must match the policy ``normalization_mode``; used for ``sigma_delta``.
        **kwargs: Forwarded to :class:`LeRobotDataModule`.

    Raises:
        TypeError: If the wrapped datamodule does not use the physicalai data format.
    """

    train_dataset: Dataset
    val_eval_dataset: Dataset | None

    def __init__(
        self,
        *,
        waypoint_spec: WaypointSpec | None = None,
        waypoint_cache: str | Path | None = None,
        max_blocks: int = 7,
        jitter: int = 1,
        normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "QUANTILES",
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the datamodule and wrap its datasets."""
        super().__init__(**kwargs)
        if not isinstance(self.train_dataset, _LeRobotDatasetAdapter):
            msg = "WaypointLeRobotDataModule requires data_format='physicalai'"
            raise TypeError(msg)

        spec = waypoint_spec or WaypointSpec()
        adapters = [self.train_dataset]
        if isinstance(self.val_eval_dataset, _LeRobotDatasetAdapter):
            adapters.append(self.val_eval_dataset)

        table = self._load_or_extract(adapters, spec, waypoint_cache)
        state_stats = self.train_dataset.stats["observation.state"]
        wp_stats = waypoint_stats(table, state_stats, normalization_mode)

        self.waypoint_table = table
        self.train_dataset = WaypointDataset(
            self.train_dataset,
            table,
            episode_start_rows(self.train_dataset),
            wp_stats,
            max_blocks=max_blocks,
            jitter=jitter,
        )
        if isinstance(self.val_eval_dataset, _LeRobotDatasetAdapter):
            self.val_eval_dataset = WaypointDataset(
                self.val_eval_dataset,
                table,
                episode_start_rows(self.val_eval_dataset),
                wp_stats,
                max_blocks=max_blocks,
                jitter=0,
            )

    @staticmethod
    def _load_or_extract(
        adapters: list[_LeRobotDatasetAdapter],
        spec: WaypointSpec,
        cache: str | Path | None,
    ) -> WaypointTable:
        if cache is not None and Path(cache).is_file():
            table = WaypointTable.load(cache)
            if table.spec != spec:
                msg = f"Waypoint cache {cache} was built with {table.spec}, expected {spec}"
                raise ValueError(msg)
            logger.info("Loaded waypoints for %d episodes from %s", len(table.episodes), cache)
            return table

        table = WaypointTable(spec=spec)
        for adapter in adapters:
            table.episodes.update(extract_table_from_lerobot(adapter._lerobot_dataset, spec).episodes)  # noqa: SLF001
        logger.info("Extracted waypoints: %s", table.statistics())
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            table.save(cache)
        return table
