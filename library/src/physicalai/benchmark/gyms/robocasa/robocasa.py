# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RoboCasa benchmark - specialized benchmark for RoboCasa task groups.

This module provides `RoboCasaBenchmark`, a convenience class that auto-creates
gyms for RoboCasa task groups with sensible defaults.

Example:
    >>> benchmark = RoboCasaBenchmark(task="atomic_seen", num_episodes=20)
    >>> results = benchmark.evaluate(policy)
    >>> print(results.summary())

    # Compare multiple policies
    >>> results = {p.name: benchmark.evaluate(p) for p in [act, pi0, groot]}
    >>> for name, r in results.items():
    ...     print(f"{name}: {r.overall_success_rate:.1%}")
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from physicalai.benchmark.gyms.benchmark import Benchmark

if TYPE_CHECKING:
    from pathlib import Path


class RoboCasaMaxSteps(IntEnum):
    """Maximum steps per episode for each RoboCasa task group.

    Used to resolve the default ``max_steps`` when none is provided;
    falls back to ``DEFAULT`` for unrecognised group or task names.
    """

    atomic_seen = 1000
    composite_seen = 1000
    composite_unseen = 1000
    pretrain50 = 1000
    pretrain100 = 1000
    pretrain200 = 1000
    pretrain300 = 1000
    DEFAULT = 1000


class RoboCasaBenchmark(Benchmark):
    """Specialized benchmark for RoboCasa task groups.

    Auto-creates `RoboCasaGym` instances for all tasks in the specified group.
    Provides sensible defaults for RoboCasa evaluation.

    Args:
        task: RoboCasa task group keyword, single task name, or
            comma-separated task names. Group keywords:
            - "atomic_seen" (18 atomic tasks, target split)
            - "composite_seen" (composite tasks, target split)
            - "composite_unseen" (composite tasks, target split)
            - "pretrain50" / "pretrain100" / "pretrain200" / "pretrain300"
              (pretrain splits of increasing size)
        num_episodes: Number of episodes per task (default: 20).
        max_steps: Maximum steps per episode (default: 1000).
        seed: Random seed for reproducibility (default: 42).
        observation_height: Height of observation images (default: 256).
        observation_width: Width of observation images (default: 256).
        video_dir: Directory to save videos. None disables recording.
        record_mode: Video recording mode - "all", "successes", "failures", "none".

    Example:
        >>> # Full atomic_seen benchmark
        >>> benchmark = RoboCasaBenchmark(task="atomic_seen", num_episodes=20)
        >>> results = benchmark.evaluate(policy)

        >>> # Quick test on specific tasks
        >>> benchmark = RoboCasaBenchmark(
        ...     task="CloseFridge,OpenDrawer",
        ...     num_episodes=5,
        ... )
        >>> results = benchmark.evaluate(policy)
    """

    def __init__(
        self,
        task: str = "atomic_seen",
        num_episodes: int = 20,
        max_steps: int | None = None,
        seed: int = 42,
        observation_height: int = 256,
        observation_width: int = 256,
        video_dir: str | Path | None = None,
        record_mode: str = "failures",
    ) -> None:
        """Initialize RoboCasa benchmark with task group configuration."""
        self.task = task
        self.observation_height = observation_height
        self.observation_width = observation_width

        # Use RoboCasa default max_steps if not specified
        if max_steps is None:
            max_steps = getattr(RoboCasaMaxSteps, task, RoboCasaMaxSteps.DEFAULT).value

        # Create gyms for the task group
        gyms = self._create_gyms()

        super().__init__(
            gyms=gyms,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
            video_dir=video_dir,
            record_mode=record_mode,
        )

    def _create_gyms(self) -> list:
        """Create RoboCasaGym instances for the task group.

        Sets ``task_id`` and ``task_suite_name`` on each gym so the base
        class ``_get_task_id``/``_get_task_name`` protocol can build
        per-task result keys.

        Returns:
            List of RoboCasaGym instances.
        """
        from physicalai.gyms import create_robocasa_gyms  # noqa: PLC0415

        gyms = create_robocasa_gyms(
            tasks=self.task,
            observation_height=self.observation_height,
            observation_width=self.observation_width,
        )
        for gym in gyms:
            gym.task_id = gym.task  # type: ignore[attr-defined]
            gym.task_suite_name = self.task  # type: ignore[attr-defined]
        return gyms

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RoboCasaBenchmark(task={self.task!r}, num_episodes={self.num_episodes}, max_steps={self.max_steps})"
