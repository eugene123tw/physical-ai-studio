# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities for computing delta timestamps from LeRobot policy configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lerobot.policies.factory import make_policy_config

if TYPE_CHECKING:
    from pathlib import Path

# RLDX-1 VTC video window defaults (upstream ``video_length`` / ``video_stride``).
# The backbone always consumes ``video_length`` temporal frames per step; the
# released FT checkpoints were trained with 4 frames at stride 2.
RLDX1_VIDEO_LENGTH = 4
RLDX1_VIDEO_STRIDE = 2
RLDX1_ACTION_HORIZON = 16


def _load_lerobot_metadata(
    *,
    dataset: Any = None,  # noqa: ANN401
    repo_id: str | None = None,
    root: str | Path | None = None,
    revision: str | None = None,
) -> Any:  # noqa: ANN401
    """Return LeRobot dataset metadata from a dataset object or a ``repo_id``.

    Args:
        dataset: A dataset exposing ``.meta`` (``LeRobotDataset`` or the Studio
            adapter). Preferred when a dataset is already built.
        repo_id: Dataset repo id, used to load metadata only (no episode data).
        root: Local dataset root for the ``repo_id`` lookup.
        revision: Dataset git revision for the ``repo_id`` lookup.

    Returns:
        A ``LeRobotDatasetMetadata`` (or the dataset's ``.meta``) exposing
        ``camera_keys`` and ``fps``.

    Raises:
        ValueError: If neither ``dataset`` nor ``repo_id`` is provided.
        TypeError: If ``dataset`` exposes no ``.meta``.
    """
    if dataset is not None:
        meta = getattr(dataset, "meta", None) or getattr(
            getattr(dataset, "_lerobot_dataset", None), "meta", None
        )
        if meta is None:
            msg = "`dataset` exposes no `.meta`; pass `repo_id` or explicit `obs_image_key` and `fps`."
            raise TypeError(msg)
        return meta

    if repo_id is None:
        msg = "Provide `dataset`, `repo_id`, or explicit `obs_image_key` and `fps`."
        raise ValueError(msg)

    from lerobot.datasets import LeRobotDatasetMetadata  # noqa: PLC0415

    return LeRobotDatasetMetadata(repo_id, root=root, revision=revision)


def _resolve_image_keys_and_fps(
    obs_image_key: str | list[str] | None,
    fps: float | None,
    *,
    dataset: Any = None,  # noqa: ANN401
    repo_id: str | None = None,
    root: str | Path | None = None,
    revision: str | None = None,
) -> tuple[list[str], float]:
    """Resolve the camera keys and fps, reading dataset metadata when omitted.

    Returns:
        ``(image_keys, fps)``. Camera keys default to every ``camera_keys`` entry
        in the dataset metadata; fps defaults to the dataset fps.

    Raises:
        ValueError: If metadata carries no camera keys.
    """
    if obs_image_key is not None and fps is not None:
        keys = [obs_image_key] if isinstance(obs_image_key, str) else list(obs_image_key)
        return keys, fps

    meta = _load_lerobot_metadata(dataset=dataset, repo_id=repo_id, root=root, revision=revision)
    if obs_image_key is None:
        keys = list(meta.camera_keys)
        if not keys:
            msg = "No camera keys found in dataset metadata; pass `obs_image_key` explicitly."
            raise ValueError(msg)
    else:
        keys = [obs_image_key] if isinstance(obs_image_key, str) else list(obs_image_key)
    resolved_fps = fps if fps is not None else meta.fps
    return keys, resolved_fps


def get_rldx1_delta_timestamps(
    fps: float | None = None,
    obs_image_key: str | list[str] | None = None,
    obs_state_key: str = "observation.state",
    *,
    dataset: Any = None,  # noqa: ANN401
    repo_id: str | None = None,
    root: str | Path | None = None,
    revision: str | None = None,
    video_length: int = RLDX1_VIDEO_LENGTH,
    video_stride: int = RLDX1_VIDEO_STRIDE,
    action_horizon: int = RLDX1_ACTION_HORIZON,
) -> dict[str, list[float]]:
    """Build RLDX-1 delta timestamps with the VTC multi-frame video window.

    RLDX-1 feeds the backbone ``video_length`` temporal frames per observation
    step, sampled at ``video_stride`` action-steps. The frame offsets are
    ``{(i - (video_length - 1)) * video_stride : i in [0, video_length)}`` --
    e.g. ``[-6, -4, -2, 0]`` for ``video_length=4, video_stride=2`` -- so every
    camera key returns 4 frames per step (matching ``FT-ROBOCASA``). State is
    read at the current step only; actions span the prediction horizon.

    Camera keys and fps are read from the dataset metadata by default (pass a
    built ``dataset`` or a ``repo_id``), so callers do not need to know the
    dataset's camera keys. Explicit ``obs_image_key`` / ``fps`` override the
    metadata and skip the metadata load.

    Args:
        fps: Frames per second. ``None`` reads it from the dataset metadata.
        obs_image_key: One camera key or a list of camera keys. ``None`` uses
            every ``camera_keys`` entry from the dataset metadata. Each key gets
            the full video window.
        obs_state_key: Key for state observations.
        dataset: A built dataset (``LeRobotDataset`` / Studio adapter) whose
            ``.meta`` provides the camera keys and fps.
        repo_id: Dataset repo id, used to load metadata when ``dataset`` is not
            given and keys/fps are omitted.
        root: Local dataset root for the ``repo_id`` lookup.
        revision: Dataset git revision for the ``repo_id`` lookup.
        video_length: Number of temporal frames per step.
        video_stride: Action-step stride between frames.
        action_horizon: Number of future action steps to predict.

    Returns:
        Delta timestamps for the camera key(s), state, and action.
    """
    image_keys, resolved_fps = _resolve_image_keys_and_fps(
        obs_image_key,
        fps,
        dataset=dataset,
        repo_id=repo_id,
        root=root,
        revision=revision,
    )

    video_offsets = [(i - (video_length - 1)) * video_stride for i in range(video_length)]
    video_deltas = [i / resolved_fps for i in video_offsets]

    delta_timestamps: dict[str, list[float]] = {key: list(video_deltas) for key in image_keys}
    delta_timestamps[obs_state_key] = [0.0]
    delta_timestamps["action"] = [i / resolved_fps for i in range(action_horizon)]
    return delta_timestamps


def get_delta_timestamps_from_policy(
    policy_name: str,
    fps: int = 10,
    obs_image_key: str = "observation.images.top",
    obs_state_key: str = "observation.state",
) -> dict[str, list[float]]:
    """Derive delta timestamps configuration from LeRobot policy config.

    This extracts n_obs_steps and action chunk/horizon size from the policy's
    default configuration to automatically compute the correct delta timestamps
    for use with LeRobotDataModule.

    For policies like Groot that have action_delta_indices with a capped horizon,
    we use the length of action_delta_indices rather than chunk_size to ensure
    the generated delta timestamps match what the policy expects.

    RLDX-1 is a first-party policy (not a LeRobot policy), so ``"rldx1"`` routes
    to :func:`get_rldx1_delta_timestamps`, which emits the VTC multi-frame video
    window instead of a single frame.

    Args:
        policy_name: Name of the policy (e.g., "act", "diffusion", "groot",
            "rldx1").
        fps: Frames per second of the dataset.
        obs_image_key: Key for image observations in the dataset.
        obs_state_key: Key for state observations in the dataset.

    Returns:
        Dictionary with delta timestamps for observation and action keys.

    Example:
        >>> from physicalai.data.lerobot import get_delta_timestamps_from_policy
        >>> from physicalai.data.lerobot import LeRobotDataModule

        >>> delta_timestamps = get_delta_timestamps_from_policy("act", fps=10)
        >>> datamodule = LeRobotDataModule(
        ...     repo_id="lerobot/aloha_sim_insertion_human",
        ...     delta_timestamps=delta_timestamps,
        ... )
    """
    if policy_name == "rldx1":
        return get_rldx1_delta_timestamps(
            fps=fps,
            obs_image_key=obs_image_key,
            obs_state_key=obs_state_key,
        )

    config = make_policy_config(policy_name)

    n_obs_steps: int = getattr(config, "n_obs_steps", 1)

    # Initialize delta_timestamps dictionary
    delta_timestamps: dict[str, list[float]] = {}

    # For policies with action_delta_indices (e.g., Groot), use that length as the source of truth
    # This respects the model's capped horizon (e.g., Groot caps at 16 steps even though chunk_size=50)
    action_delta_indices = getattr(config, "action_delta_indices", None)
    if action_delta_indices is not None:
        # Observation timestamps: indices from -(n_obs_steps-1) to 0
        if n_obs_steps > 1:
            obs_indices = list(range(-(n_obs_steps - 1), 1))  # e.g., [-1, 0] for n_obs_steps=2
            delta_timestamps[obs_image_key] = [i / fps for i in obs_indices]
            delta_timestamps[obs_state_key] = [i / fps for i in obs_indices]

        # Action timestamps: use the action_delta_indices directly
        delta_timestamps["action"] = [i / fps for i in action_delta_indices]

        return delta_timestamps

    # Fallback for policies without action_delta_indices (ACT, Diffusion)
    # Get action sequence length - different policies use different attribute names
    action_length_raw = (
        getattr(config, "chunk_size", None)
        or getattr(config, "horizon", None)
        or getattr(config, "action_chunk_size", None)
        or getattr(config, "n_action_steps", None)
    )
    action_length: int = int(action_length_raw) if action_length_raw is not None else 1

    # Observation timestamps: indices from -(n_obs_steps-1) to 0
    if n_obs_steps > 1:
        obs_indices = list(range(-(n_obs_steps - 1), 1))  # e.g., [-1, 0] for n_obs_steps=2
        delta_timestamps[obs_image_key] = [i / fps for i in obs_indices]
        delta_timestamps[obs_state_key] = [i / fps for i in obs_indices]

    # Action timestamps: depends on policy type (diffusion starts from -1)
    action_indices = list(range(-1, action_length - 1)) if policy_name == "diffusion" else list(range(action_length))

    delta_timestamps["action"] = [i / fps for i in action_indices]

    return delta_timestamps
