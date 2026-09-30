# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Automatic Waypoint Extraction (AWE) via dynamic programming.

Implements the waypoint selection of Shi et al., "Waypoint-Based Imitation Learning
for Robotic Manipulation" (CoRL 2023), with the two constraints used by
"Fast Plans, Faithful Actions" (Xie et al., 2026):

- every gripper state transition is a forced keyframe, and
- every segment spans at most ``max_gap`` control steps (the expert horizon ``D``).

Among all keyframe sets satisfying the error threshold, the one with the fewest
segments is selected; ties are broken by the smallest summed segment error.

The default error metric and gripper keyframe convention follow the reference
implementation (github.com/lucys0/awe, MIT): ``pos_only_geometric_waypoint_trajectory``
(max point-to-segment distance) and forcing the last frame before each gripper switch.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ErrorMetric = Literal["geometric", "temporal"]


def segment_errors(traj: np.ndarray, max_gap: int, metric: ErrorMetric = "geometric") -> np.ndarray:
    """Compute piecewise-linear reconstruction errors for all short segments.

    Args:
        traj: ``(T, D)`` trajectory.
        max_gap: Maximum segment length in steps.
        metric: ``"geometric"`` = distance to the closest point on the segment (AWE);
            ``"temporal"`` = distance to the time-interpolated point (stricter).

    Returns:
        ``(T, max_gap)`` array where ``err[i, L - 1]`` is the maximum distance of the
        interior frames ``i+1 .. i+L-1`` from the segment ``traj[i] -> traj[i+L]``.
        Entries with ``i + L >= T`` are ``inf``.
    """
    num_frames = traj.shape[0]
    err = np.full((num_frames, max_gap), np.inf)
    for length in range(1, max_gap + 1):
        n = num_frames - length
        if n <= 0:
            break
        start = traj[:n]
        delta = traj[length : length + n] - start
        denom = np.maximum(np.einsum("nd,nd->n", delta, delta), 1e-12)
        seg_err = np.zeros(n)
        for k in range(1, length):
            offset = traj[k : k + n] - start
            if metric == "geometric":
                frac = np.clip(np.einsum("nd,nd->n", offset, delta) / denom, 0.0, 1.0)
            else:
                frac = np.full(n, k / length)
            dist = np.linalg.norm(offset - frac[:, None] * delta, axis=1)
            np.maximum(seg_err, dist, out=seg_err)
        err[:n, length - 1] = seg_err
    return err


def gripper_transitions(gripper: np.ndarray, *, before: bool = True) -> np.ndarray:
    """Return gripper switch frames.

    Args:
        gripper: ``(T,)`` discrete gripper state.
        before: Return ``t`` with ``gripper[t] != gripper[t + 1]`` (AWE convention) instead
            of the first frame after the switch.

    Returns:
        Sorted frame indices.
    """
    gripper = np.asarray(gripper)
    switches = np.flatnonzero(gripper[1:] != gripper[:-1])
    return switches if before else switches + 1


def extract_keyframes(  # noqa: PLR0914
    traj: np.ndarray,
    *,
    eta: float,
    max_gap: int = 32,
    forced: np.ndarray | None = None,
    metric: ErrorMetric = "geometric",
    reference_dp: bool = False,
) -> np.ndarray:
    """Select a minimal keyframe set whose piecewise-linear reconstruction error is at most ``eta``.

    Args:
        traj: ``(T, D)`` configuration trajectory.
        eta: Maximum allowed per-frame reconstruction error inside a segment.
        max_gap: Maximum number of steps between consecutive keyframes.
        forced: Frame indices that must be keyframes (e.g. gripper transitions).
        metric: Segment error metric, see :func:`segment_errors`.
        reference_dp: Replicate the reference ``dp_waypoint_selection``, which fits the
            segment ending at keyframe ``j`` from the frame *after* the previous keyframe
            (``memo[k - 1]`` with a line from ``k``), leaving each first step unchecked.
            This yields fewer, longer segments at the same ``eta``.

    Returns:
        Sorted ``int64`` keyframe indices, always including ``0`` and ``T - 1``.

    Raises:
        ValueError: If ``traj`` is not 2-D or ``max_gap < 1``.
    """
    if traj.ndim != 2:  # noqa: PLR2004
        msg = f"traj must be (T, D), got shape {traj.shape}"
        raise ValueError(msg)
    if max_gap < 1:
        msg = f"max_gap must be >= 1, got {max_gap}"
        raise ValueError(msg)

    num_frames = traj.shape[0]
    if num_frames <= 1:
        return np.zeros(min(num_frames, 1), dtype=np.int64)

    # A segment (i, j) may not jump over a forced keyframe strictly inside it.
    next_forced = np.full(num_frames, num_frames - 1, dtype=np.int64)
    if forced is not None and len(forced) > 0:
        forced_sorted = np.unique(np.asarray(forced, dtype=np.int64))
        forced_sorted = forced_sorted[(forced_sorted > 0) & (forced_sorted < num_frames)]
        pos = np.searchsorted(forced_sorted, np.arange(num_frames), side="right")
        has_next = pos < len(forced_sorted)
        next_forced[has_next] = forced_sorted[pos[has_next]]

    err = segment_errors(traj, max_gap, metric)

    count = np.full(num_frames, np.iinfo(np.int64).max, dtype=np.int64)
    err_sum = np.full(num_frames, np.inf)
    # Adjacent keyframes (no fitted frame) are impossible upstream; used only when nothing else reaches j.
    adjacent = np.zeros(num_frames, dtype=np.int64)
    prev = np.full(num_frames, -1, dtype=np.int64)
    count[0] = 0
    err_sum[0] = 0.0

    for j in range(1, num_frames):
        lo = max(0, j - max_gap)
        cand = np.arange(lo, j)
        if reference_dp:
            fit_len = j - cand - 1
            seg = np.where(fit_len > 0, err[np.minimum(cand + 1, j), np.maximum(fit_len - 1, 0)], 0.0)
        else:
            fit_len = np.ones_like(cand)
            seg = err[cand, j - cand - 1]
        ok = (seg <= eta) & (next_forced[cand] >= j) & (count[cand] < np.iinfo(np.int64).max)
        # A one-step segment has no interior frames, so j-1 is always a valid predecessor.
        cand, seg, fit_len = cand[ok], seg[ok], fit_len[ok]
        new_count = count[cand] + 1
        new_err = err_sum[cand] + seg
        new_adjacent = adjacent[cand] + (fit_len == 0)
        # The reference keeps the earliest predecessor among equal counts; lexsort's last key is primary.
        order = (cand, new_count, new_adjacent) if reference_dp else (new_err, new_count)
        best = np.lexsort(order)[0]
        count[j] = new_count[best]
        err_sum[j] = new_err[best]
        adjacent[j] = new_adjacent[best]
        prev[j] = cand[best]

    keyframes = [num_frames - 1]
    while keyframes[-1] != 0:
        keyframes.append(int(prev[keyframes[-1]]))
    return np.asarray(keyframes[::-1], dtype=np.int64)


def extract_waypoints(
    config_traj: np.ndarray,
    gripper: np.ndarray,
    *,
    eta: float,
    max_gap: int = 32,
    metric: ErrorMetric = "geometric",
    reference_dp: bool = False,
) -> np.ndarray:
    """Run AWE with gripper transitions as forced keyframes.

    Args:
        config_traj: ``(T, D)`` configuration trajectory (e.g. end-effector pose or joints).
        gripper: ``(T,)`` or ``(T, n_arms)`` discrete gripper state per frame.
        eta: Segment error threshold.
        max_gap: Maximum segment length (expert horizon ``D``).
        metric: Segment error metric, see :func:`segment_errors`.
        reference_dp: See :func:`extract_keyframes`.

    Returns:
        Keyframe indices including ``0`` (episode start) and ``T - 1`` (episode end).
    """
    gripper = np.asarray(gripper)
    if gripper.ndim == 1:
        gripper = gripper[:, None]
    forced = np.unique(np.concatenate([gripper_transitions(gripper[:, a]) for a in range(gripper.shape[1])]))
    return extract_keyframes(
        config_traj,
        eta=eta,
        max_gap=max_gap,
        forced=forced,
        metric=metric,
        reference_dp=reference_dp,
    )


def reconstruct(config_traj: np.ndarray, keyframes: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``config_traj`` between keyframes (for visualization / checks).

    Returns:
        ``(T, D)`` reconstructed trajectory.
    """
    num_frames = config_traj.shape[0]
    t = np.arange(num_frames)
    return np.stack(
        [np.interp(t, keyframes, config_traj[keyframes, d]) for d in range(config_traj.shape[1])],
        axis=1,
    )
