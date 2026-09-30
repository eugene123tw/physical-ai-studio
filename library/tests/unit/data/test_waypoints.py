# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for AWE waypoint extraction and waypoint targets."""

from __future__ import annotations

import numpy as np
import pytest

from physicalai.data.waypoints import (
    EpisodeWaypoints,
    WaypointSpec,
    WaypointTable,
    build_planner_targets,
    compute_sigma_delta,
    extract_keyframes,
    extract_table,
    extract_waypoints,
    gripper_transitions,
    segment_errors,
)


def _reference_dp(traj: np.ndarray, eta: float) -> list[int]:
    """Oracle: port of lucys0/awe ``dp_waypoint_selection(pos_only=True)`` (MIT)."""

    def point_line(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        v, w = b - a, p - a
        t = float(np.clip(np.dot(w, v) / np.dot(v, v), 0, 1)) if np.dot(v, v) > 0 else 0.0
        return float(np.linalg.norm(p - (a + t * v)))

    def traj_err(sub: np.ndarray, wps: list[int]) -> float:
        wps = [0, *wps] if wps[0] != 0 else wps
        errs = [
            point_line(sub[t], sub[wps[s]], sub[wps[s + 1]])
            for s in range(len(wps) - 1)
            for t in range(wps[s], wps[s + 1])
        ]
        return max(errs)

    n = len(traj)
    memo: dict[int, tuple[float, list[int]]] = {i: (0, []) for i in range(n)}
    for i in range(1, n):
        best_n, best = float("inf"), []
        for k in range(1, i):
            if traj_err(traj[k : i + 1], [i - k]) < eta and memo[k - 1][0] + 1 < best_n:
                best_n, best = memo[k - 1][0] + 1, [*memo[k - 1][1], i]
        memo[i] = (best_n, best)
    return sorted({*memo[n - 1][1], n - 1})


def _piecewise_linear(corners: list[int], dim: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(len(corners), dim))
    t = np.arange(corners[-1] + 1)
    return np.stack([np.interp(t, corners, points[:, d]) for d in range(dim)], axis=1)


class TestSegmentErrors:
    def test_straight_line_has_zero_error(self) -> None:
        traj = np.linspace(0, 1, 20)[:, None] * np.ones((1, 3))
        err = segment_errors(traj, max_gap=8)
        assert np.allclose(err[np.isfinite(err)], 0.0)

    def test_out_of_range_is_inf(self) -> None:
        err = segment_errors(np.zeros((5, 2)), max_gap=8)
        assert np.isinf(err[0, 4:]).all()
        assert np.isinf(err[4, 0])

    def test_geometric_not_larger_than_temporal(self) -> None:
        traj = np.random.default_rng(1).normal(size=(30, 4)).cumsum(axis=0)
        geo = segment_errors(traj, 10, "geometric")
        tem = segment_errors(traj, 10, "temporal")
        finite = np.isfinite(geo)
        assert np.all(geo[finite] <= tem[finite] + 1e-12)


class TestExtractKeyframes:
    def test_recovers_corners_of_piecewise_linear(self) -> None:
        corners = [0, 7, 15, 30, 41]
        traj = _piecewise_linear(corners)
        kf = extract_keyframes(traj, eta=1e-6, max_gap=64, metric="temporal")
        assert kf.tolist() == corners

    def test_endpoints_and_max_gap(self) -> None:
        traj = np.random.default_rng(2).normal(size=(200, 6)).cumsum(axis=0) * 0.01
        kf = extract_keyframes(traj, eta=0.05, max_gap=16)
        assert kf[0] == 0
        assert kf[-1] == len(traj) - 1
        assert np.all(np.diff(kf) >= 1)
        assert np.all(np.diff(kf) <= 16)

    def test_error_below_threshold(self) -> None:
        traj = np.random.default_rng(3).normal(size=(150, 3)).cumsum(axis=0) * 0.02
        eta = 0.03
        kf = extract_keyframes(traj, eta=eta, max_gap=32)
        err = segment_errors(traj, 32)
        seg = [err[a, b - a - 1] for a, b in zip(kf[:-1], kf[1:], strict=True)]
        assert max(seg) <= eta

    def test_forced_keyframes_are_kept(self) -> None:
        traj = np.linspace(0, 1, 60)[:, None] * np.ones((1, 2))
        kf = extract_keyframes(traj, eta=1.0, max_gap=64, forced=np.array([13, 40]))
        assert kf.tolist() == [0, 13, 40, 59]

    @pytest.mark.parametrize("seed", range(6))
    def test_reference_dp_matches_upstream(self, seed: int) -> None:
        traj = np.random.default_rng(seed).normal(size=(45, 3)).cumsum(axis=0) * 0.01
        eta = 0.01
        ours = extract_keyframes(traj, eta=eta * (1 - 1e-9), max_gap=len(traj), reference_dp=True)
        assert ours[1:].tolist() == _reference_dp(traj, eta)

    def test_reference_dp_uses_fewer_or_equal_waypoints(self) -> None:
        traj = np.random.default_rng(5).normal(size=(80, 3)).cumsum(axis=0) * 0.01
        strict = extract_keyframes(traj, eta=0.01, max_gap=80)
        loose = extract_keyframes(traj, eta=0.01, max_gap=80, reference_dp=True)
        assert len(loose) <= len(strict)

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="traj"):
            extract_keyframes(np.zeros(5), eta=0.1)
        with pytest.raises(ValueError, match="max_gap"):
            extract_keyframes(np.zeros((5, 2)), eta=0.1, max_gap=0)


class TestGripper:
    def test_transitions_before_switch(self) -> None:
        g = np.array([0, 0, 1, 1, 1, 0])
        assert gripper_transitions(g).tolist() == [1, 4]
        assert gripper_transitions(g, before=False).tolist() == [2, 5]

    def test_extract_waypoints_forces_gripper_frames(self) -> None:
        traj = np.linspace(0, 1, 50)[:, None] * np.ones((1, 3))
        g = np.zeros(50, dtype=int)
        g[20:35] = 1
        kf = extract_waypoints(traj, g, eta=1.0, max_gap=64)
        assert {19, 34}.issubset(set(kf.tolist()))


def _episode() -> EpisodeWaypoints:
    return EpisodeWaypoints(
        episode_index=0,
        length=41,
        keyframes=np.array([0, 5, 12, 20, 40]),
        config=np.arange(5 * 2, dtype=np.float32).reshape(5, 2),
        gripper=np.array([[0], [0], [1], [1], [0]]),
    )


class TestPlannerTargets:
    def test_window_with_terminal(self) -> None:
        t = build_planner_targets(_episode(), segment=1, start_frame=5, max_blocks=7)
        assert t["wp_valid"].tolist() == [True, True, True, True, False, False, False]
        assert t["wp_d"][:4].tolist() == [7, 8, 20, 0]
        assert int(t["seg_d"]) == 7
        assert t["wp_cur_g"].tolist() == [0]
        np.testing.assert_array_equal(t["seg_q"], [4.0, 5.0])

    def test_window_truncated_without_terminal(self) -> None:
        t = build_planner_targets(_episode(), segment=0, start_frame=0, max_blocks=2)
        assert t["wp_valid"].tolist() == [True, True]
        assert t["wp_d"].tolist() == [5, 7]

    def test_jitter_changes_first_duration_only(self) -> None:
        t = build_planner_targets(_episode(), segment=1, start_frame=4, max_blocks=7)
        assert t["wp_d"][:3].tolist() == [8, 8, 20]


class TestTable:
    def test_extract_table_and_roundtrip(self, tmp_path) -> None:  # noqa: ANN001
        rng = np.random.default_rng(0)
        ep = np.repeat([3, 7], [60, 45])
        state = rng.normal(size=(105, 8)).cumsum(axis=0) * 0.01
        action = np.ones((105, 7))
        action[20:40, -1] = -1
        table = extract_table(ep, state, action, WaypointSpec(eta=0.02))
        assert set(table.episodes) == {3, 7}
        stats = table.statistics()
        assert stats["num_segments"] == len(table.segment_durations())
        assert stats["max"] <= 32

        path = tmp_path / "wp.npz"
        table.save(path)
        loaded = WaypointTable.load(path)
        assert loaded.spec == table.spec
        for k in table.episodes:
            np.testing.assert_array_equal(loaded.episodes[k].keyframes, table.episodes[k].keyframes)
            np.testing.assert_allclose(loaded.episodes[k].config, table.episodes[k].config)

    def test_sigma_delta_floor_and_shape(self) -> None:
        table = WaypointTable(spec=WaypointSpec(config_dims=(0, 1)), episodes={0: _episode()})
        stats = {"q01": [0.0, 0.0], "q99": [10.0, 10.0], "mean": [0, 0], "std": [1, 1]}
        sigma = compute_sigma_delta(table, stats, "QUANTILES")
        assert sigma.shape == (2,)
        assert np.all(sigma >= 1e-3)


class _FrameDataset:
    """Minimal frame dataset: row i holds frame index in state[0]."""

    def __init__(self, length: int) -> None:
        self.length = length
        self.delta_indices: dict = {}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):  # noqa: ANN204
        import torch

        from physicalai.data import Observation

        return Observation(state=torch.tensor([float(idx), 0.0]), extra={"action_is_pad": torch.zeros(4, dtype=torch.bool)})

    @property
    def stats(self) -> dict:
        return {"observation.state": {"name": "state"}}


class TestWaypointDataset:
    def test_items_are_segment_starts(self) -> None:
        from physicalai.data.waypoints import WaypointDataset

        table = WaypointTable(spec=WaypointSpec(config_dims=(0, 1)), episodes={0: _episode()})
        ds = WaypointDataset(_FrameDataset(41), table, {0: 0}, {"name": "waypoint"}, max_blocks=3, jitter=0)  # type: ignore[arg-type]
        assert len(ds) == 4  # noqa: PLR2004
        obs = ds[1]
        assert float(obs.state[0]) == 5.0  # noqa: PLR2004
        assert obs.extra["wp_d"].tolist() == [7, 8, 20]
        assert "action_is_pad" in obs.extra
        assert ds.stats["waypoint"] == {"name": "waypoint"}

    def test_jitter_keeps_first_duration_positive(self) -> None:
        from physicalai.data.waypoints import WaypointDataset

        table = WaypointTable(spec=WaypointSpec(config_dims=(0, 1)), episodes={0: _episode()})
        ds = WaypointDataset(_FrameDataset(41), table, {0: 0}, {}, max_blocks=3, jitter=1)  # type: ignore[arg-type]
        for _ in range(50):
            for i in range(len(ds)):
                obs = ds[i]
                assert int(obs.extra["seg_d"]) >= 1
                assert abs(float(obs.state[0]) - float(_episode().keyframes[i])) <= 1
