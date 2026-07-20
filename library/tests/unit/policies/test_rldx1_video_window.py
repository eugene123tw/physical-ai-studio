# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rollout-time VTC frame-window tests for the RLDX-1 policy.

The released FT checkpoints were trained with the VTC video path always on --
4 frames per observation step at strides ``[-6, -4, -2, 0]``. During a gym
rollout the environment feeds one frame per step, so the policy must reassemble
that temporal window from a per-view history buffer (:meth:`Rldx1.select_action`
records every env-step; :meth:`Rldx1.predict_action_chunk` samples the window).
These tests cover the buffer mechanics offline -- no model weights required.
"""

from __future__ import annotations

import numpy as np
import torch

from physicalai.policies.rldx1 import Rldx1

_VIEW = "observation.images.cam0"
_VIDEO_LENGTH = 4
_VIDEO_STRIDE = 2


def _frame(step: int) -> torch.Tensor:
    """Return a ``(B=1, C=3, H=2, W=2)`` frame filled with ``step``."""
    return torch.full((1, 3, 2, 2), float(step))


def _bare_policy() -> Rldx1:
    """Construct a model-less policy (offline, no weight download).

    The default config ships ``video_length=4`` / ``video_stride=2``.
    """
    return Rldx1()


def test_window_samples_expected_strides() -> None:
    """A full history yields frames at offsets ``[-6, -4, -2, 0]``."""
    policy = _bare_policy()
    for step in range(10):
        policy._record_video_frames({_VIEW: _frame(step)})  # noqa: SLF001

    windowed = policy._apply_video_window({_VIEW: _frame(9)}, [_VIEW])  # noqa: SLF001
    stacked = windowed[_VIEW]

    # (B, T, C, H, W) with T == video_length.
    assert stacked.shape == (1, _VIDEO_LENGTH, 3, 2, 2)
    # Offsets [-6, -4, -2, 0] from the most recent step (9) -> [3, 5, 7, 9].
    sampled = [int(stacked[0, t, 0, 0, 0].item()) for t in range(_VIDEO_LENGTH)]
    assert sampled == [3, 5, 7, 9]


def test_window_clamps_when_history_short() -> None:
    """Before the window span fills, offsets clamp to the oldest frame."""
    policy = _bare_policy()
    for step in range(3):  # only steps 0, 1, 2 recorded
        policy._record_video_frames({_VIEW: _frame(step)})  # noqa: SLF001

    windowed = policy._apply_video_window({_VIEW: _frame(2)}, [_VIEW])  # noqa: SLF001
    sampled = [int(windowed[_VIEW][0, t, 0, 0, 0].item()) for t in range(_VIDEO_LENGTH)]
    # -6, -4, -2 clamp to the first frame (0); 0 is the latest (2).
    assert sampled == [0, 0, 0, 2]


def test_history_buffer_bounded_to_span() -> None:
    """The per-view deque holds only ``span + 1`` frames."""
    policy = _bare_policy()
    for step in range(20):
        policy._record_video_frames({_VIEW: _frame(step)})  # noqa: SLF001

    span = (_VIDEO_LENGTH - 1) * _VIDEO_STRIDE
    assert policy._frame_history is not None  # noqa: SLF001
    assert len(policy._frame_history[_VIEW]) == span + 1  # noqa: SLF001


def test_reset_clears_history() -> None:
    """``reset`` drops the frame history so a new episode starts fresh."""
    policy = _bare_policy()
    policy._record_video_frames({_VIEW: _frame(0)})  # noqa: SLF001
    assert policy._frame_history is not None  # noqa: SLF001

    policy.reset()
    assert policy._frame_history is None  # noqa: SLF001


def test_prepare_window_seeds_history_on_direct_call() -> None:
    """A direct single-frame call (no prior record) seeds and stacks the frame."""
    policy = _bare_policy()
    prepared = policy._prepare_video_window({_VIEW: _frame(5)})  # noqa: SLF001

    stacked = prepared[_VIEW]
    assert stacked.shape == (1, _VIDEO_LENGTH, 3, 2, 2)
    # All offsets clamp to the single seeded frame.
    sampled = [int(stacked[0, t, 0, 0, 0].item()) for t in range(_VIDEO_LENGTH)]
    assert sampled == [5, 5, 5, 5]


def test_prepare_window_passes_through_multiframe_batch() -> None:
    """An already-multi-frame ``(B, T, C, H, W)`` batch is not re-stacked."""
    policy = _bare_policy()
    multiframe = torch.stack([_frame(t) for t in range(_VIDEO_LENGTH)], dim=1)  # (1, T, 3, 2, 2)
    batch = {_VIEW: multiframe}

    prepared = policy._prepare_video_window(batch)  # noqa: SLF001

    # Same object returned unchanged; history untouched.
    assert prepared is batch
    assert policy._frame_history is None  # noqa: SLF001


def test_multiframe_input_not_recorded() -> None:
    """Recording skips batches that already carry a temporal axis."""
    policy = _bare_policy()
    multiframe = torch.stack([_frame(t) for t in range(_VIDEO_LENGTH)], dim=1)
    policy._record_video_frames({_VIEW: multiframe})  # noqa: SLF001
    assert policy._frame_history is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Rldx1Model.observation_delta_indices
# ---------------------------------------------------------------------------


def _bare_model(video_length: int = 4, video_stride: int = 2) -> "Rldx1Model":
    from physicalai.policies.rldx1.model import Rldx1Model  # noqa: PLC0415

    return Rldx1Model(video_length=video_length, video_stride=video_stride)


def test_observation_delta_indices_default() -> None:
    """Default (length=4, stride=2) yields [-6, -4, -2, 0]."""
    model = _bare_model()
    assert model.observation_delta_indices == [-6, -4, -2, 0]


def test_observation_delta_indices_ends_at_zero() -> None:
    """The last index is always 0 (current timestep)."""
    for vl, vs in [(1, 1), (2, 3), (4, 2), (6, 1)]:
        assert _bare_model(vl, vs).observation_delta_indices[-1] == 0


def test_observation_delta_indices_length_matches_video_length() -> None:
    """Number of indices equals video_length."""
    for vl in [1, 2, 4, 8]:
        assert len(_bare_model(video_length=vl).observation_delta_indices) == vl


def test_observation_delta_indices_stride_1() -> None:
    """Stride 1, length 4 yields contiguous [-3, -2, -1, 0]."""
    assert _bare_model(video_length=4, video_stride=1).observation_delta_indices == [-3, -2, -1, 0]


def test_observation_delta_indices_single_frame() -> None:
    """video_length=1 always yields [0] regardless of stride."""
    assert _bare_model(video_length=1, video_stride=5).observation_delta_indices == [0]


def test_observation_delta_indices_returns_list_of_int() -> None:
    """Return type is list[int]."""
    indices = _bare_model().observation_delta_indices
    assert isinstance(indices, list)
    assert all(isinstance(i, int) for i in indices)


def test_numpy_frames_are_coerced() -> None:
    """Numpy view arrays are accepted and coerced to tensors."""
    policy = _bare_policy()
    policy._record_video_frames({_VIEW: np.full((1, 3, 2, 2), 4.0, dtype=np.float32)})  # noqa: SLF001

    windowed = policy._apply_video_window({_VIEW: _frame(4)}, [_VIEW])  # noqa: SLF001
    assert torch.is_tensor(windowed[_VIEW])
    assert int(windowed[_VIEW][0, -1, 0, 0, 0].item()) == 4
