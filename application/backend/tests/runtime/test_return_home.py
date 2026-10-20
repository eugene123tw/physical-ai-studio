import numpy as np
import pytest

from runtime.return_home import ReturnHomeBlockedError, ReturnHomeSource

from .fakes import FakeObservation


def _observation(values: list[float], timestamp: float) -> FakeObservation:
    return FakeObservation(np.array(values, dtype=np.float32), timestamp)


def _source(start: list[float], target: list[float]) -> ReturnHomeSource:
    return ReturnHomeSource(
        start=np.array(start, dtype=np.float32),
        target=np.array(target, dtype=np.float32),
        duration_s=2.5,
    )


def test_moves_every_joint_at_a_constant_velocity_so_they_arrive_together() -> None:
    source = _source([10, 1], [0, 0])

    commands = [
        source.update(_observation([10, 1], 100.0), {}, 0),
        source.update(_observation([10, 1], 101.0), {}, 1),
        source.update(_observation([6, 0.6], 102.0), {}, 2),
    ]

    np.testing.assert_allclose(commands, [[10, 1], [6, 0.6], [2, 0.2]], rtol=1e-6)
    assert not source.finished


def test_a_late_cycle_catches_up_instead_of_stretching_the_move() -> None:
    source = _source([10], [0])
    source.update(_observation([10], 100.0), {}, 0)

    command = source.update(_observation([10], 102.0), {}, 1)

    np.testing.assert_allclose(command, [2], rtol=1e-6)


def test_finishes_once_an_observation_tracks_the_final_command() -> None:
    source = _source([10], [0])
    source.update(_observation([10], 100.0), {}, 0)

    arrived = source.update(_observation([10], 100.1), {}, 1)
    np.testing.assert_allclose(arrived, [9.6], rtol=1e-6)
    np.testing.assert_allclose(source.update(_observation([9.6], 103.0), {}, 2), [0])
    assert not source.finished

    np.testing.assert_allclose(source.update(_observation([0.5], 103.1), {}, 3), [0])
    assert source.finished


def test_raises_when_a_joint_stops_following_mid_move() -> None:
    source = _source([0, 10], [0, 0])
    source.update(_observation([0, 10], 100.0), {}, 0)
    source.update(_observation([0, 10], 101.0), {}, 1)

    with pytest.raises(ReturnHomeBlockedError) as blocked:
        source.update(_observation([0, 10], 101.5), {}, 2)

    assert blocked.value.joint_index == 1
    assert blocked.value.error == pytest.approx(0.4)
    np.testing.assert_allclose(blocked.value.measured, [0, 10])


def test_raises_when_the_arm_is_blocked_short_of_the_target() -> None:
    source = _source([10], [0])
    source.update(_observation([10], 100.0), {}, 0)
    source.update(_observation([10], 102.5), {}, 1)

    # 1.6 units short of the final command is 16% of the move.
    with pytest.raises(ReturnHomeBlockedError):
        source.update(_observation([1.6], 102.6), {}, 2)
    assert not source.finished


def test_finishes_immediately_when_already_at_the_target() -> None:
    source = _source([3, 4], [3, 4])

    np.testing.assert_allclose(source.update(_observation([3, 4], 100.0), {}, 0), [3, 4])
    assert source.finished


def test_rejects_a_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        ReturnHomeSource(start=np.zeros(1), target=np.ones(1), duration_s=0)
