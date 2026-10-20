import queue

import numpy as np
import pytest

from runtime.action_source import StudioActionSource
from runtime.contract import (
    InMemoryCommandMailbox,
    QueueEventSink,
    ReturnHomeCommand,
    SetFollowerSourceCommand,
    SetInterpolationCommand,
    StopTaskCommand,
)
from runtime.interpolation import AdjustableLinearInterpolator

from .fakes import FakeObservation, FakeRobot


def _observation(values: list[float], timestamp: float, *, efforts: list[float] | None = None) -> FakeObservation:
    sensor_data = None if efforts is None else {"efforts": np.array(efforts, dtype=np.float32)}
    return FakeObservation(np.array(values, dtype=np.float32), timestamp, sensor_data)


def _source(*, follower: FakeRobot, leader: FakeRobot | None, fps: float = 30):
    mailbox = InMemoryCommandMailbox()
    events = QueueEventSink()
    source = StudioActionSource(
        follower=follower,
        leader=leader,
        mailbox=mailbox,
        event_sink=events,
        fps=fps,
    )
    source.connect(bus=object(), session_id="test")
    return source, mailbox, events


def test_hold_latches_the_target_on_entry() -> None:
    follower = FakeRobot([_observation([1, 2], 1), _observation([3, 4], 2)])
    source, _, _ = _source(follower=follower, leader=None)

    first = source.update(follower.get_observation(), {}, 0)
    second = source.update(follower.get_observation(), {}, 1)

    np.testing.assert_array_equal(first, [1, 2])
    np.testing.assert_array_equal(second, [1, 2])


def test_teleop_forwards_leader_positions_on_next_tick() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    leader = FakeRobot([_observation([4, 5], 1)])
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))

    action = source.update(follower.get_observation(), {}, 0)

    np.testing.assert_array_equal(action, [4, 5])


def test_connect_rejects_mismatched_joint_names() -> None:
    follower = FakeRobot([_observation([0], 1)], joint_names=["follower_joint"])
    leader = FakeRobot([_observation([0], 1)], joint_names=["leader_joint"])

    with pytest.raises(ValueError, match="joint names must match"):
        _source(follower=follower, leader=leader)


def test_leader_read_error_uses_hold_and_session_survives() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    leader = FakeRobot([_observation([4, 5], 1)], observation_error="lost")
    source, mailbox, _ = _source(follower=follower, leader=leader)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))

    action = source.update(follower.get_observation(), {}, 0)

    np.testing.assert_array_equal(action, [1, 2])
    assert source.follower_source == "teleop"


def test_leader_failure_is_bounded_and_emits_one_error() -> None:
    follower = FakeRobot([_observation([1, 2], 1)])
    leader = FakeRobot([_observation([4, 5], 1)], observation_error="lost")
    source, mailbox, events = _source(follower=follower, leader=leader, fps=1)
    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    state = follower.get_observation()

    for step in range(6):
        source.update(state, {}, step)

    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except Exception:
            break
    assert source.follower_source == "hold"
    assert sum(event.event == "error" for event in emitted) == 1

    mailbox.apply(SetFollowerSourceCommand(follower_source="teleop"))
    source.update(state, {}, 7)
    assert source.follower_source == "hold"


def _drain(events: QueueEventSink) -> list:
    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except queue.Empty:
            return emitted


def test_return_home_glides_back_to_the_session_start_pose_then_holds() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    source, mailbox, _ = _source(follower=follower, leader=None)
    source.update(_observation([0, 0], 1), {}, 0)

    mailbox.apply(ReturnHomeCommand())
    # The move lasts RETURN_HOME_DURATION_S (2.5 s) of observation time.
    start = source.update(_observation([10, 20], 3), {}, 1)
    assert source.follower_source == "hold"
    assert source.state_data().returning_home is True
    middle = source.update(_observation([10, 20], 4), {}, 2)
    end = source.update(_observation([6, 12], 5.5), {}, 3)

    np.testing.assert_allclose(start, [10, 20])
    np.testing.assert_allclose(middle, [6, 12])
    np.testing.assert_allclose(end, [0, 0])
    # The return stays active until an observation confirms the final command.
    assert source.state_data().returning_home is True
    np.testing.assert_allclose(source.update(_observation([0, 0], 6), {}, 4), [0, 0])
    assert source.state_data().returning_home is False
    np.testing.assert_allclose(source.update(_observation([0, 0], 7), {}, 5), [0, 0])


def test_return_home_aborts_and_holds_the_measured_pose_when_the_arm_is_blocked() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    source, mailbox, events = _source(follower=follower, leader=None)
    source.update(_observation([0, 0], 1), {}, 0)
    mailbox.apply(ReturnHomeCommand())
    source.update(_observation([10, 0], 2), {}, 1)
    source.update(_observation([10, 0], 2.25), {}, 2)
    source.update(_observation([10, 0], 2.5), {}, 3)
    _drain(events)

    # The arm stays at 10 while the 8 command asks it to move: 20% of a 10-unit move.
    action = source.update(_observation([10, 0], 2.75), {}, 4)

    np.testing.assert_allclose(action, [10, 0])
    emitted = _drain(events)
    assert any(event.event == "error" and event.error_code == "return_home_aborted" for event in emitted)
    assert source.state_data().returning_home is False


def test_a_mode_change_cancels_return_home() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    source, mailbox, _ = _source(follower=follower, leader=None)
    source.update(_observation([0, 0], 1), {}, 0)
    mailbox.apply(ReturnHomeCommand())
    first = source.update(_observation([10, 0], 2), {}, 1)

    mailbox.apply(StopTaskCommand())
    held = source.update(_observation(list(first), 3), {}, 2)

    np.testing.assert_allclose(held, first)
    assert source.state_data().returning_home is False


def test_stop_return_holds_the_measured_pose() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    source, mailbox, _ = _source(follower=follower, leader=None)
    source.update(_observation([0, 0], 1), {}, 0)
    mailbox.apply(ReturnHomeCommand())
    source.update(_observation([10, 0], 2), {}, 1)

    # The source is already "hold", so this request only cancels the return.
    mailbox.apply(SetFollowerSourceCommand(follower_source="hold"))
    held = source.update(_observation([9.5, 0], 3), {}, 2)

    np.testing.assert_allclose(held, [9.5, 0])
    assert source.state_data().returning_home is False


def _interpolated_source(
    follower: FakeRobot,
) -> tuple[StudioActionSource, InMemoryCommandMailbox, AdjustableLinearInterpolator]:
    mailbox = InMemoryCommandMailbox()
    interpolator = AdjustableLinearInterpolator(multiplier=3)
    source = StudioActionSource(
        follower=follower,
        leader=None,
        mailbox=mailbox,
        event_sink=QueueEventSink(),
        fps=10,
        interpolator=interpolator,
    )
    source.connect(bus=object(), session_id="test")
    return source, mailbox, interpolator


def _first_substep(interpolator: AdjustableLinearInterpolator, action: np.ndarray) -> np.ndarray:
    # Mirror RobotRuntime: add the source action, then send substep 0.
    interpolator.add(action)
    return interpolator.get(0)


def test_return_home_does_not_blend_from_a_stale_target() -> None:
    follower = FakeRobot([_observation([0], 1)])
    source, mailbox, interpolator = _interpolated_source(follower)
    source.update(_observation([0], 1), {}, 0)
    interpolator.add(np.array([100.0], dtype=np.float32))

    mailbox.apply(ReturnHomeCommand())
    action = source.update(_observation([10], 2), {}, 1)

    np.testing.assert_allclose(action, [10])
    np.testing.assert_allclose(_first_substep(interpolator, action), [10])


def test_return_home_abort_does_not_blend_from_a_stale_target() -> None:
    follower = FakeRobot([_observation([0], 1)])
    source, mailbox, interpolator = _interpolated_source(follower)
    source.update(_observation([0], 1), {}, 0)
    mailbox.apply(ReturnHomeCommand())
    first = source.update(_observation([10], 2), {}, 1)
    interpolator.add(first)

    # The arm is 2 units (20% of the move) off the 10 command.
    action = source.update(_observation([12], 2.25), {}, 2)

    np.testing.assert_allclose(action, [12])
    np.testing.assert_allclose(_first_substep(interpolator, action), [12])
    assert source.state_data().returning_home is False


def test_set_interpolation_requests_a_new_multiplier_and_publishes_it() -> None:
    follower = FakeRobot([_observation([0, 0], 1)])
    interpolator = AdjustableLinearInterpolator()
    mailbox = InMemoryCommandMailbox()
    events = QueueEventSink()
    source = StudioActionSource(
        follower=follower,
        leader=None,
        mailbox=mailbox,
        event_sink=events,
        fps=30,
        interpolator=interpolator,
    )
    source.connect(bus=object(), session_id="test")

    mailbox.apply(SetInterpolationCommand(multiplier=3))
    source.update(_observation([0, 0], 1), {}, 0)

    assert interpolator.requested_multiplier == 3
    states = [event for event in _drain(events) if event.event == "state"]
    assert states[-1].data.interpolation_multiplier == 3
