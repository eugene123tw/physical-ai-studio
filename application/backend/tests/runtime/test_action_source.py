import queue

import numpy as np
import pytest

from runtime.action_source import StudioActionSource
from runtime.contract import InMemoryCommandMailbox, QueueEventSink, SetFollowerSourceCommand, SetInterpolationCommand
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


def test_state_reports_the_return_home_flag_owned_by_the_session() -> None:
    returning = {"value": False}
    source = StudioActionSource(
        follower=FakeRobot([_observation([0, 0], 1)]),
        leader=None,
        mailbox=InMemoryCommandMailbox(),
        event_sink=QueueEventSink(),
        fps=30,
        returning_home=lambda: returning["value"],
    )

    assert source.state_data().returning_home is False
    returning["value"] = True
    assert source.state_data().returning_home is True


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
