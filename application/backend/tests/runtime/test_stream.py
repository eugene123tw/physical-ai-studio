import queue
import threading

import numpy as np
from physicalai.runtime import LifecycleEvent, TickEvent

from runtime.callbacks.stream import StreamCallback
from runtime.contract import ErrorEvent, ObservationEvent, QueueEventSink, StateData, StateEvent

from .fakes import FakeObservation


def test_lifecycle_start_emits_connected_state_once() -> None:
    events = QueueEventSink()
    callback = StreamCallback(event_sink=events, follower_source=lambda: "hold")

    callback.on_lifecycle(
        LifecycleEvent(session_id="test", timestamp=1, event="start", metadata={"joint_names": ["joint"]})
    )

    event = events.get_nowait()
    assert event.model_dump() == {
        "event": "state",
        "data": {
            "connected": True,
            "follower_source": "hold",
            "has_leader": False,
            "model_loaded": None,
            "task": None,
            "dataset_loaded": None,
            "is_recording": None,
            "episodes_recorded": None,
            "returning_home": None,
            "interpolation_multiplier": None,
        },
    }


def test_lifecycle_start_is_suppressed_after_disconnect() -> None:
    ready = threading.Event()
    events = QueueEventSink()
    callback = StreamCallback(
        event_sink=events,
        follower_source=lambda: "hold",
        ready=ready,
        start_allowed=lambda: False,
    )

    callback.on_lifecycle(
        LifecycleEvent(session_id="test", timestamp=1, event="start", metadata={"joint_names": ["joint"]})
    )

    assert not ready.is_set()


def test_tick_emits_exact_observation_shape() -> None:
    events = QueueEventSink()
    callback = StreamCallback(event_sink=events, follower_source=lambda: "hold")
    callback.on_lifecycle(
        LifecycleEvent(session_id="test", timestamp=1, event="start", metadata={"joint_names": ["joint"]})
    )
    events.get_nowait()

    callback.on_tick(
        TickEvent(
            session_id="test",
            step=0,
            timestamp=1,
            robot_state=FakeObservation(np.array([2]), 1),
            camera_frames={},
            action_sent=np.array([2]),
            loop_duration_s=0,
            sleep_time_s=0,
            stale_obs=False,
        )
    )

    assert events.get_nowait().model_dump() == {
        "event": "observation",
        "data": {"joint.pos": 2.0},
        "actions": {"joint.pos": 2.0},
    }


def test_state_and_error_events_keep_the_websocket_contract() -> None:
    assert StateEvent(data=StateData(connected=True, follower_source="teleop")).model_dump() == {
        "event": "state",
        "data": {
            "connected": True,
            "follower_source": "teleop",
            "has_leader": False,
            "model_loaded": None,
            "task": None,
            "dataset_loaded": None,
            "is_recording": None,
            "episodes_recorded": None,
            "returning_home": None,
            "interpolation_multiplier": None,
        },
    }
    assert ErrorEvent(message="lost", error_code="leader_connection_lost").model_dump() == {
        "event": "error",
        "message": "lost",
        "error_code": "leader_connection_lost",
    }


def test_event_sink_coalesces_observations_without_dropping_state() -> None:
    events = QueueEventSink()
    events.emit(ObservationEvent(data={"joint.pos": 1}))
    events.emit(StateEvent(data=StateData(connected=True, follower_source="hold")))
    events.emit(ObservationEvent(data={"joint.pos": 2}))

    assert events.get_nowait().event == "state"
    assert events.get_nowait().model_dump() == {
        "event": "observation",
        "data": {"joint.pos": 2.0},
        "actions": None,
    }


def _shutdown(reason: str = "stop_requested") -> LifecycleEvent:
    return LifecycleEvent(session_id="test", timestamp=1, event="shutdown", metadata={"reason": reason, "steps": 3})


def _drain(events: QueueEventSink) -> list:
    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except queue.Empty:
            return emitted


def test_shutdown_is_published_by_default() -> None:
    events = QueueEventSink()
    callback = StreamCallback(event_sink=events, follower_source=lambda: "hold")

    callback.on_lifecycle(_shutdown())

    (event,) = _drain(events)
    assert event.event == "lifecycle"
    assert event.data.event == "shutdown"
    assert event.data.reason == "stop_requested"


def test_deferred_shutdown_is_dropped_when_the_run_restarts() -> None:
    events = QueueEventSink()
    callback = StreamCallback(event_sink=events, follower_source=lambda: "hold", defer_shutdown=lambda: True)

    callback.on_lifecycle(_shutdown())
    assert _drain(events) == []

    callback.release_deferred_shutdown(publish=False)
    callback.release_deferred_shutdown(publish=True)
    assert _drain(events) == []


def test_deferred_shutdown_is_published_when_the_session_ends_instead() -> None:
    events = QueueEventSink()
    callback = StreamCallback(event_sink=events, follower_source=lambda: "hold", defer_shutdown=lambda: True)

    callback.on_lifecycle(_shutdown())
    callback.release_deferred_shutdown(publish=True)

    (event,) = _drain(events)
    assert event.data.event == "shutdown"
    assert event.data.metadata == {"reason": "stop_requested", "steps": 3}
