import asyncio
import queue
import threading
from typing import Any

import pytest
from physicalai.config import Config
from physicalai.runtime import RobotRuntime

from runtime.contract import DisconnectCommand, QueueEventSink, ReturnHomeCommand
from runtime.session import RuntimeSession
from tests.runtime.fakes import max_concurrent_connects, reset_connect_tracking


def _fake_robot_args(**kwargs: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"positions": [[1.0]], "joint_names": ["joint"]}
    args.update(kwargs)
    return args


def _document(*, connect_error: str | None = None, **follower_kwargs: Any) -> dict:
    robot_args = _fake_robot_args(**follower_kwargs)
    if connect_error is not None:
        robot_args["connect_error"] = connect_error
    return Config(
        "physicalai.runtime.RobotRuntime",
        {
            "robot": {
                "class_path": "tests.runtime.fakes.FakeRobot",
                "init_args": robot_args,
            },
            "cameras": {},
            "fps": 30.0,
        },
    ).to_dict()


def _camera_config(**kwargs: Any) -> dict[str, Any]:
    return {
        "class_path": "tests.runtime.fakes.FakeCamera",
        "init_args": kwargs,
    }


def _document_with_cameras(*names: str, **follower_kwargs: Any) -> dict:
    document = _document(**follower_kwargs)
    document["init_args"]["cameras"] = {name: _camera_config(name=name) for name in names}
    return document


def _document_with_leader(*, leader_kwargs: dict[str, Any] | None = None, **follower_kwargs: Any) -> dict:
    document = _document(**follower_kwargs)
    document["init_args"]["action_source"] = {
        "init_args": {
            "leader": {
                "class_path": "tests.runtime.fakes.FakeRobot",
                "init_args": _fake_robot_args(**(leader_kwargs or {})),
            },
        },
    }
    return document


async def test_session_runs_until_stop_signal() -> None:
    events = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=events)
    stop = threading.Event()
    await session.setup()
    thread = threading.Thread(target=session.run, args=(stop,), daemon=True)
    thread.start()
    await asyncio.to_thread(session.ready.wait, 2)
    stop.set()
    await asyncio.to_thread(thread.join, 2)
    await session.teardown()

    assert session._runtime is not None
    assert session._runtime.last_run_reason == "stop_requested"
    assert session._follower is not None
    assert not session._follower.is_connected()
    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except queue.Empty:
            break
    assert emitted[0].event == "state"
    assert emitted[-1].event == "lifecycle"


async def test_session_propagates_connect_error() -> None:
    session = RuntimeSession(_document(connect_error="connect failed"), event_sink=QueueEventSink())
    await session.setup()
    with pytest.raises(ConnectionError, match="connect failed"):
        session.run(threading.Event())
    await session.teardown()

    assert session._follower is not None
    assert not session._follower.is_connected()


async def test_preconnect_runs_follower_and_leader_connects_in_parallel() -> None:
    reset_connect_tracking()
    session = RuntimeSession(
        _document_with_leader(
            connect_delay=0.05,
            name="follower",
            leader_kwargs={"connect_delay": 0.05, "name": "leader"},
        ),
        event_sink=QueueEventSink(),
    )
    await session.setup()

    session._preconnect_devices()

    assert session._follower is not None
    assert session._leader is not None
    assert session._follower.is_connected()
    assert session._leader.is_connected()
    assert max_concurrent_connects() == 2


async def test_preconnect_disconnects_robots_when_one_connect_fails() -> None:
    session = RuntimeSession(
        _document_with_leader(
            name="follower",
            leader_kwargs={"connect_error": "leader connect failed", "name": "leader"},
        ),
        event_sink=QueueEventSink(),
    )
    await session.setup()

    with pytest.raises(ConnectionError, match="leader connect failed"):
        session._preconnect_devices()

    assert session._follower is not None
    assert session._leader is not None
    assert not session._follower.is_connected()
    assert not session._leader.is_connected()


async def test_disconnect_before_runtime_construction_stops_first_run() -> None:
    session = RuntimeSession(_document(), event_sink=QueueEventSink())
    session.apply(DisconnectCommand())
    await session.setup()

    session.run(threading.Event())

    assert session._runtime is None
    assert session._follower is not None
    assert not session._follower.is_connected()


async def test_worker_stop_before_run_does_not_connect_hardware() -> None:
    session = RuntimeSession(_document_with_leader(), event_sink=QueueEventSink())
    await session.setup()
    stop = threading.Event()
    stop.set()

    session.run(stop)

    assert session._runtime is None
    assert session._follower is not None
    assert session._leader is not None
    assert not session._follower.is_connected()
    assert not session._leader.is_connected()


def _drain(events: QueueEventSink) -> list:
    emitted = []
    while True:
        try:
            emitted.append(events.get_nowait())
        except queue.Empty:
            return emitted


def _record_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], list[RobotRuntime], threading.Event]:
    """Count run() calls, flag the second one, and replace the return move with a recorder."""
    runs: list[dict[str, Any]] = []
    returns: list[RobotRuntime] = []
    restarted = threading.Event()
    original_run = RobotRuntime.run

    def counting_run(self: RobotRuntime, **kwargs: Any) -> int:
        runs.append(kwargs)
        if len(runs) == 2:
            restarted.set()
        return original_run(self, **kwargs)

    monkeypatch.setattr(RobotRuntime, "run", counting_run)
    monkeypatch.setattr(RobotRuntime, "_return_to_initial_state", lambda self: returns.append(self))
    return runs, returns, restarted


async def test_return_home_returns_through_the_runtime_and_keeps_the_session_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs, returns, restarted = _record_runs(monkeypatch)
    events = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=events)
    stop = threading.Event()
    await session.setup()
    thread = threading.Thread(target=session.run, args=(stop,), daemon=True)
    thread.start()
    await asyncio.to_thread(session.ready.wait, 2)

    session.apply(ReturnHomeCommand())
    assert await asyncio.to_thread(restarted.wait, 2)
    stop.set()
    await asyncio.to_thread(thread.join, 2)
    await session.teardown()

    assert len(runs) == 2
    assert len(returns) == 1
    assert session._runtime is not None
    assert session._runtime.last_run_reason == "stop_requested"
    emitted = _drain(events)
    # Only the run that ends the session publishes "shutdown"; clients treat it as the session ending.
    assert [event.data.event for event in emitted if event.event == "lifecycle"] == ["shutdown"]
    assert emitted[-1].event == "lifecycle"
    returning = [event.data.returning_home for event in emitted if event.event == "state"]
    assert True in returning
    assert returning[-1] is False


async def test_disconnect_during_return_home_ends_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    runs, _, _ = _record_runs(monkeypatch)
    events = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=events)
    monkeypatch.setattr(RobotRuntime, "_return_to_initial_state", lambda self: session.apply(DisconnectCommand()))
    await session.setup()
    thread = threading.Thread(target=session.run, args=(threading.Event(),), daemon=True)
    thread.start()
    await asyncio.to_thread(session.ready.wait, 2)

    session.apply(ReturnHomeCommand())
    await asyncio.to_thread(thread.join, 2)
    await session.teardown()

    assert not thread.is_alive()
    assert len(runs) == 1
    emitted = _drain(events)
    assert [event.data.event for event in emitted if event.event == "lifecycle"] == ["shutdown"]


async def test_return_home_before_the_session_is_ready_is_ignored() -> None:
    events = QueueEventSink()
    session = RuntimeSession(_document(), event_sink=events)
    await session.setup()

    session.apply(ReturnHomeCommand())

    assert session._return_home_requested is False
    assert _drain(events) == []
