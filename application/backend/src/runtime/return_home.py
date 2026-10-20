from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

# Same duration and collision threshold as RobotRuntime's return_to_initial_state.
RETURN_HOME_DURATION_S = 2.5
RETURN_HOME_MAX_TRACKING_ERROR = 0.15


class ReturnHomeBlockedError(RuntimeError):
    """A joint stopped following the return path, for example because of a collision."""

    def __init__(self, *, joint_index: int, error: float, measured: np.ndarray) -> None:
        super().__init__(f"Joint {joint_index} is off by {error:.0%} of the move")
        self.joint_index = joint_index
        self.error = error
        self.measured = measured


class ReturnHomeSource:
    """Action source that moves the robot in a straight line from ``start`` to ``target``.

    Each joint moves at a constant velocity chosen so all joints arrive after
    ``duration_s``. Elapsed time comes from observation timestamps, so slow
    control cycles do not stretch the move, and the move pauses while the
    robot stops reporting.

    Every observation is checked against the previous command. If a joint lags
    by more than ``max_tracking_error`` of the largest joint travel, ``update``
    raises :class:`ReturnHomeBlockedError`. The move is :attr:`finished` once an
    observation has tracked the final command.
    """

    def __init__(
        self,
        start: np.ndarray,
        target: np.ndarray,
        duration_s: float = RETURN_HOME_DURATION_S,
        *,
        max_tracking_error: float = RETURN_HOME_MAX_TRACKING_ERROR,
    ) -> None:
        if duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        self._start = np.array(start, dtype=np.float32, copy=True)
        self._target = np.array(target, dtype=np.float32, copy=True)
        self._duration_s = duration_s
        self._velocity = (self._target - self._start) / duration_s
        self._travel = float(np.max(np.abs(self._target - self._start), initial=0.0))
        self._max_tracking_error = max_tracking_error
        self._started_at: float | None = None
        self._last_command: np.ndarray | None = None
        self._arrived = False
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def connect(self, *, bus: object, session_id: str) -> None:
        """Nothing to set up: the move holds no resources."""

    def disconnect(self) -> None:
        """Nothing to tear down: the move holds no resources."""

    def update(
        self,
        robot_state: RobotObservation,
        camera_frames: Mapping[str, Frame],  # noqa: ARG002 — ActionSource protocol
        step: int,  # noqa: ARG002 — ActionSource protocol
    ) -> np.ndarray:
        """Return the command on the return path at this observation's time.

        Raises:
            ReturnHomeBlockedError: If the robot stopped following the previous command.
        """
        if self._travel == 0.0:
            self._finished = True
            return self._target.copy()
        measured = np.asarray(robot_state.joint_positions, dtype=np.float32)
        if self._last_command is not None:
            error = np.abs(measured - self._last_command) / self._travel
            worst = int(np.argmax(error))
            if error[worst] > self._max_tracking_error:
                raise ReturnHomeBlockedError(joint_index=worst, error=float(error[worst]), measured=measured.copy())
            if self._arrived:
                self._finished = True
                return self._target.copy()
        if self._started_at is None:
            self._started_at = robot_state.timestamp
        elapsed = robot_state.timestamp - self._started_at
        if elapsed >= self._duration_s:
            self._arrived = True
            command = self._target.copy()
        else:
            command = (self._start + self._velocity * max(elapsed, 0.0)).astype(np.float32)
        self._last_command = command
        return command.copy()
