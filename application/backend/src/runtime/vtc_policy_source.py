from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicalai.inference.callbacks import Rldx1VtcWindowCallback
from physicalai.runtime import PolicySource

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.capture.frame import Frame
    from physicalai.inference import InferenceModel
    from physicalai.robot.interface import RobotObservation
    from physicalai.runtime.execution.base import Execution
    from physicalai.runtime.execution.queue import ActionQueue


def detach_vtc_window(model: InferenceModel) -> Rldx1VtcWindowCallback | None:
    """Remove the manifest's RLDX-1 VTC callback from ``model`` and return it, if any."""
    windows = [callback for callback in model.callbacks if isinstance(callback, Rldx1VtcWindowCallback)]
    if not windows:
        return None
    model.callbacks = [callback for callback in model.callbacks if not isinstance(callback, Rldx1VtcWindowCallback)]
    return windows[0]


class VtcPolicySource(PolicySource):
    """Record one VTC frame per control tick instead of one per inference call.

    Training spaces the video window in dataset frames, which correspond to
    control ticks. The manifest callback only sees frames when the model runs,
    which stretches the stride by the re-infer interval.
    """

    def __init__(
        self,
        model: InferenceModel,
        execution: Execution | None = None,
        action_queue: ActionQueue | None = None,
        *,
        vtc_window: Rldx1VtcWindowCallback,
        task: str | None = None,
    ) -> None:
        super().__init__(model, execution, action_queue, task=task)
        self._vtc_window = vtc_window

    def connect(self, *, bus: object, session_id: str) -> None:
        with self._state_lock:
            if not self._connected:
                self._vtc_window.on_reset()
            super().connect(bus=bus, session_id=session_id)  # type: ignore[arg-type]

    def reset(self, *, reset_model: bool = True) -> None:
        with self._state_lock:
            try:
                super().reset(reset_model=reset_model)
            finally:
                self._vtc_window.on_reset()

    def _to_model_input(self, robot_obs: RobotObservation, camera_frames: Mapping[str, Frame]) -> dict[str, Any]:
        model_input = super()._to_model_input(robot_obs, camera_frames)
        windowed = self._vtc_window.on_predict_start(model_input)
        return windowed if windowed is not None else model_input
