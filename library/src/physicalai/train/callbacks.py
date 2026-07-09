# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Callbacks for training."""

import time
from typing import Any, cast

import lightning as L  # noqa: N812
import torch
from lightning.pytorch.callbacks import Callback

from physicalai.train.utils import reformat_dataset_to_match_policy


class IterationTimer(Callback):
    """Log wall-clock time per training step in seconds.

    Logs ``train/iter_time_s`` on every training batch end.

    Example:
        >>> from physicalai.train.callbacks import IterationTimer
        >>> trainer = Trainer(callbacks=[IterationTimer()])
    """

    def on_train_batch_start(
        self,
        _trainer: L.Trainer,
        _pl_module: L.LightningModule,
        _batch: object,
        _batch_idx: int,
    ) -> None:
        """Record the batch start time."""
        self._start = time.perf_counter()

    def on_train_batch_end(
        self,
        _trainer: L.Trainer,
        pl_module: L.LightningModule,
        _outputs: object,
        _batch: object,
        _batch_idx: int,
    ) -> None:
        """Log elapsed time since batch start."""
        elapsed_s = time.perf_counter() - self._start
        pl_module.log("train/iter_time_s", elapsed_s, prog_bar=True)


class PolicyDatasetInteraction(Callback):
    """Callback to interact the policy and dataset before training starts."""

    @staticmethod
    def _interact_policy_dataset(trainer: L.Trainer, model: L.LightningModule) -> None:
        # Assumes trainer has a datamodule attached
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            reformat_dataset_to_match_policy(policy=model, datamodule=trainer.datamodule)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called at the start of `trainer.fit()`."""
        self._interact_policy_dataset(trainer, pl_module)


class XPUMemoryUtilization(Callback):
    """Log Intel XPU memory stats during training.

    Logs ``xpu/memory_allocated_mb``, ``xpu/memory_reserved_mb``, percentage
    counterparts, and peak values per process/rank. This mirrors the CUDA memory
    callback idea but uses ``torch.xpu`` APIs when available.
    """

    def __init__(self, *, log_every_n_steps: int = 20, reset_peak_stats_each_log: bool = True) -> None:
        """Initialize XPU memory logger.

        Args:
            log_every_n_steps: Log cadence in training steps.
            reset_peak_stats_each_log: Reset peak counters after each log event.
        """
        if log_every_n_steps < 1:
            msg = "log_every_n_steps must be >= 1"
            raise ValueError(msg)

        self.log_every_n_steps = log_every_n_steps
        self.reset_peak_stats_each_log = reset_peak_stats_each_log

    @staticmethod
    def _bytes_to_mb(value: int | float) -> float:
        return float(value) / (1024.0 * 1024.0)

    @staticmethod
    def _fraction_to_percent(value: int | float, total: int | float) -> float:
        if float(total) <= 0.0:
            return 0.0
        return (float(value) / float(total)) * 100.0

    @staticmethod
    def _get_xpu_metrics(device: torch.device) -> dict[str, float] | None:
        """Collect XPU memory metrics if torch.xpu runtime is available."""
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            return None

        # Some torch/xpu builds do not expose all CUDA-like memory APIs.
        allocated: int | float = 0
        reserved: int | float = 0
        max_allocated: int | float = 0
        max_reserved: int | float = 0

        total_memory: int | float = 0

        if hasattr(torch.xpu, "memory_allocated"):
            allocated = cast(float, torch.xpu.memory_allocated(device))
        if hasattr(torch.xpu, "memory_reserved"):
            reserved = cast(float, torch.xpu.memory_reserved(device))
        if hasattr(torch.xpu, "max_memory_allocated"):
            max_allocated = cast(float, torch.xpu.max_memory_allocated(device))
        if hasattr(torch.xpu, "max_memory_reserved"):
            max_reserved = cast(float, torch.xpu.max_memory_reserved(device))
        if hasattr(torch.xpu, "get_device_properties"):
            props = torch.xpu.get_device_properties(device)
            if hasattr(props, "total_memory"):
                total_memory = cast(float, props.total_memory)

        return {
            "xpu/memory_allocated_mb": XPUMemoryUtilization._bytes_to_mb(allocated),
            "xpu/memory_reserved_mb": XPUMemoryUtilization._bytes_to_mb(reserved),
            "xpu/max_memory_allocated_mb": XPUMemoryUtilization._bytes_to_mb(max_allocated),
            "xpu/max_memory_reserved_mb": XPUMemoryUtilization._bytes_to_mb(max_reserved),
            "xpu/memory_allocated_pct": XPUMemoryUtilization._fraction_to_percent(
                allocated,
                total_memory,
            ),
            "xpu/memory_reserved_pct": XPUMemoryUtilization._fraction_to_percent(
                reserved,
                total_memory,
            ),
            "xpu/max_memory_allocated_pct": XPUMemoryUtilization._fraction_to_percent(
                max_allocated,
                total_memory,
            ),
            "xpu/max_memory_reserved_pct": XPUMemoryUtilization._fraction_to_percent(
                max_reserved,
                total_memory,
            ),
        }

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        _outputs: Any,
        _batch: Any,
        batch_idx: int,
    ) -> None:
        """Log XPU memory usage every ``log_every_n_steps``."""
        if not trainer.training:
            return
        if (batch_idx + 1) % self.log_every_n_steps != 0:
            return
        if pl_module.device.type != "xpu":
            return

        metrics = self._get_xpu_metrics(pl_module.device)
        if metrics is None:
            return

        for key, value in metrics.items():
            pl_module.log(key, value, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

        if self.reset_peak_stats_each_log and hasattr(torch.xpu, "reset_peak_memory_stats"):
            torch.xpu.reset_peak_memory_stats(pl_module.device)
