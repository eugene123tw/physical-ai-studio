# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Faithfulness and planning diagnostics (paper Section 3 / Tables 3-4).

- ``B``: RMS action change when re-sampling the expert's own flow noise.
- ``S_x``: median RMS action change under intervention ``x`` divided by the median ``B``.
  Interventions: endpoint erasure (``q := s``), duration ``+4``, main-camera swap, prompt swap.
- End-of-plan recall / precision and first-waypoint error on held-out planning windows.

RMS is taken over the executed steps ``d`` and the valid action dims, then the median over
segments. All interventions share the same flow noise as the unmodified prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from physicalai.data.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import IMAGES, STATE

from .model import KEY_SEG_D, KEY_SEG_G, KEY_SEG_Q, KEY_WP_CUR_G, KEY_WP_D, KEY_WP_Q, KEY_WP_VALID

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .model import Pi05WaypointModel


def executed_rms(a: torch.Tensor, b: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS of ``a - b`` over the first ``steps[i]`` time steps.

    Returns:
        ``(B,)`` tensor.
    """
    horizon = a.shape[1]
    mask = torch.arange(horizon, device=a.device)[None] < steps.clamp(1, horizon)[:, None]
    sq = ((a - b) ** 2).mean(dim=-1)
    return ((sq * mask).sum(dim=1) / mask.sum(dim=1)).sqrt()


@dataclass
class SensitivityReport:
    """Medians over segments (``S_*`` are ratios to ``B``; ``image_abs`` is absolute)."""

    num_segments: int
    B: float  # noqa: N815
    S_endpoint: float  # noqa: N815
    S_duration: float  # noqa: N815
    S_image: float  # noqa: N815
    S_language: float  # noqa: N815
    image_abs: float

    def as_dict(self) -> dict[str, float]:
        """Plain dict for logging.

        Returns:
            Dict of metrics.
        """
        return dict(self.__dict__)


def _roll(batch: dict[str, Any], key: str, dim: int) -> dict[str, Any]:
    out = dict(batch)
    out[key] = torch.roll(batch[key], shifts=1, dims=dim)
    return out


def _swap_main_camera(batch: dict[str, Any]) -> dict[str, Any]:
    out = dict(batch)
    images = batch[IMAGES].clone()
    images[0] = torch.roll(images[0], shifts=1, dims=0)
    out[IMAGES] = images
    return out


@torch.no_grad()
def sensitivity(  # noqa: PLR0914  # noqa: PLR0914  # noqa: PLR0914
    model: Pi05WaypointModel,
    batches: Iterable[dict[str, Any]],
    *,
    duration_shift: int = 4,
    seed: int = 0,
) -> SensitivityReport:
    """Measure action sensitivity to waypoint / image / language interventions.

    Args:
        model: Trained model in eval mode.
        batches: Preprocessed batches with ``extra.seg_*`` targets (batch size >= 2 for swaps).
        duration_shift: Added to the duration for ``S_duration`` (clamped to ``D``).
        seed: Seed for the two flow-noise draws.

    Returns:
        :class:`SensitivityReport`.
    """
    gen = torch.Generator().manual_seed(seed)
    base_rms, endpoint, duration, image, language = [], [], [], [], []
    horizon = model._chunk_size  # noqa: SLF001
    for batch in batches:
        q, g, d = batch[KEY_SEG_Q], batch[KEY_SEG_G], batch[KEY_SEG_D].long()
        bsize = q.shape[0]
        shape = (bsize, horizon, model._max_action_dim)  # noqa: SLF001
        n1 = torch.randn(shape, generator=gen).to(q.device)
        n2 = torch.randn(shape, generator=gen).to(q.device)

        def run(
            b: dict[str, Any],
            goal: torch.Tensor,
            dur: torch.Tensor,
            noise: torch.Tensor,
            grip: torch.Tensor = g,
        ) -> torch.Tensor:
            return model.predict_segment(b, goal, grip, dur, noise=noise.clone())

        ref = run(batch, q, d, n1)
        base_rms.append(executed_rms(ref, run(batch, q, d, n2), d))
        endpoint.append(executed_rms(ref, run(batch, model._state_cfg(batch[STATE]), d, n1), d))  # noqa: SLF001
        duration.append(executed_rms(ref, run(batch, q, (d + duration_shift).clamp(max=horizon), n1), d))
        if bsize > 1:
            image.append(executed_rms(ref, run(_swap_main_camera(batch), q, d, n1), d))
            swapped = _roll(_roll(batch, TOKENIZED_PROMPT, 0), TOKENIZED_PROMPT_MASK, 0)
            language.append(executed_rms(ref, run(swapped, q, d, n1), d))

    def med(parts: list[torch.Tensor]) -> float:
        return float(torch.cat(parts).median()) if parts else float("nan")

    b = med(base_rms)
    return SensitivityReport(
        num_segments=int(sum(p.numel() for p in base_rms)),
        B=b,
        S_endpoint=med(endpoint) / b,
        S_duration=med(duration) / b,
        S_image=med(image) / b,
        S_language=med(language) / b,
        image_abs=med(image),
    )


@dataclass
class PlanningReport:
    """End-of-plan detection and first-waypoint accuracy on planning windows."""

    num_windows: int
    true_endings: int
    end_recall: float
    end_precision: float
    first_wp_rmse: float
    first_d_mae: float
    mean_passes: float

    def as_dict(self) -> dict[str, float]:
        """Plain dict for logging.

        Returns:
            Dict of metrics.
        """
        return dict(self.__dict__)


@torch.no_grad()
def planning_metrics(model: Pi05WaypointModel, batches: Iterable[dict[str, Any]]) -> PlanningReport:  # noqa: PLR0914  # noqa: PLR0914  # noqa: PLR0914
    """Decode plans for held-out windows and compare with ground-truth waypoints.

    A window has a *true ending* when its target plan contains the terminal ``d = 0`` block;
    a *hit* is a decoded plan that also contains an end marker.

    Returns:
        :class:`PlanningReport` (first-waypoint RMSE is in normalized configuration units).
    """
    windows = hits = false_alarms = true_end = 0
    sq_err, d_err, passes = [], [], []
    for batch in batches:
        state = batch[STATE]
        bsize = state.shape[0]
        prefix_pad, cache = model.prefix_cache(batch)
        cur_g = batch.get(KEY_WP_CUR_G)
        cur_g = torch.zeros(bsize, model.wp_tokenizer.num_arms, dtype=torch.long) if cur_g is None else cur_g
        plan = model.decode_plan(prefix_pad, cache, state, cur_g.long().to(state.device))
        gt_end = (batch[KEY_WP_VALID].bool() & (batch[KEY_WP_D] == 0)).any(dim=1)
        pred_end = (plan.valid & (plan.d == 0)).any(dim=1)
        windows += bsize
        true_end += int(gt_end.sum())
        hits += int((gt_end & pred_end).sum())
        false_alarms += int((~gt_end & pred_end).sum())
        sq_err.append(((plan.q[:, 0] - batch[KEY_WP_Q][:, 0].float()) ** 2).mean(dim=-1))
        d_err.append((plan.d[:, 0] - batch[KEY_WP_D][:, 0]).abs().float())
        passes.append(plan.num_passes)
    predicted = hits + false_alarms
    return PlanningReport(
        num_windows=windows,
        true_endings=true_end,
        end_recall=hits / true_end if true_end else float("nan"),
        end_precision=hits / predicted if predicted else float("nan"),
        first_wp_rmse=float(torch.cat(sq_err).mean().sqrt()) if sq_err else float("nan"),
        first_d_mae=float(torch.cat(d_err).mean()) if d_err else float("nan"),
        mean_passes=sum(passes) / len(passes) if passes else float("nan"),
    )
