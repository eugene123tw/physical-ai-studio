# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: file-ignore[print]

"""Build a GraphSafeVLA from a loaded Physical AI Studio Rldx1 policy.

Bridges Studio's checkpoint-loading path (``Rldx1(pretrained_name_or_path=...)``
or ``Rldx1.load_from_checkpoint(...)``) to the upstream RLWRLD/RLDX-1
graph-safe inference stack, which upstream instead loads via its own
``inference/utils/loader.py`` (an ``AutoModel.from_pretrained`` against a
fixed ``MODEL_REGISTRY`` of RLDX-1's own HF Hub checkpoints). Studio owns a
different checkpoint set (e.g. ``RLWRLD/RLDX-1-FT-LIBERO``) and preprocessing
pipeline (:class:`~physicalai.policies.rldx1.preprocessor.Rldx1Preprocessor`),
so this loader re-derives the same ``backbone``/``action_model`` submodule
split and static shapes from a live ``Rldx1`` instance instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from physicalai.policies.rldx1.inference.action_model.graph_safe_action_model import GraphSafeActionModel
from physicalai.policies.rldx1.inference.backbone.graph_safe_backbone import GraphSafeQwen3VLBackbone
from physicalai.policies.rldx1.inference.graph_safe_vla import GraphSafeVLA

if TYPE_CHECKING:
    from physicalai.data import Observation
    from physicalai.policies.rldx1.policy import Rldx1


# RLDX-1's action head always encodes a single current-state token:
# state shape is (B, 1, state_dim), never a history window.
_N_STATE_TOKENS = 1


def _scalar_int(value: torch.Tensor | int | None, default: int) -> int:
    """Coerce a possibly-batched scalar (tensor or Python int) to a plain int.

    Returns:
        The scalar as a plain Python int, or ``default`` when ``value`` is
        ``None``.
    """
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return int(value.flatten()[0].item())
    return int(value)


def build_graph_safe_vla(
    policy: Rldx1,
    observation: Observation,
    *,
    num_inference_timesteps: int | None = None,
) -> GraphSafeVLA:
    """Build a Path-C ``GraphSafeVLA`` specialized to one representative observation.

    The returned module bakes in static shapes derived from ``observation``
    (image/token counts, cog-token count). Reuse it across steps of the same
    episode/task where those shapes stay constant (e.g. a full LIBERO
    rollout); build a new instance if the shape changes (a different task
    suite, camera count, or prompt-token length).

    Args:
        policy: A loaded :class:`~physicalai.policies.rldx1.policy.Rldx1`
            (e.g. ``Rldx1(pretrained_name_or_path="RLWRLD/RLDX-1-FT-LIBERO")``),
            moved to its target device.
        observation: One representative observation, in the same format
            passed to ``policy.predict_action_chunk`` / ``policy.select_action``.
        num_inference_timesteps: Override the checkpoint's denoising step
            count. Defaults to ``policy.config.num_inference_timesteps``.

    Note:
        This installs the graph-safe vision/text submodules onto
        ``policy.model.backbone`` in place -- its vanilla forward is no
        longer usable on this ``policy`` instance afterward. Run any
        vanilla-path comparison before calling this.

    Returns:
        A ``GraphSafeVLA`` wrapping ``policy.model.backbone`` and
        ``policy.model.action_model`` in eval mode.

    Raises:
        RuntimeError: If ``policy.model`` has not been initialized.
    """
    if policy.model is None:
        msg = "Rldx1.model is not initialized; call trainer.fit() or construct with pretrained_name_or_path=..."
        raise RuntimeError(msg)

    policy.eval()
    device = next(policy.model.parameters()).device
    dtype = policy.model.dtype

    model_input = policy._vtc_buffer.prepare(observation)  # ruff: ignore[private-member-access]
    preprocessed = policy._preprocessor(model_input)  # ruff: ignore[private-member-access]
    preprocessed = {k: v.to(device) if torch.is_tensor(v) else v for k, v in preprocessed.items()}

    backbone_inputs, _action_inputs = policy.model.prepare_input(preprocessed)

    num_frames = _scalar_int(backbone_inputs.get("num_frames"), default=1)
    num_views = _scalar_int(backbone_inputs.get("num_views"), default=1)

    print(f"[GraphSafeVLA] Building backbone (num_frames={num_frames}, num_views={num_views})...")
    gs_backbone = GraphSafeQwen3VLBackbone(
        policy.model.backbone,
        backbone_inputs,
        num_frames=num_frames,
        num_views=num_views,
    ).eval()

    with torch.no_grad():
        vl_out = gs_backbone(backbone_inputs)
    n_vl = vl_out.shape[1]

    config = policy.config
    n_sa_pure = _N_STATE_TOKENS + config.action_horizon

    print(f"[GraphSafeVLA] Building action model (n_vl={n_vl}, n_sa_pure={n_sa_pure})...")
    gs_action_model = GraphSafeActionModel(
        action_model=policy.model.action_model,
        n_vl=n_vl,
        n_sa_pure=n_sa_pure,
        action_horizon=config.action_horizon,
        action_dim=config.max_action_dim,
        num_inference_timesteps=num_inference_timesteps or config.num_inference_timesteps,
        device=device,
        dtype=dtype,
    ).eval()

    return GraphSafeVLA(gs_backbone, gs_action_model).eval()
