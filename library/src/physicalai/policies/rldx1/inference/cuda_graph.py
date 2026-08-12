# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: file-ignore[print]

"""CUDA Graph capture/replay for the full ``GraphSafeVLA`` pipeline (completes Path C).

Captures ``GraphSafeVLA.forward()`` (backbone + action-model denoising loop)
as a single CUDA graph. All data-dependent operations are pre-resolved by
the GraphSafe wrappers, so the entire pipeline is graph-safe.

Note: ``torch.randn`` in the action-model denoising loop is captured with
fixed RNG state -- replayed noise will differ between captures but is
deterministic within a captured graph. Pass an explicit ``init_noise`` for
reproducible replay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from physicalai.policies.rldx1.inference.graph_safe_vla import GraphSafeVLA

    class ReplayFn(Protocol):
        """Callable signature returned by :func:`setup_vla_cuda_graph`."""

        def __call__(
            self,
            vl_input: dict,
            state: torch.Tensor,
            embodiment_id: torch.Tensor,
            init_noise: torch.Tensor | None = None,
            prefix_actions: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Replay the captured graph against new inputs of the same shape."""
            ...


def setup_vla_cuda_graph(
    gs_vla: GraphSafeVLA,
    vl_input: dict,
    state: torch.Tensor,
    embodiment_id: torch.Tensor,
    init_noise: torch.Tensor | None = None,
    prefix_actions: torch.Tensor | None = None,
) -> tuple[ReplayFn, torch.Tensor]:
    """Capture the full VLA forward as a single CUDA graph.

    Args:
        gs_vla: A built ``GraphSafeVLA`` instance.
        vl_input: Sample backbone input dict (fixes the static shapes this
            capture is specialized for).
        state: ``(B, 1, state_dim)`` sample state.
        embodiment_id: ``(B,)`` sample embodiment ids.
        init_noise: ``(B, action_horizon, action_dim)`` or ``None``.
        prefix_actions: ``(B, prefix_len, action_dim)`` for RTC trained
            mode, or ``None`` when ``gs_action_model.prefix_len == 0``.

    Returns:
        ``(replay_fn, static_output)`` -- ``replay_fn(vl_input, state,
        embodiment_id, init_noise=, prefix_actions=)`` replays the captured
        graph against new inputs of the same shape.
    """
    static_vl_input = {}
    for k, v in vl_input.items():
        if isinstance(v, torch.Tensor):
            t = v.clone()
            if k == "pixel_values" and t.ndim == 3:  # ruff: ignore[magic-value-comparison]
                t = t.reshape(-1, t.shape[-1])
            static_vl_input[k] = t
        else:
            static_vl_input[k] = v

    static_state = state.clone()
    static_embodiment_id = embodiment_id.clone()
    static_init_noise = init_noise.clone() if init_noise is not None else None
    static_prefix_actions = prefix_actions.clone() if prefix_actions is not None else None

    # Warmup in a side stream.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream), torch.no_grad():
        gs_vla(
            static_vl_input,
            static_state,
            static_embodiment_id,
            init_noise=static_init_noise,
            prefix_actions=static_prefix_actions,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        static_output = gs_vla(
            static_vl_input,
            static_state,
            static_embodiment_id,
            init_noise=static_init_noise,
            prefix_actions=static_prefix_actions,
        )
    torch.cuda.synchronize()

    print("  CUDA graph captured (full VLA pipeline)")

    # Replay determinism check.
    graph.replay()
    torch.cuda.synchronize()
    r1 = static_output.clone()
    graph.replay()
    torch.cuda.synchronize()
    r2 = static_output.clone()
    print(f"  Replay determinism: max_diff={(r1 - r2).abs().max().item():.6f}")

    def replay_fn(
        vl_input_: dict,
        state_: torch.Tensor,
        embodiment_id_: torch.Tensor,
        init_noise: torch.Tensor | None = None,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for k, v in vl_input_.items():
            if k in static_vl_input and isinstance(v, torch.Tensor):
                t = v
                if k == "pixel_values" and t.ndim == 3:  # ruff: ignore[magic-value-comparison]
                    t = t.reshape(-1, t.shape[-1])
                static_vl_input[k].copy_(t)
        static_state.copy_(state_)
        static_embodiment_id.copy_(embodiment_id_)
        if init_noise is not None and static_init_noise is not None:
            static_init_noise.copy_(init_noise)
        if prefix_actions is not None and static_prefix_actions is not None:
            static_prefix_actions.copy_(prefix_actions)

        graph.replay()
        return static_output.clone()

    return replay_fn, static_output
