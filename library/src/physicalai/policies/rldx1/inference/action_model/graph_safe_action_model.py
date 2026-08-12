# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)
# ruff: file-ignore[print]

"""Graph-safe wrapper for the full RLDX-1 action-head pipeline.

Composes :class:`~physicalai.policies.rldx1.inference.action_model.graph_safe_msat.GraphSafeMSAT`
with the state/action encoders/decoders and runs the denoising loop.
Pre-determines data-dependent variables (position ids, timestep schedule,
``dt``) in ``__init__``.

Data-dependent operations replaced:
  - ``torch.arange(action_horizon)`` per forward -> ``static_pos_ids`` buffer
  - timestep schedule computation -> ``static_timesteps`` buffer
  - ``dt = 1/N`` -> Python float

Physics support (RLDX-1 midtrain add-ons): when
``action_model.use_physics`` is ``True``, the denoising loop is extended
with physics conditioning (history) and flow-matching (future) streams.
Physical AI Studio's LIBERO checkpoint does not use physics, so this path
is inert (``use_physics=False``) for that checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from physicalai.policies.rldx1.inference.action_model.graph_safe_msat import GraphSafeMSAT


class GraphSafeActionModel(nn.Module):
    """Graph-safe full action-head pipeline.

    Without physics:
      ``vlln + state_enc(1x) + N_steps x [action_enc + MSAT + action_dec + Euler]``

    With physics:
      ``vlln + state_enc(1x) + physics_hist_enc(1x) + init physics_fut noise
      + N_steps x [action_enc + physics_fut_enc + MSAT(+physics) + action_dec
      + Euler + physics_dec + physics_Euler]``
    """

    def __init__(  # ruff: ignore[too-many-arguments, too-many-locals, too-many-statements]
        self,
        action_model: nn.Module | None = None,
        *,
        msat: nn.Module | None = None,
        state_encoder: nn.Module | None = None,
        action_encoder: nn.Module | None = None,
        action_decoder: nn.Module | None = None,
        position_embedding: nn.Module | None = None,
        vlln: nn.Module | None = None,
        n_vl: int,
        n_sa_pure: int,
        action_horizon: int,
        action_dim: int,
        num_inference_timesteps: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        physics_cond_encoder: nn.Module | None = None,
        physics_fut_encoder: nn.Module | None = None,
        physics_decoder: nn.Module | None = None,
        physics_hist_len: int = 0,
        physics_fut_len: int = 0,
        physics_dim: int = 0,
        prefix_len: int = 0,
    ) -> None:
        """Build the graph-safe action-head pipeline.

        Args:
            action_model: The vanilla action model (e.g.
                :class:`~physicalai.policies.rldx1.model.RLDXActionModel`).
                When provided, sub-modules are extracted automatically;
                otherwise pass them individually via the keyword-only args
                (standalone benchmark use).
            msat: MSAT diffusion model (``action_model.model``).
            state_encoder: State encoder.
            action_encoder: Action encoder.
            action_decoder: Action decoder.
            position_embedding: Action-horizon position embedding.
            vlln: Backbone output normalization (``nn.Identity()`` if none).
            n_vl: Number of VL (backbone) tokens.
            n_sa_pure: Number of pure SA tokens (state + action, excluding
                the time token).
            action_horizon: Predicted action-chunk length.
            action_dim: Action dimensionality.
            num_inference_timesteps: Number of Euler denoising steps.
            device: Target device.
            dtype: Target dtype.
            physics_cond_encoder: Physics history encoder (physics add-on).
            physics_fut_encoder: Physics future (flow-matching) encoder.
            physics_decoder: Physics decoder.
            physics_hist_len: Physics history token count.
            physics_fut_len: Physics future token count.
            physics_dim: Physics feature dimensionality.
            prefix_len: RTC trained-mode frozen-prefix length (0 = disabled).
        """
        super().__init__()

        if action_model is not None:
            msat = action_model.model
            vlln = action_model.vlln
            state_encoder = action_model.state_encoder
            action_encoder = action_model.action_encoder
            action_decoder = action_model.action_decoder
            position_embedding = action_model.position_embedding

        use_physics = False
        if action_model is not None and getattr(action_model, "use_physics", False):
            use_physics = True
            physics = action_model.physics
            if physics_cond_encoder is None:
                physics_cond_encoder = physics.physics_cond_encoder
            if physics_fut_encoder is None:
                physics_fut_encoder = physics.physics_fut_encoder
            if physics_decoder is None:
                physics_decoder = physics.physics_decoder
            if physics_hist_len == 0:
                physics_hist_len = physics.physics_hist_len
            if physics_fut_len == 0:
                physics_fut_len = physics.physics_fut_len
            if physics_dim == 0:
                physics_dim = physics.physics_dim
        elif physics_cond_encoder is not None:
            use_physics = True

        self.use_physics = use_physics
        n_physics = physics_hist_len + physics_fut_len if use_physics else 0

        self.gs_msat = GraphSafeMSAT(msat, n_vl, n_sa_pure, device, n_physics=n_physics)
        # Cache inner_dim for physics zero-tensor creation (avoids a
        # ``gs_msat._msat`` access at forward time, which breaks once
        # ``gs_msat`` is swapped out for a CUDA-Graph/compiled variant).
        self._inner_dim = msat.inner_dim

        self.vlln = vlln if vlln is not None else nn.Identity()

        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.position_embedding = position_embedding

        if use_physics:
            self.physics_cond_encoder = physics_cond_encoder
            self.physics_fut_encoder = physics_fut_encoder
            self.physics_decoder = physics_decoder
            self.physics_hist_len = physics_hist_len
            self.physics_fut_len = physics_fut_len
            self.physics_dim = physics_dim

        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.num_inference_timesteps = num_inference_timesteps
        self.dt = 1.0 / num_inference_timesteps
        self.prefix_len = int(prefix_len)

        pos_ids = torch.arange(action_horizon, dtype=torch.long, device=device)
        self.register_buffer("static_pos_ids", pos_ids)

        timesteps = torch.tensor(
            [t / float(num_inference_timesteps) for t in range(num_inference_timesteps)],
            dtype=dtype,
            device=device,
        )
        self.register_buffer("static_timesteps", timesteps)

        print(
            f"  [GraphSafeActionModel] pos_ids={list(pos_ids.shape)}, "
            f"timesteps={list(timesteps.shape)}, "
            f"action_horizon={action_horizon}, "
            f"num_steps={num_inference_timesteps}, dt={self.dt}",
        )
        if use_physics:
            print(
                f"  [GraphSafeActionModel] physics: dim={physics_dim}, "
                f"hist_len={physics_hist_len}, fut_len={physics_fut_len}, "
                f"n_physics={n_physics}",
            )
        if self.prefix_len > 0:
            print(f"  [GraphSafeActionModel] RTC trained: prefix_len={self.prefix_len}")

    def forward(  # ruff: ignore[too-many-branches, too-many-locals]
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        init_noise: torch.Tensor | None = None,
        physics_hist: torch.Tensor | None = None,
        physics_init_noise: torch.Tensor | None = None,
        prefix_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Graph-safe action-head forward.

        Args:
            vl_embs: ``(B, n_vl, vl_dim)`` VL embeddings from the backbone.
            state: ``(B, 1, state_dim)`` current-state observation.
            embodiment_id: ``(B,)`` embodiment ids.
            init_noise: ``(B, action_horizon, action_dim)`` action noise, or
                ``None`` to sample fresh.
            physics_hist: ``(B, hist_len, physics_dim)`` physics history, or
                ``None``.
            physics_init_noise: ``(B, fut_len, physics_dim)`` physics noise,
                or ``None``.
            prefix_actions: ``(B, prefix_len, action_dim)`` frozen actions
                from the previous chunk (RTC trained mode). Required when
                ``prefix_len > 0``; ignored otherwise.

        Returns:
            ``(B, action_horizon, action_dim)`` predicted actions.
        """
        vl_embs = self.vlln(vl_embs)

        b = vl_embs.shape[0]
        d = self.prefix_len

        state_features = self.state_encoder(state, embodiment_id)

        pos_embs = self.position_embedding(self.static_pos_ids).unsqueeze(0)

        if init_noise is not None:
            current_state = init_noise.clone() if d > 0 else init_noise
        else:
            current_state = torch.randn(
                (b, self.action_horizon, self.action_dim),
                dtype=vl_embs.dtype,
                device=vl_embs.device,
            )
        if d > 0:
            current_state[:, :d] = prefix_actions

        physics_hist_tok = None
        physics_fut = None
        if self.use_physics:
            if physics_hist is not None and self.physics_hist_len > 0:
                physics_hist_tok = self.physics_cond_encoder(physics_hist)
            else:
                physics_hist_tok = torch.zeros(b, 0, self._inner_dim, dtype=vl_embs.dtype, device=vl_embs.device)

            if physics_init_noise is not None:
                physics_fut = physics_init_noise
            else:
                physics_fut = torch.randn(
                    b,
                    self.physics_fut_len,
                    self.physics_dim,
                    dtype=vl_embs.dtype,
                    device=vl_embs.device,
                )

        for t in range(self.num_inference_timesteps):
            t_scalar = self.static_timesteps[t].expand(b)

            t_tok = t_scalar.unsqueeze(1).expand(-1, self.action_horizon).clone()
            if d > 0:
                t_tok[:, :d] = 1.0
                current_state[:, :d] = prefix_actions

            action_features = self.action_encoder(current_state, t_tok, embodiment_id)
            action_features += pos_embs

            sa_embs = torch.cat([state_features, action_features], dim=1)

            physics_embs = None
            if physics_hist_tok is not None and physics_fut is not None:
                physics_fut_tok = self.physics_fut_encoder(physics_fut, t_scalar)
                physics_embs = torch.cat([physics_hist_tok, physics_fut_tok], dim=1)

            model_output = self.gs_msat(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=t_scalar,
                physics_embs=physics_embs,
            )

            action_output = model_output["action"] if isinstance(model_output, dict) else model_output

            pred = self.action_decoder(action_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            current_state += self.dt * pred_velocity
            if d > 0:
                current_state[:, :d] = prefix_actions

            if physics_fut is not None and isinstance(model_output, dict) and "physics" in model_output:
                physics_hidden_fut = model_output["physics"][:, -self.physics_fut_len :]
                physics_pred_vel = self.physics_decoder(physics_hidden_fut)
                physics_fut += self.dt * physics_pred_vel

        return current_state

    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to :attr:`gs_msat`.

        Returns:
            The resolved attribute from :attr:`gs_msat`.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.gs_msat, name)
