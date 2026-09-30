# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pi0.5 waypoint hierarchy: VLM waypoint planner + flow-matching action expert.

Implements "Fast Plans, Faithful Actions" (Xie et al., 2026) on top of :class:`Pi05Model`:

- **Planner** (PaliGemma backbone) decodes up to ``M`` waypoint blocks ``(q, g, d)`` after the
  image/prompt prefix, either token-by-token (``token_ar``) or one block per forward pass
  with bidirectional attention inside a block (``block_ar``).
- **Executor** (Gemma action expert) denoises a ``D``-step action chunk conditioned on three
  suffix tokens (start state, goal, duration) and, optionally, on the goal through every
  AdaRMS layer (``ngm``: gated normalized goal modulation, ``naive``: ungated).

Training runs both in one forward pass: ``[prefix][plan tokens][expert suffix]`` where the
expert attends to the prefix only, and optimizes ``CE_plan + lambda * FM``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, cast, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, EXTRA, IMAGES, STATE
from physicalai.data.waypoints.dataset import SEG_D, SEG_G, SEG_Q, WP_CUR_G, WP_D, WP_G, WP_Q, WP_VALID
from physicalai.data.waypoints.extract import WAYPOINT_STATS_KEY
from physicalai.policies.pi05.model import (  # noqa: PLC2701
    Pi05Model,
    _clone_kv_cache,
    _create_sinusoidal_pos_embedding,
    _make_att_2d_masks,
)
from physicalai.policies.utils import in_episode_bound, reduce_losses

from .ngm import GoalModulation
from .tokenizer import SlotKind, WaypointTokenizer

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)

PlannerMode = Literal["token_ar", "block_ar"]
GoalConditioning = Literal["suffix", "naive", "ngm"]

KEY_WP_Q = f"{EXTRA}.{WP_Q}"
KEY_WP_G = f"{EXTRA}.{WP_G}"
KEY_WP_D = f"{EXTRA}.{WP_D}"
KEY_WP_VALID = f"{EXTRA}.{WP_VALID}"
KEY_WP_CUR_G = f"{EXTRA}.{WP_CUR_G}"
KEY_SEG_Q = f"{EXTRA}.{SEG_Q}"
KEY_SEG_G = f"{EXTRA}.{SEG_G}"
KEY_SEG_D = f"{EXTRA}.{SEG_D}"

_NUM_COND_TOKENS = 3
_NEG_INF = -1e9


@dataclass
class ExpertCondition:
    """Inputs that condition one action segment.

    Attributes:
        state: ``(B, max_state_dim)`` normalized, zero-padded start state.
        goal_q: ``(B, num_cfg)`` normalized goal configuration (possibly noised).
        goal_g: ``(B, num_arms)`` goal gripper command.
        duration: ``(B,)`` duration code.
        delta_hat: ``(B, num_cfg)`` ``(goal_q - s) / sigma_delta``.
        drop: Optional ``(B,)`` bool; replace the goal on both routes by null embeddings.
    """

    state: Tensor
    goal_q: Tensor
    goal_g: Tensor
    duration: Tensor
    delta_hat: Tensor
    drop: Tensor | None = None


@dataclass
class WaypointPlan:
    """Decoded plan.

    Attributes:
        q: ``(B, M, num_cfg)`` normalized waypoint configurations.
        g: ``(B, M, num_arms)`` gripper commands.
        d: ``(B, M)`` durations (``0`` = end of plan).
        valid: ``(B, M)`` blocks actually decoded (terminal block included).
        num_passes: Serial backbone passes including the prefix prefill.
    """

    q: Tensor
    g: Tensor
    d: Tensor
    valid: Tensor
    num_passes: int

    def executable(self, b: int = 0) -> list[tuple[Tensor, Tensor, int]]:
        """Waypoints before the end marker for batch row ``b``.

        Returns:
            List of ``(q, g, d)`` with ``d >= 1``.
        """
        out = []
        for k in range(self.q.shape[1]):
            if not bool(self.valid[b, k]) or int(self.d[b, k]) == 0:
                break
            out.append((self.q[b, k], self.g[b, k], int(self.d[b, k])))
        return out


class Pi05WaypointModel(Pi05Model):
    """Waypoint planner + goal-conditioned flow-matching expert.

    Args:
        dataset_stats: Dataset stats; must contain the ``"waypoint"`` entry produced by
            :func:`physicalai.data.waypoints.waypoint_stats`.
        planner_mode: ``"token_ar"`` or ``"block_ar"``.
        goal_conditioning: ``"suffix"`` (suffix tokens only), ``"naive"`` (plus ungated deep
            route), or ``"ngm"`` (plus gated deep route with condition noise / dropout).
        max_blocks: Maximum plan blocks ``M``.
        num_bins: Value bins for configuration tokens.
        max_state_dim: Padded state size fed to the state token.
        planner_loss_weight: Weight of the planner cross-entropy.
        action_loss_weight: Weight ``lambda`` of the flow-matching loss.
        goal_noise_std: Training noise on ``delta_hat`` (default: 0.7 for ``ngm``, else 0).
        goal_dropout: Training null-goal probability (default: 0.15 for ``ngm``, else 0).
        gate_lo: NGM gate closed at or below this flow time.
        gate_hi: NGM gate open at or above this flow time.
        **kwargs: Forwarded to :class:`Pi05Model` (``chunk_size`` is the expert horizon ``D``).

    Raises:
        ValueError: If ``dataset_stats`` lacks the ``"waypoint"`` entry.
    """

    def __init__(  # noqa: PLR0913
        self,
        dataset_stats: dict[str, Any],
        *,
        planner_mode: PlannerMode = "block_ar",
        goal_conditioning: GoalConditioning = "ngm",
        max_blocks: int = 7,
        num_bins: int = 300,
        max_state_dim: int = 32,
        planner_loss_weight: float = 1.0,
        action_loss_weight: float = 1.7,
        goal_noise_std: float | None = None,
        goal_dropout: float | None = None,
        gate_lo: float = 0.2,
        gate_hi: float = 0.5,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Build the Pi0.5 backbone and the waypoint heads."""
        if WAYPOINT_STATS_KEY not in dataset_stats:
            msg = f"dataset_stats must contain '{WAYPOINT_STATS_KEY}' (use WaypointLeRobotDataModule)"
            raise ValueError(msg)
        kwargs.setdefault("chunk_size", 32)
        kwargs["snapflow_enabled"] = False
        super().__init__(dataset_stats, **kwargs)

        wp_stats = dataset_stats[WAYPOINT_STATS_KEY]
        config_dims = [int(x) for x in wp_stats["config_dims"]]
        num_cfg = len(config_dims)
        num_arms = int(wp_stats["num_arms"])

        self.planner_mode: PlannerMode = planner_mode
        self.goal_conditioning: GoalConditioning = goal_conditioning
        self.max_blocks = max_blocks
        self.max_state_dim = max_state_dim
        self.planner_loss_weight = planner_loss_weight
        self.action_loss_weight = action_loss_weight
        is_ngm = goal_conditioning == "ngm"
        self.goal_noise_std = (0.7 if is_ngm else 0.0) if goal_noise_std is None else goal_noise_std
        self.goal_dropout = (0.15 if is_ngm else 0.0) if goal_dropout is None else goal_dropout

        self.wp_tokenizer = WaypointTokenizer(
            num_cfg=num_cfg,
            num_arms=num_arms,
            num_bins=num_bins,
            max_duration=self._chunk_size,
        )
        tok = self.wp_tokenizer
        self.register_buffer("config_dims", torch.tensor(config_dims, dtype=torch.long), persistent=False)
        self.register_buffer(
            "sigma_delta",
            torch.tensor(wp_stats["sigma_delta"], dtype=torch.float32).reshape(num_cfg),
            persistent=False,
        )
        self.register_buffer("value_positions", tok.value_positions(), persistent=False)
        self.register_buffer("family_mask", tok.value_family_mask(), persistent=False)

        expert_width = self.action_in_proj.out_features
        vlm_width = self._lm.config.hidden_size

        self.state_in_proj = nn.Linear(max_state_dim, expert_width)
        self.goal_in_proj = nn.Linear(num_cfg, expert_width)
        self.goal_gripper_emb = nn.Embedding(2, expert_width)
        self.null_goal = nn.Parameter(torch.zeros(expert_width))
        self.dur_emb = nn.Embedding(tok.num_duration_codes, expert_width)
        self.block_query = nn.Parameter(torch.zeros(tok.block_len, vlm_width)) if planner_mode == "block_ar" else None
        self.goal_route = (
            GoalModulation(
                num_cfg,
                num_arms,
                expert_width,
                gated=goal_conditioning == "ngm",
                gate_lo=gate_lo,
                gate_hi=gate_hi,
            )
            if goal_conditioning in {"ngm", "naive"}
            else None
        )

    # ------------------------------------------------------------------ helpers

    def waypoint_head_modules(self) -> list[nn.Module | nn.Parameter]:
        """New (non-Pi0.5) modules that must stay fully trainable under LoRA.

        Returns:
            Modules and parameters owned by the waypoint hierarchy.
        """
        items: list[nn.Module | nn.Parameter] = [
            self.state_in_proj,
            self.goal_in_proj,
            self.goal_gripper_emb,
            self.null_goal,
            self.dur_emb,
        ]
        if self.block_query is not None:
            items.append(self.block_query)
        if self.goal_route is not None:
            items.append(self.goal_route)
        return items

    def enable_waypoint_head_training(self) -> None:
        """Re-enable gradients on the waypoint heads (``inject_lora`` freezes everything)."""
        for item in self.waypoint_head_modules():
            params = [item] if isinstance(item, nn.Parameter) else list(item.parameters())
            for p in params:
                p.requires_grad = True

    @property
    def _lm(self) -> Any:  # noqa: ANN401
        return self.paligemma_with_expert.paligemma.model.language_model

    def _joint_forward(self, **kwargs: Any) -> tuple[list[Tensor | None], DynamicCache | None]:  # noqa: ANN401
        # PaliGemmaWithExpertModel.forward is annotated with list[FloatTensor]; route calls through Any.
        return cast("Any", self.paligemma_with_expert).forward(**kwargs)

    def _reserved_embeddings(self) -> Tensor:
        tok = self.wp_tokenizer
        weight = self._lm.get_input_embeddings().weight
        return weight[tok.base_id : tok.base_id + tok.num_reserved]

    def _state_cfg(self, state: Tensor) -> Tensor:
        if state.ndim == 3:  # noqa: PLR2004
            state = state[:, -1]
        return state[:, self.config_dims].float()

    def _padded_state(self, state: Tensor) -> Tensor:
        if state.ndim == 3:  # noqa: PLR2004
            state = state[:, -1]
        state = state.float()
        if state.shape[-1] >= self.max_state_dim:
            return state[:, : self.max_state_dim]
        return F.pad(state, (0, self.max_state_dim - state.shape[-1]))

    def _state_block_ids(self, s_cfg: Tensor, cur_g: Tensor) -> Tensor:
        tok = self.wp_tokenizer
        code = torch.full((s_cfg.shape[0],), tok.state_code, dtype=torch.long, device=s_cfg.device)
        return tok.encode(s_cfg, cur_g, code)

    def _current_gripper(self, batch: dict[str, Any], bsize: int, device: torch.device) -> Tensor:
        cur = batch.get(KEY_WP_CUR_G)
        if cur is None:
            return torch.zeros(bsize, self.wp_tokenizer.num_arms, dtype=torch.long, device=device)
        return cur.long().reshape(bsize, self.wp_tokenizer.num_arms).to(device)

    def _plan_attention(self, num_blocks: int, bsize: int, device: torch.device) -> Tensor:
        n = num_blocks * self.wp_tokenizer.block_len
        if self.planner_mode == "token_ar":
            att = torch.ones(n, dtype=torch.long, device=device)
        else:
            att = torch.zeros(n, dtype=torch.long, device=device)
            att[:: self.wp_tokenizer.block_len] = 1
        return att[None].expand(bsize, n)

    def _embed_plan(self, ids: Tensor) -> Tensor:
        embs = self.paligemma_with_expert.embed_language_tokens(ids)
        if self.block_query is not None:
            n = self.wp_tokenizer.block_len
            first = embs[:, :n] + self.block_query.to(embs.dtype)
            embs = torch.cat([first, embs[:, n:]], dim=1)
        return embs

    def _reserved_logits(self, hidden: Tensor) -> Tensor:
        """``hidden (..., V, W)`` at value slots -> family-masked logits ``(..., V, R)``.

        Returns:
            Logits over the reserved vocabulary.
        """
        logits = hidden.float() @ self._reserved_embeddings().float().T
        return logits.masked_fill(~self.family_mask, _NEG_INF)

    def expert_condition(  # noqa: PLR0913
        self,
        state: Tensor,
        goal_q: Tensor,
        goal_g: Tensor,
        duration: Tensor,
        *,
        perturb: bool = False,
    ) -> ExpertCondition:
        """Build the expert condition; ``perturb`` applies training noise and null dropout.

        Returns:
            The :class:`ExpertCondition`.
        """
        s_cfg = self._state_cfg(state)
        goal_q = goal_q.float()
        delta_hat = (goal_q - s_cfg) / self.sigma_delta
        drop = None
        if perturb and self.goal_noise_std > 0:
            delta_hat = delta_hat + self.goal_noise_std * torch.randn_like(delta_hat)  # noqa: PLR6104  # noqa: PLR6104  # noqa: PLR6104
            goal_q = s_cfg + delta_hat * self.sigma_delta
        if perturb and self.goal_dropout > 0:
            drop = torch.rand(goal_q.shape[0], device=goal_q.device) < self.goal_dropout
        return ExpertCondition(
            state=self._padded_state(state),
            goal_q=goal_q,
            goal_g=goal_g.long(),
            duration=duration.long().reshape(-1),
            delta_hat=delta_hat,
            drop=drop,
        )

    # ------------------------------------------------------------------ expert

    def embed_expert_suffix(
        self,
        noisy_actions: Tensor,
        timestep: Tensor,
        cond: ExpertCondition,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed ``[state, goal, duration, actions]`` and the AdaRMS condition.

        Returns:
            Tuple of (embeddings, padding masks, attention masks, adarms condition).
        """
        time_emb = _create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self._min_period,
            max_period=self._max_period,
            device=timestep.device,
        ).type(dtype=timestep.dtype)

        def time_mlp_func(emb: Tensor) -> Tensor:
            return F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(emb))))

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

        state_tok = self.state_in_proj(cond.state)
        goal_tok = self.goal_in_proj(cond.goal_q) + self.goal_gripper_emb(cond.goal_g).sum(dim=1)
        if cond.drop is not None:
            goal_tok = torch.where(cond.drop[:, None], self.null_goal.to(goal_tok.dtype), goal_tok)
        dur_tok = self.dur_emb(cond.duration.clamp(0, self.wp_tokenizer.state_code))
        cond_toks = torch.stack([state_tok, goal_tok, dur_tok], dim=1).to(action_emb.dtype)
        embs = torch.cat([cond_toks, action_emb], dim=1)

        adarms_cond = time_emb
        if self.goal_route is not None:
            adarms_cond = time_emb + self.goal_route(cond.delta_hat, cond.goal_g, timestep, cond.drop).to(
                time_emb.dtype,
            )

        bsize, seq = embs.shape[:2]
        pad_masks = torch.ones(bsize, seq, dtype=torch.bool, device=embs.device)
        att = torch.zeros(seq, dtype=torch.long, device=embs.device)
        att[0] = 1
        return embs, pad_masks, att[None].expand(bsize, seq), adarms_cond

    def _expert_denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: DynamicCache,
        x_t: Tensor,
        timestep: Tensor,
        cond: ExpertCondition,
    ) -> Tensor:
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_expert_suffix(x_t, timestep, cond)
        bsize, suffix_len = suffix_pad.shape
        prefix_len = prefix_pad_masks.shape[1]
        prefix_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        full_2d = torch.cat([prefix_2d, _make_att_2d_masks(suffix_pad, suffix_att)], dim=2)
        position_ids = prefix_pad_masks.sum(dim=-1, keepdim=True) + torch.cumsum(suffix_pad, dim=1) - 1
        cast("Any", self.paligemma_with_expert.gemma_expert.model).config._attn_implementation = "eager"  # noqa: SLF001
        outputs, _ = self._joint_forward(
            attention_mask=self._prepare_attention_masks_4d(full_2d),
            position_ids=position_ids,
            past_key_values=_clone_kv_cache(past_key_values),
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs[1][:, -self._chunk_size :].to(dtype=torch.float32)  # type: ignore[index]
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_segment_from_cache(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: DynamicCache,
        cond: ExpertCondition,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Denoise one ``(B, D, max_action_dim)`` chunk from a cached prefix.

        Returns:
            Normalized, padded action chunk.
        """
        num_steps = num_steps or self._num_inference_steps
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device
        x_t = noise if noise is not None else self.sample_noise((bsize, self._chunk_size, self._max_action_dim), device)
        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            # Not in place: x_t may alias the caller's noise tensor.
            # Not in place: x_t may alias the caller's noise tensor.
            x_t = x_t + dt * self._expert_denoise_step(prefix_pad_masks, past_key_values, x_t, t, cond)  # noqa: PLR6104
            x_t = x_t + dt * self._expert_denoise_step(prefix_pad_masks, past_key_values, x_t, t, cond)  # noqa: PLR6104  # noqa: PLR6104
        return x_t

    # ------------------------------------------------------------------ prefix / planner

    @torch.no_grad()
    def prefix_cache(self, batch: dict[str, Any]) -> tuple[Tensor, DynamicCache]:
        """Prefill the image/prompt prefix once; shared by planner and expert.

        Returns:
            Tuple of (prefix padding mask, KV cache).
        """
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(
            batch[IMAGES],
            batch[IMAGE_MASKS],
            batch[TOKENIZED_PROMPT],
            batch[TOKENIZED_PROMPT_MASK],
        )
        att_2d = _make_att_2d_masks(prefix_pad, prefix_att)
        self._lm.config._attn_implementation = "eager"  # noqa: SLF001
        _, cache = self._joint_forward(
            attention_mask=self._prepare_attention_masks_4d(att_2d),
            position_ids=torch.cumsum(prefix_pad, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_pad, cache  # type: ignore[return-value]

    def _lm_step(
        self,
        embs: Tensor,
        past_pad: Tensor,
        new_mask: Tensor,
        cache: DynamicCache,
        start_pos: Tensor,
    ) -> Tensor:
        bsize, n = embs.shape[:2]
        past_cols = past_pad[:, None, :].expand(bsize, n, past_pad.shape[1])
        full = torch.cat([past_cols, new_mask[None].expand(bsize, n, n)], dim=2)
        position_ids = start_pos[:, None] + torch.arange(n, device=embs.device)[None]
        (hidden, _), _ = self._joint_forward(
            attention_mask=self._prepare_attention_masks_4d(full),
            position_ids=position_ids,
            past_key_values=cache,
            inputs_embeds=[embs, None],
            use_cache=True,
        )
        return hidden  # type: ignore[return-value]

    def _masked_argmax(self, logits: Tensor, *, first_block: bool) -> Tensor:
        return self.decode_constraints(logits, first_block=first_block).argmax(dim=-1)

    def decode_constraints(self, logits: Tensor, *, first_block: bool) -> Tensor:
        """Forbid the current-state code everywhere and the end code on the first block.

        Args:
            logits: ``(..., V, R)`` value-slot logits of one block.
            first_block: Whether these are the logits of the first plan block.

        Returns:
            Constrained logits.
        """
        d_lo = self.wp_tokenizer.family_ranges()[SlotKind.D][0]
        logits = logits.clone()
        logits[..., -1, d_lo + self.wp_tokenizer.state_code] = _NEG_INF
        if first_block:
            logits[..., -1, d_lo] = _NEG_INF
        return logits

    @torch.no_grad()  # noqa: PLR0914, PLR0915
    def decode_plan(  # noqa: PLR0914, PLR0915  # noqa: PLR0914, PLR0915
        self,
        prefix_pad_masks: Tensor,
        past_key_values: DynamicCache,
        state: Tensor,
        cur_gripper: Tensor,
    ) -> WaypointPlan:
        """Decode up to ``M`` waypoint blocks after the cached prefix.

        Returns:
            The decoded :class:`WaypointPlan`.
        """
        tok = self.wp_tokenizer
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device
        cache = _clone_kv_cache(past_key_values)
        past_pad = prefix_pad_masks.clone()
        next_pos = prefix_pad_masks.sum(dim=-1)
        state_ids = self._state_block_ids(self._state_cfg(state), cur_gripper)

        m = self.max_blocks
        q_out = torch.zeros(bsize, m, tok.num_cfg, device=device)
        g_out = torch.zeros(bsize, m, tok.num_arms, dtype=torch.long, device=device)
        d_out = torch.zeros(bsize, m, dtype=torch.long, device=device)
        valid = torch.zeros(bsize, m, dtype=torch.bool, device=device)
        done = torch.zeros(bsize, dtype=torch.bool, device=device)
        passes = 1

        def feed(ids: Tensor, mask: Tensor, *, query: bool = False) -> Tensor:
            nonlocal past_pad, next_pos
            embs = self.paligemma_with_expert.embed_language_tokens(ids)
            if query and self.block_query is not None:
                embs = embs + self.block_query.to(embs.dtype)
            hidden = self._lm_step(embs, past_pad, mask, cache, next_pos)
            past_pad = torch.cat([past_pad, torch.ones_like(ids, dtype=torch.bool)], dim=1)
            next_pos += ids.shape[1]
            return hidden

        if self.planner_mode == "block_ar":
            block_ids = state_ids
            full = torch.ones(tok.block_len, tok.block_len, dtype=torch.bool, device=device)
            for k in range(m):
                hidden = feed(block_ids, full, query=k == 0)
                passes += 1
                logits = self._reserved_logits(hidden[:, self.value_positions])
                reserved = self._masked_argmax(logits, first_block=k == 0)
                q, g, d = tok.values_from_reserved(reserved)
                q_out[:, k], g_out[:, k], d_out[:, k] = q, g, d
                valid[:, k] = ~done
                done |= d == 0
                if bool(done.all()):
                    break
                block_ids = tok.encode(q, g, d)
        else:
            family = self.family_mask
            n_qg = tok.num_cfg + tok.num_arms
            pending = torch.cat([state_ids, torch.full((bsize, 1), tok.wp_id, device=device)], dim=1)
            reserved_emb = self._reserved_embeddings().float()
            for k in range(m):
                slots = []
                for j in range(tok.num_values):
                    n = pending.shape[1]
                    hidden = feed(pending, torch.tril(torch.ones(n, n, dtype=torch.bool, device=device)))
                    passes += 1
                    logits = (hidden[:, -1].float() @ reserved_emb.T).masked_fill(~family[j], _NEG_INF)
                    if j == tok.num_values - 1:
                        full_logits = logits[:, None].expand(-1, tok.num_values, -1)
                        logits = self.decode_constraints(full_logits, first_block=k == 0)[:, -1]
                    r = logits.argmax(dim=-1)
                    slots.append(r)
                    tid = (r + tok.base_id)[:, None]
                    if j == n_qg - 1:
                        pending = torch.cat([tid, torch.full_like(tid, tok.dur_id)], dim=1)
                    elif j == tok.num_values - 1:
                        pending = torch.cat([tid, torch.full_like(tid, tok.wp_id)], dim=1)
                    else:
                        pending = tid
                q, g, d = tok.values_from_reserved(torch.stack(slots, dim=1))
                q_out[:, k], g_out[:, k], d_out[:, k] = q, g, d
                valid[:, k] = ~done
                done |= d == 0
                if bool(done.all()):
                    break

        return WaypointPlan(q=q_out, g=g_out, d=d_out, valid=valid, num_passes=passes)

    # ------------------------------------------------------------------ training

    def _plan_sequence(self, batch: dict[str, Any], cur_gripper: Tensor) -> tuple[Tensor, Tensor]:
        tok = self.wp_tokenizer
        state_ids = self._state_block_ids(self._state_cfg(batch[STATE]), cur_gripper)
        wp_ids = tok.encode(batch[KEY_WP_Q].float(), batch[KEY_WP_G].long(), batch[KEY_WP_D].long())
        ids = torch.cat([state_ids[:, None], wp_ids], dim=1).flatten(1)
        block_valid = torch.cat([torch.ones_like(batch[KEY_WP_VALID][:, :1]), batch[KEY_WP_VALID]], dim=1).bool()
        return ids, block_valid.repeat_interleave(tok.block_len, dim=1)

    def planner_logits(self, plan_hidden: Tensor) -> Tensor:
        """Teacher-forced value-slot logits from the hidden states of the plan sequence.

        Args:
            plan_hidden: ``(B, (M + 1) * block_len, W)`` VLM outputs at plan positions.

        Returns:
            ``(B, M, V, R)`` family-masked logits for waypoint blocks ``1..M``.
        """
        tok = self.wp_tokenizer
        m, n = self.max_blocks, tok.block_len
        vpos = self.value_positions
        blocks = torch.arange(m, device=plan_hidden.device)[:, None]
        # Token-AR predicts slot p from position p - 1 of the next block; Block-AR from the same slot one block earlier.
        idx = (blocks + 1) * n + vpos[None] - 1 if self.planner_mode == "token_ar" else blocks * n + vpos[None]
        hidden = plan_hidden[:, idx.flatten()].reshape(plan_hidden.shape[0], m, tok.num_values, -1)
        return self._reserved_logits(hidden)

    def _planner_loss(self, plan_hidden: Tensor, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        tok = self.wp_tokenizer
        logits = self.planner_logits(plan_hidden)
        targets = tok.reserved_targets(batch[KEY_WP_Q].float(), batch[KEY_WP_G].long(), batch[KEY_WP_D].long())

        valid = batch[KEY_WP_VALID].bool()
        terminal = batch[KEY_WP_D].long() == 0
        is_d = torch.zeros(tok.num_values, dtype=torch.bool, device=valid.device)
        is_d[-1] = True
        weight = (valid[..., None] & (~terminal[..., None] | is_d)).float()

        ce = F.cross_entropy(logits.flatten(0, 2), targets.flatten(), reduction="none").reshape(targets.shape)
        loss = (ce * weight).sum() / weight.sum().clamp(min=1.0)
        correct = (logits.argmax(-1) == targets).float()
        acc = (correct * weight).sum() / weight.sum().clamp(min=1.0)
        d_weight = weight[..., -1]
        d_acc = (correct[..., -1] * d_weight).sum() / d_weight.sum().clamp(min=1.0)
        return loss, {"planner_ce": loss.detach(), "planner_acc": acc.detach(), "planner_d_acc": d_acc.detach()}

    def teacher_forced_planner_logits(self, batch: dict[str, Any]) -> Tensor:
        """Run prefix + ground-truth plan tokens through the VLM only.

        Returns:
            ``(B, M, V, R)`` value-slot logits.
        """
        bsize, device = batch[STATE].shape[0], batch[STATE].device
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(
            batch[IMAGES],
            batch[IMAGE_MASKS],
            batch[TOKENIZED_PROMPT],
            batch[TOKENIZED_PROMPT_MASK],
        )
        plan_ids, plan_pad = self._plan_sequence(batch, self._current_gripper(batch, bsize, device))
        embs = torch.cat([prefix_embs, self._embed_plan(plan_ids).to(prefix_embs.dtype)], dim=1)
        pad = torch.cat([prefix_pad, plan_pad], dim=1)
        att = torch.cat([prefix_att.long(), self._plan_attention(self.max_blocks + 1, bsize, device)], dim=1)
        n_prefix = prefix_pad.sum(dim=-1, keepdim=True)
        position_ids = torch.cat(
            [torch.cumsum(prefix_pad, dim=1) - 1, n_prefix + torch.cumsum(plan_pad, dim=1) - 1],
            dim=1,
        )
        self._lm.config._attn_implementation = "eager"  # noqa: SLF001
        (hidden, _), _ = self._joint_forward(
            attention_mask=self._prepare_attention_masks_4d(_make_att_2d_masks(pad, att)),
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[embs, None],
            use_cache=False,
        )
        return self.planner_logits(hidden[:, prefix_pad.shape[1] :])  # type: ignore[index]

    def compute_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:  # noqa: PLR0914
        """Joint planner cross-entropy and expert flow-matching loss in one forward pass.

        Returns:
            Tuple of (loss, loss dict).
        """
        actions = batch[ACTION]
        bsize, device = actions.shape[0], actions.device
        cur_gripper = self._current_gripper(batch, bsize, device)

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(
            batch[IMAGES],
            batch[IMAGE_MASKS],
            batch[TOKENIZED_PROMPT],
            batch[TOKENIZED_PROMPT_MASK],
        )
        plan_ids, plan_pad = self._plan_sequence(batch, cur_gripper)
        plan_embs = self._embed_plan(plan_ids)
        plan_att = self._plan_attention(self.max_blocks + 1, bsize, device)

        noise = self.sample_noise(actions.shape, device)
        time = self.sample_time(bsize, device)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions
        cond = self.expert_condition(
            batch[STATE],
            batch[KEY_SEG_Q],
            batch[KEY_SEG_G],
            batch[KEY_SEG_D],
            perturb=self.training,
        )
        suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_expert_suffix(x_t, time, cond)

        vlm_embs = torch.cat([prefix_embs, plan_embs.to(prefix_embs.dtype)], dim=1)
        if self._lm.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            vlm_embs = vlm_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        pad = torch.cat([prefix_pad, plan_pad, suffix_pad], dim=1)
        att = torch.cat([prefix_att.long(), plan_att, suffix_att.long()], dim=1)
        att_2d = _make_att_2d_masks(pad, att).clone()
        p_len, v_len = prefix_pad.shape[1], vlm_embs.shape[1]
        att_2d[:, v_len:, p_len:v_len] = False
        n_prefix = prefix_pad.sum(dim=-1, keepdim=True)
        position_ids = torch.cat(
            [
                torch.cumsum(prefix_pad, dim=1) - 1,
                n_prefix + torch.cumsum(plan_pad, dim=1) - 1,
                n_prefix + torch.cumsum(suffix_pad, dim=1) - 1,
            ],
            dim=1,
        )
        mask_4d = self._prepare_attention_masks_4d(att_2d)

        def forward_func(vlm: Tensor, suffix: Tensor, mask: Tensor, pos: Tensor, adarms: Tensor) -> tuple[Tensor, Tensor]:
            (vlm_out, suffix_out), _ = self._joint_forward(
                attention_mask=mask,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[vlm, suffix],
                use_cache=False,
                adarms_cond=[None, adarms],
            )
            return vlm_out, suffix_out  # type: ignore[return-value]

        vlm_out, suffix_out = self._apply_checkpoint(forward_func, vlm_embs, suffix_embs, mask_4d, position_ids, adarms_cond)

        plan_loss, plan_stats = self._planner_loss(vlm_out[:, p_len:], batch)

        v_t = self.action_out_proj(suffix_out[:, -self._chunk_size :].to(torch.float32))
        action_dim = int(self._dataset_stats[ACTION]["shape"][-1])
        fm_losses = F.mse_loss(u_t, v_t, reduction="none")[:, :, :action_dim]
        fm_loss = reduce_losses(fm_losses, in_episode_bound(batch))

        loss = self.planner_loss_weight * plan_loss + self.action_loss_weight * fm_loss
        return loss, {"loss": loss.detach(), "fm_loss": fm_loss.detach(), **plan_stats}

    @torch.no_grad()
    def compute_val_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Teacher-forced planner CE plus action MSE of segments conditioned on ground-truth waypoints.

        Returns:
            Tuple of (action MSE, metrics dict).
        """
        _, stats = self.compute_loss(batch)
        prefix_pad, cache = self.prefix_cache(batch)
        cond = self.expert_condition(batch[STATE], batch[KEY_SEG_Q], batch[KEY_SEG_G], batch[KEY_SEG_D])
        pred = self.sample_segment_from_cache(prefix_pad, cache, cond)
        action_dim = int(self._dataset_stats[ACTION]["shape"][-1])
        losses = F.mse_loss(pred[:, :, :action_dim], batch[ACTION][:, :, :action_dim], reduction="none")
        mse = reduce_losses(losses, in_episode_bound(batch))
        return mse, {
            "loss": mse.item(),
            "planner_ce": float(stats["planner_ce"]),
            "planner_acc": float(stats["planner_acc"]),
        }

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def plan_and_segment(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
    ) -> tuple[WaypointPlan, Tensor]:
        """Decode a plan and the action chunk for its first waypoint, sharing one prefix prefill.

        Returns:
            Tuple of (plan, unpadded normalized action chunk ``(B, D, action_dim)``).
        """
        state = batch[STATE]
        bsize, device = state.shape[0], state.device
        prefix_pad, cache = self.prefix_cache(batch)
        plan = self.decode_plan(prefix_pad, cache, state, self._current_gripper(batch, bsize, device))
        cond = self.expert_condition(state, plan.q[:, 0], plan.g[:, 0], plan.d[:, 0].clamp(min=1))
        actions = self.sample_segment_from_cache(prefix_pad, cache, cond, noise=noise)
        action_dim = int(self._dataset_stats[ACTION]["shape"][-1])
        return plan, actions[:, :, :action_dim]

    @torch.no_grad()
    def predict_segment(
        self,
        batch: dict[str, Any],
        goal_q: Tensor,
        goal_g: Tensor,
        duration: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Action chunk for a given (normalized) waypoint from the current observation.

        Returns:
            Unpadded normalized action chunk ``(B, D, action_dim)``.
        """
        prefix_pad, cache = self.prefix_cache(batch)
        cond = self.expert_condition(batch[STATE], goal_q, goal_g, duration)
        actions = self.sample_segment_from_cache(prefix_pad, cache, cond, noise=noise)
        return actions[:, :, : int(self._dataset_stats[ACTION]["shape"][-1])]

    def predict_action_chunk(self, batch: dict[str, Any]) -> Tensor:
        """Plan, then return the chunk for the first waypoint (``(B, D, action_dim)``).

        Returns:
            Normalized action chunk.
        """
        _, actions = self.plan_and_segment(batch)
        return actions
