# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pi05Waypoint policy: Lightning wrapper with a plan -> segment execution loop."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from physicalai.data.dataset import Dataset
from physicalai.data.observation import ACTION, STATE
from physicalai.data.waypoints.extract import WAYPOINT_STATS_KEY
from physicalai.policies.base import Policy
from physicalai.policies.pi05.policy import Pi05
from physicalai.policies.pi05.pretrained_utils import fix_state_dict_keys
from physicalai.train.utils import reformat_dataset_to_match_policy

from .config import Pi05WaypointConfig
from .model import KEY_WP_CUR_G, Pi05WaypointModel, WaypointPlan
from .preprocessor import make_pi05_waypoint_preprocessors

if TYPE_CHECKING:
    from physicalai.data import Observation
    from physicalai.export import ExportBackend
    from physicalai.policies.pi05.preprocessor import Pi05Postprocessor

    from .preprocessor import ConfigNormalizer, Pi05WaypointPreprocessor

logger = logging.getLogger(__name__)


def _resolve_weights(pretrained_name_or_path: str | Path) -> Path:
    path = Path(pretrained_name_or_path)
    if path.is_dir():
        return path / "model.safetensors"
    return Path(hf_hub_download(str(pretrained_name_or_path), "model.safetensors"))  # nosec B615


class Pi05Waypoint(Pi05):
    """Hierarchical Pi0.5: VLM waypoint planner + goal-conditioned flow-matching expert.

    Training requires a :class:`~physicalai.data.waypoints.WaypointLeRobotDataModule`. At
    inference :meth:`select_action` decodes a plan, then for every waypoint prefills the
    current observation once, denoises a ``D``-step chunk and executes its first ``d`` actions.

    Args:
        pretrained_name_or_path: Pi0.5 weights to initialize from (HF repo or local dir with
            ``model.safetensors``). Waypoint heads are always freshly initialized.
        dataset_stats: Stats for eager construction (must include ``"waypoint"``).
        Other arguments: see :class:`Pi05WaypointConfig` and :class:`~physicalai.policies.Pi05`.

    Example:
        >>> policy = Pi05Waypoint(
        ...     pretrained_name_or_path="lerobot/pi05_base",
        ...     planner_mode="block_ar",
        ...     goal_conditioning="ngm",
        ...     lora_enabled=True,
        ...     lora_rank=16,
        ... )
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        pretrained_name_or_path: str | Path | None = None,
        *,
        # Waypoint hierarchy
        planner_mode: Literal["token_ar", "block_ar"] = "block_ar",
        goal_conditioning: Literal["suffix", "naive", "ngm"] = "ngm",
        max_blocks: int = 7,
        num_bins: int = 300,
        planner_loss_weight: float = 1.0,
        action_loss_weight: float = 1.7,
        goal_noise_std: float | None = None,
        goal_dropout: float | None = None,
        gate_lo: float = 0.2,
        gate_hi: float = 0.5,
        replan_mode: Literal["full_plan", "receding"] = "full_plan",
        # Pi0.5 backbone / expert
        paligemma_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_2b",
        action_expert_variant: Literal["gemma_300m", "gemma_2b"] = "gemma_300m",
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        chunk_size: int = 32,
        n_action_steps: int = 32,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        num_inference_steps: int = 10,
        use_random_input_noise: bool = True,
        image_resolution: tuple[int, int] = (224, 224),
        empty_cameras: int = 0,
        tokenizer_max_length: int = 200,
        gradient_checkpointing: bool = True,
        freeze_vision_encoder: bool = False,
        normalization_mode: Literal["MEAN_STD", "QUANTILES"] = "QUANTILES",
        # LoRA
        lora_enabled: bool = False,
        lora_rank: int = 16,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        lora_target_modules: str | tuple[str, ...] | None = None,
        lora_adapter_dtype: Literal["float32", "auto"] = "float32",
        lora_use_dora: bool = False,
        lora_lr_scale: float = 10.0,
        # Optimizer / scheduler
        optimizer_lr: float = 2.5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.01,
        optimizer_grad_clip_norm: float = 1.0,
        scheduler_warmup_steps: int = 1_000,
        scheduler_decay_steps: int | None = None,
        scheduler_decay_lr: float = 2.5e-6,
        # Eager initialization
        dataset_stats: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the policy (model is built lazily in ``setup`` unless stats are given)."""
        self.config = Pi05WaypointConfig(
            planner_mode=planner_mode,
            goal_conditioning=goal_conditioning,
            max_blocks=max_blocks,
            num_bins=num_bins,
            planner_loss_weight=planner_loss_weight,
            action_loss_weight=action_loss_weight,
            goal_noise_std=goal_noise_std,
            goal_dropout=goal_dropout,
            gate_lo=gate_lo,
            gate_hi=gate_hi,
            replan_mode=replan_mode,
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            dtype=dtype,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            num_inference_steps=num_inference_steps,
            use_random_input_noise=use_random_input_noise,
            image_resolution=image_resolution,
            empty_cameras=empty_cameras,
            tokenizer_max_length=tokenizer_max_length,
            gradient_checkpointing=gradient_checkpointing,
            freeze_vision_encoder=freeze_vision_encoder,
            normalization_mode=normalization_mode,
            lora_enabled=lora_enabled,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_adapter_dtype=lora_adapter_dtype,
            lora_use_dora=lora_use_dora,
            lora_lr_scale=lora_lr_scale,
            optimizer_lr=optimizer_lr,
            optimizer_betas=optimizer_betas,
            optimizer_eps=optimizer_eps,
            optimizer_weight_decay=optimizer_weight_decay,
            optimizer_grad_clip_norm=optimizer_grad_clip_norm,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_decay_steps=scheduler_decay_steps,
            scheduler_decay_lr=scheduler_decay_lr,
        )
        Policy.__init__(self, n_action_steps=self.config.n_action_steps)
        self.save_hyperparameters(ignore=["pretrained_name_or_path"])
        self.hparams["config"] = self.config.to_dict()

        self._weights_file = _resolve_weights(pretrained_name_or_path) if pretrained_name_or_path else None
        self.model: Pi05WaypointModel | None = None  # type: ignore[assignment]
        self._preprocessor: Pi05WaypointPreprocessor | None = None  # type: ignore[assignment]
        self._postprocessor: Pi05Postprocessor | None = None
        self._config_normalizer: ConfigNormalizer | None = None
        self._dataset_stats = dataset_stats

        self.erase_endpoint = False
        self.reset()

        if dataset_stats is not None:
            self._initialize_model(dataset_stats, self._weights_file)

    # ------------------------------------------------------------------ construction

    def _initialize_model(
        self,
        dataset_stats: dict[str, Any],
        weights_file: Path | None = None,
    ) -> None:
        """Build model, load Pi0.5 weights, inject LoRA and create preprocessors.

        Raises:
            ValueError: If waypoint stats are missing or were computed with another normalization.
        """
        wp_stats = dataset_stats.get(WAYPOINT_STATS_KEY)
        if wp_stats is None:
            msg = "dataset_stats has no 'waypoint' entry; train with WaypointLeRobotDataModule"
            raise ValueError(msg)
        if str(wp_stats.get("normalization_mode", self.config.normalization_mode)) != self.config.normalization_mode:
            msg = (
                f"Waypoint sigma_delta was computed with {wp_stats['normalization_mode']} but the policy uses "
                f"{self.config.normalization_mode}; set the same normalization_mode on the datamodule."
            )
            raise ValueError(msg)

        cfg = self.config
        self.model = Pi05WaypointModel(
            dataset_stats,
            planner_mode=cfg.planner_mode,
            goal_conditioning=cfg.goal_conditioning,
            max_blocks=cfg.max_blocks,
            num_bins=cfg.num_bins,
            max_state_dim=cfg.max_state_dim,
            planner_loss_weight=cfg.planner_loss_weight,
            action_loss_weight=cfg.action_loss_weight,
            goal_noise_std=cfg.goal_noise_std,
            goal_dropout=cfg.goal_dropout,
            gate_lo=cfg.gate_lo,
            gate_hi=cfg.gate_hi,
            paligemma_variant=cfg.paligemma_variant,
            action_expert_variant=cfg.action_expert_variant,
            dtype=cfg.dtype,
            chunk_size=cfg.chunk_size,
            max_action_dim=cfg.max_action_dim,
            n_action_steps=cfg.n_action_steps,
            num_inference_steps=cfg.num_inference_steps,
            time_sampling_beta_alpha=cfg.time_sampling_beta_alpha,
            time_sampling_beta_beta=cfg.time_sampling_beta_beta,
            time_sampling_scale=cfg.time_sampling_scale,
            time_sampling_offset=cfg.time_sampling_offset,
            min_period=cfg.min_period,
            max_period=cfg.max_period,
            image_resolution=cfg.image_resolution,
            tokenizer_max_length=cfg.tokenizer_max_length,
            freeze_vision_encoder=cfg.freeze_vision_encoder,
            train_expert_only=False,
            gradient_checkpointing=cfg.gradient_checkpointing,
            compile_model=False,
            use_random_input_noise=cfg.use_random_input_noise,
        )
        if weights_file is not None:
            state_dict = fix_state_dict_keys(load_file(str(weights_file)))
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False, assign=True)
            new_heads = [k for k in missing if not k.startswith("paligemma_with_expert.")]
            logger.info("Loaded Pi0.5 weights; %d freshly initialized waypoint-head tensors", len(new_heads))
            backbone_missing = [k for k in missing if k.startswith("paligemma_with_expert.")]
            if backbone_missing or unexpected:
                logger.warning(
                    "Pretrained load: %d missing backbone keys, %d unexpected keys (e.g. %s)",
                    len(backbone_missing),
                    len(unexpected),
                    (backbone_missing + unexpected)[:5],
                )
            self.model.paligemma_with_expert.to_bfloat16_for_selected_params(cfg.dtype)
            self.model.paligemma_with_expert._set_requires_grad()  # noqa: SLF001

        if cfg.use_lora:
            self._inject_lora()
            self.model.enable_waypoint_head_training()

        self._build_preprocessors(dataset_stats)
        self._dataset_stats = dataset_stats

    def _build_preprocessors(self, dataset_stats: dict[str, Any]) -> None:
        cfg = self.config
        self._preprocessor, self._postprocessor, self._config_normalizer = make_pi05_waypoint_preprocessors(
            dataset_stats,
            dataset_stats[WAYPOINT_STATS_KEY]["config_dims"],
            max_action_dim=cfg.max_action_dim,
            image_resolution=cfg.image_resolution,
            max_token_len=cfg.tokenizer_max_length,
            empty_cameras=cfg.empty_cameras,
            normalization_mode=cfg.normalization_mode,
        )

    def setup(self, stage: str) -> None:
        """Build the model from the waypoint datamodule stats.

        Raises:
            TypeError: If the train dataset is not a physicalai Dataset.
        """
        del stage
        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset
        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)
        stats = train_dataset.stats
        if self.model is None:
            self.hparams["dataset_stats"] = stats
            self._initialize_model(stats, self._weights_file)
        else:
            self._update_preprocessor_stats(stats)
        reformat_dataset_to_match_policy(self, datamodule)

    def _update_preprocessor_stats(self, dataset_stats: dict[str, Any]) -> None:
        self._build_preprocessors(dataset_stats)
        self._dataset_stats = dataset_stats
        self.hparams["dataset_stats"] = dataset_stats
        if self.model is not None:
            self.model.set_dataset_stats(dataset_stats)

    # ------------------------------------------------------------------ training

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Joint planner + expert loss with per-term logging.

        Returns:
            Training loss.
        """
        del batch_idx
        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        for key in ("fm_loss", "planner_ce", "planner_acc", "planner_d_acc"):
            if key in loss_dict:
                self.log(f"train/{key}", loss_dict[key])
        return loss

    # ------------------------------------------------------------------ inference

    def reset(self) -> None:
        """Clear action queue, pending plan and telemetry."""
        super().reset()
        self._plan_queue: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        self._segments_since_plan = 0
        self._cur_gripper: torch.Tensor | None = None
        self.last_plan: WaypointPlan | None = None
        self.telemetry: list[dict[str, Any]] = []

    def plan_to_raw(self, plan: WaypointPlan) -> torch.Tensor:
        """Denormalize plan configurations to dataset units (for logging / visualization).

        Returns:
            ``(B, M, num_cfg)`` raw configurations.

        Raises:
            ValueError: If the policy is not initialized.
        """
        if self._config_normalizer is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        return self._config_normalizer.inverse(plan.q)

    @torch.no_grad()
    def select_action(self, batch: Observation) -> torch.Tensor:  # noqa: PLR0914
        """Execute the plan one action at a time, replanning per ``replan_mode``.

        Returns:
            Action ``(1, action_dim)``.

        Raises:
            ValueError: If the model is not initialized.
            NotImplementedError: For batch sizes other than 1.
        """
        queued = self._get_queued_action()
        if queued is not None:
            return queued
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)

        processed = self._preprocessor(batch.to(self.device).to_dict())
        state = processed[STATE]
        if state.shape[0] != 1:
            msg = "Pi05Waypoint.select_action supports batch size 1 (plans differ per environment)"
            raise NotImplementedError(msg)
        model = self.model
        if self._cur_gripper is not None:
            processed[KEY_WP_CUR_G] = self._cur_gripper

        sync = torch.xpu.synchronize if state.device.type == "xpu" else torch.cuda.synchronize
        timed = state.device.type in {"cuda", "xpu"}
        t0 = time.perf_counter()
        prefix_pad, cache = model.prefix_cache(processed)
        need_plan = not self._plan_queue or (self.config.replan_mode == "receding" and self._segments_since_plan >= 1)
        if need_plan:
            cur_g = model._current_gripper(processed, 1, state.device)  # noqa: SLF001
            plan = model.decode_plan(prefix_pad, cache, state, cur_g)
            if timed:
                sync()
            t_plan = time.perf_counter()
            self.last_plan = plan
            waypoints = plan.executable(0) or [(plan.q[0, 0], plan.g[0, 0], max(int(plan.d[0, 0]), 1))]
            self._plan_queue = waypoints
            self._segments_since_plan = 0
            self.telemetry.append({
                "event": "plan",
                "ms": (t_plan - t0) * 1e3,
                "num_passes": plan.num_passes,
                "num_waypoints": len(waypoints),
                "ended": bool((plan.d[0][plan.valid[0]] == 0).any()),
            })
            t0 = t_plan

        q, g, d = self._plan_queue.pop(0)
        if self.erase_endpoint:
            q = model._state_cfg(state)[0]  # noqa: SLF001
        cond = model.expert_condition(state, q[None], g[None], torch.tensor([d], device=state.device))
        actions = model.sample_segment_from_cache(prefix_pad, cache, cond)
        if timed:
            sync()
        self.telemetry.append({"event": "segment", "ms": (time.perf_counter() - t0) * 1e3, "d": d})
        self._segments_since_plan += 1
        self._cur_gripper = g[None]

        action_dim = int(self._dataset_stats[ACTION]["shape"][-1])  # type: ignore[index]
        actions = self._postprocessor({ACTION: actions[:, :d, :action_dim]})[ACTION]
        self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Plan and return the full ``D``-step chunk for the first waypoint.

        Returns:
            Denormalized ``(B, D, action_dim)`` actions.

        Raises:
            ValueError: If the model is not initialized.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model is not initialized"
            raise ValueError(msg)
        processed = self._preprocessor(batch.to(self.device).to_dict())
        actions = self.model.predict_action_chunk(processed)
        return self._postprocessor({ACTION: actions})[ACTION]

    # ------------------------------------------------------------------ export

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Export of the planner/expert loop is not implemented yet.

        Returns:
            Empty list.
        """
        return []

    @property
    def extra_export_args(self) -> dict[str, Any]:
        """Not exportable yet.

        Raises:
            NotImplementedError: Always.
        """
        msg = "Pi05Waypoint export (planner prefill / block step / expert graphs) is not implemented yet"
        raise NotImplementedError(msg)

    def freeze_vlm(self) -> None:
        """Not supported: the planner lives in the VLM.

        Raises:
            NotImplementedError: Always.
        """
        msg = "freeze_vlm() would freeze the waypoint planner"
        raise NotImplementedError(msg)

    def _set_hparam_keys(self) -> None:
        self.hparams["config"] = self.config.to_dict()

    @property
    def planner_mode(self) -> Literal["token_ar", "block_ar"]:
        """Configured planner decoding mode."""
        return self.config.planner_mode
