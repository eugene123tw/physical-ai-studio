# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RLDX-1 Policy - first-party Lightning wrapper for RLWRLD's RLDX-1 VLA.

This module provides the PyTorch Lightning policy for training and inference
with RLDX-1, a flow-matching Vision-Language-Action model built on a Qwen3-VL-8B
backbone and a Multi-Stream Action Transformer (MSAT) action head.

Scope (v1): pre-train (PT) -> fine-tune (FT) path only, starting from
``RLWRLD/RLDX-1-PT``. See ``library/docs/rldx-1-integration.md``.

.. note::
    The architecture port is landing incrementally. The Lightning contract
    (config, dual-path init, action queue) is in place; the underlying
    :class:`~physicalai.policies.rldx1.model.Rldx1Model` forward passes are
    filled in by subsequent component ports.

## Quick Start

```python
from physicalai.data.lerobot import LeRobotDataModule
from physicalai.policies.rldx1 import Rldx1
from physicalai.train import Trainer

# Default: LoRA on both backbone and action model
policy = Rldx1(base_model_path="RLWRLD/RLDX-1-PT")

# Paper Table 6, Row 1: Full FT backbone + LoRA action (62.67% success)
policy = Rldx1(
    base_model_path="RLWRLD/RLDX-1-PT",
    backbone_use_lora=False,  # Full fine-tune top-4 LLM layers
    action_use_lora=True,     # LoRA on MSAT (r=64)
)

datamodule = LeRobotDataModule(repo_id="<user dataset>", train_batch_size=4)
trainer = Trainer(max_steps=60000, precision="bf16-mixed")
trainer.fit(policy, datamodule)
```
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from physicalai.data import Observation
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler

from .config import Rldx1Config
from .model import Rldx1Model

if TYPE_CHECKING:
    from pathlib import Path

    from .transforms import Rldx1Postprocessor, Rldx1Preprocessor

logger = logging.getLogger(__name__)


class Rldx1(Policy):
    """RLDX-1 Policy - first-party Lightning wrapper.

    All hyperparameters are explicit in the signature for discoverability.
    Supports the dual-path initialization shared across Studio policies:

    - **Lazy path**: ``Rldx1()`` + ``trainer.fit()`` - model built in ``setup()``
      once dataset features are known.
    - **Eager path**: ``Rldx1.load_from_checkpoint()`` or ``Rldx1(env_action_dim=...)``
      - model built immediately.

    Args:
        chunk_size: Number of action predictions per forward pass (action horizon).
        n_action_steps: Number of action steps to execute per chunk.
        max_state_dim: Maximum state dimension (shorter states zero-padded).
        max_action_dim: Maximum action dimension (shorter actions zero-padded).
        base_model_path: HuggingFace model ID or path to the base checkpoint.
        revision: Pinned git commit SHA for the checkpoint download (lib.security rule 9).
        model_name: HuggingFace ID of the Qwen3-VL backbone.
        select_layer: VLM hidden layer used as cognition features.
        attn_implementation: Attention backend ('sdpa', 'flash_attention_2', 'eager').
        n_cog_tokens: Number of cognition tokens routed to MSAT.
        tune_top_llm_layers: Number of top LLM layers to fine-tune.
        tune_llm: Whether to fine-tune the entire LLM backbone (all decoder
            layers + input embeddings + lm_head). Overrides tune_top_llm_layers.
        backbone_trainable_params_fp32: Whether to cast trainable backbone
            parameters to float32 after bf16 loading for optimizer stability.
        tune_visual: Whether to fine-tune the vision tower.
        tune_projector: Whether to fine-tune the projectors.
        tune_diffusion_model: Whether to fine-tune the MSAT action model.
        tune_vlln: Whether to fine-tune the VLM-output layer norm in the action head.
        num_inference_timesteps: Number of flow-matching denoising steps at inference.
        backbone_use_lora: Whether to use LoRA on the backbone top layers.
            Default False (full fine-tuning). Set to True for LoRA.
        action_use_lora: Whether to use LoRA on the MSAT action model.
            Default False (full fine-tuning). Set to True for LoRA (Paper Table 6, row 1).
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        warmup_ratio: Warmup ratio (0.0-1.0) of total training steps.
        use_bf16: Whether to use bfloat16 precision.
        compile_model: Whether to torch.compile the model.
        gradient_checkpointing: Whether to enable activation checkpointing in
            MSAT during training.
        color_jitter_params: Train-time ``A.ColorJitter`` params
            (``{"brightness", "contrast", "saturation", "hue"}``). ``None``
            (default) disables color augmentation.
        clip_outliers: Clip normalized state/action to ``[-1, 1]`` at train and
            inference (upstream default ``True``). Set ``False`` (Pi05-style, no
            clip) for wide-range action spaces where ``QUANTILES`` bounds would
            truncate task-critical extremes (e.g. PushT).
        env_action_dim: Environment action dimension. If provided, enables eager init.
        dataset_stats: Dataset normalization statistics for eager init.
    """

    def __init__(  # noqa: PLR0913
        self,
        # Model architecture
        chunk_size: int = 16,
        n_action_steps: int = 16,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
        # Model source
        base_model_path: str = "RLWRLD/RLDX-1-PT",
        revision: str | None = None,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        # Backbone
        select_layer: int = 18,
        attn_implementation: str = "sdpa",
        n_cog_tokens: int = 64,
        # Fine-tuning control
        *,
        tune_top_llm_layers: int = 4,
        tune_llm: bool = False,
        backbone_trainable_params_fp32: bool = True,
        tune_visual: bool = False,
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        tune_vlln: bool = True,
        num_inference_timesteps: int = 4,
        backbone_use_lora: bool = False,
        action_use_lora: bool = False,
        # Optimizer
        optim: Literal["adamw_torch", "adamw_torch_fused", "adafactor"] = "adamw_torch",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        # Precision / compilation
        use_bf16: bool = True,
        compile_model: bool = False,
        gradient_checkpointing: bool = True,
        # Image augmentation (train only)
        color_jitter_params: dict[str, float] | None = None,
        image_min_area: int | None = None,
        # Normalization
        clip_outliers: bool = True,
        # Eager initialization (optional)
        env_action_dim: int | None = None,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Initialize the RLDX-1 policy and save hyperparameters."""
        super().__init__(n_action_steps=n_action_steps)

        self.config = Rldx1Config(
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            base_model_path=base_model_path,
            revision=revision,
            model_name=model_name,
            select_layer=select_layer,
            attn_implementation=attn_implementation,
            n_cog_tokens=n_cog_tokens,
            tune_top_llm_layers=tune_top_llm_layers,
            tune_llm=tune_llm,
            backbone_trainable_params_fp32=backbone_trainable_params_fp32,
            tune_visual=tune_visual,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            tune_vlln=tune_vlln,
            num_inference_timesteps=num_inference_timesteps,
            backbone_use_lora=backbone_use_lora,
            action_use_lora=action_use_lora,
            optim=optim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            use_bf16=use_bf16,
            compile_model=compile_model,
            gradient_checkpointing=gradient_checkpointing,
            color_jitter_params=color_jitter_params,
            image_min_area=image_min_area,
            clip_outliers=clip_outliers,
        )

        # Save individual args (not the config object) for checkpoint restoration.
        self.save_hyperparameters(ignore=["config"])
        self.hparams["config"] = self.config.to_dict()

        self.model: Rldx1Model | None = None
        self._preprocessor: Rldx1Preprocessor | None = None
        self._postprocessor: Rldx1Postprocessor | None = None
        self._is_setup_complete: bool = False

        # Per-view VTC frame history for rollout. Populated every env-step by
        # ``select_action`` and sampled at the video window offsets in
        # ``predict_action_chunk`` so the backbone sees the same multi-frame
        # stack it was trained on (see ``_apply_video_window``). ``None`` until
        # the first rollout step; cleared on ``reset``.
        self._frame_history: dict[str, deque[torch.Tensor]] | None = None

        if env_action_dim is not None:
            self._initialize_model(env_action_dim, dataset_stats)

    @classmethod
    def from_config(cls, config: Rldx1Config) -> Rldx1:
        """Build a policy from a :class:`Rldx1Config`.

        Args:
            config: The policy configuration.

        Returns:
            An initialized :class:`Rldx1` policy.
        """
        return cls(
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            base_model_path=config.base_model_path,
            revision=config.revision,
            model_name=config.model_name,
            select_layer=config.select_layer,
            attn_implementation=config.attn_implementation,
            n_cog_tokens=config.n_cog_tokens,
            tune_top_llm_layers=config.tune_top_llm_layers,
            tune_llm=config.tune_llm,
            backbone_trainable_params_fp32=config.backbone_trainable_params_fp32,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vlln=config.tune_vlln,
            num_inference_timesteps=config.num_inference_timesteps,
            backbone_use_lora=config.backbone_use_lora,
            action_use_lora=config.action_use_lora,
            optim=config.optim,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            use_bf16=config.use_bf16,
            compile_model=config.compile_model,
            gradient_checkpointing=config.gradient_checkpointing,
            color_jitter_params=config.color_jitter_params,
            clip_outliers=config.clip_outliers,
        )
    def get_delta_timestamps(
        self,
        dataset: Any = None,
        *,
        repo_id: str | None = None,
        root: str | Path | None = None,
        revision: str | None = None,
        fps: float | None = None,
        image_keys: str | list[str] | None = None,
        obs_state_key: str = "observation.state",
    ) -> dict[str, list[float]]:
        """Build ``LeRobotDataModule`` delta timestamps for the VTC video window.

        Wires the config's ``video_length`` / ``video_stride`` / ``chunk_size``
        into per-camera frame offsets so the datamodule returns ``video_length``
        frames per step (offsets ``[-6, -4, -2, 0]`` for the defaults). Pass the
        result to ``LeRobotDataModule(delta_timestamps=...)`` so training feeds
        the backbone the same multi-frame stack the released FT checkpoints saw.

        Camera keys and fps are read from the dataset metadata, so the caller
        normally passes only the dataset (or its ``repo_id``) -- every camera in
        the dataset gets the window automatically, no per-view key lists.

        Args:
            dataset: A built dataset (``LeRobotDataset`` / Studio adapter) whose
                metadata provides the camera keys and fps.
            repo_id: Dataset repo id, used to load metadata when ``dataset`` is
                not given.
            root: Local dataset root for the ``repo_id`` lookup.
            revision: Dataset git revision for the ``repo_id`` lookup.
            fps: Override the dataset fps (defaults to the metadata fps).
            image_keys: Override the auto-detected camera keys (rarely needed).
            obs_state_key: State observation key.

        Returns:
            A ``delta_timestamps`` dict mapping each camera key to the video
            window, plus the state and action offsets.

        Examples:
            >>> policy = Rldx1()
            >>> dt = policy.get_delta_timestamps(repo_id="<dataset>")  # keys auto-detected
            >>> datamodule = LeRobotDataModule(repo_id="<dataset>", delta_timestamps=dt)
        """
        from physicalai.data.lerobot import get_rldx1_delta_timestamps  # noqa: PLC0415

        return get_rldx1_delta_timestamps(
            fps=fps,
            obs_image_key=image_keys,
            obs_state_key=obs_state_key,
            dataset=dataset,
            repo_id=repo_id,
            root=root,
            revision=revision,
            video_length=self.config.video_length,
            video_stride=self.config.video_stride,
            action_horizon=self.config.chunk_size,
        )

    def _initialize_model(
        self,
        env_action_dim: int,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
    ) -> None:
        """Build the model (and preprocessors) for both init paths.

        Args:
            env_action_dim: Environment action dimension.
            dataset_stats: Dataset normalization statistics.
        """
        from .transforms import make_rldx1_transforms  # noqa: PLC0415

        config = self.config
        self.model = Rldx1Model.from_pretrained(
            config.base_model_path,
            revision=config.revision,
            attn_implementation=config.attn_implementation,
            use_bf16=config.use_bf16,
            chunk_size=config.chunk_size,
            n_action_steps=config.n_action_steps,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            select_layer=config.select_layer,
            n_cog_tokens=config.n_cog_tokens,
            num_inference_timesteps=config.num_inference_timesteps,
            compile_model=config.compile_model,
            gradient_checkpointing=config.gradient_checkpointing,
            # Fine-tuning / PEFT control -> bridged onto the vendored RLDXNetworkConfig.
            # Use independent backbone/action PEFT control with fallback to master switch.
            backbone_peft_mode="lora" if config.backbone_use_lora else "full",
            tune_top_llm_layers=config.tune_top_llm_layers,
            tune_llm=config.tune_llm,
            backbone_trainable_params_fp32=config.backbone_trainable_params_fp32,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            use_vlln=config.use_vlln,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vlln=config.tune_vlln,
            backbone_lora_rank=config.backbone_lora_rank,
            backbone_lora_alpha=config.backbone_lora_alpha,
            backbone_lora_dropout=config.backbone_lora_dropout,
            backbone_lora_targets=config.backbone_lora_targets,
            action_peft_mode="lora" if config.action_use_lora else "full",
            action_lora_rank=config.action_lora_rank,
            action_lora_alpha=config.action_lora_alpha,
            action_lora_dropout=config.action_lora_dropout,
            action_lora_targets=config.action_lora_targets,
        )
        self._preprocessor, self._postprocessor = make_rldx1_transforms(
            stats=dataset_stats,  # type: ignore[arg-type]
            env_action_dim=env_action_dim,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=config.chunk_size,
            model_name=config.model_name,
            revision=config.revision,
            use_percentiles=config.use_percentiles,
            clip_outliers=config.clip_outliers,
            image_max_area=config.image_max_area,
            image_min_area=config.image_min_area,
            image_resize_m=config.image_resize_m,
            random_crop_fraction=config.random_crop_fraction,
            random_rotation_angle=config.random_rotation_angle,
            color_jitter_params=config.color_jitter_params,
            embodiment_id=config.embodiment_id,
        )
        self._is_setup_complete = True ## ??? is it neceesary???

    def setup(self, stage: str) -> None:  # noqa: ARG002
        """Lazy-init the model from the datamodule before fit/validate/test.

        Skips if already initialized via the eager path.

        Args:
            stage: Lightning stage ('fit', 'validate', 'test', 'predict').

        Raises:
            TypeError: If the train dataset is not a physicalai Dataset.
        """
        if self._is_setup_complete or self.model is not None:
            return

        from physicalai.data import Dataset  # noqa: PLC0415

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai.data.Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        dataset_stats = train_dataset.stats
        action_features = train_dataset.action_features
        env_action_dim = 0
        for feature in action_features.values():
            if feature.shape:
                env_action_dim = feature.shape[0]
                break

        self.hparams["env_action_dim"] = env_action_dim
        self.hparams["dataset_stats"] = dataset_stats
        self._initialize_model(env_action_dim, dataset_stats)

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Forward pass: training loss in train mode, action chunk in eval.

        Args:
            batch: Input observation batch.

        Returns:
            Training: ``(loss, loss_dict)``. Eval: action chunk tensor.

        Raises:
            RuntimeError: If the model has not been initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model not initialized. Call trainer.fit() or pass env_action_dim."
            raise RuntimeError(msg)
        if not self.training:
            return self.predict_action_chunk(batch)
        preprocessed = self._preprocessor(batch)
        return self.model.compute_loss(preprocessed)

    def training_step(self, batch: Observation, batch_idx: int) -> torch.Tensor:
        """Training step that computes and logs the optimization loss.

        Args:
            batch: Input observation batch.
            batch_idx: Batch index (unused).

        Returns:
            Scalar training loss tensor.
        """
        del batch_idx

        loss, loss_dict = self(batch)
        self.log("train/loss", loss_dict["loss"], prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute validation loss for the current batch.

        Args:
            batch: Input observation batch.

        Returns:
            Tuple of ``(loss, loss_dict)``.

        Raises:
            RuntimeError: If model or preprocessor has not been initialized.
        """
        if self.model is None or self._preprocessor is None:
            msg = "Model not initialized. Call trainer.fit() or pass env_action_dim."
            raise RuntimeError(msg)
        preprocessed = self._preprocessor(batch)
        return self.model.compute_val_loss(preprocessed)

    def configure_optimizers(self) -> dict[str, Any]:
        """Create the configured optimizer and a cosine-decay-with-warmup scheduler.

        The optimizer is selected by ``config.optim``:
        ``"adamw_torch"``/``"adamw_torch_fused"`` use AdamW; ``"adafactor"`` uses
        transformers' Adafactor with a fixed learning rate to cut optimizer memory.

        The LR schedule matches pi05: linear warmup for ``warmup_ratio`` of total
        steps, then cosine decay from ``learning_rate`` down to
        ``config.scheduler_decay_lr`` over the remaining training steps.

        Returns:
            Lightning optimizer configuration dictionary.

        Raises:
            RuntimeError: If model has not been initialized.
        """
        if self.model is None:
            msg = "Model not initialized. Call trainer.fit() or pass env_action_dim."
            raise RuntimeError(msg)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(trainable_params)

        total_steps = 10000
        if hasattr(self, "trainer") and self.trainer is not None:
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", total_steps))
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        scheduler = cosine_decay_with_warmup_scheduler(
            optimizer,
            peak_lr=self.config.learning_rate,
            decay_lr=self.config.scheduler_decay_lr,
            num_warmup_steps=warmup_steps,
            num_decay_steps=total_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _build_optimizer(
        self, params: list[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        """Build the optimizer selected by ``config.optim``.

        Args:
            params: Trainable parameters to optimize.

        Returns:
            The instantiated optimizer.

        Raises:
            ValueError: If ``config.optim`` is not a supported value.
        """
        optim = self.config.optim
        if optim in ("adamw_torch", "adamw_torch_fused"):
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
                fused=optim == "adamw_torch_fused",
            )
        if optim == "adafactor":
            # Fixed-LR Adafactor: factoring the second moment cuts optimizer
            # state memory. relative_step/scale_parameter disabled so the
            # external warmup scheduler drives the LR.
            from transformers.optimization import Adafactor  # noqa: PLC0415

            return Adafactor(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        msg = f"Unsupported optim {optim!r}; expected one of 'adamw_torch', 'adamw_torch_fused', 'adafactor'."
        raise ValueError(msg)

    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict a chunk of actions of shape ``(B, T, D)``.

        Assembles the VTC temporal window (``video_length`` frames at
        ``video_stride`` env-steps, offsets ``[-6, -4, -2, 0]`` for the
        defaults) so the backbone receives the same multi-frame stack it was
        trained on. A batch that already carries a temporal axis (the training /
        validation ``delta_timestamps`` path) is passed through unchanged; a
        single-frame rollout observation is stacked from the per-view history
        buffer maintained by :meth:`select_action` (see :meth:`_apply_video_window`).

        Args:
            batch: Input observation batch.

        Returns:
            Action chunk tensor of shape ``(B, chunk_size, action_dim)``.

        Raises:
            RuntimeError: If the model has not been initialized.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model not initialized. Call trainer.fit() or pass env_action_dim."
            raise RuntimeError(msg)
        self.model.eval()
        model_input = self._prepare_video_window(batch)
        preprocessed = self._preprocessor(model_input)
        preprocessed = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in preprocessed.items()
        }
        actions = self.model.get_action(preprocessed)
        return self._postprocessor(actions)

    # -- VTC rollout frame stacking --------------------------------------- #

    def reset(self) -> None:
        """Reset the policy state at an episode boundary.

        Clears the action queue (base) and the VTC frame-history buffer so the
        next episode's video window is rebuilt from scratch.
        """
        super().reset()
        self._frame_history = None

    def select_action(self, batch: Observation) -> torch.Tensor:
        """Select a single action, recording the frame for the VTC window.

        Records the current observation into the per-view history every
        env-step (regardless of the action queue) so
        :meth:`predict_action_chunk` can sample the ``[-6, -4, -2, 0]`` temporal
        window from the correct env-step strides, then delegates to the base
        action-chunking logic.

        During the warmup phase (buffer not yet full), returns zero actions to
        allow the frame history to accumulate. This matches training conditions
        where observations have complete temporal structure from delta_timestamps.

        Args:
            batch: Input observation batch.

        Returns:
            Single action tensor of shape ``(B, D)`` or ``(D,)``.
        """
        self._record_video_frames(batch)
        
        # Warmup phase: delay inference until buffer is full so temporal window is stable.
        # During training, delta_timestamps ensures observations have complete temporal
        # structure from the first step. During eval rollout, we build frame history
        # incrementally, so the temporal window shifts until we have enough frames.
        # This check ensures we only inference when the model sees a window matching
        # training conditions (full temporal context, not growing).
        if self._frame_history is not None:
            # Get any view key to check buffer fullness
            view_keys = list(self._frame_history.keys())
            if view_keys:
                buffer = self._frame_history[view_keys[0]]
                buffer_len = len(buffer)
                # Warmup needed until buffer is full (accounts for video_stride gaps)
                target_len = (self.config.video_length - 1) * self.config.video_stride + 1
                if buffer_len < target_len:
                    env_action_dim = self.hparams.get("env_action_dim", self.config.max_action_dim)
                    return self._get_warmup_hold_action(batch, env_action_dim)
        
        return super().select_action(batch)

    def _prepare_video_window(self, batch: Observation | dict[str, Any]) -> Observation | dict[str, Any]:
        """Return a model input with the VTC temporal window applied.

        Passes the batch through unchanged when the window is a single frame
        (``video_length <= 1``), when there are no camera views, or when the
        batch already carries a temporal axis (training / validation
        ``delta_timestamps`` path). Otherwise stacks the per-view history into a
        ``(B, T, C, H, W)`` window.

        Returns:
            The original batch, or a flat dict with stacked multi-frame views.
        """
        if self.config.video_length <= 1:
            return batch

        batch_dict = self._to_flat_dict(batch)
        view_keys = self._image_keys(batch_dict)
        if not view_keys or self._is_multiframe(batch_dict, view_keys):
            return batch

        # Direct call that bypassed ``select_action`` (e.g. eval ``forward``):
        # seed the history with the current frame so the window is well-defined.
        if self._frame_history is None or not all(key in self._frame_history for key in view_keys):
            self._record_video_frames(batch_dict, view_keys)

        return self._apply_video_window(batch_dict, view_keys)

    def _record_video_frames(
        self,
        batch: Observation | dict[str, Any],
        view_keys: list[str] | None = None,
    ) -> None:
        """Append the current per-view frames to the rollout history buffer.

        No-op when the window is a single frame, when there are no camera views,
        or when the batch is already multi-frame (temporal axis present).

        Args:
            batch: Current observation (single frame per view).
            view_keys: Precomputed camera keys; resolved from ``batch`` when None.
        """
        if self.config.video_length <= 1:
            return

        batch_dict = self._to_flat_dict(batch)
        if view_keys is None:
            view_keys = self._image_keys(batch_dict)
        if not view_keys or self._is_multiframe(batch_dict, view_keys):
            return

        if self._frame_history is None:
            span = (self.config.video_length - 1) * self.config.video_stride
            self._frame_history = {key: deque(maxlen=span + 1) for key in view_keys}

        for key in view_keys:
            self._frame_history[key].append(self._as_frame_tensor(batch_dict[key]))

    def _apply_video_window(
        self,
        batch_dict: dict[str, Any],
        view_keys: list[str],
    ) -> dict[str, Any]:
        """Stack each view's history into a ``(B, T, C, H, W)`` VTC window.

        Samples the history at offsets ``[(i - (L - 1)) * S : i in [0, L)]``
        (``[-6, -4, -2, 0]`` for the defaults), clamping to the oldest available
        frame when the episode is shorter than the window span -- matching the
        upstream reset-fill behaviour.

        Returns:
            A shallow copy of ``batch_dict`` with the view arrays replaced by
            their stacked multi-frame windows.
        """
        vl = self.config.video_length
        vs = self.config.video_stride
        offsets = [(i - (vl - 1)) * vs for i in range(vl)]

        out = dict(batch_dict)
        assert self._frame_history is not None  # noqa: S101 - populated in _prepare_video_window
        for key in view_keys:
            buffer = self._frame_history[key]
            count = len(buffer)
            frames = [buffer[max(0, count - 1 + offset)] for offset in offsets]
            out[key] = torch.stack(frames, dim=1)  # (B, T, C, H, W)
        return out

    def _get_warmup_hold_action(self, batch: Observation, env_action_dim: int) -> torch.Tensor:
        """Return a safe warmup action while frame history is filling.

        For PushT-like position-control tasks, action represents the target
        position. Returning zeros would command motion toward ``(0, 0)``.
        During warmup we instead try to hold the current state position.

        Args:
            batch: Current observation batch.
            env_action_dim: Real environment action dimension.

        Returns:
            Warmup action tensor of shape ``(B, env_action_dim)``.
        """
        batch_size = batch.batch_size

        state = batch.state
        if isinstance(state, torch.Tensor):
            state_t = state
        elif isinstance(state, dict):
            state_t = None
            for key in ("agent_pos", "state"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    state_t = value
                    break
        else:
            state_t = None

        if state_t is not None:
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)
            if state_t.shape[-1] >= env_action_dim:
                return state_t[..., :env_action_dim].to(device=self.device, dtype=torch.bfloat16)

        # Fallback for tasks without compatible state-to-action mapping.
        return torch.zeros(batch_size, env_action_dim, dtype=torch.bfloat16, device=self.device)

    @staticmethod
    def _to_flat_dict(batch: Observation | dict[str, Any]) -> dict[str, Any]:
        """Return a flat observation dict for both Observation and dict inputs.

        Returns:
            The flattened observation dict.
        """
        if isinstance(batch, Observation):
            return batch.to_dict(flatten=True)
        return dict(batch)

    @staticmethod
    def _image_keys(batch_dict: dict[str, Any]) -> list[str]:
        """Return the camera view keys, reusing the preprocessor's detection.

        Returns:
            Sorted list of image keys (possibly empty).
        """
        from .transforms import Rldx1Preprocessor  # noqa: PLC0415

        return Rldx1Preprocessor._image_keys(batch_dict)  # noqa: SLF001

    @staticmethod
    def _as_frame_tensor(value: Any) -> torch.Tensor:  # noqa: ANN401
        """Coerce a single-frame view value to a ``(B, C, H, W)`` tensor.

        Returns:
            A torch tensor view of ``value``.
        """
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(np.asarray(value))

    @staticmethod
    def _is_multiframe(batch_dict: dict[str, Any], view_keys: list[str]) -> bool:
        """Return True when the views already carry a temporal axis.

        A batched single frame is ``(B, C, H, W)`` (4-D); a batched VTC window is
        ``(B, T, C, H, W)`` (5-D). The rollout always supplies batched
        observations, so a 5-D view means the ``delta_timestamps`` path already
        produced the window and no history stacking is needed.

        Returns:
            True when the first view is 5-D (already multi-frame).
        """
        value = batch_dict[view_keys[0]]
        ndim = value.dim() if isinstance(value, torch.Tensor) else np.asarray(value).ndim
        return ndim == 5  # noqa: PLR2004 - (B, T, C, H, W)
