# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

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
from typing import TYPE_CHECKING, Any, Literal

import torch
from physicalai.train.utils import reformat_dataset_to_match_policy
from physicalai.data import Dataset
from physicalai.data import Observation
from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler

from .config import Rldx1Config
from .model import Rldx1Model
from .vtc_buffer import VtcWindowBuffer

from .preprocessor import make_rldx1_transforms  # noqa: PLC0415

if TYPE_CHECKING:
    from .preprocessor import Rldx1Postprocessor, Rldx1Preprocessor

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
        scheduler_decay_lr: Final learning rate after cosine decay (default 1e-5).
        use_bf16: Whether to use bfloat16 precision.
        compile_model: Whether to torch.compile the model.
        gradient_checkpointing: Whether to enable activation checkpointing in
            MSAT during training.
        color_jitter_params: Train-time ``A.ColorJitter`` params
            (``{"brightness", "contrast", "saturation", "hue"}``). ``None``
            (default) disables color augmentation.
        video_length: Number of VTC temporal frames per observation step (default 4).
        video_stride: Action-step stride between VTC video frames (default 2).
            With ``video_length=4, video_stride=2`` the offsets are ``[-6,-4,-2,0]``
            (600 ms at 10 fps). Set ``video_stride=1`` for contiguous frames.
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
        scheduler_decay_lr: float = 1e-5,
        # Precision / compilation
        use_bf16: bool = True,
        compile_model: bool = False,
        gradient_checkpointing: bool = True,
        # VTC video window
        video_length: int = 4,
        video_stride: int = 2,
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
            scheduler_decay_lr=scheduler_decay_lr,
            use_bf16=use_bf16,
            compile_model=compile_model,
            gradient_checkpointing=gradient_checkpointing,
            video_length=video_length,
            video_stride=video_stride,
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

        # Per-view VTC frame buffer for rollout. Populated every env-step via
        # ``select_action``; ``prepare`` assembles the temporal window for
        # ``predict_action_chunk``. Cleared on ``reset``.
        self._vtc_buffer = VtcWindowBuffer(
            video_length=self.config.video_length,
            video_stride=self.config.video_stride,
        )

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
            scheduler_decay_lr=config.scheduler_decay_lr,
            use_bf16=config.use_bf16,
            compile_model=config.compile_model,
            gradient_checkpointing=config.gradient_checkpointing,
            color_jitter_params=config.color_jitter_params,
            clip_outliers=config.clip_outliers,
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
            video_length=config.video_length,
            video_stride=config.video_stride,
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
        del stage 

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai.data.Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats
        action_features = train_dataset.action_features
        env_action_dim = 0
        for feature in action_features.values():
            if feature.shape:
                env_action_dim = feature.shape[0]
                break

        self.hparams["env_action_dim"] = env_action_dim
        self.hparams["dataset_stats"] = stats_dict
        self._initialize_model(env_action_dim, stats_dict)
        reformat_dataset_to_match_policy(self, datamodule)

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

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Remap legacy state-dict keys before loading.

        Checkpoints saved before the action denormalizer was moved from
        ``_preprocessor`` to ``_postprocessor`` carry keys such as::

            _preprocessor._action_denormalizer.buffer_action.q01

        These are remapped to the current layout::

            _postprocessor._action_denormalizer.buffer_action.q01
        """
        state_dict = checkpoint.get("state_dict", {})
        old_prefix = "_preprocessor._action_denormalizer."
        new_prefix = "_postprocessor._action_denormalizer."
        remapped = {
            (new_prefix + k[len(old_prefix):] if k.startswith(old_prefix) else k): v
            for k, v in state_dict.items()
        }
        if remapped != state_dict:
            checkpoint["state_dict"] = remapped

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
        buffer maintained by :meth:`select_action` (see :class:`~physicalai.policies.rldx1.vtc_buffer.VtcWindowBuffer`).

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
        model_input = self._vtc_buffer.prepare(batch)
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
        self._vtc_buffer.reset()

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
        self._vtc_buffer.record(batch)

        # Warmup phase: delay inference until the buffer is full so the temporal
        # window is stable. During training, delta_timestamps ensures complete
        # temporal structure from the first step. During eval rollout, we build
        # frame history incrementally, so we hold until the window matches
        # training conditions.
        if self._vtc_buffer.is_warming_up:
            env_action_dim = self.hparams.get("env_action_dim", self.config.max_action_dim)
            return self._get_warmup_hold_action(batch, env_action_dim)

        return super().select_action(batch)

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
