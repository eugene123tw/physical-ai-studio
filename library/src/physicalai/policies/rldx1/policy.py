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
from physicalai.policies.rldx1 import Rldx1, Rldx1Config
from physicalai.train import Trainer

policy = Rldx1(
    base_model_path="RLWRLD/RLDX-1-PT",
    use_lora=True,
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

from physicalai.policies.base import Policy
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler

from .config import Rldx1Config
from .model import Rldx1Model

if TYPE_CHECKING:
    from pathlib import Path

    from physicalai.data import Observation

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
        use_lora: Master LoRA switch. True -> LoRA on both the backbone top
            layers and the MSAT action model; False -> full fine-tune instead.
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        warmup_ratio: Warmup ratio (0.0-1.0) of total training steps.
        grad_clip_norm: Gradient clipping norm (0.0 = disabled).
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
        use_lora: bool = True,
        # Optimizer
        optim: Literal["adamw_torch", "adamw_torch_fused", "adafactor"] = "adamw_torch",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_ratio: float = 0.05,
        grad_clip_norm: float = 1.0,
        # Precision / compilation
        use_bf16: bool = True,
        compile_model: bool = False,
        gradient_checkpointing: bool = True,
        # Image augmentation (train only)
        color_jitter_params: dict[str, float] | None = None,
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
            use_lora=use_lora,
            optim=optim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            grad_clip_norm=grad_clip_norm,
            use_bf16=use_bf16,
            compile_model=compile_model,
            gradient_checkpointing=gradient_checkpointing,
            color_jitter_params=color_jitter_params,
            clip_outliers=clip_outliers,
        )

        # Save individual args (not the config object) for checkpoint restoration.
        self.save_hyperparameters(ignore=["config"])
        self.hparams["config"] = self.config.to_dict()

        self.model: Rldx1Model | None = None
        self._preprocessor: Rldx1Preprocessor | None = None
        self._postprocessor: Rldx1Postprocessor | None = None
        self._is_setup_complete: bool = False

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
            use_lora=config.use_lora,
            optim=config.optim,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            grad_clip_norm=config.grad_clip_norm,
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
            # use_lora=False => full fine-tune the backbone top layers + MSAT.
            backbone_peft_mode="lora" if config.use_lora else "full",
            tune_top_llm_layers=config.tune_top_llm_layers,
            tune_llm=config.tune_llm,
            backbone_trainable_params_fp32=config.backbone_trainable_params_fp32,
            tune_visual=config.tune_visual,
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vlln=config.tune_vlln,
            backbone_lora_rank=config.backbone_lora_rank,
            backbone_lora_alpha=config.backbone_lora_alpha,
            backbone_lora_dropout=config.backbone_lora_dropout,
            backbone_lora_targets=config.backbone_lora_targets,
            action_peft_mode="lora" if config.use_lora else "full",
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
        preprocessed = self._preprocessor(batch)
        preprocessed = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in preprocessed.items()
        }
        actions = self.model.get_action(preprocessed)
        return self._postprocessor(actions)
