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
    action_model_use_lora=True,     # LoRA on MSAT (r=64)
)

datamodule = LeRobotDataModule(repo_id="<user dataset>", train_batch_size=4)
trainer = Trainer(max_steps=60000, precision="bf16-mixed")
trainer.fit(policy, datamodule)
```
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError
from physicalai.inference.data import InferenceFeature, InferenceFeatureDtype, InferenceFeatureType
from physicalai.inference.manifest import ComponentSpec
from transformers.optimization import Adafactor

from physicalai.data import Dataset, Observation
from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Feature, FeatureType
from physicalai.export import ExportablePolicyMixin, ExportBackend
from physicalai.export.backends import (
    ExportParameters,
    ONNXExportParameters,
    OpenVINOExportParameters,
    TorchExportParameters,
)
from physicalai.policies.base import Policy
from physicalai.policies.rldx1.config import Rldx1Config
from physicalai.policies.rldx1.model import Rldx1Model
from physicalai.policies.rldx1.pretrained_utils import (
    extract_camera_names,
    extract_dataset_stats,
    retrieve_safetensors_shards,
)
from physicalai.policies.utils.features import infer_shape_from_stats
from physicalai.train.schedulers import cosine_decay_with_warmup_scheduler
from physicalai.train.utils import reformat_dataset_to_match_policy

try:
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
except ImportError:
    OptimizerLRScheduler = Any  # type: ignore[assignment, misc]

from .preprocessor import make_rldx1_transforms
from .vtc_buffer import VtcWindowBuffer

if TYPE_CHECKING:
    from collections.abc import Generator
    from os import PathLike

    from .preprocessor import Rldx1Postprocessor, Rldx1Preprocessor

logger = logging.getLogger(__name__)


def _merge_explicit_features(
    dataset_stats: dict[str, dict[str, Any]] | None,
    input_features: dict[str, Feature] | None,
    output_features: dict[str, Feature] | None,
) -> dict[str, dict[str, Any]] | None:
    """Merge explicit ``Feature`` overrides into a ``dataset_stats``-shaped dict.

    User-supplied features take precedence over anything already in
    ``dataset_stats`` (e.g. auto-fetched state/action stats) -- required for
    RLWRLD checkpoints, whose ``statistics.json`` never records camera shapes.

    Returns:
        The merged dict, or ``None`` if there is nothing to merge.
    """
    merged = dict(dataset_stats or {})
    for name, feature in {**(input_features or {}), **(output_features or {})}.items():
        if feature.shape is None:
            continue
        if feature.ftype == FeatureType.VISUAL:
            key = f"observation.{IMAGES}.{name}"
        elif feature.ftype == FeatureType.STATE:
            key = f"observation.{STATE}"
        elif feature.ftype == FeatureType.ACTION:
            key = ACTION
        else:
            key = name
        merged[key] = {"name": feature.name or name, "shape": feature.shape, "type": str(feature.ftype)}
    return merged or None


def _resolve_feature_shape(feature: dict[str, Any]) -> tuple[int, ...]:
    """Return a feature's shape, raising if it can't be inferred from stats.

    RLDX1's own ``extract_dataset_stats`` (used when loading a raw HF release
    checkpoint via ``_from_hf``) returns bare ``min``/``max``/``mean``/``std``/
    ``q01``/``q99`` vectors with no ``"shape"`` key at all -- unlike the
    LeRobot-style enriched stats (e.g. a Studio-trained checkpoint's full
    ``train_dataset.stats``, or an explicit ``input_features``/``output_features``
    override) which carry an explicit ``"shape"``. Both are valid dataset_stats
    entries for this policy; :func:`infer_shape_from_stats` handles both.

    Returns:
        The feature's shape as a tuple.

    Raises:
        ValueError: If neither ``"shape"`` nor a stat vector is present.
    """
    shape = infer_shape_from_stats(feature)
    if shape is None:
        msg = f"Cannot resolve a shape for feature {feature!r}: no 'shape' key and no stat vector to infer it from."
        raise ValueError(msg)
    return shape


def _get_dataset_stats_entry(dataset_stats: dict[str, dict[str, Any]], *keys: str) -> dict[str, Any]:
    """Return the first present entry among candidate ``dataset_stats`` keys.

    ``extract_dataset_stats`` (raw HF release checkpoints) uses bare keys like
    ``"state"``; a Studio-trained checkpoint's full LeRobot-style stats use
    ``"observation.state"``. Callers pass both spellings as candidates.

    Returns:
        The matching stats dict.

    Raises:
        KeyError: If none of ``keys`` is present in ``dataset_stats``.
    """
    for key in keys:
        if key in dataset_stats:
            return dataset_stats[key]
    msg = f"None of {keys!r} found in dataset_stats (keys present: {sorted(dataset_stats)!r})"
    raise KeyError(msg)


class Rldx1(ExportablePolicyMixin, Policy):
    """RLDX-1 Policy - first-party Lightning wrapper.

    All hyperparameters are explicit in the signature for discoverability.
    Supports the dual-path initialization shared across Studio policies:

    - **Lazy path**: ``Rldx1()`` + ``trainer.fit()`` - model built in ``setup()``
      once dataset features are known.
    - **Eager path**: ``Rldx1.load_from_checkpoint()`` or ``Rldx1(env_action_dim=...)``
      - model built immediately.

    Args:
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
            Default ``False`` here, diverging from upstream's ``True``.
            TODO(Eugene): upstream defaults this to ``True``, but the fp32
            copies of trainable backbone params OOM on an A100. DeepSpeed
            ZeRO-Offload (CPU) avoids the OOM but is very slow in practice.
            Explore a better way (e.g. selective fp32 casting, ZeRO-3 without
            offload, or partial offload) to re-enable ``True`` by default.
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
        gradient_checkpointing: Whether to enable activation checkpointing in
            MSAT during training.
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
        input_features: Explicit observation feature overrides (e.g. camera shapes), merged
            into ``dataset_stats`` and taking precedence over it. Required for export when
            ``dataset_stats`` carries no visual entries -- true for every RLWRLD release
            checkpoint (their ``statistics.json`` is pretrain-stage state/action stats only;
            no file in the repo records pixel resolution). Camera *names* are auto-discovered
            from the checkpoint's ``processor_config.json`` in :meth:`_from_hf` (see
            ``self._camera_names``) purely to guide this override -- only the shape must be
            supplied. For ``RLWRLD/RLDX-1-FT-LIBERO``:
            ``{"front_view": Feature(ftype=FeatureType.VISUAL, shape=(3, 256, 256)),
            "left_wrist_view": Feature(ftype=FeatureType.VISUAL, shape=(3, 256, 256))}``.
        output_features: Explicit action feature overrides, merged into ``dataset_stats``
            the same way as ``input_features``.
    """

    def __init__(  # noqa: PLR0913
        self,
        # Model architecture
        n_action_steps: int = 16,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
        # Model source
        pretrained_name_or_path: str | None = "RLWRLD/RLDX-1-PT",
        revision: str | None = None,
        # Backbone
        attn_implementation: Literal["sdpa", "flash_attention_2"] = "sdpa",
        # Fine-tuning control
        *,
        tune_top_llm_layers: int = 6,
        tune_llm: bool = False,
        backbone_trainable_params_fp32: bool = False,
        tune_visual: bool = True,
        tune_projector: bool = True,
        use_vlln: bool = False,
        tune_diffusion_model: bool = True,
        tune_vlln: bool = True,
        num_inference_timesteps: int = 4,
        backbone_use_lora: bool = False,
        action_model_use_lora: bool = False,
        # Optimizer
        optim: Literal["adamw_torch", "adamw_torch_fused", "adafactor"] = "adamw_torch",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        scheduler_decay_lr: float = 1e-5,
        # Precision / compilation
        use_bf16: bool = True,
        gradient_checkpointing: bool = True,
        # VTC video window
        video_length: int = 4,
        video_stride: int = 2,
        # Image geometry
        image_min_area: int | None = None,
        # Normalization
        clip_outliers: bool = True,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]] | None = None,
        embodiment_tag: str = "general_embodiment",
        input_features: dict[str, Feature] | None = None,
        output_features: dict[str, Feature] | None = None,
        tokenizer_max_length: int = 512,
    ) -> None:
        """Initialize the RLDX-1 policy and save hyperparameters."""
        super().__init__(n_action_steps=n_action_steps)
        self._camera_names: list[str] = []

        shard_files = None
        if pretrained_name_or_path is not None:
            # dataset_stats already provided (e.g. restored from a checkpoint's
            # hparams during load_from_checkpoint) takes precedence -- _from_hf's
            # own extract_dataset_stats() is only a narrow, defaulted fallback.
            self.config, hf_dataset_stats, shard_files, self._camera_names = self._from_hf(
                pretrained_name_or_path,
                revision=revision,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                attn_implementation=attn_implementation,
                # Fine-tuning
                tune_diffusion_model=tune_diffusion_model,
                use_vlln=use_vlln,
                tune_vlln=tune_vlln,
                tune_top_llm_layers=tune_top_llm_layers,
                tune_llm=tune_llm,
                backbone_trainable_params_fp32=backbone_trainable_params_fp32,
                tune_visual=tune_visual,
                tune_projector=tune_projector,
                num_inference_timesteps=num_inference_timesteps,
                backbone_use_lora=backbone_use_lora,
                action_model_use_lora=action_model_use_lora,
                # optimizer
                optim=optim,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                scheduler_decay_lr=scheduler_decay_lr,
                # Precision / compilation
                use_bf16=use_bf16,
                gradient_checkpointing=gradient_checkpointing,
                # VTC video window
                video_length=video_length,
                video_stride=video_stride,
                # Image geometry
                image_min_area=image_min_area,
                # Normalization
                clip_outliers=clip_outliers,
                # Action prediciton
                action_horizon=n_action_steps,
                embodiment_tag=embodiment_tag,
                # tokenizer config
                tokenizer_max_length=tokenizer_max_length,
            )
            if dataset_stats is None:
                dataset_stats = hf_dataset_stats
        else:
            self.config = Rldx1Config(
                base_model_path=pretrained_name_or_path,
                revision=revision,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                attn_implementation=attn_implementation,
                # Fine-tuning
                tune_diffusion_model=tune_diffusion_model,
                use_vlln=use_vlln,
                tune_vlln=tune_vlln,
                tune_top_llm_layers=tune_top_llm_layers,
                tune_llm=tune_llm,
                backbone_trainable_params_fp32=backbone_trainable_params_fp32,
                tune_visual=tune_visual,
                tune_projector=tune_projector,
                num_inference_timesteps=num_inference_timesteps,
                backbone_use_lora=backbone_use_lora,
                action_model_use_lora=action_model_use_lora,
                # optimizer
                optim=optim,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                scheduler_decay_lr=scheduler_decay_lr,
                # Precision / compilation
                use_bf16=use_bf16,
                gradient_checkpointing=gradient_checkpointing,
                # VTC video window
                video_length=video_length,
                video_stride=video_stride,
                # Image geometry
                image_min_area=image_min_area,
                # Normalization
                clip_outliers=clip_outliers,
                action_horizon=n_action_steps,
                embodiment_tag=embodiment_tag,
                tokenizer_max_length=tokenizer_max_length,
            )

        # Save `pretrained_name_or_path` so load_from_checkpoint() reconstructs
        # from the same base repo the checkpoint was actually fine-tuned from,
        # instead of silently falling back to this constructor's default.
        self.save_hyperparameters(ignore=["config"])

        self.model: Rldx1Model | None = None  # type: ignore[assignment]
        self._preprocessor: Rldx1Preprocessor | None = None
        self._postprocessor: Rldx1Postprocessor | None = None

        # Per-view VTC frame buffer for rollout. Populated every env-step via
        # ``select_action``; ``prepare`` assembles the temporal window for
        # ``predict_action_chunk``. Cleared on ``reset``.
        self._vtc_buffer = VtcWindowBuffer(
            video_length=self.config.video_length,
            video_stride=self.config.video_stride,
        )
        # Explicit Feature overrides win over anything auto-fetched/user-supplied above --
        # required for RLWRLD checkpoints, which never record camera shapes anywhere.
        dataset_stats = _merge_explicit_features(dataset_stats, input_features, output_features)


        # TODO(Eugene): hardcode this first
        self.config.export_dynamic_prompt = True

        # TODO(Eugene): Make this nicer        
        if input_features is not None:
            num_views = 0
            for feature in input_features.values():
                if feature.ftype == FeatureType.VISUAL:
                    num_views += 1
                self.config.num_views = num_views

        self._dataset_stats = dataset_stats
        if dataset_stats is not None:
            self._initialize_model(dataset_stats, shard_files)

    def _from_hf(  # noqa: PLR6301, PLR0913, PLR0917
        self,
        pretrained_name_or_path: str,
        revision: str | None,
        max_state_dim: int,
        max_action_dim: int,
        attn_implementation: Literal["sdpa", "flash_attention_2"],
        # Fine-tuning
        tune_diffusion_model: bool,  # noqa: FBT001
        use_vlln: bool,  # noqa: FBT001
        tune_vlln: bool,  # noqa: FBT001
        tune_top_llm_layers: int,
        tune_llm: bool,  # noqa: FBT001
        backbone_trainable_params_fp32: bool,  # noqa: FBT001
        tune_visual: bool,  # noqa: FBT001
        tune_projector: bool,  # noqa: FBT001
        num_inference_timesteps: int,
        backbone_use_lora: bool,  # noqa: FBT001
        action_model_use_lora: bool,  # noqa: FBT001
        # optimizer
        optim: Literal["adamw_torch", "adamw_torch_fused", "adafactor"],
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        scheduler_decay_lr: float,
        # Precision / compilation
        use_bf16: bool,  # noqa: FBT001
        gradient_checkpointing: bool,  # noqa: FBT001
        # VTC video window
        video_length: int,
        video_stride: int,
        # Image geometry
        image_min_area: int | None,
        # Normalization
        clip_outliers: bool,  # noqa: FBT001
        embodiment_tag: str,
        # Action prediction
        action_horizon: int,
        # Tokenizer
        tokenizer_max_length: int,
    ) -> tuple[Rldx1Config, dict[str, dict[str, list[float] | str | tuple]], list[Path], list[str]]:
        config_file = Path(hf_hub_download(pretrained_name_or_path, "config.json", revision=revision))  # nosec B615
        shard_files = retrieve_safetensors_shards(pretrained_name_or_path, revision=revision)
        try:
            stats_file = Path(hf_hub_download(pretrained_name_or_path, "processor/statistics.json", revision=revision))  # nosec B615
        except RemoteEntryNotFoundError:
            try:
                stats_file = Path(hf_hub_download(pretrained_name_or_path, "statistics.json", revision=revision))  # nosec B615
            except RemoteEntryNotFoundError as e2:
                msg = "statistics.json not found in the root of the repo. Falling back to processor/statistics.json"
                raise RuntimeError(msg) from e2
        try:
            processor_config_file = Path(
                hf_hub_download(pretrained_name_or_path, "processor_config.json", revision=revision),  # nosec B615
            )
        except RemoteEntryNotFoundError:
            processor_config_file = None

        # --- parse config.json ---
        with Path(config_file).open(encoding="utf-8") as f:
            hf_config = json.load(f)

        hf_config["base_model_path"] = pretrained_name_or_path
        hf_config["backbone_trainable_params_fp32"] = backbone_trainable_params_fp32
        hf_config["max_state_dim"] = max_state_dim
        hf_config["max_action_dim"] = max_action_dim
        hf_config["attn_implementation"] = attn_implementation
        hf_config["tune_diffusion_model"] = tune_diffusion_model
        hf_config["use_vlln"] = use_vlln
        hf_config["tune_vlln"] = tune_vlln
        hf_config["tune_top_llm_layers"] = tune_top_llm_layers
        hf_config["tune_llm"] = tune_llm
        hf_config["tune_visual"] = tune_visual
        hf_config["tune_projector"] = tune_projector
        hf_config["num_inference_timesteps"] = num_inference_timesteps
        hf_config["backbone_use_lora"] = backbone_use_lora
        hf_config["action_model_use_lora"] = action_model_use_lora
        hf_config["optim"] = optim
        hf_config["learning_rate"] = learning_rate
        hf_config["weight_decay"] = weight_decay
        hf_config["warmup_ratio"] = warmup_ratio
        hf_config["scheduler_decay_lr"] = scheduler_decay_lr
        hf_config["use_bf16"] = use_bf16
        hf_config["gradient_checkpointing"] = gradient_checkpointing
        hf_config["video_length"] = video_length
        hf_config["video_stride"] = video_stride
        hf_config["clip_outliers"] = clip_outliers
        hf_config["image_min_area"] = image_min_area
        hf_config["action_horizon"] = action_horizon
        hf_config["embodiment_tag"] = embodiment_tag
        hf_config["tokenizer_max_length"] = tokenizer_max_length

        # strict=False: ignore upstream config.json keys with no Rldx1Config field
        # (e.g. architectures, model_type, rtc_inference_*) instead of denylisting
        # them one by one.
        config = Rldx1Config.from_dict(hf_config, strict=False)

        # --- build dataset_stats from HF artefacts ---
        dataset_stats = extract_dataset_stats(
            stats_file,
            embodiment_tag=embodiment_tag,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
        )
        camera_names = extract_camera_names(processor_config_file, embodiment_tag=embodiment_tag)
        return config, dataset_stats, shard_files, camera_names

    def _initialize_model(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
        shard_files: list[Path] | None = None,
    ) -> None:
        """Build the model (and preprocessors) for both init paths.

        Args:
            dataset_stats: Dataset normalization statistics.
            shard_files: List of shard files containing model weights.
        """
        config: Rldx1Config = self.config
        self.model = Rldx1Model(
            base_model_path=config.base_model_path,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=config.action_horizon,
            attn_implementation=config.attn_implementation,
            num_inference_timesteps=config.num_inference_timesteps,
            use_vlln=config.use_vlln,
            tune_projector=config.tune_projector,
            tune_visual=config.tune_visual,
            tune_vlln=config.tune_vlln,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_top_llm_layers=config.tune_top_llm_layers,
            tune_llm=config.tune_llm,
            backbone_trainable_params_fp32=config.backbone_trainable_params_fp32,
            backbone_use_lora=config.backbone_use_lora,
            action_model_use_lora=config.action_model_use_lora,
            gradient_checkpointing=config.gradient_checkpointing,
            diffusion_model_cfg=config.diffusion_model_cfg,
            backbone_lora_target_modules=config.backbone_lora_target_modules,
            action_model_lora_target_modules=config.action_model_lora_target_modules,
            video_length=config.video_length,
            video_stride=config.video_stride,
        )

        if shard_files is not None:
            self.model.load_sharded_weights(shard_files)

        if config.use_bf16:
            self.model.to(torch.bfloat16)

            if config.backbone_trainable_params_fp32:
                for _, p in self.model.backbone.named_parameters():
                    if p.requires_grad:
                        p.data = p.data.to(torch.float32)

        self._preprocessor, self._postprocessor = make_rldx1_transforms(
            stats=dataset_stats,  # type: ignore[arg-type]
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=config.action_horizon,
            revision=config.revision,
            use_percentiles=config.use_percentiles,
            clip_outliers=config.clip_outliers,
            image_max_area=config.image_max_area,
            image_min_area=config.image_min_area or 0,  # type: ignore[arg-type]
            image_resize_m=config.image_resize_m,
            embodiment_id=int(config.embodiment_id),  # type: ignore[arg-type]
            max_token_len=config.tokenizer_max_length,
        )

    def setup(self, stage: str) -> None:
        """Set up model from datamodule (lazy or fine-tuning path).

        Called by Lightning before fit/validate/test/predict.

        - **Lazy path**: model is None → build model + preprocessors from dataset stats.
        - **Fine-tuning path**: model already loaded from pretrained → rebuild
          preprocessors with the training dataset's stats so normalization
          matches the new data distribution.

        Raises:
            TypeError: If the train dataset is not a physicalai Dataset.
        """
        del stage

        datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
        train_dataset = datamodule.train_dataset

        if not isinstance(train_dataset, Dataset):
            msg = f"Expected physicalai Dataset, got {type(train_dataset)}"
            raise TypeError(msg)

        stats_dict = train_dataset.stats

        if self.model is not None:
            # Fine-tuning path: model exists from pretrained, but the
            # preprocessor stats must match the training data distribution.
            self._update_preprocessor_stats(stats_dict)
            reformat_dataset_to_match_policy(self, datamodule)
            return

        self.hparams["dataset_stats"] = stats_dict

        self._initialize_model(stats_dict)

        reformat_dataset_to_match_policy(self, datamodule)

    def _update_preprocessor_stats(
        self,
        dataset_stats: dict[str, dict[str, list[float] | str | tuple]],
    ) -> None:
        """Rebuild preprocessor/postprocessor with new dataset stats.

        Used when fine-tuning a pretrained model on a new dataset: the model
        weights come from the checkpoint, but normalization statistics must
        reflect the training data.
        """
        logger.info("Updating preprocessor stats for fine-tuning dataset")
        config = self.config
        self._preprocessor, self._postprocessor = make_rldx1_transforms(
            stats=dataset_stats,  # type: ignore[arg-type]
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
            action_horizon=config.action_horizon,
            revision=config.revision,
            use_percentiles=config.use_percentiles,
            clip_outliers=config.clip_outliers,
            image_max_area=config.image_max_area,
            image_min_area=config.image_min_area or 0,  # type: ignore[arg-type]
            image_resize_m=config.image_resize_m,
            embodiment_id=int(config.embodiment_id),  # type: ignore[arg-type]
            max_token_len=config.tokenizer_max_length,
        )
        self._dataset_stats = dataset_stats
        self.hparams["dataset_stats"] = dataset_stats

    def forward(self, batch: Observation) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
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

    def compute_val_loss(self, batch: Observation) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
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

    def configure_optimizers(self) -> OptimizerLRScheduler:  # type: ignore[override]
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

        total_steps = int(self.trainer.estimated_stepping_batches)
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
        self,
        params: list[torch.nn.Parameter],
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
        if optim in {"adamw_torch", "adamw_torch_fused"}:
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
                fused=optim == "adamw_torch_fused",
            )
        if optim == "adafactor":
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
            Action chunk tensor of shape ``(B, n_action_steps, action_dim)``.

        Raises:
            RuntimeError: If the model has not been initialized.
        """
        if self.model is None or self._preprocessor is None or self._postprocessor is None:
            msg = "Model not initialized. Call trainer.fit() or pass env_action_dim."
            raise RuntimeError(msg)
        self.model.eval()
        model_input = self._vtc_buffer.prepare(batch)
        preprocessed = self._preprocessor(model_input)
        preprocessed = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in preprocessed.items()}
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

        Args:
            batch: Input observation batch.

        Returns:
            Single action tensor of shape ``(B, D)`` or ``(D,)``.
        """
        self._vtc_buffer.record(batch)

        # A malformed or externally managed history may be incomplete. Normal
        # rollout histories are seeded with their first frame and skip this path.
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

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        """Get a list of export backends supported by policy.

        This method returns a list of supported export backends as strings.

        Returns:
            list[str | ExportBackend]: A list of supported export backends.
        """
        return [ExportBackend.TORCH, ExportBackend.OPENVINO]

    @property
    def inputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's expected model inputs for export tracing.

        Returns:
            A list of feature descriptors matching the model's expected input format,
            covering the robot state, image observations, language task, and any
            real-time chunking control tensors when ``enable_rtc`` is set on the model.
            Returns ``None`` if the underlying model or dataset stats have not been
            initialized yet.
        """
        if self.model is None or self._dataset_stats is None:
            return None

        dataset_stats = self._dataset_stats

        schema: list[InferenceFeature] = []

        num_image_features = sum(
            1 for feature in dataset_stats.values() if str(FeatureType.VISUAL) in str(feature.get("type", ""))
        )
        for feature_id, feature in dataset_stats.items():
            feature_type = str(feature.get("type", ""))
            if STATE in feature_id:
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.STATE,
                        shape=infer_shape_from_stats(feature),
                        name=STATE,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )
            elif str(FeatureType.VISUAL) in feature_type:
                feature_name = (
                    str(feature.get("name", feature_id)).removeprefix("observation.").removeprefix(f"{IMAGES}.")
                )
                name = IMAGES if num_image_features == 1 else f"{IMAGES}.{feature_name}"
                schema.append(
                    InferenceFeature(
                        ftype=InferenceFeatureType.VISUAL,
                        shape=infer_shape_from_stats(feature),
                        name=name,
                        dtype=InferenceFeatureDtype.FLOAT32,
                    ),
                )

        if num_image_features == 0:
            msg = (
                "dataset_stats carries no visual features. Pass input_features={'<view>': "
                "Feature(ftype=FeatureType.VISUAL, shape=(3, height, width)), ...} to "
                f"Rldx1(...) to export this policy. Camera names discovered from "
                f"processor_config.json: {self._camera_names or '(none found)'}."
            )
            raise ValueError(msg)

        schema.append(
            InferenceFeature(
                ftype=InferenceFeatureType.LANGUAGE,
                shape=(),
                name=TASK,
                dtype=InferenceFeatureDtype.STRING,
            ),
        )

        return schema

    @property
    def outputs_schema(self) -> list[InferenceFeature] | None:
        """Describe the policy's model output for export.

        Returns:
            A list with a single ``action`` feature of shape
            ``(chunk_size, *action_dim)``, where ``action_dim`` is the actual
            action dimension taken from the dataset stats. Returns ``None`` if the
            underlying model or dataset stats have not been initialized yet.
        """
        if self.model is None or self._dataset_stats is None:
            return None

        action_shape = _resolve_feature_shape(_get_dataset_stats_entry(self._dataset_stats, ACTION))

        return [
            InferenceFeature(
                ftype=InferenceFeatureType.ACTION,
                shape=(self.config.action_horizon, *action_shape),
                name=ACTION,
                dtype=InferenceFeatureDtype.FLOAT32,
            ),
        ]

    def extra_export_args(self) -> dict[str, ExportParameters]:
        """Additional export arguments for model conversion.

        Returns:
            dict[str, ExportParameters]: A dictionary mapping format names to their export parameters.

        Raises:
            ValueError: If dataset stats are not available for export.
        """
        if self._dataset_stats is None:
            msg = (
                "Dataset stats are required for export. Initialize the policy with dataset_stats"
                " or train for at least one epoch to populate them."
            )
            raise ValueError(msg)

        # RLWRLD release checkpoints never carry visual entries in dataset_stats
        # (see inputs_schema) unless input_features merged one in -- find the first
        # visual entry, whatever its key (single observation.images or per-camera
        # observation.images.<name>); the Runtime rldx1 preprocessor assumes one
        # shared resolution across views.
        image_resolution = next(
            (
                tuple(feature["shape"])[-2:]
                for feature in self._dataset_stats.values()
                if str(FeatureType.VISUAL) in str(feature.get("type", ""))
            ),
            None,
        )
        if image_resolution is None:
            msg = (
                "dataset_stats carries no visual features. Pass input_features={'<view>': "
                "Feature(ftype=FeatureType.VISUAL, shape=(3, height, width)), ...} to "
                "Rldx1(...) to export this policy."
            )
            raise ValueError(msg)

        base_preproc_specs = [
            ComponentSpec(
                type="normalize",
                stats={STATE: _get_dataset_stats_entry(self._dataset_stats, f"observation.{STATE}", STATE)},
                mode="quantiles",
            ),
            ComponentSpec(
                type="rldx1",
                image_resolution=image_resolution,
            ),
        ]
        postproc_specs = [
            ComponentSpec(
                type="denormalize",
                stats={ACTION: self._dataset_stats[ACTION]},
                mode="quantiles",
            ),
        ]


        # Dynamic-prompt export keeps input_ids dynamic, so the runtime rebuilds
        # position_ids / attention_mask (numpy get_rope_index) after tokenization.
        rope_specs: list[ComponentSpec] = []
        if self.config.export_dynamic_prompt and self.model is not None:
            # During export self.model may be a GraphSafeRldx1Model (gs_backbone)
            # or the regular Rldx1Model (backbone).
            backbone = getattr(self.model, "backbone", None) or getattr(self.model, "gs_backbone", None)
            qwen_config = backbone.qwen_config
            rope_specs = [
                ComponentSpec(
                    type="rldx1_rope",
                    image_token_id=qwen_config.image_token_id,
                    vision_start_token_id=qwen_config.vision_start_token_id,
                    spatial_merge_size=qwen_config.vision_config.spatial_merge_size,
                    n_cog_tokens=self.config.n_cog_tokens,
                ),
            ]

        # TODO(Eugene): make this a bit generic?
        openvino_input_names = list(self.model.input_keys)

        extra_args: dict[str, ExportParameters] = {}
        output_names = [feature.name for feature in (self.outputs_schema or [])]
        extra_args["onnx"] = ONNXExportParameters(
            exporter_kwargs={
                "output_names": output_names,
            },
            export_tokenizer=False,
            preprocessors_specs=[
                *base_preproc_specs,
                ComponentSpec(
                    type="hf_tokenizer",
                    tokenizer_name="RLWRLD/RLDX-1-VLM",
                    max_token_len=self.config.tokenizer_max_length,
                ),
                *rope_specs,
            ],
            postprocessors_specs=postproc_specs,
        )
        extra_args["openvino"] = OpenVINOExportParameters(
            inputs=openvino_input_names,
            outputs=output_names,
            compress_to_fp16=True,
            via_onnx=False,
            export_tokenizer=True,
            exporter_kwargs={},
            preprocessors_specs=[
                *base_preproc_specs,
                ComponentSpec(
                    type="ov_tokenizer",
                    artifact="tokenizer.xml",
                ),
                *rope_specs,
            ],
            postprocessors_specs=postproc_specs,
        )
        extra_args["torch"] = TorchExportParameters(
            preprocessors_specs=[ComponentSpec(type="to_float_tensor")],
            postprocessors_specs=[],
        )

        return extra_args

    # ------------------------------------------------------------------ #
    # Export (graph-safe tracing)                                        #
    # ------------------------------------------------------------------ #
    # RLDX-1's eager forward runs data-dependent ops (fast_pos_embed_interpolate,
    # rot_pos_emb, cu_seqlens, VTC token compression, M-RoPE get_rope_index) that
    # torch.export / torch.onnx.export cannot trace. For tracing backends we swap
    # ``self.model`` for a graph-safe view that shares the trained parameters by
    # reference and precomputes every data-dependent tensor from the eager sample.

    def _build_graph_safe_model(
        self,
        model: Rldx1Model,
        input_sample: dict[str, torch.Tensor],
    ) -> torch.nn.Module:
        """Build the export-only graph-safe view over a trained model.

        The returned module shares parameters with ``model`` by reference (no
        state-dict copy) and precomputes all data-dependent buffers from
        ``input_sample`` in its constructor, so its ``forward(batch)`` is a
        static graph. Imported lazily to keep the graph-safe port off the
        training import path.

        Returns:
            An ``nn.Module`` with the same ``forward(batch)`` contract as
            :class:`Rldx1Model`, safe to hand to the tracer.
        """
        from physicalai.policies.rldx1.components.backbone.graph_safe_rldx1 import GraphSafeRldx1Model

        outputs_schema = self.outputs_schema or []
        action_dim = self.config.max_action_dim
        if outputs_schema and outputs_schema[0].shape:
            action_dim = int(outputs_schema[0].shape[-1])

        return GraphSafeRldx1Model(model, input_sample, self.config, output_action_dim=action_dim)

    @contextmanager
    def _graph_safe_export_model(self, input_sample: dict[str, torch.Tensor]) -> Generator[None, None, None]:
        """Temporarily swap ``self.model`` for its graph-safe export view.

        Restores the trained model on exit (including on error) so exporting
        never mutates the in-memory model.

        Raises:
            RuntimeError: If the model has not been initialized yet.
        """
        if self.model is None:
            msg = "Cannot export before the model is initialized (call setup / load a checkpoint first)."
            raise RuntimeError(msg)
        original = self.model
        graph_safe = self._build_graph_safe_model(original, input_sample)
        try:
            self.model = graph_safe  # type: ignore[assignment]
            yield
        finally:
            # Undo any in-place submodule swaps before restoring the trained model.
            if hasattr(graph_safe, "restore"):
                graph_safe.restore()
            self.model = original

    @torch.no_grad()
    def to_onnx(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: Any,
    ) -> None:
        """Export to ONNX, tracing the graph-safe view of the model."""
        if input_sample is None:
            input_sample = self._get_default_export_input_sample()
        with self._graph_safe_export_model(input_sample or {}):
            traced_sample = self._trim_export_sample(input_sample)
            super().to_onnx(output_path, input_sample=traced_sample, **export_kwargs)

    @torch.no_grad()
    def to_openvino(
        self,
        output_path: PathLike | str,
        input_sample: dict[str, torch.Tensor] | None = None,
        **export_kwargs: Any,
    ) -> None:
        """Export to OpenVINO, tracing the graph-safe view of the model."""
        if input_sample is None:
            input_sample = self._get_default_export_input_sample()
        with self._graph_safe_export_model(input_sample or {}):
            # TODO(Eugene): find a tidier way
            traced_sample = self._trim_export_sample(input_sample)
            super().to_openvino(output_path, input_sample=traced_sample, **export_kwargs)

    # TODO(Eugene): would there be a better way than trimming? Assuming the input data are succinct?
    def _trim_export_sample(
        self,
        input_sample: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor] | None:
        """Trim the export sample to the inputs the static graph consumes.

        The graph-safe model bakes ``input_ids`` / ``image_grid_thw`` / position
        ids into static buffers, so they are not graph inputs. Passing them to
        the tracer makes OpenVINO's ``input=`` shapes reference tensors absent
        from the converted model. Restrict the sample to the model's declared
        ``input_keys``.

        Returns:
            The sample filtered to the consumed inputs (unchanged when the model
            declares no ``input_keys``).
        """
        keys = getattr(self.model, "input_keys", None)
        if not keys or input_sample is None:
            return input_sample
        # Dynamic-prompt export may attach a padded prompt sample. Only copy the
        # static prompt tensors from that sample; keep live observation tensors
        # (pixel_values/state) from input_sample.
        source = dict(input_sample)
        export_sample = getattr(self.model, "export_sample", None)
        if export_sample is not None:
            for key in ("input_ids", "position_ids", "attention_mask"):
                if key in export_sample:
                    source[key] = export_sample[key]
        return {key: source[key] for key in keys if key in source}
