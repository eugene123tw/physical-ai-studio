# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the RLDX-1 policy.

This module provides the dataclass configuration for the RLDX-1 policy, a
flow-matching Vision-Language-Action model built on a Qwen3-VL-8B backbone and
a Multi-Stream Action Transformer (MSAT) action head.

The configuration mirrors the **pre-train (PT) shape** of the upstream RLDX
config (RLWRLD/RLDX-1), vendored here as ``RLDXNetworkConfig``. It deliberately
omits the mid-train add-on streams (motion / memory / physics) and the RECAP RL
plumbing, which are
deferred to phase 2. See ``library/docs/rldx-1-integration.md`` for the full
scope decision.

Architecture dimensions (MSAT depth, attention heads, ``diffusion_model_cfg``,
etc.) are read from the checkpoint ``config.json`` at load time, following the
same pattern as :class:`physicalai.policies.groot.GrootConfig`. This config
holds the Studio-level knobs: model source, fine-tuning / PEFT control, and
flow-matching sampling parameters.

For CLI usage, use a YAML config under ``configs/physicalai/``:

    physicalai fit --config configs/physicalai/rldx1-ft-default.yaml

Example (API):
    >>> from physicalai.policies.rldx1 import Rldx1Config
    >>> config = Rldx1Config(
    ...     base_model_path="RLWRLD/RLDX-1-PT",
    ...     use_lora=True,
    ...     action_lora_rank=64,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from physicalai.config import Config


# Size of the per-embodiment projector bank in RLDX-1-PT (``W.shape[0]`` of each
# CategorySpecificLinear). Valid ``embodiment_id`` slots are ``[0, 36)``.
MAX_NUM_EMBODIMENTS = 36


@dataclass
class Rldx1Config(Config):
    """Configuration for the RLDX-1 policy (PT -> FT path only).

    RLDX-1 is RLWRLD's flow-matching VLA. v1 supports the single reproducible
    training path: fine-tuning ``RLWRLD/RLDX-1-PT`` on a LeRobot dataset, the
    same recipe that produced every released ``RLDX-1-FT-*`` checkpoint.

    The motion / memory / physics add-on streams and the RECAP RL trainer are
    out of v1 scope; their toggle fields are kept (defaulting to ``False``) only
    so the loader can tolerate FT ``config.json`` files that carry them. Setting
    any add-on to ``True`` raises -- see :meth:`__post_init__`.

    Attributes:
        chunk_size: Number of action predictions per forward pass (action horizon).
        n_action_steps: Number of action steps to execute per chunk.
        max_state_dim: Maximum state dimension (shorter states zero-padded).
        max_action_dim: Maximum action dimension (shorter actions zero-padded).
        base_model_path: HuggingFace model ID or local path to the base checkpoint.
        revision: Pinned git commit SHA for the base checkpoint download. Should be
            a concrete SHA, never a branch name (lib.security rule 9).
        model_name: HuggingFace ID of the Qwen3-VL backbone.
        embodiment_id: Per-embodiment projector slot in the MSAT action head.
            Either a slot ``int`` in ``[0, 36)`` or a tag-name ``str`` (e.g.
            ``"fractal20220817_data"``) resolved via
            ``EMBODIMENT_TAG_TO_PROJECTOR_INDEX``. Default 0 (general_embodiment)
            for a fresh new-robot fine-tune; set the slot a released checkpoint
            was trained on (0/1/3) to load it. Normalized to an int at init.
        select_layer: Index of the VLM hidden layer used as cognition features.
        backbone_embedding_dim: Backbone hidden dimension projected into the action head.
        attn_implementation: Attention backend ('sdpa', 'flash_attention_2', 'eager').
        n_cog_tokens: Number of cognition tokens routed from the backbone to MSAT.
        tune_top_llm_layers: Number of top LLM layers to fine-tune (lower layers frozen).
        tune_llm: Whether to fine-tune the entire LLM backbone (all decoder layers,
            input embeddings, and lm_head). Overrides ``tune_top_llm_layers``.
        backbone_trainable_params_fp32: Whether to cast trainable backbone
            parameters to float32 after bf16 loading for optimizer stability.
        tune_visual: Whether to fine-tune the vision tower.
        tune_projector: Whether to fine-tune the cognition/state/action projectors.
        tune_diffusion_model: Whether to fine-tune the MSAT action model.
        tune_vlln: Whether to fine-tune the VLM-output layer norm.
        use_lora: Master LoRA switch. True -> LoRA adapters on both the backbone
            top layers and the MSAT action model; False -> full fine-tune the
            backbone top layers and (when tune_diffusion_model) the MSAT.
        backbone_lora_rank: LoRA rank for the backbone (used when use_lora=True).
        backbone_lora_alpha: LoRA alpha for the backbone.
        backbone_lora_dropout: LoRA dropout for the backbone.
        backbone_lora_targets: Linear module names to wrap with LoRA in the backbone.
        action_lora_rank: LoRA rank for the action model (paper App. D free-lunch default).
        action_lora_alpha: LoRA alpha for the action model.
        action_lora_dropout: LoRA dropout for the action model.
        action_lora_targets: Linear module names to wrap with LoRA in the action model.
        num_inference_timesteps: Number of flow-matching denoising steps at inference.
        noise_beta_alpha: Alpha of the Beta distribution used to sample flow time.
        noise_beta_beta: Beta of the Beta distribution used to sample flow time.
        noise_s: Time-shift parameter for the flow-matching schedule.
        num_timestep_buckets: Number of buckets for the timestep embedding.
        state_dropout_prob: Probability of dropping the state input during training.
        state_additive_noise_scale: Scale of additive Gaussian noise on state features.
        image_max_area: Target max area (pixels) for aspect-preserving image resize.
        image_resize_m: Alignment multiple for the resized/cropped image dimensions.
        video_length: Number of VTC temporal frames per observation step.
        video_stride: Action-step stride between VTC video frames.
        random_crop_fraction: Train-time fractional crop size in ``(0, 1]``
            (``None`` disables the crop stage).
        random_rotation_angle: Train-time rotation limit in degrees (``None`` /
            ``0`` disables rotation).
        color_jitter_params: Train-time ``A.ColorJitter`` params (``None``
            disables color jitter).
        use_relative_action: Must be False in v1. Relative actions are not supported.
        use_percentiles: Whether to normalize with 1st/99th percentiles (vs min/max).
        clip_outliers: Whether to clip normalized state/action to ``[-1, 1]`` (upstream
            ``clip_outliers``). ``True`` (default) matches the upstream RLDX-1 recipe:
            training targets and decoded actions are clamped to the normalization
            bounds. Set ``False`` (Pi05-style, no clip) for wide-range action spaces
            where ``QUANTILES`` bounds would truncate task-critical extremes (e.g.
            PushT). Gates both the train-time clip and the inference denormalize clamp.
        rtc_inference_mode: Real-Time Chunking inference mode ('none', 'trained').
        rtc_training_max_delay: Max prefix delay sampled per step during RTC training.
        rtc_inference_delay: Inference-time prefix delay for RTC.
        rtc_inference_exec_horizon: RTC execution horizon (0 => action_horizon - delay).
        use_memory: Phase-2 memory stream. Must be False in v1.
        use_motion: Phase-2 motion stream. Must be False in v1.
        use_physics: Phase-2 physics stream. Must be False in v1.
        optim: Optimizer to build in ``configure_optimizers``. ``"adamw_torch"``
            (default) and ``"adamw_torch_fused"`` keep full Adam moment state;
            ``"adafactor"`` factors the second moment to cut optimizer memory
            (roughly halves optimizer state) at a small quality/throughput cost.
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        warmup_ratio: Warmup ratio (0.0-1.0) of total training steps.
        scheduler_decay_lr: Final learning rate after cosine decay. The LR is
            cosine-decayed from ``learning_rate`` down to this value over the
            remaining (post-warmup) training steps.
        grad_clip_norm: Gradient clipping norm (0.0 = disabled).
        use_bf16: Whether to use bfloat16 precision.
        compile_model: Whether to torch.compile the model.
        gradient_checkpointing: Whether to enable activation checkpointing in
            the MSAT action model during training.

    Examples:
        LoRA on both the backbone top layers and the MSAT action model (default):

        >>> config = Rldx1Config()
        >>> config.use_lora
        True

        Full fine-tune instead of LoRA:

        >>> config = Rldx1Config(use_lora=False)
    """

    # Model architecture / action chunking
    chunk_size: int = 16  # action_horizon
    n_action_steps: int = 16
    max_state_dim: int = 64
    max_action_dim: int = 64

    # Model source
    base_model_path: str = "RLWRLD/RLDX-1-PT"
    revision: str | None = None
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"

    # Embodiment projector slot in the MSAT action head. Default 0
    # (general_embodiment), the slot RLDX-1-PT reserves and pre-conditions for
    # downstream fine-tuning (highest-norm projector in PT); every released FT
    # used it. Accepts either a slot int in [0, 36) or a tag-name string resolved
    # via EMBODIMENT_TAG_TO_PROJECTOR_INDEX. To load a released benchmark
    # checkpoint, set the slot it was trained on:
    #   0  general_embodiment   -> FT-ROBOCASA, FT-RC365, FT-LIBERO, FT-GR1
    #   1  fractal20220817_data -> FT-SIMPLER-GOOGLE
    #   3  bridge_orig          -> FT-SIMPLER-WIDOWX
    # (35 = new_embodiment is the legacy GR00T slot, superseded by general_embodiment.)
    # Normalized to an int in __post_init__.
    embodiment_id: int | str = 0

    # Backbone
    select_layer: int = 18
    backbone_embedding_dim: int = 4096
    attn_implementation: str = "sdpa"
    n_cog_tokens: int = 64

    # Fine-tuning control (full / partial)
    tune_top_llm_layers: int = 4
    tune_llm: bool = False
    backbone_trainable_params_fp32: bool = True
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # LoRA master switch. True -> LoRA adapters on both the backbone top layers
    # and the MSAT action model; False -> full fine-tune (backbone top layers,
    # plus the MSAT when tune_diffusion_model=True). Paper App. D / Table 6.
    use_lora: bool = True

    # Backbone (Qwen3-VL top layers) LoRA hyperparameters (used when use_lora).
    backbone_lora_rank: int = 64
    backbone_lora_alpha: int = 64
    backbone_lora_dropout: float = 0.0
    backbone_lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Action model (MSAT) LoRA hyperparameters (used when use_lora). r=64 is the
    # paper-recommended free lunch. Targets are MSAT block module names (V-L /
    # state-action / physics QKV + output projections and the MMDiT inner FFN
    # linears), not Qwen attention names. Absent targets (e.g. p_qkv/p_proj when
    # physics is disabled) are filtered before the PEFT call.
    action_lora_rank: int = 64
    action_lora_alpha: int = 64
    action_lora_dropout: float = 0.0
    action_lora_targets: tuple[str, ...] = (
        "vl_qkv",
        "vl_proj",
        "sa_qkv",
        "sa_proj",
        "p_qkv",
        "p_proj",
        "linear1",
        "linear2",
    )

    # Flow matching
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # State augmentation
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0

    # Image / language pipeline
    image_max_area: int = 65536  # 256 * 256
    image_resize_m: int = 32
    # VTC video window: each camera view carries ``video_length`` temporal frames
    # sampled at ``video_stride`` action-steps (offsets [-6, -4, -2, 0]). Used to
    # build ``delta_timestamps`` (see ``get_rldx1_delta_timestamps``); the
    # released FT checkpoints were all trained with 4 frames at stride 2.
    video_length: int = 4
    video_stride: int = 2
    # Train-time image augmentation (upstream ReplayCompose). All default off so
    # eval / inference is deterministic; set to reproduce an FT training recipe.
    # One sampled param set is shared across a sample's frames and views.
    random_crop_fraction: float | None = None
    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_relative_action: bool = False
    use_percentiles: bool = True
    # Clip normalized state/action to [-1, 1] (upstream clip_outliers). True keeps
    # upstream parity; False (Pi05-style) preserves out-of-percentile action tails
    # for wide-range tasks like PushT. Gates both the train clip and infer clamp.
    clip_outliers: bool = True

    # Real-Time Chunking (optional; no released checkpoint enables it).
    # "guided" mode is intentionally unsupported (autograd VJP, not exportable).
    rtc_inference_mode: Literal["none", "trained"] = "none"
    rtc_training_max_delay: int = 0
    rtc_inference_delay: int = 0
    rtc_inference_exec_horizon: int = 0

    # Phase-2 add-on streams. Kept so FT configs that carry them load cleanly;
    # must remain False in v1.
    use_memory: bool = False
    use_motion: bool = False
    use_physics: bool = False

    # Optimizer / training hyperparameters
    optim: Literal["adamw_torch", "adamw_torch_fused", "adafactor"] = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    scheduler_decay_lr: float = 1e-5
    grad_clip_norm: float = 1.0

    # Precision / compilation
    use_bf16: bool = True
    compile_model: bool = False
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        """Enforce the v1 scope boundary and normalize ``embodiment_id``.

        Raises:
            NotImplementedError: If a phase-2 add-on stream or the unsupported
                RTC ``guided`` mode is requested.
            ValueError: If ``embodiment_id`` is an unknown tag name or an int
                outside ``[0, MAX_NUM_EMBODIMENTS)``.
        """
        for name in ("use_memory", "use_motion", "use_physics"):
            if getattr(self, name):
                msg = (
                    f"Rldx1Config.{name}=True is a phase-2 feature and is not "
                    "supported in v1. See library/docs/rldx-1-integration.md."
                )
                raise NotImplementedError(msg)

        if self.use_relative_action:
            msg = (
                "Rldx1Config.use_relative_action=True is not supported in v1; "
                "only absolute actions are implemented."
            )
            raise NotImplementedError(msg)

        self.embodiment_id = self._resolve_embodiment_id(self.embodiment_id)

    @staticmethod
    def _resolve_embodiment_id(value: int | str) -> int:
        """Resolve an embodiment tag name or slot index to a projector slot int.

        Args:
            value: Either a projector slot ``int`` in ``[0, MAX_NUM_EMBODIMENTS)``
                or a tag-name ``str`` key of ``EMBODIMENT_TAG_TO_PROJECTOR_INDEX``
                (e.g. ``"fractal20220817_data"``).

        Returns:
            The resolved projector slot index.

        Raises:
            ValueError: If a string tag is unknown or an int is out of range.
        """
        if isinstance(value, str):
            from physicalai.policies.rldx1.components.embodiments import (  # noqa: PLC0415
                EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
            )

            try:
                return EMBODIMENT_TAG_TO_PROJECTOR_INDEX[value]
            except KeyError:
                known = ", ".join(EMBODIMENT_TAG_TO_PROJECTOR_INDEX)
                msg = (
                    f"Unknown embodiment tag {value!r}. Known tags: {known}. "
                    f"Alternatively pass an int slot in [0, {MAX_NUM_EMBODIMENTS})."
                )
                raise ValueError(msg) from None
        if not 0 <= value < MAX_NUM_EMBODIMENTS:
            msg = (
                f"embodiment_id={value} is out of range; must be a slot int in "
                f"[0, {MAX_NUM_EMBODIMENTS}) or a known tag name."
            )
            raise ValueError(msg)
        return value

    @property
    def action_horizon(self) -> int:
        """Alias for chunk_size (action horizon)."""
        return self.chunk_size
