# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RLDX-1 Model - pure-PyTorch nn.Module (MSAT action head + Qwen3-VL backbone).

This is the exportable model entrypoint for the RLDX-1 policy. It wraps the
Qwen3-VL-8B backbone and the Multi-Stream Action Transformer (MSAT)
flow-matching action head, ported from the upstream RLWRLD/RLDX-1 codebase
(``rldx/model/core/rldx.py``) with the Lightning glue removed so the graph is
clean to export to OpenVINO / ONNX.

Scope (v1): pre-train (PT) -> fine-tune (FT) path only. The motion / memory /
physics add-on streams are not constructed. See
``library/docs/rldx-1-integration.md``.

The architecture itself is the upstream RLDX-1 model code, vendored under
``components/`` (Apache-2.0, attribution preserved). This module is the thin
Studio-facing wrapper that builds / loads that network and adapts its
forward / action interface to the :class:`~physicalai.policies.base.Model`
contract.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from physicalai.policies.base import Model

if TYPE_CHECKING:
    from collections.abc import Mapping

    from physicalai.policies.rldx1.components.core_rldx import RLDX

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_PATH = "RLWRLD/RLDX-1-PT"
DEFAULT_BACKBONE_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Safe allowlist for files pulled from the checkpoint repo (lib.security rule 8).
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md"]


class Rldx1Model(Model):
    """RLDX-1 Vision-Language-Action model (MSAT + Qwen3-VL).

    Pure ``nn.Module`` wrapping the Qwen3-VL backbone and the MSAT
    flow-matching action head. Can be used standalone for inference / export,
    or wrapped in the :class:`~physicalai.policies.rldx1.policy.Rldx1` Lightning
    policy for training.

    Args:
        chunk_size: Action horizon (number of actions predicted per pass).
        n_action_steps: Number of action steps to execute per chunk.
        max_state_dim: Maximum state dimension (shorter states zero-padded).
        max_action_dim: Maximum action dimension (shorter actions zero-padded).
        select_layer: VLM hidden layer used as cognition features.
        backbone_embedding_dim: Backbone hidden dim projected into the action head.
        n_cog_tokens: Number of cognition tokens routed to MSAT.
        attn_implementation: Attention backend ('sdpa', 'flash_attention_2', 'eager').
        num_inference_timesteps: Flow-matching denoising steps at inference.
        use_bf16: Whether to use bfloat16 compute.
        compile_model: Whether to ``torch.compile`` the model.
        gradient_checkpointing: Whether to enable activation checkpointing in
            the MSAT action model during training.
    """

    def __init__(
        self,
        chunk_size: int = 16,
        n_action_steps: int = 16,
        *,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
        select_layer: int = 18,
        backbone_embedding_dim: int = 4096,
        n_cog_tokens: int = 64,
        attn_implementation: str = "sdpa",
        num_inference_timesteps: int = 4,
        use_bf16: bool = True,
        compile_model: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        """Build the RLDX-1 model skeleton (weights loaded via from_pretrained)."""
        super().__init__()
        self._chunk_size = chunk_size
        self._n_action_steps = n_action_steps
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.select_layer = select_layer
        self.backbone_embedding_dim = backbone_embedding_dim
        self.n_cog_tokens = n_cog_tokens
        self.attn_implementation = attn_implementation
        self.num_inference_timesteps = num_inference_timesteps
        self.use_bf16 = use_bf16
        self.compile_model = compile_model
        self.gradient_checkpointing = gradient_checkpointing
        # The vendored RLDX module (backbone + MSAT action head). Populated by
        # ``from_pretrained``; ``None`` until then.
        self.net: RLDX | None = None

    @classmethod
    def from_pretrained(  # noqa: PLR0913
        cls,
        base_model_path: str = DEFAULT_BASE_MODEL_PATH,
        *,
        revision: str | None = None,
        attn_implementation: str = "sdpa",
        use_bf16: bool = True,
        gradient_checkpointing: bool = False,
        # Fine-tuning / PEFT control (bridged onto the vendored RLDXConfig).
        backbone_peft_mode: str = "full",
        tune_top_llm_layers: int = 4,
        tune_llm: bool = False,
        tune_visual: bool = False,
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        tune_vlln: bool = True,
        backbone_lora_rank: int = 64,
        backbone_lora_alpha: int = 64,
        backbone_lora_dropout: float = 0.0,
        backbone_lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        action_peft_mode: str = "lora",
        action_lora_rank: int = 64,
        action_lora_alpha: int = 64,
        action_lora_dropout: float = 0.0,
        action_lora_targets: tuple[str, ...] = (
            "vl_qkv",
            "vl_proj",
            "sa_qkv",
            "sa_proj",
            "p_qkv",
            "p_proj",
            "linear1",
            "linear2",
        ),
        **kwargs: Any,  # noqa: ANN401
    ) -> Rldx1Model:
        """Load RLDX-1 weights from a HuggingFace repo or local path.

        Reads the checkpoint ``config.json`` to recover architecture
        dimensions, builds the model, and loads the ``safetensors`` state dict.

        Args:
            base_model_path: HuggingFace model ID or local path (e.g. RLWRLD/RLDX-1-PT).
            revision: Pinned git commit SHA. Should be a concrete SHA, never a
                branch name (lib.security rule 9).
            attn_implementation: Attention backend.
            use_bf16: Whether to load/compute in bfloat16.
            gradient_checkpointing: Whether to enable activation checkpointing
                in the MSAT action model during training.
            backbone_peft_mode: Backbone fine-tuning mode ('full', 'lora', 'frozen').
                'lora' injects PEFT adapters into the top ``tune_top_llm_layers``
                LLM layers; 'frozen' freezes the backbone entirely.
            tune_top_llm_layers: Number of top LLM layers to adapt (full-tune in
                'full' mode, LoRA scope in 'lora' mode).
            tune_llm: Full-tune the entire LLM backbone (all decoder layers +
                input embeddings + lm_head). Only applies in 'full' mode;
                overrides ``tune_top_llm_layers``.
            tune_visual: Whether to fine-tune the vision tower.
            tune_projector: Whether to fine-tune the cognition/state/action projectors.
            tune_diffusion_model: Whether to full-tune the MSAT action model
                (only applies when ``action_peft_mode='full'``).
            tune_vlln: Whether to fine-tune the VLM-output layer norm.
            backbone_lora_rank: LoRA rank for the backbone.
            backbone_lora_alpha: LoRA alpha for the backbone.
            backbone_lora_dropout: LoRA dropout for the backbone.
            backbone_lora_targets: Linear module names to wrap with LoRA in the backbone.
            action_peft_mode: MSAT action-model fine-tuning mode ('full', 'lora').
            action_lora_rank: LoRA rank for the action model.
            action_lora_alpha: LoRA alpha for the action model.
            action_lora_dropout: LoRA dropout for the action model.
            action_lora_targets: Linear module names to wrap with LoRA in the action model.
            **kwargs: Architecture overrides forwarded to ``__init__``.

        Returns:
            An ``Rldx1Model`` with the vendored ``RLDX`` network loaded.

        Raises:
            ValueError: If the checkpoint contains parameters the model does not
                expect (architecture mismatch).
        """
        # Lazy import: the vendored stack pulls in transformers + diffusers and
        # allocates the 8B backbone, so keep it out of module import time.
        #
        # The Qwen3-VL backbone reads its attention backend from the
        # ``RLDX_ATTN_IMPL`` env var at adapter-import time (upstream default
        # ``flash_attention_2`` requires CUDA + the flash-attn package). Map the
        # requested ``attn_implementation`` onto it *before* importing the
        # vendored stack so CPU / XPU loads default to ``sdpa``.
        import os  # noqa: PLC0415

        backbone_attn = (
            attn_implementation
            if attn_implementation in {"sdpa", "eager", "flash_attention_2"}
            else "sdpa"
        )
        os.environ.setdefault("RLDX_ATTN_IMPL", backbone_attn)

        from physicalai.policies.rldx1.components.config_rldx import RLDXConfig  # noqa: PLC0415
        from physicalai.policies.rldx1.components.core_rldx import RLDX  # noqa: PLC0415
        from physicalai.policies.rldx1.pretrained_utils import (  # noqa: PLC0415
            load_rldx_state_dict,
        )

        cfg: RLDXConfig = RLDXConfig.from_pretrained(base_model_path, revision=revision)
        cfg.diffusion_model_cfg["gradient_checkpointing"] = gradient_checkpointing

        # Bridge the Studio-level fine-tuning / PEFT knobs onto the vendored
        # RLDXConfig. Without this the checkpoint config.json values win and the
        # policy-interface flags (backbone_peft_mode, action_peft_mode, tune_*)
        # are silently ignored -- e.g. backbone_use_lora stays False. Applied
        # before ``RLDX(cfg, ...)`` because the LoRA injection and requires_grad
        # bookkeeping run in the model constructor.
        cfg.tune_visual = tune_visual
        cfg.tune_projector = tune_projector
        cfg.tune_vlln = tune_vlln
        if backbone_peft_mode == "lora":
            cfg.backbone_use_lora = True
            cfg.tune_llm = False
            cfg.tune_top_llm_layers = 0
            cfg.backbone_lora_num_layers = tune_top_llm_layers if tune_top_llm_layers > 0 else -1
            cfg.backbone_lora_rank = backbone_lora_rank
            cfg.backbone_lora_alpha = backbone_lora_alpha
            cfg.backbone_lora_dropout = backbone_lora_dropout
            cfg.backbone_lora_target_modules = list(backbone_lora_targets)
        elif backbone_peft_mode == "frozen":
            cfg.backbone_use_lora = False
            cfg.tune_llm = False
            cfg.tune_top_llm_layers = 0
        else:  # "full"
            cfg.backbone_use_lora = False
            cfg.tune_llm = tune_llm
            cfg.tune_top_llm_layers = tune_top_llm_layers
        if action_peft_mode == "lora":
            cfg.action_model_use_lora = True
            cfg.action_model_lora_rank = action_lora_rank
            cfg.action_model_lora_alpha = action_lora_alpha
            cfg.action_model_lora_dropout = action_lora_dropout
            cfg.action_model_lora_target_modules = list(action_lora_targets)
        else:  # "full"
            cfg.action_model_use_lora = False
            cfg.tune_diffusion_model = tune_diffusion_model

        # pop unused fields in cfg.diffusion_model_cfg that are not in the vendored RLDXConfig
        cfg.diffusion_model_cfg.pop("final_dropout", None)
        cfg.diffusion_model_cfg.pop("use_swiglu", None)

        # The RLDX wrapper holds no attention modules of its own; ``eager``
        # satisfies the transformers attn-dispatch check. The real backbone
        # attention is selected via ``RLDX_ATTN_IMPL`` (set above).
        cfg._attn_implementation = "eager"  # noqa: SLF001

        # Build directly (not via ``RLDX.from_pretrained``) so ``__init__`` runs
        # on the real device: transformers' meta-device lazy init breaks the
        # ``torch.distributions.Beta`` constructed in the action head. Skip the
        # redundant backbone weight download — the checkpoint state dict below
        # supplies the full backbone + action-head weights. ``trust_remote_code``
        # is force-disabled (lib.security): Qwen3-VL is a native transformers
        # model, so no remote code execution is required.
        # LoRA adapters must be injected AFTER the base checkpoint weights are
        # loaded. PEFT wraps each target Linear, renaming its weight
        # (``...q_proj.weight`` -> ``...q_proj.base_layer.weight``); injecting
        # during construction makes every base weight key mismatch on load and
        # trips the ``unexpected`` guard below. Suppress the constructor's
        # auto-injection, load the base weights, then inject on top.
        want_backbone_lora = bool(getattr(cfg, "backbone_use_lora", False))
        want_action_lora = bool(getattr(cfg, "action_model_use_lora", False))
        if want_backbone_lora or want_action_lora:
            logger.info(
                "Deferring LoRA injection until after base checkpoint weights load "
                "(backbone_lora=%s, action_lora=%s).",
                want_backbone_lora,
                want_action_lora,
            )
        cfg.backbone_use_lora = False
        cfg.action_model_use_lora = False

        net = RLDX(
            cfg,
            transformers_loading_kwargs={
                "trust_remote_code": False,
                "skip_pretrained_weights": True,
            },
        )

        dtype = torch.bfloat16 if use_bf16 else torch.float32
        net = net.to(dtype)

        state_dict = load_rldx_state_dict(base_model_path, revision=revision)
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if unexpected:
            msg = (
                f"Checkpoint {base_model_path} contains {len(unexpected)} "
                f"unexpected parameter(s) (architecture mismatch), e.g. "
                f"{unexpected[:5]}"
            )
            raise ValueError(msg)
        # ``missing`` is expected to hold only non-persistent buffers (e.g. RoPE
        # ``inv_freq``) that are recomputed at construction, not trained weights.
        if missing:
            logger.warning(
                "%d parameter(s) not found in checkpoint (expected to be "
                "non-persistent buffers): %s",
                len(missing),
                missing[:5],
            )

        # Inject LoRA on top of the loaded base weights. The base Linear moves to
        # ``base_layer`` (weights preserved); fresh adapters are marked trainable.
        if want_backbone_lora:
            cfg.backbone_use_lora = True
            net._apply_backbone_lora()  # noqa: SLF001
        if want_action_lora:
            cfg.action_model_use_lora = True
            net.action_model.set_trainable_parameters(
                cfg.tune_projector, cfg.tune_diffusion_model, cfg.tune_vlln
            )
        if want_backbone_lora or want_action_lora:
            logger.info(
                "LoRA injection complete after checkpoint load "
                "(backbone_lora=%s, action_lora=%s).",
                want_backbone_lora,
                want_action_lora,
            )

        # Re-assert the fp32 contract on LoRA adapter params: the ``net.to(dtype)``
        # above cast the base network to bf16, so adapters injected afterwards
        # land in bf16 too, whose AdamW state underflows on the first optimizer
        # step and yields NaN losses. Only LoRA params are promoted -- negligible
        # memory, unlike full-tuned trainable weights.
        for pname, p in net.named_parameters():
            if p.requires_grad and any(seg.startswith("lora_") for seg in pname.split(".")):
                p.data = p.data.to(torch.float32)

        net.eval()

        # Allow call-site overrides while avoiding duplicate keyword forwarding
        # when values are provided in ``kwargs`` (e.g. from policy config).
        chunk_size = int(kwargs.pop("chunk_size", cfg.action_horizon))
        n_action_steps = int(
            kwargs.pop("n_action_steps", getattr(cfg, "n_action_steps", cfg.action_horizon))
        )
        max_state_dim = int(kwargs.pop("max_state_dim", cfg.max_state_dim))
        max_action_dim = int(kwargs.pop("max_action_dim", cfg.max_action_dim))
        select_layer = int(kwargs.pop("select_layer", cfg.select_layer))
        backbone_embedding_dim = int(
            kwargs.pop("backbone_embedding_dim", cfg.backbone_embedding_dim)
        )
        n_cog_tokens = int(kwargs.pop("n_cog_tokens", getattr(cfg, "n_cog_tokens", 64)))
        num_inference_timesteps = int(
            kwargs.pop("num_inference_timesteps", cfg.num_inference_timesteps)
        )
        gradient_checkpointing = bool(
            kwargs.pop("gradient_checkpointing", gradient_checkpointing)
        )
        use_bf16 = bool(kwargs.pop("use_bf16", use_bf16))

        model = cls(
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            select_layer=select_layer,
            backbone_embedding_dim=backbone_embedding_dim,
            n_cog_tokens=n_cog_tokens,
            attn_implementation=attn_implementation,
            num_inference_timesteps=num_inference_timesteps,
            use_bf16=use_bf16,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        model.net = net
        return model

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]] | torch.Tensor:
        """Dispatch between training loss and action prediction.

        Args:
            batch: Preprocessed batch dict.

        Returns:
            Training: ``(loss, loss_dict)``. Eval: predicted action tensor.
        """
        if self.training:
            return self.compute_loss(batch)
        return self.get_action(batch)

    def compute_loss(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the flow-matching training loss.

        Args:
            batch: Collated RLDX inputs dict (produced by the preprocessor).

        Returns:
            ``(loss, loss_dict)`` where ``loss_dict`` holds scalar sub-losses.

        Raises:
            RuntimeError: If called before ``from_pretrained`` loads the network.
        """
        if self.net is None:
            msg = "Rldx1Model has no loaded network; call from_pretrained first."
            raise RuntimeError(msg)
        outputs = self.net(dict(batch))
        loss = outputs["loss"]
        loss_dict = {
            key: float(value.detach())
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        }
        return loss, loss_dict

    @torch.no_grad()
    def compute_val_loss(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute deterministic action prediction MSE for validation.

        Runs the full denoising path via :meth:`get_action` and compares it to
        ground-truth actions from the preprocessed batch.

        Args:
            batch: Collated RLDX inputs dict (must include ``action`` during
                validation).

        Returns:
            ``(loss, loss_dict)`` where ``loss`` is action-space MSE.

        Raises:
            RuntimeError: If called before ``from_pretrained`` loads the network.
            ValueError: If the batch does not contain ground-truth ``action``.
        """
        if self.net is None:
            msg = "Rldx1Model has no loaded network; call from_pretrained first."
            raise RuntimeError(msg)

        if "action" not in batch:
            msg = "Validation batch must include 'action' to compute action MSE."
            raise ValueError(msg)

        predicted = self.get_action(batch)
        target = batch["action"]

        min_len = min(predicted.shape[1], target.shape[1])
        min_dim = min(predicted.shape[2], target.shape[2])
        pred_trimmed = predicted[:, :min_len, :min_dim]
        target_trimmed = target[:, :min_len, :min_dim]

        if "action_mask" in batch:
            mask = batch["action_mask"][:, :min_len, :min_dim].to(dtype=pred_trimmed.dtype)
            sq_err = (pred_trimmed - target_trimmed).pow(2)
            denom = mask.sum().clamp_min(1.0)
            loss = (sq_err * mask).sum() / denom
        else:
            loss = F.mse_loss(pred_trimmed, target_trimmed)

        return loss, {"loss": float(loss.detach()), "action_mse": float(loss.detach())}

    def get_action(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Predict an action chunk via the flow-matching denoising loop.

        Args:
            batch: Collated RLDX inputs dict (produced by the preprocessor).

        Returns:
            Predicted action chunk of shape ``(B, action_horizon, action_dim)``.

        Raises:
            RuntimeError: If called before ``from_pretrained`` loads the network.
        """
        if self.net is None:
            msg = "Rldx1Model has no loaded network; call from_pretrained first."
            raise RuntimeError(msg)
        outputs = self.net.get_action(dict(batch))
        return outputs["action_pred"]

    @property
    def reward_delta_indices(self) -> None:
        """Reward indices (rewards not implemented)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Indices of actions relative to the current timestep."""
        return list(range(self._chunk_size))

    @property
    def observation_delta_indices(self) -> None:
        """Indices of observations relative to the current timestep."""
        return None
