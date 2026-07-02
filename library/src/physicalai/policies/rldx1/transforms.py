# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Transforms for the RLDX-1 policy inputs and outputs.

This module provides ``nn.Module``-based transforms that bridge the Studio
:class:`~physicalai.data.observation.Observation` format and the vendored RLDX
network (``components/core_rldx.py``):

- :class:`Rldx1Preprocessor`: ``Observation`` -> RLDX ``inputs`` dict
- :class:`Rldx1Postprocessor`: RLDX ``action_pred`` -> environment action space

Both run the PAS-native pipeline in ``preprocessing.py`` -- batched torch,
library normalization, ``cv2`` image geometry, HF Qwen tokenization. The native
pipeline is parity-tested against the original vendored processor; see
``tests/unit/policies/test_rldx1_preprocessing.py`` for the numerical-parity
proofs. The preprocessor produces the exact keys consumed by
:meth:`RLDX.forward` / :meth:`RLDX.get_action`:

Backbone (Qwen3-VL) keys:
    ``input_ids``, ``attention_mask``, ``pixel_values``, ``image_grid_thw``,
    ``image_wise_encoding``, ``num_views``, ``num_frames``.

Action-head keys:
    ``state`` ``(B, 1, max_state_dim)``, ``action``
    ``(B, action_horizon, max_action_dim)``, ``embodiment_id`` ``(B,)``.

VTC video: each camera view can carry ``num_frames`` temporal frames per step
(the ``delta_timestamps`` video window ``[-6, -4, -2, 0]``). Frames are ordered
frame-major / view-inner into the Qwen conversation, matching upstream
``_get_vlm_inputs``; ``num_frames`` is reported per sample. Train mode applies
replay-consistent augmentation (:mod:`augmentations`) so all frames of a sample
share one sampled param set; eval uses the deterministic ``cv2`` geometry.

Scope (v1): absolute actions, no relative actions / motion / memory / physics
streams.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


from PIL import Image  # noqa: PLC0415

import numpy as np
import torch
from torch import nn

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Observation
from physicalai.policies.utils.normalization import FeatureNormalizeTransform

from transformers import AutoProcessor

from .preprocessing import (
    build_qwen_conversation,
    build_state_action_features,
    build_state_action_norm_map,
    clip_state_action,
    formalize_language,
    pad_state_action,
    resize_and_center_crop,
    tokenize_vlm_batch,
)

if TYPE_CHECKING:
    from transformers import ProcessorMixin

logger = logging.getLogger(__name__)


def _module_device(module: nn.Module) -> torch.device:
    """Return a module's device from its parameters, then buffers, else CPU.

    ``next(iter(module.parameters()))`` raises ``StopIteration`` on a
    parameter-less module (e.g. ``nn.Identity`` or a buffer-only normalizer).
    Inside ``training_step`` that exception is silently caught by Lightning's
    data-fetch loop and misread as end-of-epoch, skipping training entirely.
    """
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


# ============================================================================ #
# Constants                                                                    #
# ============================================================================ #

# LeRobot-format keys (accepted alongside the native Observation keys).
OBSERVATION_STATE = "observation.state"
OBSERVATION_IMAGES_PREFIX = "observation.images."
OBSERVATION_IMAGE = "observation.image"

# Transform output keys (RLDX inputs dict).
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
PIXEL_VALUES = "pixel_values"
IMAGE_GRID_THW = "image_grid_thw"
MM_TOKEN_TYPE_IDS = "mm_token_type_ids"  # noqa: S105  (output key, not a secret)
IMAGE_WISE_ENCODING = "image_wise_encoding"
NUM_VIEWS = "num_views"
NUM_FRAMES = "num_frames"
ACTION_MASK = "action_mask"

# Default padded dimensions (match RLDX PT config.json).
MAX_STATE_DIM = 64
MAX_ACTION_DIM = 64
ACTION_HORIZON = 16

# Default embodiment projector slot. 0 = general_embodiment, the slot RLDX-1
# reserves and pre-conditions for downstream / new-robot fine-tunes (every
# released FT used it). Released benchmark checkpoints use the slot they were
# trained on (see EMBODIMENT_TAG_TO_PROJECTOR_INDEX in components.embodiments):
# 0 = general_embodiment (FT-ROBOCASA/RC365/LIBERO/GR1),
# 1 = fractal20220817_data (FT-SIMPLER-GOOGLE), 3 = bridge_orig (FT-SIMPLER-WIDOWX).
# (35 = new_embodiment is the legacy GR00T slot, superseded by general_embodiment.)
DEFAULT_EMBODIMENT_ID = 0

# Backbone / processor defaults. RLWRLD/RLDX-1-VLM ships the exact Qwen3-VL
# processor used to train the released checkpoints.
DEFAULT_MODEL_NAME = "RLWRLD/RLDX-1-VLM"
DEFAULT_TASK = "Perform the task."

# Fixed upstream RLDX-1 recipe (FT config.json ground truth): text-first prompt
# order, lowercase + strip punctuation.
FORMALIZE_LANGUAGE = True
CONVERSATION_IMAGE_FIRST = False


# ============================================================================ #
# Preprocessor                                                                 #
# ============================================================================ #


class Rldx1Preprocessor(nn.Module):
    """Preprocessor for RLDX-1 policy inputs.

    Converts a Studio :class:`~physicalai.data.observation.Observation` (or an
    equivalent flat dict) into the exact ``inputs`` dict consumed by
    :meth:`RLDX.forward` / :meth:`RLDX.get_action`.

    Runs the PAS-native pipeline (:meth:`forward`): library state/action
    normalization, ``cv2`` aspect-area resize + center crop,
    ``formalize_language``, text-first Qwen3-VL chat template, and HF
    tokenization. The native pipeline is parity-tested against the original
    vendored processor (see ``tests/unit/policies/test_rldx1_preprocessing.py``).
    The HF Qwen3-VL processor is loaded lazily on the first call.

    Scope (v1): absolute actions, single observation step. Relative actions are
    not supported.

    Args:
        max_state_dim: Padded state dimension (shorter states zero-padded).
        max_action_dim: Padded action dimension (shorter actions zero-padded).
        action_horizon: Number of action steps predicted per chunk.
        model_name: HuggingFace ID / local path of the Qwen3-VL processor.
        revision: Pinned git commit SHA for the processor download.
        normalize: Retained for API compatibility; normalization is always
            applied from ``stats``.
        use_percentiles: Use 1st/99th percentile bounds instead of min/max.
        image_max_area: Target max area (pixels) for aspect-preserving resize.
        image_resize_m: Alignment multiple for resized/cropped dimensions.
        random_crop_fraction: Train-time fractional crop size in ``(0, 1]``.
            ``None`` (default) disables the crop stage. Eval always uses a
            deterministic center crop / no crop.
        random_rotation_angle: Train-time ``A.Rotate`` limit in degrees. ``None``
            or ``0`` (default) disables rotation.
        color_jitter_params: Train-time ``A.ColorJitter`` params
            ``{"brightness", "contrast", "saturation", "hue"}``. ``None``
            (default) disables color jitter.
        default_task: Fallback instruction when an observation has no task.
        embodiment_id: Per-embodiment projector slot in the MSAT action head.
            Default 0 (general_embodiment) for a fresh new-robot fine-tune; set
            the slot a released checkpoint was trained on to reuse it.
        stats: Dataset statistics ``{key: {min, max, mean, std, q01, q99}}``.

    Examples:
        >>> pre = Rldx1Preprocessor(stats=dataset.stats)
        >>> inputs = pre(observation)
        >>> out = model.net.get_action(inputs)
    """

    def __init__(
        self,
        *,
        max_state_dim: int = MAX_STATE_DIM,
        max_action_dim: int = MAX_ACTION_DIM,
        action_horizon: int = ACTION_HORIZON,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str | None = None,
        normalize: bool = True,
        use_percentiles: bool = True,
        image_max_area: int = 65536,
        image_resize_m: int = 32,
        random_crop_fraction: float | None = None,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        default_task: str = DEFAULT_TASK,
        embodiment_id: int = DEFAULT_EMBODIMENT_ID,
        stats: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Initialize the preprocessor and build the normalization blocks."""
        super().__init__()

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.model_name = model_name
        self.revision = revision
        self.normalize = normalize
        self.use_percentiles = use_percentiles
        self.image_max_area = image_max_area
        self.image_resize_m = image_resize_m
        self.default_task = default_task
        self.embodiment_id = embodiment_id

        # Train-time image augmentation (upstream ReplayCompose). Engaged only
        # when a stochastic stage is configured; otherwise both train and eval
        # use the deterministic ``resize_and_center_crop`` geometry (pixel-
        # identical to the eval ``AspectAreaResizeAndCrop``). The albumentations
        # transform is built lazily so the default no-aug path stays light.
        self.random_crop_fraction = random_crop_fraction
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self._augment = (
            random_crop_fraction is not None or bool(random_rotation_angle) or color_jitter_params is not None
        )
        self._train_transform_cache: Any = None

        # PAS-native state/action normalizer (Stage 1). An nn.Module submodule so
        # its min/max/q01/q99 buffers move with `.to(device)` and export cleanly.
        sa_features = build_state_action_features(stats)
        self._has_sa_features = bool(sa_features)
        if sa_features:
            norm_map = build_state_action_norm_map(use_percentiles=use_percentiles)
            self._state_action_normalizer: nn.Module = FeatureNormalizeTransform(sa_features, norm_map)
        else:
            self._state_action_normalizer = nn.Identity()

        # PAS-native action denormalizer (Stage 5). Inverse of the action half of
        # the forward normalizer; the postprocessor uses it to decode predicted
        # action chunks back to the environment space. `_action_dim` is the
        # unpadded action width.
        action_feature = sa_features.get(ACTION)
        if action_feature is not None:
            self._action_dim = int(action_feature.shape[0])
            self._action_denormalizer: nn.Module = FeatureNormalizeTransform(
                {ACTION: action_feature},
                build_state_action_norm_map(use_percentiles=use_percentiles),
                inverse=True,
            )
        else:
            self._action_dim = 0
            self._action_denormalizer = nn.Identity()

        # Lazily-loaded HF Qwen processor for the PAS-native VLM path (Stage 4).
        self._vlm_processor_cache: ProcessorMixin | None = None

    # -- native VLM processor ---------------------------------------------- #
    @property
    def _vlm_processor(self) -> ProcessorMixin:
        """Lazily load the HF Qwen processor for the PAS-native VLM path.

        Returns:
            The cached HF processor with left padding (Flash-Attention
            compatible), loaded with ``trust_remote_code=False`` and the pinned
            ``revision``.
        """
        if self._vlm_processor_cache is None:
            # lib.security: never trust_remote_code; pin the processor revision.
            loading_kwargs: dict[str, Any] = {"trust_remote_code": False, "use_fast": True}
            if self.revision is not None:
                loading_kwargs["revision"] = self.revision
            processor = AutoProcessor.from_pretrained(self.model_name, **loading_kwargs)
            processor.tokenizer.padding_side = "left"
            self._vlm_processor_cache = processor
        return self._vlm_processor_cache

    # -- train-time augmentation ------------------------------------------- #
    @property
    def _train_transform(self) -> Any:  # noqa: ANN401
        """Lazily build the albumentations ``ReplayCompose`` train transform.

        Returns:
            The train ``ReplayCompose`` (step-1 geometry + configured stochastic
            stages). Only reached when ``self._augment`` is True.
        """
        if self._train_transform_cache is None:
            from .augmentations import build_image_transformations_albumentations  # noqa: PLC0415

            train_transform, _eval = build_image_transformations_albumentations(
                image_max_area=self.image_max_area,
                image_resize_m=self.image_resize_m,
                random_crop_fraction=self.random_crop_fraction,
                random_rotation_angle=self.random_rotation_angle,
                color_jitter_params=self.color_jitter_params,
            )
            self._train_transform_cache = train_transform
        return self._train_transform_cache

    def _transform_frames(
        self,
        frames: list[np.ndarray],
        replay: dict | None,
    ) -> tuple[list[Image.Image], dict | None]:
        """Resize (eval) or augment (train) a sample's frames into PIL images.

        In train mode with augmentation configured, one ``ReplayCompose`` blob
        is shared across every frame of the sample (threaded via ``replay``) so
        the random params are identical across frames and views -- matching
        upstream ``_get_vlm_inputs``. Otherwise the deterministic ``cv2``
        geometry is applied.

        Args:
            frames: ``(H, W, 3)`` uint8 frames for one view of one sample.
            replay: Running replay blob for the sample (``None`` for the first
                view; reused for subsequent views/frames).

        Returns:
            ``(pil_frames, replay)``.
        """
        if self.training and self._augment:
            from .augmentations import apply_with_replay  # noqa: PLC0415

            tensors, replay = apply_with_replay(self._train_transform, frames, replay)
            pil = [Image.fromarray(np.ascontiguousarray(t.permute(1, 2, 0).numpy())) for t in tensors]
            return pil, replay

        pil = [
            Image.fromarray(resize_and_center_crop(frame, max_area=self.image_max_area, m=self.image_resize_m))
            for frame in frames
        ]
        return pil, replay

    # -- native forward ---------------------------------------------------- #
    def _normalize_pad_state_action(
        self,
        batch_dict: dict[str, Any],
        *,
        has_action: bool,
    ) -> dict[str, torch.Tensor]:
        """Normalize, clip and pad state/action with the PAS-native blocks.

        Returns:
            Dict with padded ``state`` ``(B, 1, max_state_dim)`` and, when an
            action is present, ``action`` ``(B, action_horizon, max_action_dim)``
            plus ``action_mask``.
        """
        # Run on the normalizer's buffer device (falls back to CPU for Identity).
        sa_device = _module_device(self._state_action_normalizer)

        state_raw = batch_dict.get(OBSERVATION_STATE, batch_dict.get(STATE))
        sa_batch: dict[str, Any] = {STATE: self._as_float_tensor(state_raw).to(sa_device)}
        if has_action:
            sa_batch[ACTION] = self._as_float_tensor(batch_dict[ACTION]).to(sa_device)

        sa_batch = self._state_action_normalizer(sa_batch)
        if self._has_sa_features:
            sa_batch = clip_state_action(sa_batch)
        sa_batch = pad_state_action(
            sa_batch,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            max_action_horizon=self.action_horizon,
        )

        dtype = torch.get_default_dtype()
        return {key: value.to(dtype) for key, value in sa_batch.items()}

    def _build_conversations(
        self,
        batch_dict: dict[str, Any],
        view_keys: list[str],
        tasks: list[str],
        batch_size: int,
    ) -> tuple[list[list[dict[str, Any]]], int]:
        """Build one Qwen conversation per sample and report the frame count.

        Each view can carry ``num_frames`` temporal frames (the VTC video
        window). Frames are resized (eval) or replay-augmented (train) then
        flattened frame-major / view-inner -- ``[t0v0, t0v1, ..., t1v0, ...]`` --
        matching upstream ``_get_vlm_inputs``.

        Returns:
            ``(conversations, num_frames)`` -- ``batch_size`` conversations and
            the per-sample temporal frame count.
        """
        conversations: list[list[dict[str, Any]]] = []
        num_frames = 1
        for index in range(batch_size):
            # One replay blob per sample: shared across all views and frames so
            # the sampled augmentation params are identical (upstream parity).
            replay: dict | None = None
            per_view_frames: list[list[Image.Image]] = []
            for view in view_keys:
                frames = self._frames_for_view(batch_dict[view], index)
                pil_frames, replay = self._transform_frames(frames, replay)
                per_view_frames.append(pil_frames)

            num_frames = len(per_view_frames[0])
            # Frame-major, view-inner ordering (upstream: for t: for view).
            images = [per_view_frames[view_idx][t] for t in range(num_frames) for view_idx in range(len(view_keys))]

            language = formalize_language(tasks[index]) if FORMALIZE_LANGUAGE else tasks[index]
            conversations.append(
                build_qwen_conversation(images, language, image_first=CONVERSATION_IMAGE_FIRST),
            )
        return conversations, num_frames

    def _frames_for_view(self, view_value: Any, index: int) -> list[np.ndarray]:  # noqa: ANN401
        """Return the ``(H, W, 3)`` uint8 frames for one sample and view.

        Handles both single-frame (``(C, H, W)`` / ``(H, W, C)``) and multi-frame
        (``(T, C, H, W)`` / ``(T, H, W, C)``) layouts; ``delta_timestamps`` adds a
        leading temporal axis when a video window is requested.

        Returns:
            A list of per-frame ``(H, W, 3)`` uint8 arrays.
        """
        sample = self._index(view_value, index)
        return [self._to_hwc_uint8(frame) for frame in self._split_frames(sample)]

    @staticmethod
    def _split_frames(sample: Any) -> list[np.ndarray]:  # noqa: ANN401
        """Split a per-sample view array into a list of single-frame arrays.

        Returns:
            One array per temporal frame. A 4-D input is split along its leading
            (temporal) axis; a 3-D input is returned as a single frame.
        """
        arr = np.asarray(sample)
        if arr.ndim == 4:  # noqa: PLR2004 - (T, C, H, W) or (T, H, W, C)
            return [arr[t] for t in range(arr.shape[0])]
        return [arr]


    # -- forward ----------------------------------------------------------- #

    def forward(self, batch: Observation | dict[str, Any]) -> dict[str, torch.Tensor]:
        """Preprocess an observation batch into RLDX ``inputs``.

        Runs the PAS-native pipeline: state/action normalization + padding,
        ``cv2`` image geometry, and Qwen3-VL tokenization (see
        ``preprocessing.py``). Supports the v1 configuration only: absolute
        actions, single observation step, deterministic image geometry.

        Args:
            batch: Input as :class:`Observation` or a flat dict with keys
                ``state``, ``images.*`` / ``observation.images.*``, ``task``,
                and optionally ``action`` (for the training loss).

        Returns:
            Dict of tensors consumed by :meth:`RLDX.forward` /
            :meth:`RLDX.get_action`.

        Raises:
            ValueError: If the observation carries no camera image.
        """
        batch_dict = batch.to_dict(flatten=True) if isinstance(batch, Observation) else dict(batch)
        view_keys = self._image_keys(batch_dict)
        if not view_keys:
            msg = "RLDX-1 preprocessor requires at least one camera image."
            raise ValueError(msg)

        batch_size, device = self._infer_batch_info(batch_dict)
        tasks = self._task_strings(batch_dict.get(TASK), batch_size)
        has_action = batch_dict.get(ACTION) is not None

        sa_inputs = self._normalize_pad_state_action(batch_dict, has_action=has_action)
        conversations, num_frames = self._build_conversations(batch_dict, view_keys, tasks, batch_size)
        vlm = tokenize_vlm_batch(self._vlm_processor, conversations)

        inputs: dict[str, torch.Tensor] = {
            INPUT_IDS: vlm[INPUT_IDS],
            ATTENTION_MASK: vlm[ATTENTION_MASK],
            PIXEL_VALUES: vlm[PIXEL_VALUES],
            IMAGE_GRID_THW: vlm[IMAGE_GRID_THW],
            # vtc scalars: bool image-wise flag, per-sample view/frame counts.
            IMAGE_WISE_ENCODING: torch.tensor([True] * batch_size),
            NUM_VIEWS: torch.tensor([len(view_keys)] * batch_size),
            NUM_FRAMES: torch.tensor([num_frames] * batch_size),
            STATE: sa_inputs[STATE],
            "embodiment_id": torch.tensor([self.embodiment_id] * batch_size),
        }
        if MM_TOKEN_TYPE_IDS in vlm:
            inputs[MM_TOKEN_TYPE_IDS] = vlm[MM_TOKEN_TYPE_IDS]
        if has_action:
            inputs[ACTION] = sa_inputs[ACTION]
            inputs[ACTION_MASK] = sa_inputs[ACTION_MASK]

        return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in inputs.items()}

    @staticmethod
    def _as_float_tensor(value: Any) -> torch.Tensor:  # noqa: ANN401
        """Coerce a state/action value to a float32 tensor.

        Returns:
            A ``float32`` tensor view of ``value``.
        """
        if isinstance(value, torch.Tensor):
            return value.to(torch.float32)
        return torch.as_tensor(np.asarray(value, dtype=np.float32))

    # -- native denormalization (postprocessor) ---------------------------- #

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Decode a normalized action chunk to the environment space (Stage 5).

        PAS-native inverse of the forward action normalization: slices to the
        unpadded action width and prediction horizon, clamps to ``[-1, 1]``
        (matching the vendored ``unnormalize_values_minmax`` clip), then applies
        the inverse :class:`FeatureNormalizeTransform`.

        Args:
            action: Predicted action ``(B, T, max_action_dim)``.

        Returns:
            Denormalized action ``(B, action_horizon, action_dim)`` in the
            original environment space.

        Raises:
            RuntimeError: If no action statistics were provided.
        """
        if self._action_dim <= 0:
            msg = "Cannot denormalize action: no action statistics were provided to the preprocessor."
            raise RuntimeError(msg)

        sliced = action[..., : self.action_horizon, : self._action_dim].clamp(-1.0, 1.0)

        denorm_device = sliced.device
        for buf in self._action_denormalizer.buffers():
            denorm_device = buf.device
            break

        batch = {ACTION: sliced.to(denorm_device)}
        batch = self._action_denormalizer(batch)
        return batch[ACTION].to(action.device, action.dtype)

    @staticmethod
    def _index(value: Any, index: int) -> np.ndarray | None:  # noqa: ANN401
        """Return the ``index``-th element of a batched tensor/array as numpy.

        Returns:
            The selected sample as a numpy array, or ``None`` when ``value`` is None.
        """
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value[index].detach().cpu().numpy()
        return np.asarray(value)[index]

    @staticmethod
    def _to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
        """Convert an image array to contiguous ``(H, W, 3)`` uint8 layout.

        Returns:
            A contiguous ``(H, W, 3)`` uint8 array.
        """
        arr = np.asarray(arr)
        # CHW -> HWC when the leading axis is the channel axis.
        if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:  # noqa: PLR2004
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:  # noqa: PLR2004
            arr = np.repeat(arr, 3, axis=-1)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _infer_batch_info(batch: dict[str, Any]) -> tuple[int, torch.device]:
        """Infer batch size and device from the first tensor in ``batch``.

        Returns:
            Tuple of ``(batch_size, device)``.
        """
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0], value.device
        return 1, torch.device("cpu")

    @staticmethod
    def _image_keys(batch: dict[str, Any]) -> list[str]:
        """Find sorted image keys, supporting LeRobot and Observation formats.

        Returns:
            Sorted list of image keys (possibly empty).
        """
        keys = sorted(k for k in batch if k.startswith(OBSERVATION_IMAGES_PREFIX))
        if not keys:
            keys = sorted(k for k in batch if k.startswith(f"{IMAGES}.") and k != IMAGES)
        if not keys and OBSERVATION_IMAGE in batch:
            keys = [OBSERVATION_IMAGE]
        if not keys and isinstance(batch.get(IMAGES), torch.Tensor):
            keys = [IMAGES]
        return keys

    @staticmethod
    def _task_strings(task: Any, batch_size: int) -> list[str]:  # noqa: ANN401
        """Normalize the task field into a list of ``batch_size`` strings.

        Returns:
            List of task strings, one per batch element.
        """
        if task is None:
            return [DEFAULT_TASK] * batch_size
        if isinstance(task, str):
            return [task] * batch_size
        if isinstance(task, (list, tuple)):
            tasks = [str(t) for t in task]
            if len(tasks) == 1:
                return tasks * batch_size
            return tasks
        return [str(task)] * batch_size


# ============================================================================ #
# Postprocessor                                                                #
# ============================================================================ #


class Rldx1Postprocessor(nn.Module):
    """Postprocessor for RLDX-1 policy outputs.

    Decodes the RLDX ``action_pred`` ``(B, action_horizon, max_action_dim)``
    back to the environment action space using the paired preprocessor's
    PAS-native inverse normalization
    (:meth:`Rldx1Preprocessor.denormalize_action`).

    Args:
        preprocessor: The paired preprocessor holding the normalization stats.
        env_action_dim: Original (unpadded) action dimension. ``0`` keeps the
            full decoded width.

    Examples:
        >>> pre, post = make_rldx1_transforms(stats=dataset.stats, env_action_dim=7)
        >>> inputs = pre(observation)
        >>> action = post(model.net.get_action(inputs))
    """

    def __init__(
        self,
        *,
        preprocessor: Rldx1Preprocessor,
        env_action_dim: int = 0,
    ) -> None:
        """Initialize with the paired preprocessor and env action dimension."""
        super().__init__()
        # Stored in a tuple so nn.Module does not register the preprocessor as a
        # submodule (avoids duplicate state tracking under Lightning).
        self._pre_ref = (preprocessor,)
        self.env_action_dim = env_action_dim

    def forward(
        self,
        action: torch.Tensor,
        state: dict[str, np.ndarray] | None = None,  # noqa: ARG002 - reserved for relative-action decoding
    ) -> torch.Tensor:
        """Decode the predicted action chunk to the environment action space.

        Args:
            action: RLDX prediction ``(B, T, D)`` or ``(B, D)``.
            state: Reserved for future relative-action decoding; ignored.

        Returns:
            Action chunk ``(B, T, env_action_dim)`` (or ``(B, env_action_dim)``
            when the input was 2-D) in the original environment space.
        """
        pre = self._pre_ref[0]

        squeeze = action.dim() == 2  # noqa: PLR2004
        if squeeze:
            action = action.unsqueeze(1)

        out_t = pre.denormalize_action(action)

        if self.env_action_dim > 0 and out_t.shape[-1] >= self.env_action_dim:
            out_t = out_t[..., : self.env_action_dim]

        return out_t.squeeze(1) if squeeze else out_t


# ============================================================================ #
# Factory                                                                      #
# ============================================================================ #


def make_rldx1_transforms(
    *,
    stats: dict[str, dict[str, list[float]]] | None,
    env_action_dim: int = 0,
    max_state_dim: int = MAX_STATE_DIM,
    max_action_dim: int = MAX_ACTION_DIM,
    action_horizon: int = ACTION_HORIZON,
    model_name: str = DEFAULT_MODEL_NAME,
    revision: str | None = None,
    normalize: bool = True,
    use_percentiles: bool = True,
    image_max_area: int = 65536,
    image_resize_m: int = 32,
    random_crop_fraction: float | None = None,
    random_rotation_angle: int | None = None,
    color_jitter_params: dict[str, float] | None = None,
    embodiment_id: int = DEFAULT_EMBODIMENT_ID,
) -> tuple[Rldx1Preprocessor, Rldx1Postprocessor]:
    """Build the matched RLDX-1 preprocessor / postprocessor pair.

    Both share the same normalization statistics, so the forward and inverse
    transforms are guaranteed consistent.

    Args:
        stats: Dataset statistics for (de)normalization.
        env_action_dim: Original environment action dimension for decoding.
        max_state_dim: Padded state dimension.
        max_action_dim: Padded action dimension.
        action_horizon: Number of action steps per chunk.
        model_name: HuggingFace ID / local path of the Qwen3-VL processor.
        revision: Pinned git commit SHA for the processor download.
        normalize: Retained for API compatibility (normalization always applied).
        use_percentiles: Prefer 1st/99th percentile bounds over min/max.
        image_max_area: Target max area (pixels) for aspect-preserving resize.
        image_resize_m: Alignment multiple for resized/cropped dimensions.
        random_crop_fraction: Train-time fractional crop size in ``(0, 1]``
            (``None`` disables the crop stage).
        random_rotation_angle: Train-time rotation limit in degrees (``None`` /
            ``0`` disables rotation).
        color_jitter_params: Train-time ``A.ColorJitter`` params (``None``
            disables color jitter).
        embodiment_id: Per-embodiment projector slot in the MSAT action head
            (default 0 = general_embodiment for a fresh new-robot fine-tune).

    Returns:
        Tuple of ``(preprocessor, postprocessor)``.
    """
    preprocessor = Rldx1Preprocessor(
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        action_horizon=action_horizon,
        model_name=model_name,
        revision=revision,
        normalize=normalize,
        use_percentiles=use_percentiles,
        image_max_area=image_max_area,
        image_resize_m=image_resize_m,
        random_crop_fraction=random_crop_fraction,
        random_rotation_angle=random_rotation_angle,
        color_jitter_params=color_jitter_params,
        embodiment_id=embodiment_id,
        stats=stats,
    )
    postprocessor = Rldx1Postprocessor(
        preprocessor=preprocessor,
        env_action_dim=env_action_dim,
    )
    return preprocessor, postprocessor
