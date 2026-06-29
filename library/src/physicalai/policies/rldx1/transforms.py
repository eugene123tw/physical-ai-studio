# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Transforms for the RLDX-1 policy inputs and outputs.

This module provides ``nn.Module``-based transforms that bridge the Studio
:class:`~physicalai.data.observation.Observation` format and the vendored RLDX
network (``components/core_rldx.py``):

- :class:`Rldx1Preprocessor`: ``Observation`` -> RLDX ``inputs`` dict
- :class:`Rldx1Postprocessor`: RLDX ``action_pred`` -> environment action space

Both are thin adapters that drive the vendored upstream ``RLDXProcessor`` /
``RLDXDataCollator`` (``components/processing/processing_rldx.py``, copied
verbatim from ``rldx/model/core/processing_rldx.py``). Driving the vendored
processor guarantees byte-for-byte parity with the upstream RLDX pipeline; see
``scripts/rldx1_parity_check.py`` for the numerical-parity proof. The
preprocessor produces the exact keys consumed by :meth:`RLDX.forward` /
:meth:`RLDX.get_action`:

Backbone (Qwen3-VL) keys:
    ``input_ids``, ``attention_mask``, ``pixel_values``, ``image_grid_thw``,
    ``image_wise_encoding``, ``num_views``, ``num_frames``.

Action-head keys:
    ``state`` ``(B, 1, max_state_dim)``, ``action``
    ``(B, action_horizon, max_action_dim)``, ``action_mask`` (same shape as
    ``action``), ``embodiment_id`` ``(B,)``.

Scope (v1): single observation step, no motion / memory / physics streams.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from physicalai.data.observation import ACTION, IMAGES, STATE, TASK, Observation

if TYPE_CHECKING:
    from .components.processing import RLDXProcessor

logger = logging.getLogger(__name__)

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
IMAGE_WISE_ENCODING = "image_wise_encoding"
NUM_VIEWS = "num_views"
NUM_FRAMES = "num_frames"
ACTION_MASK = "action_mask"

# Stats keys.
STATS_MIN = "min"
STATS_MAX = "max"
STATS_MEAN = "mean"
STATS_STD = "std"
STATS_Q01 = "q01"
STATS_Q99 = "q99"
STAT_KEYS = (STATS_MIN, STATS_MAX, STATS_MEAN, STATS_STD, STATS_Q01, STATS_Q99)

# Default padded dimensions (match RLDX PT config.json).
MAX_STATE_DIM = 64
MAX_ACTION_DIM = 64
ACTION_HORIZON = 16

# Default embodiment projector slot. 0 = general_embodiment, the slot RLDX-1
# reserves and pre-conditions for downstream / new-robot fine-tunes (every
# released FT used it). Released benchmark checkpoints use the slot they were
# trained on (see EMBODIMENT_TAG_TO_PROJECTOR_INDEX in components.processing):
# 0 = general_embodiment (FT-ROBOCASA/RC365/LIBERO/GR1),
# 1 = fractal20220817_data (FT-SIMPLER-GOOGLE), 3 = bridge_orig (FT-SIMPLER-WIDOWX).
# (35 = new_embodiment is the legacy GR00T slot, superseded by general_embodiment.)
DEFAULT_EMBODIMENT_ID = 0

# Backbone / processor defaults. RLWRLD/RLDX-1-VLM ships the exact Qwen3-VL
# processor used to train the released checkpoints.
DEFAULT_MODEL_NAME = "RLWRLD/RLDX-1-VLM"
DEFAULT_TASK = "Perform the task."

# Vision-temporal-conditioned backbone variant.
MODEL_TYPE = "vtc_qwen3_vl"

# Single-group modality names fed to the vendored RLDXProcessor.
STATE_GROUP = "state"
ACTION_GROUP = "action"
RELATIVE_ACTION_GROUP = "relative_action"

# Fixed upstream RLDX-1 recipe (FT config.json ground truth): text-first prompt
# order, lowercase + strip punctuation, clip outliers during normalization.
FORMALIZE_LANGUAGE = True
CONVERSATION_IMAGE_FIRST = False
CLIP_OUTLIERS = True

# Studio dataset-stats keys that may carry relative-action statistics.
RELATIVE_ACTION_STAT_KEYS = ("relative_action", "action.relative")


# ============================================================================ #
# Preprocessor                                                                 #
# ============================================================================ #


class Rldx1Preprocessor(nn.Module):
    """Preprocessor for RLDX-1 policy inputs.

    Thin adapter over the vendored upstream :class:`RLDXProcessor`
    (``components/processing/processing_rldx.py``). It converts a Studio
    :class:`~physicalai.data.observation.Observation` (or an equivalent flat
    dict) into per-sample content dicts, runs the verbatim upstream processor
    and data collator, and returns the exact ``inputs`` dict consumed by
    :meth:`RLDX.forward` / :meth:`RLDX.get_action`.

    Driving the vendored processor guarantees byte-for-byte parity with the
    upstream RLDX pipeline (state/action per-joint-group normalization,
    relative-action pose math, albumentations image resize, Qwen3-VL smart
    resize, text-first chat template, ``formalize_language``). See
    ``scripts/rldx1_parity_check.py`` for the numerical-parity proof.

    The processor is built lazily on the first call (it loads the Qwen3-VL
    tokenizer / image processor and depends on the observed camera-view keys).

    Args:
        max_state_dim: Padded state dimension (shorter states zero-padded).
        max_action_dim: Padded action dimension (shorter actions zero-padded).
        action_horizon: Number of action steps predicted per chunk.
        model_name: HuggingFace ID / local path of the Qwen3-VL processor.
        revision: Pinned git commit SHA for the processor download.
        normalize: Retained for API compatibility; the vendored processor
            always normalizes from ``stats``.
        use_percentiles: Use 1st/99th percentile bounds instead of min/max.
        use_relative_action: Encode actions relative to the current state.
            Requires ``relative_action`` statistics; otherwise falls back to
            absolute actions with a warning.
        image_max_area: Target max area (pixels) for aspect-preserving resize.
        image_resize_m: Alignment multiple for resized/cropped dimensions.
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
        use_relative_action: bool = False,
        image_max_area: int = 65536,
        image_resize_m: int = 32,
        default_task: str = DEFAULT_TASK,
        embodiment_id: int = DEFAULT_EMBODIMENT_ID,
        stats: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Initialize the adapter and build the processor statistics."""
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

        self._statistics, self._relative = self._build_statistics(stats, use_relative_action=use_relative_action)

        # Lazily-built vendored RLDXProcessor (not an nn.Module). Rebuilt if the
        # observed set of camera-view keys changes.
        self._processor: RLDXProcessor | None = None
        self._view_keys: list[str] | None = None

    # -- statistics -------------------------------------------------------- #

    def _build_statistics(
        self,
        stats: dict[str, dict[str, list[float]]] | None,
        *,
        use_relative_action: bool,
    ) -> tuple[dict[str, Any], bool]:
        """Translate Studio dataset stats into the processor statistics schema.

        Returns:
            Tuple ``(statistics, relative)`` where ``statistics`` is keyed by
            ``{"state"/"action"/"relative_action": {group: stat}}``
            and ``relative`` indicates whether relative actions are active.
        """
        state_stat = self._extract_stat(stats, (OBSERVATION_STATE, STATE))
        action_stat = self._extract_stat(stats, (ACTION, ACTION_GROUP))
        statistics: dict[str, Any] = {
            STATE_GROUP: {STATE_GROUP: state_stat},
            ACTION_GROUP: {ACTION_GROUP: action_stat},
        }

        ## ?!?! use_relative_action is set to True, but no relative_action statistics were found in the dataset stats????
        ## how do we support relative action?
        relative = False
        if use_relative_action:
            rel_stat = self._extract_stat(stats, RELATIVE_ACTION_STAT_KEYS)
            if rel_stat is not None:
                statistics[RELATIVE_ACTION_GROUP] = {ACTION_GROUP: rel_stat}
                relative = True
            else:
                logger.warning(
                    "use_relative_action=True but no 'relative_action' statistics were "
                    "found in the dataset stats; falling back to ABSOLUTE actions. Provide "
                    "relative-action stats to enable relative encoding.",
                )
        return statistics, relative

    @staticmethod
    def _extract_stat(
        stats: dict[str, dict[str, list[float]]] | None,
        keys: tuple[str, ...],
    ) -> dict[str, list[float]] | None:
        """Return the first matching, coerced stat entry for ``keys``.

        Returns:
            A stat dict with all of :data:`STAT_KEYS`, or ``None`` if absent.
        """
        if not stats:
            return None
        for key in keys:
            entry = stats.get(key)
            if entry is not None:
                return Rldx1Preprocessor._coerce_stat(entry)
        return None

    @staticmethod
    def _coerce_stat(raw: Any) -> dict[str, list[float]] | None:  # noqa: ANN401
        """Coerce a raw stat mapping to lists, filling missing sub-keys.

        Returns:
            Dict with ``min``/``max``/``mean``/``std``/``q01``/``q99`` as
            ``list[float]``, or ``None`` when no numeric vector is present.
        """

        def vec(key: str) -> list[float] | None:
            value = raw.get(key) if isinstance(raw, dict) else None
            if value is None:
                return None
            return np.asarray(value, dtype=np.float64).ravel().tolist()

        mn, mx = vec(STATS_MIN), vec(STATS_MAX)
        q01, q99 = vec(STATS_Q01), vec(STATS_Q99)
        mean, std = vec(STATS_MEAN), vec(STATS_STD)

        dim_src = next((v for v in (mn, mx, q01, q99, mean, std) if v is not None), None)
        if dim_src is None:
            return None
        dim = len(dim_src)

        mn = mn if mn is not None else (q01 if q01 is not None else [-1.0] * dim)
        mx = mx if mx is not None else (q99 if q99 is not None else [1.0] * dim)
        q01 = q01 if q01 is not None else list(mn)
        q99 = q99 if q99 is not None else list(mx)
        mean = mean if mean is not None else [0.0] * dim
        std = std if std is not None else [1.0] * dim
        return {
            STATS_MIN: mn,
            STATS_MAX: mx,
            STATS_MEAN: mean,
            STATS_STD: std,
            STATS_Q01: q01,
            STATS_Q99: q99,
        }

    # -- processor --------------------------------------------------------- #

    def _get_processor(self, view_keys: list[str]) -> RLDXProcessor:
        """Lazily build the vendored RLDXProcessor for the given camera views.

        Returns:
            The cached :class:`RLDXProcessor` instance.

        Raises:
            ValueError: If state / action statistics are missing.
        """
        if self._processor is not None and self._view_keys == view_keys:
            return self._processor

        from .components.processing import (  # noqa: PLC0415
            ActionConfig,
            ActionFormat,
            ActionRepresentation,
            ActionType,
            ModalityConfig,
            RLDXProcessor,
        )

        if (
            self._statistics[STATE_GROUP][STATE_GROUP] is None
            or self._statistics[ACTION_GROUP][ACTION_GROUP] is None
        ):
            msg = (
                "RLDX-1 preprocessor requires dataset statistics for state and action "
                "(min/max or q01/q99). None were found in the provided stats."
            )
            raise ValueError(msg)

        rep = ActionRepresentation.RELATIVE if self._relative else ActionRepresentation.ABSOLUTE
        modality_configs = {
            STATE_GROUP: ModalityConfig(delta_indices=[0], modality_keys=[STATE_GROUP]),
            ACTION_GROUP: ModalityConfig(
                delta_indices=list(range(self.action_horizon)),
                modality_keys=[ACTION_GROUP],
                action_configs=[
                    ActionConfig(
                        rep=rep,
                        type=ActionType.NON_EEF,
                        format=ActionFormat.DEFAULT,
                        state_key=STATE_GROUP,
                    ),
                ],
            ),
            "video": ModalityConfig(delta_indices=[0], modality_keys=list(view_keys)),
        }

        # lib.security: never trust_remote_code; pin the processor revision.
        loading_kwargs: dict[str, Any] = {"trust_remote_code": False, "use_fast": True}
        if self.revision is not None:
            loading_kwargs["revision"] = self.revision

        processor = RLDXProcessor(
            modality_configs=modality_configs,
            statistics=self._statistics,
            model_name=self.model_name,
            model_type=MODEL_TYPE,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            max_action_horizon=self.action_horizon,
            use_relative_action=self._relative,
            use_percentiles=self.use_percentiles,
            clip_outliers=CLIP_OUTLIERS,
            formalize_language=FORMALIZE_LANGUAGE,
            conversation_image_first=CONVERSATION_IMAGE_FIRST,
            image_max_area=self.image_max_area,
            image_resize_m=self.image_resize_m,
            embodiment_id=self.embodiment_id,
            transformers_loading_kwargs=loading_kwargs,
        )
        self._processor = processor
        self._view_keys = list(view_keys)
        return processor

    # -- forward ----------------------------------------------------------- #

    def forward(self, batch: Observation | dict[str, Any]) -> dict[str, torch.Tensor]:
        """Preprocess an observation batch into RLDX ``inputs``.

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

        processor = self._get_processor(view_keys)
        # Sync the processor's train/eval mode: selects the deterministic image
        # transform and disables stochastic embodiment routing at inference.
        if self.training:
            processor.train()
        else:
            processor.eval()

        batch_size, device = self._infer_batch_info(batch_dict)
        tasks = self._task_strings(batch_dict.get(TASK), batch_size)
        has_action = batch_dict.get(ACTION) is not None

        per_sample = [
            processor([{"content": self._to_vla(batch_dict, b, view_keys, tasks[b], has_action=has_action)}])
            for b in range(batch_size)
        ]
        collated = processor.collator(per_sample)
        inputs = dict(collated["inputs"])
        return {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in inputs.items()}

    def _to_vla(
        self,
        batch: dict[str, Any],
        index: int,
        view_keys: list[str],
        task: str,
        *,
        has_action: bool,
    ) -> dict[str, Any]:
        """Build a single per-step content dict from a batched observation.

        Returns:
            A content dict (``images``/``states``/``actions``/``text``) for
            ``batch`` element ``index``, consumed by the vendored processor.
        """
        state = self._index(batch.get(OBSERVATION_STATE, batch.get(STATE)), index)
        state_np = np.asarray(state, dtype=np.float32).reshape(1, -1)

        actions: dict[str, np.ndarray] = {}
        if has_action:
            action_np = np.asarray(self._index(batch[ACTION], index), dtype=np.float32)
            if action_np.ndim == 1:
                action_np = action_np.reshape(1, -1)
            actions = {ACTION_GROUP: action_np}

        images = {key: [self._to_hwc_uint8(self._index(batch[key], index))] for key in view_keys}
        return {
            "images": images,
            "states": {STATE_GROUP: state_np},
            "actions": actions,
            "text": task,
        }

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
    back to the environment action space using the same vendored
    :class:`RLDXProcessor` that produced the inputs. This guarantees the inverse
    transform matches the forward normalization exactly (per-joint-group
    unnormalization and, when enabled, relative-to-absolute pose conversion).

    The processor is shared with the paired :class:`Rldx1Preprocessor`; the
    preprocessor must run at least once (it builds the processor lazily) before
    postprocessing.

    Args:
        preprocessor: The paired preprocessor holding the vendored processor.
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
        state: dict[str, np.ndarray] | None = None,
    ) -> torch.Tensor:
        """Decode the predicted action chunk to the environment action space.

        Args:
            action: RLDX prediction ``(B, T, D)`` or ``(B, D)``.
            state: Optional current state ``{group: array}`` used to convert
                relative actions back to absolute. Required only when the
                preprocessor uses relative actions.

        Returns:
            Action chunk ``(B, T, env_action_dim)`` (or ``(B, env_action_dim)``
            when the input was 2-D) in the original environment space.

        Raises:
            RuntimeError: If the paired preprocessor has not built its processor.
        """
        pre = self._pre_ref[0]
        processor = pre._processor  # noqa: SLF001  (tightly-coupled paired transforms)
        if processor is None:
            msg = "Run the preprocessor before the postprocessor (processor not built yet)."
            raise RuntimeError(msg)

        squeeze = action.dim() == 2  # noqa: PLR2004
        if squeeze:
            action = action.unsqueeze(1)

        action_np = action.detach().to(torch.float32).cpu().numpy()
        decoded = processor.decode_action(
            action_np,
            state=state,
        )
        out_t = torch.from_numpy(np.asarray(decoded[ACTION_GROUP])).to(action.device, action.dtype)

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
    use_relative_action: bool = False,
    image_max_area: int = 65536,
    image_resize_m: int = 32,
    embodiment_id: int = DEFAULT_EMBODIMENT_ID,
) -> tuple[Rldx1Preprocessor, Rldx1Postprocessor]:
    """Build the matched RLDX-1 preprocessor / postprocessor pair.

    Both share a single vendored :class:`RLDXProcessor`, so the forward and
    inverse transforms are guaranteed consistent.

    Args:
        stats: Dataset statistics for (de)normalization.
        env_action_dim: Original environment action dimension for decoding.
        max_state_dim: Padded state dimension.
        max_action_dim: Padded action dimension.
        action_horizon: Number of action steps per chunk.
        model_name: HuggingFace ID / local path of the Qwen3-VL processor.
        revision: Pinned git commit SHA for the processor download.
        normalize: Retained for API compatibility (processor always normalizes).
        use_percentiles: Prefer 1st/99th percentile bounds over min/max.
        use_relative_action: Encode actions relative to the current state.
        image_max_area: Target max area (pixels) for aspect-preserving resize.
        image_resize_m: Alignment multiple for resized/cropped dimensions.
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
        use_relative_action=use_relative_action,
        image_max_area=image_max_area,
        image_resize_m=image_resize_m,
        embodiment_id=embodiment_id,
        stats=stats,
    )
    postprocessor = Rldx1Postprocessor(
        preprocessor=preprocessor,
        env_action_dim=env_action_dim,
    )
    return preprocessor, postprocessor
