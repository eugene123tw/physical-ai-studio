# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Vendored from RLWRLD/RLDX-1 (Apache-2.0)

"""Train-time image augmentation for the RLDX-1 policy (albumentations).

Faithful port of the upstream ``rldx/data/augmentations.py`` pipeline, wired to
the PAS-native geometry helper in :mod:`preprocessing`. Both the eval and train
paths run through ``AspectAreaResizeAndCrop`` (deterministic area-budget resize +
``m``-aligned center crop); the train path adds the stochastic stages that
``FT-*`` checkpoints were trained with:

1. ``AspectAreaResizeAndCrop`` -- deterministic area-budget resize + ``m``-aligned
   center crop (same geometry as the eval path, expressed as an albumentations
   ``DualTransform`` so it can compose with the stochastic stages).
2. optional fractional crop + resize-back (train: random position; eval: center).
3. optional ``Rotate`` / ``ColorJitter`` (train only).

:func:`apply_with_replay` reuses one sampled ``ReplayCompose`` blob across every
frame and view of a sample, so the 4 video frames share identical random params
(crop origin, rotation, jitter) -- matching upstream ``_get_vlm_inputs``.
"""

from __future__ import annotations

import warnings

import albumentations as A
import cv2
import numpy as np
import torch

from physicalai.policies.rldx1.preprocessing import compute_aspect_area_resize_crop


def apply_with_replay(
    transform: A.BaseCompose,
    images: list,
    replay: dict | None = None,
) -> tuple[list[torch.Tensor], dict | None]:
    """Apply an albumentations transform to multiple images with replay.

    When ``transform`` is an :class:`A.ReplayCompose`, the first image produces
    replay data that is reused for every subsequent image so the random params
    (rotation, jitter, crop origin, ...) are identical across all frames/views.

    Args:
        transform: ``A.Compose`` (deterministic) or ``A.ReplayCompose`` (train).
        images: Iterable of PIL images or numpy arrays.
        replay: Optional replay blob from an earlier call. When ``None`` the
            first image creates fresh replay data.

    Returns:
        ``(tensors, replay)`` -- a list of ``uint8`` tensors ``(C, H, W)`` and
        the replay blob (or ``None`` for a plain ``A.Compose``).

    Raises:
        ValueError: If a transform returns an unexpected dtype.
    """
    transformed_tensors: list[torch.Tensor] = []
    current_replay = replay
    has_replay = hasattr(transform, "replay")

    for img in images:
        if has_replay:
            if current_replay is None:
                augmented = transform(image=np.array(img))
                current_replay = augmented["replay"]
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    augmented = transform.replay(image=np.array(img), saved_augmentations=current_replay)
            img_array = augmented["image"]
        else:
            augmented = transform(image=np.array(img))
            img_array = augmented["image"]

        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            msg = f"Unexpected data type: {img_array.dtype}"
            raise ValueError(msg)

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        transformed_tensors.append(img_tensor)

    return transformed_tensors, current_replay


class AspectAreaResizeAndCrop(A.DualTransform):
    """Resize preserving aspect ratio (area-constrained) then center-crop.

    Step 1: resize so total area <= ``max_area`` with the short side aligned to
    ``m`` and the aspect ratio preserved. Step 2: center-crop both dims to
    multiples of ``m``. Deterministic; the integer geometry is shared with the
    eval path via :func:`compute_aspect_area_resize_crop`.
    """

    def __init__(
        self,
        max_area: int = 65536,
        m: int = 32,
        interpolation: int = cv2.INTER_AREA,
        p: float = 1.0,
        always_apply: bool | None = None,
        min_area: int | None = None,
    ) -> None:
        """Store the area budget, alignment multiple, and interpolation mode."""
        super().__init__(p=p, always_apply=always_apply)
        self.max_area = max_area
        self.m = m
        self.interpolation = interpolation
        self.min_area = min_area

    def apply(self, img, resize_hw=(0, 0), crop_coords=(0, 0, 0, 0), **params):  # noqa: ANN001, ANN201, ARG002
        """Resize to ``resize_hw`` then slice ``crop_coords`` (x_min, y_min, x_max, y_max)."""
        h_r, w_r = resize_hw
        h, w = img.shape[:2]
        # INTER_AREA is a decimation filter; use cubic when the target enlarges.
        interpolation = cv2.INTER_CUBIC if h_r * w_r > h * w else self.interpolation
        resized = cv2.resize(img, (w_r, h_r), interpolation=interpolation)
        x_min, y_min, x_max, y_max = crop_coords
        return resized[y_min:y_max, x_min:x_max]

    def get_params_dependent_on_data(self, params, data):  # noqa: ANN001, ANN201, ARG002
        """Compute the resize target and centered crop box from the input shape."""
        h, w = params["shape"][:2]
        (h_r, w_r), (h_c, w_c) = compute_aspect_area_resize_crop(
            h, w, max_area=self.max_area, m=self.m, min_area=self.min_area
        )
        y_min = (h_r - h_c) // 2
        x_min = (w_r - w_c) // 2
        return {
            "resize_hw": (h_r, w_r),
            "crop_coords": (x_min, y_min, x_min + w_c, y_min + h_c),
        }

    def get_transform_init_args_names(self):  # noqa: ANN201
        """Return the ctor arg names albumentations serializes for replay."""
        return ("max_area", "m", "interpolation", "min_area")


class _FractionalCropAndResizeBase(A.DualTransform):
    """Crop a fraction of the input then resize back to the pre-crop shape.

    Concrete subclasses choose the crop origin (random vs center). The output
    shape matches the input so downstream stages see a consistent spatial size.
    """

    def __init__(
        self,
        crop_fraction: float = 0.95,
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0,
        always_apply: bool | None = None,
    ) -> None:
        """Validate and store the crop fraction and interpolation mode."""
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            msg = "crop_fraction must be in (0.0, 1.0]"
            raise ValueError(msg)
        self.crop_fraction = crop_fraction
        self.interpolation = interpolation

    def apply(self, img, crop_coords=(0, 0, 0, 0), out_hw=(0, 0), **params):  # noqa: ANN001, ANN201, ARG002
        """Slice ``crop_coords`` then resize the crop back to ``out_hw``."""
        x_min, y_min, x_max, y_max = crop_coords
        cropped = img[y_min:y_max, x_min:x_max]
        h_out, w_out = out_hw
        return cv2.resize(cropped, (w_out, h_out), interpolation=self.interpolation)

    def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:
        """Return the (y, x) crop origin. Implemented by subclasses."""
        raise NotImplementedError

    def get_params_dependent_on_data(self, params, data):  # noqa: ANN001, ANN201, ARG002
        """Compute the crop box (from the chosen origin) and the resize-back size."""
        h, w = params["shape"][:2]
        ch = max(1, int(h * self.crop_fraction))
        cw = max(1, int(w * self.crop_fraction))
        y_min, x_min = self._origin(h - ch, w - cw)
        return {
            "crop_coords": (x_min, y_min, x_min + cw, y_min + ch),
            "out_hw": (h, w),
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return the ctor arg names albumentations serializes for replay."""
        return ("crop_fraction", "interpolation")


class FractionalRandomCropAndResize(_FractionalCropAndResizeBase):
    """Random-position fractional crop, then resize back to the pre-crop (H, W)."""

    def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:
        """Return a random (y, x) origin within the crop margin."""
        y = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0
        x = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0
        return y, x


class FractionalCenterCropAndResize(_FractionalCropAndResizeBase):
    """Center fractional crop, then resize back to the pre-crop (H, W)."""

    def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:
        """Return the centered (y, x) origin."""
        return max_y // 2, max_x // 2


def build_image_transformations_albumentations(
    image_max_area: int = 65536,
    image_resize_m: int = 32,
    random_crop_fraction: float | None = None,
    random_rotation_angle: int | None = None,
    color_jitter_params: dict[str, float] | None = None,
    image_min_area: int | None = None,
) -> tuple[A.BaseCompose, A.BaseCompose]:
    """Build the ``(train_transform, eval_transform)`` pair.

    The eval transform is deterministic (aspect-area resize + ``m``-aligned
    center crop). The train transform adds stochastic augmentation (random
    fractional crop, rotate, color jitter) on the same step-1 geometry, wrapped
    in an :class:`A.ReplayCompose` so :func:`apply_with_replay` can share one
    sampled param set across all frames and views of a sample.

    Args:
        image_max_area: Area budget for the aspect-area resize (``256*256``).
        image_resize_m: Alignment multiple for the resized/cropped dims.
        random_crop_fraction: Fractional crop size in ``(0, 1]``. ``None`` skips
            the crop stage entirely (train == eval geometry).
        random_rotation_angle: Optional ``A.Rotate(limit=...)`` (train only).
        color_jitter_params: Optional ``A.ColorJitter`` params
            ``{"brightness", "contrast", "saturation", "hue"}`` (train only).
        image_min_area: Optional minimum pixel-area floor; tiny frames are
            upscaled to it before the crop (``None`` keeps never-upscale).

    Returns:
        ``(train_transform, eval_transform)`` -- albumentations composes taking
        ``image=np.ndarray`` and returning dicts.

    Raises:
        ValueError: If ``random_crop_fraction`` is outside ``(0, 1]``.
    """
    train_list: list = [
        AspectAreaResizeAndCrop(
            max_area=image_max_area, m=image_resize_m, interpolation=cv2.INTER_AREA, min_area=image_min_area
        ),
    ]
    eval_list: list = [
        AspectAreaResizeAndCrop(
            max_area=image_max_area, m=image_resize_m, interpolation=cv2.INTER_AREA, min_area=image_min_area
        ),
    ]

    if random_crop_fraction is not None:
        if not 0.0 < random_crop_fraction <= 1.0:
            msg = f"random_crop_fraction must be in (0.0, 1.0], got {random_crop_fraction!r}"
            raise ValueError(msg)
        train_list.append(
            FractionalRandomCropAndResize(crop_fraction=random_crop_fraction, interpolation=cv2.INTER_LINEAR),
        )
        eval_list.append(
            FractionalCenterCropAndResize(crop_fraction=random_crop_fraction, interpolation=cv2.INTER_LINEAR),
        )

    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_list.append(A.Rotate(limit=random_rotation_angle, p=1.0))

    if color_jitter_params is not None:
        train_list.append(
            A.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            ),
        )

    train_transform = A.ReplayCompose(train_list, p=1.0)
    eval_transform = A.Compose(eval_list)
    return train_transform, eval_transform
