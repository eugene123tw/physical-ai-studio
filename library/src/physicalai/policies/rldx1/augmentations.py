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
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch

from physicalai.policies.rldx1.preprocessing import compute_aspect_area_resize_crop

if TYPE_CHECKING:
    # Only for static type checking; runtime import is lazy via _import_albumentations().
    import albumentations as A  # noqa: N812


def _import_albumentations() -> Any:  # noqa: ANN401
    """Lazy import of albumentations.

    Returns:
        The albumentations module.

    Raises:
        ImportError: If albumentations is not installed.
    """
    try:
        import albumentations as albumentations_module  # noqa: PLC0415
    except ImportError as e:
        msg = "RLDX-1 image augmentation requires albumentations.\n\nInstall with:\n    pip install albumentations"
        raise ImportError(msg) from e
    else:
        return albumentations_module


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
                    augmented = transform.replay(image=np.array(img), saved_augmentations=current_replay)  # type: ignore[attr-defined]
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


def _create_aspect_area_resize_and_crop_class() -> type:
    """Factory to create AspectAreaResizeAndCrop with the albumentations base class.

    Returns:
        The AspectAreaResizeAndCrop class.
    """
    albumentations_module = _import_albumentations()
    dual_transform_cls = albumentations_module.DualTransform

    class AspectAreaResizeAndCrop(dual_transform_cls):  # type: ignore[misc,valid-type]
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
            min_area: int | None = None,
        ) -> None:
            """Store the area budget, alignment multiple, and interpolation mode."""
            super().__init__(p=p)
            self.max_area = max_area
            self.m = m
            self.interpolation = interpolation
            self.min_area = min_area

        def apply(self, img, resize_hw=(0, 0), crop_coords=(0, 0, 0, 0), **params):  # noqa: ANN001, ANN003, ANN202, ARG002
            """Resize to ``resize_hw`` then slice ``crop_coords`` (x_min, y_min, x_max, y_max).

            Returns:
                Cropped image array.
            """
            h_r, w_r = resize_hw
            h, w = img.shape[:2]
            # INTER_AREA is a decimation filter; use cubic when the target enlarges.
            interpolation = cv2.INTER_CUBIC if h_r * w_r > h * w else self.interpolation
            resized = cv2.resize(img, (w_r, h_r), interpolation=interpolation)
            x_min, y_min, x_max, y_max = crop_coords
            return resized[y_min:y_max, x_min:x_max]

        def get_params_dependent_on_data(self, params, data):  # noqa: ANN001, ANN202, ARG002
            """Compute the resize target and centered crop box from the input shape.

            Returns:
                Dict with ``resize_hw`` and ``crop_coords`` keys.
            """
            h, w = params["shape"][:2]
            (h_r, w_r), (h_c, w_c) = compute_aspect_area_resize_crop(
                h,
                w,
                max_area=self.max_area,
                m=self.m,
                min_area=self.min_area,
            )
            y_min = (h_r - h_c) // 2
            x_min = (w_r - w_c) // 2
            return {
                "resize_hw": (h_r, w_r),
                "crop_coords": (x_min, y_min, x_min + w_c, y_min + h_c),
            }

        def get_transform_init_args_names(self):  # noqa: ANN202, PLR6301
            """Return the ctor arg names albumentations serializes for replay."""
            return ("max_area", "m", "interpolation", "min_area")

    return AspectAreaResizeAndCrop


def _create_fractional_crop_classes() -> tuple[type, type]:
    """Factory to create the fractional-crop transform classes.

    Returns:
        Tuple of ``(FractionalRandomCropAndResize, FractionalCenterCropAndResize)``.
    """
    albumentations_module = _import_albumentations()
    dual_transform_cls = albumentations_module.DualTransform

    class _FractionalCropAndResizeBase(dual_transform_cls):  # type: ignore[misc,valid-type]
        """Crop a fraction of the input then resize back to the pre-crop shape.

        Concrete subclasses choose the crop origin (random vs center). The output
        shape matches the input so downstream stages see a consistent spatial size.
        """

        def __init__(
            self,
            crop_fraction: float = 0.95,
            interpolation: int = cv2.INTER_LINEAR,
            p: float = 1.0,
        ) -> None:
            """Validate and store the crop fraction and interpolation mode.

            Raises:
                ValueError: If ``crop_fraction`` is not in ``(0.0, 1.0]``.
            """
            super().__init__(p=p)
            if not 0.0 < crop_fraction <= 1.0:
                msg = "crop_fraction must be in (0.0, 1.0]"
                raise ValueError(msg)
            self.crop_fraction = crop_fraction
            self.interpolation = interpolation

        def apply(self, img, crop_coords=(0, 0, 0, 0), out_hw=(0, 0), **params):  # noqa: ANN001, ANN202, ANN003, ARG002
            """Slice ``crop_coords`` then resize the crop back to ``out_hw``.

            Returns:
                Resized crop array.
            """
            x_min, y_min, x_max, y_max = crop_coords
            cropped = img[y_min:y_max, x_min:x_max]
            h_out, w_out = out_hw
            return cv2.resize(cropped, (w_out, h_out), interpolation=self.interpolation)

        def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:
            """Return the (y, x) crop origin. Implemented by subclasses."""
            raise NotImplementedError

        def get_params_dependent_on_data(self, params, data):  # noqa: ANN001, ANN202, ARG002
            """Compute the crop box (from the chosen origin) and the resize-back size.

            Returns:
                Dict with ``crop_coords`` and ``out_hw`` keys.
            """
            h, w = params["shape"][:2]
            ch = max(1, int(h * self.crop_fraction))
            cw = max(1, int(w * self.crop_fraction))
            y_min, x_min = self._origin(h - ch, w - cw)
            return {
                "crop_coords": (x_min, y_min, x_min + cw, y_min + ch),
                "out_hw": (h, w),
            }

        def get_transform_init_args_names(self) -> tuple[str, ...]:  # noqa: PLR6301
            """Return the ctor arg names albumentations serializes for replay."""
            return ("crop_fraction", "interpolation")

    class FractionalRandomCropAndResize(_FractionalCropAndResizeBase):
        """Random-position fractional crop, then resize back to the pre-crop (H, W)."""

        def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:  # noqa: PLR6301
            """Return a random (y, x) origin within the crop margin."""
            y = int(np.random.randint(0, max_y + 1)) if max_y > 0 else 0  # noqa: NPY002
            x = int(np.random.randint(0, max_x + 1)) if max_x > 0 else 0  # noqa: NPY002
            return y, x

    class FractionalCenterCropAndResize(_FractionalCropAndResizeBase):
        """Center fractional crop, then resize back to the pre-crop (H, W)."""

        def _origin(self, max_y: int, max_x: int) -> tuple[int, int]:  # noqa: PLR6301
            """Return the centered (y, x) origin."""
            return max_y // 2, max_x // 2

    return FractionalRandomCropAndResize, FractionalCenterCropAndResize


# Lazy class creation, mirroring physicalai.policies.groot.components.transformer.
_AspectAreaResizeAndCrop: type | None = None
_FractionalRandomCropAndResize: type | None = None
_FractionalCenterCropAndResize: type | None = None


def get_aspect_area_resize_and_crop_class() -> type:
    """Get AspectAreaResizeAndCrop class, creating it on first use.

    Returns:
        The AspectAreaResizeAndCrop class.
    """
    global _AspectAreaResizeAndCrop  # noqa: PLW0603
    if _AspectAreaResizeAndCrop is None:
        _AspectAreaResizeAndCrop = _create_aspect_area_resize_and_crop_class()
    return _AspectAreaResizeAndCrop


def get_fractional_crop_classes() -> tuple[type, type]:
    """Get the fractional-crop classes, creating them on first use.

    Returns:
        Tuple of ``(FractionalRandomCropAndResize, FractionalCenterCropAndResize)``.
    """
    global _FractionalRandomCropAndResize, _FractionalCenterCropAndResize
    if _FractionalRandomCropAndResize is None or _FractionalCenterCropAndResize is None:
        _FractionalRandomCropAndResize, _FractionalCenterCropAndResize = _create_fractional_crop_classes()
    return _FractionalRandomCropAndResize, _FractionalCenterCropAndResize


def __getattr__(name: str) -> type:
    """Lazy attribute access for the albumentations-backed transform classes.

    Args:
        name: Attribute name to access.

    Returns:
        The requested transform class.

    Raises:
        AttributeError: If the attribute is not found.
    """
    if name == "AspectAreaResizeAndCrop":
        return get_aspect_area_resize_and_crop_class()
    if name == "FractionalRandomCropAndResize":
        return get_fractional_crop_classes()[0]
    if name == "FractionalCenterCropAndResize":
        return get_fractional_crop_classes()[1]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


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
    albumentations_module = _import_albumentations()
    aspect_area_resize_and_crop_cls = get_aspect_area_resize_and_crop_class()

    train_list: list = [
        aspect_area_resize_and_crop_cls(
            max_area=image_max_area,
            m=image_resize_m,
            interpolation=cv2.INTER_AREA,
            min_area=image_min_area,
        ),
    ]
    eval_list: list = [
        aspect_area_resize_and_crop_cls(
            max_area=image_max_area,
            m=image_resize_m,
            interpolation=cv2.INTER_AREA,
            min_area=image_min_area,
        ),
    ]

    if random_crop_fraction is not None:
        if not 0.0 < random_crop_fraction <= 1.0:
            msg = f"random_crop_fraction must be in (0.0, 1.0], got {random_crop_fraction!r}"
            raise ValueError(msg)
        fractional_random_cls, fractional_center_cls = get_fractional_crop_classes()
        train_list.append(
            fractional_random_cls(crop_fraction=random_crop_fraction, interpolation=cv2.INTER_LINEAR),
        )
        eval_list.append(
            fractional_center_cls(crop_fraction=random_crop_fraction, interpolation=cv2.INTER_LINEAR),
        )

    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_list.append(albumentations_module.Rotate(limit=random_rotation_angle, p=1.0))

    if color_jitter_params is not None:
        train_list.append(
            albumentations_module.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            ),
        )

    train_transform = albumentations_module.ReplayCompose(train_list, p=1.0)
    eval_transform = albumentations_module.Compose(eval_list)
    return train_transform, eval_transform
