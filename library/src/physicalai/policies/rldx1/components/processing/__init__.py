# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
#
# Vendored RLDX-1 data-processing pipeline.
#
# These modules are copied verbatim (with import paths rewritten to this
# subpackage) from RLWRLD/RLDX-1, which is itself modified from NVIDIA Isaac
# GR00T N1.7. The original code is licensed under Apache-2.0; the per-file
# SPDX headers and provenance notices are preserved.
#
# Upstream: https://github.com/rlwrld/RLDX-1
#   rldx/model/core/processing_rldx.py        -> processing_rldx.py
#   rldx/data/state_action/state_action_processor.py -> state_action_processor.py
#   rldx/data/state_action/pose.py            -> pose.py
#   rldx/data/state_action/action_chunking.py -> action_chunking.py
#   rldx/data/augmentations.py                -> augmentations.py
#   rldx/data/utils.py                        -> data_utils.py
#   rldx/data/types.py                        -> data_types.py
#   rldx/utils/qwen_vision_process.py         -> qwen_vision_process.py
# Original: https://github.com/NVIDIA/Isaac-GR00T
#
# Studio modifications (documented in-file): import paths rewritten to relative
# imports; the trailing `AutoProcessor.register(RLDXConfig, RLDXProcessor)` hook
# in processing_rldx.py was removed to keep this subpackage decoupled from the
# model config and to avoid the AutoProcessor trust_remote_code auto-load path;
# the single-subclass `BaseProcessor` (rldx/data/interfaces.py) was merged into
# `RLDXProcessor`, which now subclasses `transformers.ProcessorMixin` directly.
"""Vendored RLDX-1 data-processing pipeline (Apache-2.0)."""

from .data_types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)
from .processing_rldx import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    GENERAL_EMBODIMENT_ID,
    NEW_EMBODIMENT_ID,
    RLDXDataCollator,
    RLDXProcessor,
    build_processor,
)
from .state_action_processor import StateActionProcessor


__all__ = [
    "EMBODIMENT_TAG_TO_PROJECTOR_INDEX",
    "GENERAL_EMBODIMENT_ID",
    "NEW_EMBODIMENT_ID",
    "ActionConfig",
    "ActionFormat",
    "ActionRepresentation",
    "ActionType",
    "ModalityConfig",
    "RLDXDataCollator",
    "RLDXProcessor",
    "StateActionProcessor",
    "build_processor",
]
