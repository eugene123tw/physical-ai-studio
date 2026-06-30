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
#   rldx/data/state_action/pose.py            -> pose.py
#   rldx/data/state_action/action_chunking.py -> action_chunking.py
#   rldx/data/augmentations.py                -> augmentations.py
#   rldx/data/utils.py                        -> data_utils.py
#   rldx/data/types.py                        -> data_types.py
#   rldx/utils/qwen_vision_process.py         -> qwen_vision_process.py
# Original: https://github.com/NVIDIA/Isaac-GR00T
#
# Studio modifications (documented in-file): import paths rewritten to relative
# imports. The vendored parity oracles -- ``RLDXProcessor`` / ``RLDXDataCollator`` /
# ``build_processor`` (upstream ``rldx/model/core/processing_rldx.py``) and the
# numpy ``StateActionProcessor`` (upstream
# ``rldx/data/state_action/state_action_processor.py``) -- are no longer part of
# this runtime subpackage: Studio's native :class:`Rldx1Preprocessor` replaced
# those paths, so their only consumer is the parity test. They now live at
# ``tests/unit/policies/{processing_rldx,state_action_processor}.py`` (imports
# rewritten to absolute).
"""Vendored RLDX-1 data-processing pipeline (Apache-2.0).

Exports the live config/data types (modality + action config enums) plus the
shared embodiment-projector constants. The vendored ``RLDXProcessor`` collator
pipeline and the numpy ``StateActionProcessor`` were moved to
``tests/unit/policies/`` -- they are parity oracles with no production consumer,
so keeping them out of ``src`` avoids dragging that machinery into the live
import path.
"""

from physicalai.policies.rldx1.components.embodiments import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    GENERAL_EMBODIMENT_ID,
    NEW_EMBODIMENT_ID,
)

from .data_types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

__all__ = [
    "EMBODIMENT_TAG_TO_PROJECTOR_INDEX",
    "GENERAL_EMBODIMENT_ID",
    "NEW_EMBODIMENT_ID",
    "ActionConfig",
    "ActionFormat",
    "ActionRepresentation",
    "ActionType",
    "ModalityConfig",
]
