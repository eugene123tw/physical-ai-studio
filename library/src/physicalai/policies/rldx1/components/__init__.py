# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
#
# Vendored model code for the RLDX-1 policy.
#
# These modules are copied verbatim (with import paths rewritten to this
# package) from RLWRLD/RLDX-1, which is itself modified from NVIDIA Isaac
# GR00T N1.7. The original code is licensed under Apache-2.0; the per-file
# SPDX headers and provenance notices are preserved.
#
# Upstream: https://github.com/rlwrld/RLDX-1  (rldx/model/modules, rldx/model/core)
# Original: https://github.com/NVIDIA/Isaac-GR00T
#
# The architecture (nn.Module) code needed to construct and load the
# RLDX-1-PT checkpoint is vendored here. The training/inference data pipeline
# (RLDXProcessor + StateActionProcessor + image/vision transforms) is vendored
# under the `processing/` subpackage so Studio feeds the model identical inputs
# to upstream.
"""Vendored RLDX-1 model architecture (Apache-2.0)."""
