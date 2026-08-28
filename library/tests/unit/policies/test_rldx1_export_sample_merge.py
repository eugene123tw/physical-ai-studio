# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for dynamic export sample merging in ``Rldx1``."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from physicalai.policies.rldx1.policy import Rldx1


def _make_policy_with_model(model: object) -> Rldx1:
    policy = object.__new__(Rldx1)
    policy.model = model
    return policy


def test_trim_export_sample_keeps_live_observation_tensors() -> None:
    """Only prompt tensors come from export_sample; observation tensors stay live."""
    model = SimpleNamespace(
        input_keys=("pixel_values", "input_ids", "position_ids", "attention_mask", "state"),
        export_sample={
            "pixel_values": torch.full((1, 1, 2, 2), -9.0),
            "state": torch.full((1, 4), -9.0),
            "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
            "position_ids": torch.arange(9, dtype=torch.long).view(3, 1, 3),
            "attention_mask": torch.tensor([[0, 1, 1]], dtype=torch.long),
        },
    )
    policy = _make_policy_with_model(model)

    live_input = {
        "pixel_values": torch.full((1, 1, 2, 2), 3.0),
        "state": torch.full((1, 4), 4.0),
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "position_ids": torch.zeros(3, 1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    trimmed = policy._trim_export_sample(live_input)

    assert trimmed is not None
    assert torch.equal(trimmed["pixel_values"], live_input["pixel_values"])
    assert torch.equal(trimmed["state"], live_input["state"])
    assert torch.equal(trimmed["input_ids"], model.export_sample["input_ids"])
    assert torch.equal(trimmed["position_ids"], model.export_sample["position_ids"])
    assert torch.equal(trimmed["attention_mask"], model.export_sample["attention_mask"])


def test_trim_export_sample_without_export_sample_uses_input() -> None:
    """When no export sample is present, trimming is a pure key filter."""
    model = SimpleNamespace(input_keys=("pixel_values", "state"))
    policy = _make_policy_with_model(model)

    live_input = {
        "pixel_values": torch.full((1, 1, 2, 2), 1.0),
        "state": torch.full((1, 4), 2.0),
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
    }

    trimmed = policy._trim_export_sample(live_input)

    assert trimmed is not None
    assert set(trimmed) == {"pixel_values", "state"}
    assert torch.equal(trimmed["pixel_values"], live_input["pixel_values"])
    assert torch.equal(trimmed["state"], live_input["state"])
