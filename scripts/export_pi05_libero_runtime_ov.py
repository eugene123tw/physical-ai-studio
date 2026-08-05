#!/usr/bin/env python3
"""Export a meta-loaded Pi05 LIBERO checkpoint for Runtime InferenceModel."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from physicalai.policies.pi05 import Pi05
from physicalai.policies.pi05.model import Pi05Model
from physicalai.policies.pi05.preprocessor import make_pi05_preprocessors
from physicalai.policies.pi05.pretrained_utils import (
    detect_normalization_mode,
    extract_dataset_stats,
    fix_state_dict_keys,
)

MODEL_ID = "models--lerobot--pi05_libero_finetuned_v044"
MODEL_REVISION = "dbf8a3f794a9c4297b44f40b752712f50073d945"
ACTION_HORIZON = 10
TOKEN_LENGTH = 64
SNAPFLOW_MISSING_KEYS = frozenset(
    {
        "target_time_mlp_in.weight",
        "target_time_mlp_in.bias",
        "target_time_mlp_out.weight",
        "target_time_mlp_out.bias",
    },
)


def materialize_meta_tensors(model: Pi05Model) -> None:
    """Initialize tensors absent from the pre-SnapFlow checkpoint."""
    for name, buffer in model.named_buffers():
        if buffer.device != torch.device("meta"):
            continue

        module = model
        path = name.split(".")
        for attribute in path[:-1]:
            module = getattr(module, attribute)

        if "position_ids" in name:
            value = torch.arange(buffer.shape[-1], dtype=buffer.dtype).unsqueeze(0)
        elif "rotary_emb.inv_freq" in name or "rotary_emb.original_inv_freq" in name:
            head_dim = 256
            value = 1.0 / (10_000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        else:
            value = torch.zeros(buffer.shape, dtype=buffer.dtype)
        setattr(module, path[-1], value)

    for name, parameter in model.named_parameters():
        if parameter.device != torch.device("meta"):
            continue

        module = model
        path = name.split(".")
        for attribute in path[:-1]:
            module = getattr(module, attribute)
        setattr(
            module,
            path[-1],
            torch.nn.Parameter(torch.zeros(parameter.shape, dtype=parameter.dtype), requires_grad=parameter.requires_grad),
        )


def add_feature_metadata(
    dataset_stats: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Add visual feature entries required for Pi05 export input schema."""
    features = {**config.get("input_features", {}), **config.get("output_features", {})}
    for feature_name, feature_info in features.items():
        if "empty_camera" in feature_name:
            continue
        stats = dataset_stats.setdefault(
            feature_name,
            {
                "name": feature_name.split(".")[-1],
                "shape": tuple(feature_info.get("shape", [])),
            },
        )
        stats.setdefault("type", feature_info.get("type", ""))


def load_policy(snapshot_dir: Path, export_dtype: str) -> Pi05:
    """Assemble a Pi05 policy without allocating a duplicate model instance."""
    config_path = snapshot_dir / "config.json"
    weights_path = snapshot_dir / "model.safetensors"
    preprocessor_path = snapshot_dir / "policy_preprocessor.json"
    for path in (config_path, weights_path, preprocessor_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact is missing: {path}")

    with config_path.open(encoding="utf-8") as config_file:
        config: dict[str, Any] = json.load(config_file)
    config["chunk_size"] = ACTION_HORIZON
    config["n_action_steps"] = ACTION_HORIZON
    config["tokenizer_max_length"] = TOKEN_LENGTH

    normalization_mode = detect_normalization_mode(preprocessor_path) or "QUANTILES"
    config["normalization_mode"] = normalization_mode
    dataset_stats = extract_dataset_stats(config, preprocessor_path, snapshot_dir)
    add_feature_metadata(dataset_stats, config)

    with torch.device("meta"):
        model = Pi05Model(
            dataset_stats,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype=export_dtype,
            chunk_size=ACTION_HORIZON,
            max_action_dim=config["max_action_dim"],
            n_action_steps=ACTION_HORIZON,
            num_inference_steps=config["num_inference_steps"],
            image_resolution=tuple(config["image_resolution"]),
            tokenizer_max_length=TOKEN_LENGTH,
            gradient_checkpointing=False,
            compile_model=False,
            use_random_input_noise=True,
            snapflow_enabled=False,
        )

    state_dict = fix_state_dict_keys(load_file(str(weights_path)))
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    gc.collect()

    disallowed_missing = set(missing) - SNAPFLOW_MISSING_KEYS
    if disallowed_missing or unexpected:
        raise RuntimeError(f"Unexpected weight load result: missing={missing}, unexpected={unexpected}")

    materialize_meta_tensors(model)
    if export_dtype == "bfloat16":
        model.paligemma_with_expert.to_bfloat16_for_selected_params(export_dtype)
        force_bfloat16 = (
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, parameter in model.named_parameters():
            if ".dense" not in name and parameter.dtype == torch.float32 and any(
                selector in name for selector in force_bfloat16
            ):
                parameter.data = parameter.data.to(torch.bfloat16)
        for name, buffer in model.named_buffers():
            if ".dense" not in name and buffer.dtype == torch.float32 and any(
                selector in name for selector in force_bfloat16
            ):
                buffer.data = buffer.data.to(torch.bfloat16)

    policy = Pi05(
        dtype=export_dtype,
        chunk_size=ACTION_HORIZON,
        n_action_steps=ACTION_HORIZON,
        max_state_dim=config["max_state_dim"],
        max_action_dim=config["max_action_dim"],
        num_inference_steps=config["num_inference_steps"],
        image_resolution=tuple(config["image_resolution"]),
        empty_cameras=config.get("empty_cameras", 0),
        tokenizer_max_length=TOKEN_LENGTH,
        normalization_mode=normalization_mode,
        use_random_input_noise=True,
        snapflow_enabled=False,
        gradient_checkpointing=False,
        compile_model=False,
    )
    preprocessor, postprocessor = make_pi05_preprocessors(
        max_action_dim=config["max_action_dim"],
        stats=dataset_stats,
        image_resolution=tuple(config["image_resolution"]),
        max_token_len=TOKEN_LENGTH,
        empty_cameras=config.get("empty_cameras", 0),
        normalization_mode=normalization_mode,
    )
    policy.model = model
    policy._preprocessor = preprocessor
    policy._postprocessor = postprocessor
    policy._dataset_stats = dataset_stats
    return policy.eval()


def main() -> None:
    """Export Pi05 and its Runtime preprocessing contract to OpenVINO."""
    parser = argparse.ArgumentParser(description="Export Pi05 LIBERO to OpenVINO for Runtime InferenceModel")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/pi05_libero_runtime_ov"),
        help="Directory for pi05.xml, tokenizer.xml, and manifest.json",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="PyTorch model dtype before OpenVINO conversion",
    )
    args = parser.parse_args()

    snapshot_dir = Path.home() / ".cache" / "huggingface" / "hub" / MODEL_ID / "snapshots" / MODEL_REVISION
    print("Loading Pi05Model on meta device and assigning safetensors weights...")
    policy = load_policy(snapshot_dir, args.dtype)

    print(f"Exporting Runtime artifact to {args.output_dir}...")
    policy.to_openvino(args.output_dir)

    required_artifacts = (args.output_dir / "pi05.xml", args.output_dir / "tokenizer.xml", args.output_dir / "manifest.json")
    missing_artifacts = [path for path in required_artifacts if not path.is_file()]
    if missing_artifacts:
        raise RuntimeError(f"OpenVINO export did not create required artifacts: {missing_artifacts}")
    print("Export complete: pi05.xml, tokenizer.xml, and manifest.json created.")


if __name__ == "__main__":
    main()
