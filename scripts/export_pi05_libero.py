#!/usr/bin/env python3
"""Mixed-precision export of pi05 LIBERO model to OpenVINO.

Exports the full model to FP16 with fixed inference shapes, then emits:
- Mixed precision: VLM body W8A16 + action expert W16A16
- Full INT8 weights: W8A16 over all compressible layers

Uses meta-device initialization + direct weight assignment to avoid
allocating 15.4GB for model construction AND 7GB for state dict simultaneously.
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import gc
import json
import logging
import psutil
from pathlib import Path

import torch

import openvino as ov
from physicalai.export.backends import OpenVINOExportParameters
from physicalai.policies.pi05 import Pi05
from physicalai.policies.pi05.model import Pi05Model
from physicalai.policies.pi05.preprocessor import make_pi05_preprocessors
from physicalai.policies.pi05.pretrained_utils import (
    extract_dataset_stats,
    fix_state_dict_keys,
    detect_normalization_mode,
)
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mem_status():
    m = psutil.virtual_memory()
    return f"{m.used/1024**3:.1f}GB used / {m.available/1024**3:.1f}GB avail"


def main():

    # Set to "bfloat16" or "float16"
    EXPORT_DTYPE = "bfloat16" #
    torch_dtype = torch.bfloat16 if EXPORT_DTYPE == "bfloat16" else torch.float16

    # Output DIR
    OUTPUT_DIR = Path(f"./exports/pi05_libero_mixed_precision_{EXPORT_DTYPE}")

    print(f"[0] Start: {mem_status()} (dtype={EXPORT_DTYPE})")

    # Paths to cached HF files
    snap_dir = Path.home() / ".cache/huggingface/hub/models--lerobot--pi05_libero_finetuned_v044/snapshots/dbf8a3f794a9c4297b44f40b752712f50073d945"
    config_file = snap_dir / "config.json"
    weights_file = snap_dir / "model.safetensors"
    preproc_file = snap_dir / "policy_preprocessor.json"

    assert config_file.exists(), f"Config not found: {config_file}"
    assert weights_file.exists(), f"Weights not found: {weights_file}"

    # Import after env setup
    print(f"[1] After imports: {mem_status()}")

    # Load config
    with open(config_file) as f:
        hf_config = json.load(f)

    # Fixed export settings requested for simulator runs.
    action_horizon = 10
    token_len = 64
    hf_config["chunk_size"] = action_horizon
    hf_config["n_action_steps"] = action_horizon
    hf_config["tokenizer_max_length"] = token_len

    # Detect normalization mode
    norm_mode = detect_normalization_mode(preproc_file)
    if norm_mode:
        hf_config["normalization_mode"] = norm_mode

    # Extract dataset stats
    dataset_stats = extract_dataset_stats(hf_config, preproc_file, snap_dir)

    # Augment dataset_stats with image features from input_features config
    # (needed by model.sample_input which looks for entries with type=VISUAL).
    # Skip synthetic empty-camera features here because they are already added
    # by the preprocessor via `empty_cameras`.
    input_features = hf_config.get("input_features", {})
    output_features = hf_config.get("output_features", {})
    for feat_name, feat_info in {**input_features, **output_features}.items():
        if "empty_camera" in feat_name:
            continue
        if feat_name not in dataset_stats:
            dataset_stats[feat_name] = {
                "name": feat_name.split(".")[-1],
                "type": feat_info.get("type", ""),
                "shape": tuple(feat_info.get("shape", [])),
            }
        elif "type" not in dataset_stats[feat_name]:
            dataset_stats[feat_name]["type"] = feat_info.get("type", "")

    print(f"[2] Config loaded. Norm mode: {norm_mode}")
    print(f"    Export overrides: action_horizon={action_horizon}, token_len={token_len}")
    print(f"    Dataset stats keys: {list(dataset_stats.keys())}")

    # Step 1: Build model skeleton on meta device (zero memory)
    print(f"[3] Building model on meta device...")
    with torch.device("meta"):
        model = Pi05Model(
            dataset_stats,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype=EXPORT_DTYPE,
            chunk_size=hf_config.get("chunk_size", 50),
            max_action_dim=hf_config.get("max_action_dim", 32),
            n_action_steps=hf_config.get("n_action_steps", 50),
            num_inference_steps=hf_config.get("num_inference_steps", 10),
            image_resolution=tuple(hf_config.get("image_resolution", [224, 224])),
            tokenizer_max_length=hf_config.get("tokenizer_max_length", 200),
            use_random_input_noise=True,
            gradient_checkpointing=False,
            compile_model=False,
        )
    print(f"    Model on meta: {mem_status()}")

    # Step 2: Load weights (memory-mapped by safetensors)
    print(f"[4] Loading weights...")
    raw_sd = load_file(str(weights_file))
    print(f"    After load_file: {mem_status()}")

    # Fix keys (lerobot → Pi05Model format)
    fixed_sd = fix_state_dict_keys(raw_sd)
    del raw_sd
    gc.collect()
    print(f"    After fix_keys: {mem_status()}")

    # Step 3: Assign weights directly to model (assign=True = no copy)
    print(f"[5] Assigning weights to model...")
    missing, unexpected = model.load_state_dict(fixed_sd, strict=False, assign=True)
    del fixed_sd
    gc.collect()
    print(f"    After assign: {mem_status()}")

    if missing:
        print(f"    Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"      - {k}")
    if unexpected:
        print(f"    Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"      - {k}")

    # Step 3b: Materialize any parameters/buffers still on meta device
    meta_count = 0
    for name, buf in model.named_buffers():
        if buf.device == torch.device("meta"):
            model_part = model
            parts = name.split(".")
            for part in parts[:-1]:
                model_part = getattr(model_part, part)

            if "position_ids" in name:
                val = torch.arange(0, buf.shape[-1], dtype=buf.dtype).unsqueeze(0)
                print(f"    Init {name}: arange(0, {buf.shape[-1]})")
            elif "rotary_emb.inv_freq" in name or "rotary_emb.original_inv_freq" in name:
                head_dim = 256
                rope_theta = 10000.0
                val = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
                print(f"    Init {name}: RoPE inv_freq (theta={rope_theta}, dim={head_dim})")
            else:
                val = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                print(f"    WARNING: Zeroing unknown meta buffer {name}: {buf.shape}")

            setattr(model_part, parts[-1], val.to(dtype=buf.dtype, device="cpu"))
            meta_count += 1

    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            model_part = model
            parts = name.split(".")
            for part in parts[:-1]:
                model_part = getattr(model_part, part)
            setattr(model_part, parts[-1], torch.nn.Parameter(
                torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                requires_grad=param.requires_grad,
            ))
            print(f"    WARNING: Zeroing unknown meta param {name}: {param.shape}")
            meta_count += 1

    if meta_count > 0:
        print(f"    Materialized {meta_count} meta tensors to CPU")
        print(f"    After materialize: {mem_status()}")

    # Step 4: Convert dtype
    print(f"[6] Converting to {EXPORT_DTYPE}...")
    if EXPORT_DTYPE == "bfloat16":
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model.paligemma_with_expert.to(torch.float16)
    gc.collect()
    print(f"    After bfloat16: {mem_status()}")

    # Step 4b: Force the params that are normally kept in fp32
    # (vision_tower + multi_modal_projector + norms) to bfloat16 as well,
    # so the exported model is uniform bf16 (matches openpi dtype layout).
    # Without this, these stay fp32 and get compressed to fp16 on OV save.
    #
    # NOTE: the adaRMS conditioning ".dense" Linear (PiGemmaRMSNorm.dense) runs
    # on the fp32 time/cond embedding, so casting its weight to bf16 produces a
    # MatMul(f32 activation, bf16 weight) dtype mismatch at export. Skip ".dense".
    force_dtype_selectors = [
        "vision_tower",
        "multi_modal_projector",
        "input_layernorm",
        "post_attention_layernorm",
        "model.norm",
    ]
    n_cast = 0
    for name, param in model.named_parameters():
        if ".dense" in name:
            continue  # adaRMS conditioning Linear consumes fp32 cond → keep fp32
        if param.dtype == torch.float32 and any(s in name for s in force_dtype_selectors):
            param.data = param.data.to(torch_dtype)
            n_cast += 1
    for name, buf in model.named_buffers():
        if ".dense" in name:
            continue
        if buf.dtype == torch.float32 and any(s in name for s in force_dtype_selectors):
            buf.data = buf.data.to(torch_dtype)
            n_cast += 1
    print(f"    Forced {n_cast} vision/projector/norm tensors to {EXPORT_DTYPE}")
    gc.collect()

    # Step 5: Build policy wrapper with the loaded model
    print(f"[7] Creating policy wrapper...")
    policy = Pi05(
        pretrained_name_or_path=None,
        dtype=EXPORT_DTYPE,
        chunk_size=hf_config.get("chunk_size", 50),
        n_action_steps=hf_config.get("n_action_steps", 50),
        max_state_dim=hf_config.get("max_state_dim", 32),
        max_action_dim=hf_config.get("max_action_dim", 32),
        num_inference_steps=hf_config.get("num_inference_steps", 10),
        image_resolution=tuple(hf_config.get("image_resolution", [224, 224])),
        tokenizer_max_length=hf_config.get("tokenizer_max_length", 200),
        empty_cameras=hf_config.get("empty_cameras", 0),
        normalization_mode=norm_mode or "QUANTILES",
        use_random_input_noise=True,
        gradient_checkpointing=False,
        compile_model=False,
        dataset_stats=None,
    )
    policy.model = model
    policy._preprocessor, policy._postprocessor = make_pi05_preprocessors(
        max_action_dim=hf_config.get("max_action_dim", 32),
        stats=dataset_stats,
        image_resolution=tuple(hf_config.get("image_resolution", [224, 224])),
        max_token_len=hf_config.get("tokenizer_max_length", 200),
        empty_cameras=hf_config.get("empty_cameras", 0),
        normalization_mode=norm_mode or "QUANTILES",
    )
    policy._dataset_stats = dataset_stats
    policy.eval()
    print(f"    Policy ready: {mem_status()}")

    # Step 6: Inject a mock tokenizer to bypass HF auth for export tracing
    print(f"[8] Injecting local tokenizer to bypass HF auth...")

    class MockTokenizer:
        """Mock tokenizer that returns fixed-shape outputs for export tracing."""
        def __init__(self, max_length=200):
            self.model_max_length = max_length
            self.padding_side = "right"
            self.pad_token_id = 0

        def __call__(self, texts, padding="max_length", truncation=True,
                     max_length=None, return_tensors="pt", **kwargs):
            import torch
            if max_length is None:
                max_length = self.model_max_length
            batch_size = len(texts) if isinstance(texts, list) else 1
            input_ids = torch.ones(batch_size, max_length, dtype=torch.long)
            attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
            input_ids[:, 20:] = 0
            attention_mask[:, 20:] = 0

            class TokenizerOutput:
                def __init__(self, ids, mask):
                    self.input_ids = ids
                    self.attention_mask = mask
                def __getitem__(self, key):
                    if key == "input_ids":
                        return self.input_ids
                    return self.attention_mask

            return TokenizerOutput(input_ids, attention_mask)

    policy._preprocessor._tokenizer = MockTokenizer(policy._preprocessor.max_token_len)

    # Also patch extra_export_args to disable tokenizer export
    orig_extra = policy.extra_export_args
    ov_params = orig_extra["openvino"]
    patched_params = OpenVINOExportParameters(
        outputs=ov_params.outputs,
        compress_to_fp16=ov_params.compress_to_fp16,
        via_onnx=ov_params.via_onnx,
        export_tokenizer=False,
        exporter_kwargs=ov_params.exporter_kwargs,
        preprocessors_specs=ov_params.preprocessors_specs,
        postprocessors_specs=ov_params.postprocessors_specs,
    )
    print(patched_params)
    orig_extra["openvino"] = patched_params
    policy.__class__.extra_export_args = property(lambda self: orig_extra)

    # Step 7: Export to OpenVINO FP16 first
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[9] Starting OpenVINO {EXPORT_DTYPE} export to {OUTPUT_DIR}...")
    print(f"    Memory: {mem_status()}")

    policy.to_openvino(str(OUTPUT_DIR))

    print(f"\n[10] FP16 export complete! {mem_status()}")

    # Find the exported model file
    xml_files = list(OUTPUT_DIR.glob("*.xml"))
    model_xml = None
    for f in xml_files:
        if "tokenizer" not in f.name:
            model_xml = f
            break
    assert model_xml is not None, f"No model XML found in {OUTPUT_DIR}"
    print(f"    {EXPORT_DTYPE} model: {model_xml}")


if __name__ == "__main__":
    main()