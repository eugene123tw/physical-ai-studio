#!/usr/bin/env python3
"""Benchmark raw Pi05Model latency with LIBERO-shaped synthetic inputs."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from physicalai.data.observation import Observation
from physicalai.policies.pi05 import Pi05
from physicalai.policies.pi05.model import Pi05Model
from physicalai.policies.pi05.preprocessor import make_pi05_preprocessors
from physicalai.policies.pi05.pretrained_utils import extract_dataset_stats, fix_state_dict_keys


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


def synchronize(device: str | torch.device) -> None:
    """Wait for all queued work on an accelerator device."""
    device_type = torch.device(device).type
    if device_type == "xpu":
        torch.xpu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()


class TimedPi05(Pi05):
    """Pi05 policy that records full action-chunk latency during rollouts."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Initialize the policy and its latency samples."""
        super().__init__(*args, **kwargs)
        self.latencies_ms: list[float] = []

    def predict_action_chunk(self, batch: Observation) -> torch.Tensor:
        """Predict an action chunk and record synchronized policy latency."""
        start = time.perf_counter()
        actions = super().predict_action_chunk(batch)
        synchronize(self.device)
        self.latencies_ms.append((time.perf_counter() - start) * 1000)
        return actions


def materialize_meta_tensors(model: Pi05Model) -> None:
    """Initialize model tensors that are not stored in the checkpoint."""
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


def load_policy(device: str, compile_model: bool, *, measure_latency: bool = False) -> Pi05:
    """Load Pi05Model on meta and attach it to a policy wrapper."""
    snapflow_enabled = False
    snapshot_dir = Path.home() / ".cache" / "huggingface" / "hub" / MODEL_ID / "snapshots" / MODEL_REVISION
    config_path = snapshot_dir / "config.json"
    weights_path = snapshot_dir / "model.safetensors"
    preprocessor_path = snapshot_dir / "policy_preprocessor.json"
    for path in (config_path, weights_path, preprocessor_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required model artifact is missing: {path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config["chunk_size"] = ACTION_HORIZON
    config["n_action_steps"] = ACTION_HORIZON
    config["tokenizer_max_length"] = TOKEN_LENGTH
    dataset_stats = extract_dataset_stats(config, preprocessor_path, snapshot_dir)

    with torch.device("meta"):
        model = Pi05Model(
            dataset_stats,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype="bfloat16",
            chunk_size=ACTION_HORIZON,
            max_action_dim=config["max_action_dim"],
            n_action_steps=ACTION_HORIZON,
            num_inference_steps=config["num_inference_steps"],
            image_resolution=tuple(config["image_resolution"]),
            tokenizer_max_length=TOKEN_LENGTH,
            gradient_checkpointing=False,
            compile_model=compile_model,
            use_random_input_noise=True,
            snapflow_enabled=snapflow_enabled,
        )

    state_dict = fix_state_dict_keys(load_file(str(weights_path)))
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    gc.collect()
    disallowed_missing = set(missing)
    if not snapflow_enabled:
        disallowed_missing -= SNAPFLOW_MISSING_KEYS
    if disallowed_missing or unexpected:
        raise RuntimeError(f"Unexpected weight load result: missing={missing}, unexpected={unexpected}")

    materialize_meta_tensors(model)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    force_bfloat16 = (
        "vision_tower",
        "multi_modal_projector",
        "input_layernorm",
        "post_attention_layernorm",
        "model.norm",
    )
    for name, parameter in model.named_parameters():
        if ".dense" not in name and parameter.dtype == torch.float32 and any(selector in name for selector in force_bfloat16):
            parameter.data = parameter.data.to(torch.bfloat16)
    for name, buffer in model.named_buffers():
        if ".dense" not in name and buffer.dtype == torch.float32 and any(selector in name for selector in force_bfloat16):
            buffer.data = buffer.data.to(torch.bfloat16)

    policy_type = TimedPi05 if measure_latency else Pi05
    policy = policy_type(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16",
        chunk_size=ACTION_HORIZON,
        max_action_dim=config["max_action_dim"],
        n_action_steps=ACTION_HORIZON,
        num_inference_steps=config["num_inference_steps"],
        image_resolution=tuple(config["image_resolution"]),
        tokenizer_max_length=TOKEN_LENGTH,
        gradient_checkpointing=False,
        compile_model=compile_model,
        use_random_input_noise=True,
        snapflow_enabled=snapflow_enabled,
    )
    preprocessor, postprocessor = make_pi05_preprocessors(
        max_action_dim=config["max_action_dim"],
        stats=dataset_stats,
        image_resolution=tuple(config["image_resolution"]),
        max_token_len=TOKEN_LENGTH,
        empty_cameras=config.get("empty_cameras", 0),
        normalization_mode=config.get("normalization_mode", "QUANTILES"),
    )
    policy.model = model
    policy._preprocessor = preprocessor
    policy._postprocessor = postprocessor
    policy._dataset_stats = dataset_stats
    return policy.to(device).eval()


def make_batch(device: str, batch_size: int) -> Observation:
    """Create a LIBERO-shaped observation for policy-level inference."""
    image_size = 224
    return Observation(
        state=torch.zeros(batch_size, 8, device=device),
        task=["pick up the red mug"] * batch_size,
        images={
            "agentview": torch.rand(batch_size, 3, image_size, image_size, device=device),
            "eye_in_hand": torch.rand(batch_size, 3, image_size, image_size, device=device),
        },
    )


def percentile(samples: list[float], percentile_value: float) -> float:
    """Return a nearest-rank percentile from sorted latency samples."""
    index = min(int(len(samples) * percentile_value), len(samples) - 1)
    return samples[index]


def print_latency_summary(latencies_ms: list[float], label: str) -> None:
    """Print latency percentiles for synchronized policy calls."""
    if not latencies_ms:
        print(f"No {label} samples were recorded.")
        return

    sorted_latencies = sorted(latencies_ms)
    mean_ms = sum(sorted_latencies) / len(sorted_latencies)
    print(f"\n{label} over {len(sorted_latencies)} action chunks:")
    print(f"  mean: {mean_ms:.2f} ms")
    print(f"  p50:  {percentile(sorted_latencies, 0.50):.2f} ms")
    print(f"  p90:  {percentile(sorted_latencies, 0.90):.2f} ms")
    print(f"  p99:  {percentile(sorted_latencies, 0.99):.2f} ms")
    print(f"  min:  {sorted_latencies[0]:.2f} ms")
    print(f"  max:  {sorted_latencies[-1]:.2f} ms")


def run_libero_benchmark(policy: TimedPi05, args: argparse.Namespace) -> None:
    """Evaluate the policy on real LIBERO simulator observations."""
    from physicalai.benchmark.gyms import LiberoBenchmark

    benchmark = LiberoBenchmark(
        task_suite=args.task_suite,
        task_ids=args.task_ids or None,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        record_mode="none",
    )
    results = benchmark.evaluate(policy, continue_on_error=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_json(args.output_dir / "results.json")
    results.to_csv(args.output_dir / "results.csv")

    print(results.summary())
    print_latency_summary(policy.latencies_ms, "End-to-end Pi05 policy latency")
    print(f"Results written to {args.output_dir}")


def main() -> None:
    """Run synthetic latency timing or real LIBERO rollout evaluation."""
    parser = argparse.ArgumentParser(description="Pi05 policy latency and LIBERO benchmark")
    parser.add_argument("--device", default="xpu", help="Device to run on (default: xpu)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--batch-size", type=int, default=1, help="Synthetic batch size (default: 1)")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--num-iters", type=int, default=100, help="Timed iterations (default: 100)")
    parser.add_argument("--libero", action="store_true", help="Evaluate policy success and latency in the LIBERO simulator")
    parser.add_argument("--task-suite", default="libero_10", help="LIBERO task suite (default: libero_10)")
    parser.add_argument("--task-ids", nargs="*", type=int, help="Optional LIBERO task IDs to evaluate")
    parser.add_argument("--num-episodes", type=int, default=20, help="Episodes per LIBERO task (default: 20)")
    parser.add_argument("--max-steps", type=int, help="Optional maximum simulator steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="LIBERO random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pi05_libero_xpu"),
        help="Directory for LIBERO results JSON and CSV",
    )
    args = parser.parse_args()
    if args.libero:
        if args.num_episodes < 1 or (args.max_steps is not None and args.max_steps < 1):
            parser.error("num-episodes and max-steps must be positive")
    elif args.batch_size < 1 or args.num_warmup < 0 or args.num_iters < 1:
        parser.error("batch-size and num-iters must be positive; num-warmup cannot be negative")

    print("loading Pi05 policy...")
    policy = load_policy(args.device, args.compile, measure_latency=args.libero)
    if args.libero:
        if not isinstance(policy, TimedPi05):
            msg = "LIBERO benchmarking requires a timed Pi05 policy"
            raise RuntimeError(msg)
        run_libero_benchmark(policy, args)
        return

    batch = make_batch(args.device, args.batch_size)

    with torch.inference_mode():
        print("running warmup...")
        for _ in range(args.num_warmup):
            actions = policy.predict_action_chunk(batch)
        synchronize(args.device)

        print("running latency test...")
        latencies_ms = []
        for _ in range(args.num_iters):
            start = time.perf_counter()
            actions = policy.predict_action_chunk(batch)
            synchronize(args.device)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    print(f"Actions shape: {actions.shape}")
    print(f"\nPi05 policy latency over {len(latencies_ms)} iterations (warmup={args.num_warmup}):")
    print(f"  mean: {mean_ms:.2f} ms")
    print(f"  p50:  {percentile(latencies_ms, 0.50):.2f} ms")
    print(f"  p90:  {percentile(latencies_ms, 0.90):.2f} ms")
    print(f"  p99:  {percentile(latencies_ms, 0.99):.2f} ms")
    print(f"  min:  {latencies_ms[0]:.2f} ms")
    print(f"  max:  {latencies_ms[-1]:.2f} ms")


if __name__ == "__main__":
    main()