#!/usr/bin/env python3
"""Measure end-to-end Pi05 OpenVINO Runtime latency through InferenceModel."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from physicalai.inference import InferenceModel

DEFAULT_EXPORT_DIR = Path("exports/pi05_libero_runtime_ov")
IMAGE_SIZE = 256
STATE_DIM = 8


def make_observation() -> dict[str, object]:
    """Create one raw LIBERO-shaped observation matching the export manifest."""
    return {
        "state": np.zeros((1, STATE_DIM), dtype=np.float32),
        "images": {
            "image": np.random.random((1, IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32),
            "image2": np.random.random((1, IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32),
        },
        "task": "pick up the red mug",
    }


def percentile(samples: list[float], percentile_value: float) -> float:
    """Return a nearest-rank percentile from sorted latency samples."""
    index = min(int(len(samples) * percentile_value), len(samples) - 1)
    return samples[index]


def main() -> None:
    """Load the OpenVINO policy package and report select_action latency."""
    parser = argparse.ArgumentParser(description="Pi05 OpenVINO Runtime latency benchmark")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="OpenVINO Runtime package containing pi05.xml and manifest.json",
    )
    parser.add_argument("--device", default="CPU", help="OpenVINO device, such as CPU or GPU (default: CPU)")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--num-iters", type=int, default=100, help="Timed iterations (default: 100)")
    args = parser.parse_args()

    if args.num_warmup < 0 or args.num_iters < 1:
        parser.error("num-warmup cannot be negative and num-iters must be positive")
    for filename in ("pi05.xml", "manifest.json", "tokenizer.xml"):
        path = args.export_dir / filename
        if not path.is_file():
            parser.error(f"Required export artifact is missing: {path}")

    print(f"Loading InferenceModel from {args.export_dir} on {args.device}...")
    model = InferenceModel(export_dir=args.export_dir, device=args.device)
    observation = make_observation()

    print("Running warmup...")
    for _ in range(args.num_warmup):
        model.reset()
        actions = model.select_action(observation)

    print("Running latency test...")
    latencies_ms: list[float] = []
    for _ in range(args.num_iters):
        model.reset()
        start = time.perf_counter()
        actions = model.select_action(observation)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    print(f"Actions shape: {np.asarray(actions).shape}")
    print(f"\nInferenceModel select_action latency over {len(latencies_ms)} action chunks (warmup={args.num_warmup}):")
    print("  Includes manifest preprocessors, OpenVINO tokenizer, pi05.xml, and postprocessor.")
    print("  Each timed call returns the full 10-action chunk from this export.")
    print(f"  mean: {mean_ms:.2f} ms")
    print(f"  p50:  {percentile(latencies_ms, 0.50):.2f} ms")
    print(f"  p90:  {percentile(latencies_ms, 0.90):.2f} ms")
    print(f"  p99:  {percentile(latencies_ms, 0.99):.2f} ms")
    print(f"  min:  {latencies_ms[0]:.2f} ms")
    print(f"  max:  {latencies_ms[-1]:.2f} ms")


if __name__ == "__main__":
    main()
