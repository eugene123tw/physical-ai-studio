#!/usr/bin/env python3
"""Pi05 LIBERO parity report: PyTorch (XPU) vs OpenVINO (GPU).

Verifies both backends run the same checkpoint with the same settings (dtype, action
horizon, token length, preprocessing, normalization), runs identical LIBERO rollouts on
each, and reports success rate (SR) and per-chunk latency. Exits with code 1 when the
SR gap exceeds the tolerance.

Example:
    python scripts/pi05_xpu_vs_ov_parity.py --dtype bfloat16 --task-ids 1 --num-episodes 50
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_pi05_libero_runtime_ov import ACTION_HORIZON, MODEL_ID, MODEL_REVISION, TOKEN_LENGTH, load_policy  # noqa: E402

from physicalai.benchmark.gyms import LiberoBenchmark  # noqa: E402
from physicalai.data.constants import TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK  # noqa: E402
from physicalai.data.observation import IMAGES, STATE, Observation  # noqa: E402
from physicalai.gyms.libero import create_libero_gyms  # noqa: E402
from physicalai.inference import InferenceModel  # noqa: E402

OV_PRECISION = {"float32": "f32", "bfloat16": "bf16"}
# Weight element types a faithful run may contain; f16 means the IR was FP16-compressed.
ALLOWED_WEIGHT_TYPES = {"float32": {"f32"}, "bfloat16": {"bf16", "f32"}}
FLOAT_TYPES = {"f16", "bf16", "f32", "f64"}
TORCH_SHORT = {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16", torch.float64: "f64"}
REPEAT_ATOL = 1e-4


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def copy_obs(np_obs: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy an observation so Runtime preprocessors cannot mutate the original."""
    return {key: dict(value) if isinstance(value, dict) else value for key, value in np_obs.items()}


def fmt(value: Any) -> str:
    if isinstance(value, Counter | dict):
        return ", ".join(f"{k}:{v}" for k, v in sorted(value.items()))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def timed(fn: Callable[..., Any], sink: list[float], sync: Callable[[], None] | None = None) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        if sync is not None:
            sync()
        sink.append((time.perf_counter() - start) * 1000)
        return out

    return wrapper


def latency_stats(samples_ms: list[float], skip: int) -> dict[str, float]:
    samples = np.asarray(samples_ms[skip:] or samples_ms)
    return {
        "n": int(samples.size),
        "mean": float(samples.mean()),
        "p50": float(np.percentile(samples, 50)),
        "p90": float(np.percentile(samples, 90)),
        "p99": float(np.percentile(samples, 99)),
        "min": float(samples.min()),
        "max": float(samples.max()),
    }


def make_benchmark(args: argparse.Namespace) -> LiberoBenchmark:
    return LiberoBenchmark(
        task_suite=args.task_suite,
        task_ids=args.task_ids,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        record_mode="none",
    )


def summarize_results(results: Any) -> dict[str, Any]:
    return {
        "success_rate": results.aggregate_success_rate,
        "avg_episode_length": results.aggregate_episode_length,
        "per_task": {tr.task_id: tr.success_rate for tr in results.task_results},
    }


def run_openvino(args: argparse.Namespace, np_obs: dict[str, Any]) -> dict[str, Any]:
    core = ov.Core()
    capabilities = core.get_property(args.ov_device, "OPTIMIZATION_CAPABILITIES")
    requested = OV_PRECISION[args.dtype] if args.ov_precision == "auto" else args.ov_precision
    adapter_kwargs: dict[str, str] = {}
    if requested == "default":
        pass
    elif requested == "bf16" and "BF16" not in capabilities:
        print(f"[WARN] {args.ov_device} has no BF16 support ({capabilities}); using the device default precision.")
    else:
        adapter_kwargs["INFERENCE_PRECISION_HINT"] = requested

    ir = core.read_model(str(args.export_dir / "pi05.xml"))
    weight_types = Counter(
        op.get_output_element_type(0).get_type_name() for op in ir.get_ops() if op.get_type_name() == "Constant"
    )
    del ir

    model = InferenceModel(export_dir=args.export_dir, device=args.ov_device, **adapter_kwargs)
    compiled = model.adapter.compiled_model
    shapes = {port.get_any_name(): [d.get_length() for d in port.get_partial_shape()] for port in compiled.inputs}
    action_shape = [d.get_length() for d in compiled.outputs[0].get_partial_shape()]
    manifest = json.loads((args.export_dir / "manifest.json").read_text(encoding="utf-8"))["model"]
    pre_specs = {spec.get("type"): spec for spec in manifest["preprocessors"]}
    post_specs = {spec.get("type"): spec for spec in manifest["postprocessors"]}

    processed = copy_obs(np_obs)
    for preprocessor in model.preprocessors:
        processed = preprocessor(processed)
    frame_actions = [np.asarray(model.predict_action_chunk(copy_obs(np_obs))).reshape(-1, action_shape[-1]) for _ in range(2)]

    latencies: list[float] = []
    model.predict_action_chunk = timed(model.predict_action_chunk, latencies)
    print(f"\n[OpenVINO] Running LIBERO rollouts on {args.ov_device}...")
    results = make_benchmark(args).evaluate(model, continue_on_error=False)

    info = {
        "device": f"{args.ov_device} ({core.get_property(args.ov_device, 'FULL_DEVICE_NAME')})",
        "weight_types": Counter({k: v for k, v in weight_types.items() if k in FLOAT_TYPES}),
        "compute_precision": compiled.get_property("INFERENCE_PRECISION_HINT").get_type_name(),
        "action_horizon": action_shape[1],
        "actions_per_chunk": action_shape[1],
        "token_length_input": shapes["tokenized_prompt"][1],
        "token_length_tokenizer": int(np.asarray(processed["tokenized_prompt"]).shape[1]),
        "image_resolution": tuple(pre_specs["pi05"]["image_resolution"]),
        "image_input_hw": tuple(shapes["images"][-2:]),
        "num_cameras": shapes["images"][0],
        "empty_cameras": pre_specs["pi05"]["empty_cameras"],
        "normalize_mode": pre_specs["normalize"]["mode"],
        "denormalize_mode": post_specs["denormalize"]["mode"],
        "state_stats": pre_specs["normalize"]["stats"]["state"],
        "action_stats": post_specs["denormalize"]["stats"]["action"],
        "deterministic": float(np.abs(frame_actions[0] - frame_actions[1]).max()),
        "tokens": np.asarray(processed["tokenized_prompt"]),
        "token_mask": np.asarray(processed["tokenized_prompt_mask"]).astype(bool),
        "norm_state": np.asarray(processed["state"], dtype=np.float32),
        "images": np.asarray(processed["images"], dtype=np.float32),
        "frame_actions": frame_actions[0],
        "latency_ms": latency_stats(latencies, args.warmup_chunks),
        "results": summarize_results(results),
    }
    del model, compiled, results
    gc.collect()
    return info


def run_torch(args: argparse.Namespace, obs: Observation) -> dict[str, Any]:
    snapshot_dir = Path.home() / ".cache" / "huggingface" / "hub" / MODEL_ID / "snapshots" / MODEL_REVISION
    policy = load_policy(snapshot_dir, args.dtype, compile_model=args.compile).to(args.torch_device).eval()
    device_type = torch.device(args.torch_device).type
    sync = {"xpu": torch.xpu.synchronize, "cuda": torch.cuda.synchronize}.get(device_type)
    device_name = torch.xpu.get_device_name() if device_type == "xpu" else args.torch_device

    preprocessor = policy._preprocessor
    with torch.inference_mode():
        processed = preprocessor(obs.to(args.torch_device).to_dict())
        frame_actions = [policy.predict_action_chunk(obs).float().cpu().numpy() for _ in range(2)]
        frame_actions = [actions.reshape(-1, actions.shape[-1]) for actions in frame_actions]

    latencies: list[float] = []
    policy.predict_action_chunk = timed(policy.predict_action_chunk, latencies, sync)
    print(f"\n[PyTorch] Running LIBERO rollouts on {device_name}...")
    results = make_benchmark(args).evaluate(policy, continue_on_error=False)

    stats = policy._dataset_stats
    info = {
        "device": f"{args.torch_device} ({device_name})",
        "weight_types": Counter(TORCH_SHORT.get(p.dtype, str(p.dtype)) for p in policy.model.parameters()),
        "compute_precision": OV_PRECISION[args.dtype],
        "action_horizon": policy.config.chunk_size,
        "actions_per_chunk": policy._n_action_steps,
        "token_length": preprocessor.max_token_len,
        "image_resolution": tuple(preprocessor.image_resolution),
        "num_cameras": int(processed[IMAGES].shape[0]),
        "empty_cameras": preprocessor.empty_cameras,
        "normalization_mode": preprocessor.normalization_mode.lower(),
        "state_stats": stats["observation.state"],
        "action_stats": stats["action"],
        "deterministic": float(np.abs(frame_actions[0] - frame_actions[1]).max()),
        "tokens": processed[TOKENIZED_PROMPT].cpu().numpy(),
        "token_mask": processed[TOKENIZED_PROMPT_MASK].cpu().numpy().astype(bool),
        "norm_state": processed[STATE].float().cpu().numpy(),
        "images": processed[IMAGES].float().cpu().numpy(),
        "frame_actions": frame_actions[0],
        "latency_ms": latency_stats(latencies, args.warmup_chunks),
        "results": summarize_results(results),
        "compile": args.compile,
    }
    del policy, results
    gc.collect()
    return info


def stats_match(torch_stats: dict[str, Any], ov_stats: dict[str, Any]) -> bool:
    keys = [k for k in ("mean", "std", "q01", "q99") if k in torch_stats or k in ov_stats]
    return all(
        k in torch_stats and k in ov_stats and np.allclose(torch_stats[k], ov_stats[k], atol=1e-6) for k in keys
    )


def build_checks(args: argparse.Namespace, pt: dict[str, Any], ov_: dict[str, Any]) -> list[tuple[str, Any, Any, bool | None]]:
    allowed = ALLOWED_WEIGHT_TYPES[args.dtype]
    state_diff = float(np.abs(pt["norm_state"] - ov_["norm_state"]).max())
    image_diff = float(np.abs(pt["images"] - ov_["images"]).max())
    return [
        ("checkpoint", f"{MODEL_ID}@{MODEL_REVISION[:12]}", f"{args.export_dir} (exported from same checkpoint)", None),
        ("weight dtypes", pt["weight_types"], ov_["weight_types"], set(pt["weight_types"]) <= allowed and set(ov_["weight_types"]) <= allowed),
        ("compute precision", pt["compute_precision"], ov_["compute_precision"], pt["compute_precision"] == ov_["compute_precision"]),
        ("ACTION_HORIZON (chunk)", pt["action_horizon"], ov_["action_horizon"], pt["action_horizon"] == ov_["action_horizon"] == ACTION_HORIZON),
        ("actions executed / chunk", pt["actions_per_chunk"], ov_["actions_per_chunk"], pt["actions_per_chunk"] == ov_["actions_per_chunk"]),
        (
            "TOKEN_LENGTH",
            pt["token_length"],
            f"IR input {ov_['token_length_input']}, tokenizer {ov_['token_length_tokenizer']}",
            pt["token_length"] == ov_["token_length_input"] == ov_["token_length_tokenizer"] == TOKEN_LENGTH,
        ),
        (
            "image resolution",
            pt["image_resolution"],
            f"{ov_['image_resolution']} (IR input {ov_['image_input_hw']})",
            pt["image_resolution"] == ov_["image_resolution"] == ov_["image_input_hw"],
        ),
        ("cameras (incl. empty)", pt["num_cameras"], ov_["num_cameras"], pt["num_cameras"] == ov_["num_cameras"]),
        ("empty cameras", pt["empty_cameras"], ov_["empty_cameras"], pt["empty_cameras"] == ov_["empty_cameras"]),
        (
            "normalization mode",
            pt["normalization_mode"],
            f"normalize={ov_['normalize_mode']}, denormalize={ov_['denormalize_mode']}",
            pt["normalization_mode"] == ov_["normalize_mode"] == ov_["denormalize_mode"],
        ),
        ("state stats", "dataset_stats", "manifest", stats_match(pt["state_stats"], ov_["state_stats"])),
        ("action stats", "dataset_stats", "manifest", stats_match(pt["action_stats"], ov_["action_stats"])),
        (
            "repeat-run max|diff| (same input)",
            pt["deterministic"],
            ov_["deterministic"],
            pt["deterministic"] <= REPEAT_ATOL and ov_["deterministic"] <= REPEAT_ATOL,
        ),
        (
            "tokenized prompt (frame 0)",
            "torch tokenizer",
            "OV tokenizer",
            bool(np.array_equal(pt["tokens"], ov_["tokens"]) and np.array_equal(pt["token_mask"], ov_["token_mask"])),
        ),
        ("normalized state max|diff|", 0.0, state_diff, state_diff < 1e-5),
        ("preprocessed images max|diff|", 0.0, image_diff, None),
    ]


def print_report(args: argparse.Namespace, env: dict[str, str], pt: dict[str, Any], ov_: dict[str, Any]) -> bool:
    line = "=" * 100
    print(f"\n{line}\nPi05 LIBERO PARITY REPORT: PyTorch vs OpenVINO\n{line}")
    for key, value in env.items():
        print(f"  {key:<22} {value}")

    checks = build_checks(args, pt, ov_)
    settings_ok = all(ok for *_, ok in checks if ok is not None)
    print(f"\nSETTINGS ({args.dtype})\n  {'check':<30} {'PyTorch':<32} {'OpenVINO':<44} status")
    for name, pt_value, ov_value, ok in checks:
        status = "INFO" if ok is None else ("MATCH" if ok else "MISMATCH")
        print(f"  {name:<30} {fmt(pt_value):<32} {fmt(ov_value):<44} {status}")

    action_diff = np.abs(pt["frame_actions"] - ov_["frame_actions"])
    print("\nSINGLE-FRAME ACTION PARITY (episode 0, first observation, denormalized)")
    print(f"  max|diff| = {action_diff.max():.6f}   mean|diff| = {action_diff.mean():.6f}")
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(f"  PyTorch  a[0] = {pt['frame_actions'][0]}")
    print(f"  OpenVINO a[0] = {ov_['frame_actions'][0]}")

    print(f"\nBENCHMARK  suite={args.task_suite} task_ids={args.task_ids} episodes/task={args.num_episodes} "
          f"seed={args.seed} max_steps={args.max_steps or 'suite default'} (identical for both)")
    print(f"  {'':<26} {'PyTorch':>16} {'OpenVINO':>16}")
    print(f"  {'success rate (%)':<26} {pt['results']['success_rate']:>16.1f} {ov_['results']['success_rate']:>16.1f}")
    print(f"  {'avg episode length':<26} {pt['results']['avg_episode_length']:>16.1f} {ov_['results']['avg_episode_length']:>16.1f}")
    for task_id, sr in pt["results"]["per_task"].items():
        print(f"  {'  ' + task_id + ' SR (%)':<26} {sr:>16.1f} {ov_['results']['per_task'].get(task_id, float('nan')):>16.1f}")

    print(f"\nLATENCY per action chunk, end-to-end predict_action_chunk (ms, first {args.warmup_chunks} chunks excluded)")
    print(f"  {'':<26} {'PyTorch':>16} {'OpenVINO':>16}")
    for key in ("n", "mean", "p50", "p90", "p99", "min", "max"):
        print(f"  {key:<26} {pt['latency_ms'][key]:>16.2f} {ov_['latency_ms'][key]:>16.2f}")
    print(f"  {'p50 speedup (PT/OV)':<26} {pt['latency_ms']['p50'] / ov_['latency_ms']['p50']:>16.2f}x")

    gap = round(ov_["results"]["success_rate"] - pt["results"]["success_rate"], 6)
    sr_ok = abs(gap) <= args.sr_tolerance
    print(f"\n{line}")
    print(f"SETTINGS: {'ALL MATCH' if settings_ok else 'MISMATCH (see table above)'}")
    print(f"SR GAP (OV - PT): {gap:+.1f} pp, tolerance ±{args.sr_tolerance:.1f} pp -> {'PASS' if sr_ok else 'DISPARITY'}")
    if 100 / args.num_episodes > args.sr_tolerance:
        print(f"NOTE: 1 episode = {100 / args.num_episodes:.1f} pp of SR per task; use more episodes for a ±{args.sr_tolerance:.0f} pp claim.")
    print(line)
    return sr_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, default=Path("exports/pi05_libero_runtime_ov"))
    parser.add_argument("--dtype", choices=tuple(OV_PRECISION), default="bfloat16", help="Dtype for both backends")
    parser.add_argument("--torch-device", default="xpu")
    parser.add_argument("--ov-device", default="GPU")
    parser.add_argument(
        "--ov-precision",
        choices=("auto", "default", "f16", "f32", "bf16"),
        default="auto",
        help="OV INFERENCE_PRECISION_HINT; 'auto' follows --dtype, 'default' uses the device default",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile the PyTorch model")
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-ids", nargs="+", type=int, default=[1])
    parser.add_argument("--num-episodes", type=int, default=50, help="Episodes per task (default: 50)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr-tolerance", type=float, default=5.0, help="Allowed |SR gap| in percentage points")
    parser.add_argument("--warmup-chunks", type=int, default=3, help="Chunks excluded from latency stats")
    parser.add_argument("--report", type=Path, default=Path("results/pi05_xpu_vs_ov/report.json"))
    args = parser.parse_args()

    env = {
        "host": f"{platform.node()} / {platform.processor() or platform.machine()} / {platform.platform()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "openvino": ov.__version__,
        "openvino_tokenizers": pkg_version("openvino-tokenizers"),
        "physicalai (runtime)": pkg_version("physicalai"),
        "physicalai-train": pkg_version("physicalai-train"),
        "export dir": str(args.export_dir),
    }

    gym = create_libero_gyms(args.task_suite, task_ids=[args.task_ids[0]])[0]
    obs, _ = gym.reset(seed=args.seed, episode_index=0)
    np_obs = obs.to_numpy().to_dict(flatten=False)
    if hasattr(gym, "close"):
        gym.close()

    ov_info = run_openvino(args, np_obs)
    pt_info = run_torch(args, obs)
    env["PyTorch device"] = pt_info["device"] + (" + torch.compile" if args.compile else "")
    env["OpenVINO device"] = ov_info["device"]

    sr_ok = print_report(args, env, pt_info, ov_info)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    skip = {"tokens", "token_mask", "norm_state", "images", "frame_actions", "state_stats", "action_stats"}
    report = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "env": env,
        "checks": [
            {"check": n, "pytorch": fmt(p), "openvino": fmt(o), "match": ok} for n, p, o, ok in build_checks(args, pt_info, ov_info)
        ],
        "pytorch": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in pt_info.items() if k not in skip},
        "openvino": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in ov_info.items() if k not in skip},
    }
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report written to {args.report}")
    sys.exit(0 if sr_ok else 1)


if __name__ == "__main__":
    main()
