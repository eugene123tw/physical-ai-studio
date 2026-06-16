# RLDX-1 → Physical AI Studio Integration Thought Doc

Companion to [rldx-1.md](rldx-1.md) (catalog entry + verdict) and
[rldx-1-paper.md](rldx-1-paper.md) (vendored arXiv markdown). This doc is
the **integration plan** — what we ship into [`library/src/physicalai/policies/rldx1/`](../src/physicalai/policies/), what
we ship into the `physicalai` runtime, and what we leave on the floor.

> **Standing license caveat** — `RLWRLD/RLDX-1-*` weights ship under the
> RLWRLD Model License v1.0 (non-commercial). Everything below assumes a
> research-only integration unless RLWRLD relicenses or we re-train from
> scratch. License is the only true blocker; the code is Apache-2.0 and the
> codebase is integration-ready.

> **v1 scope decision (2026-06): ship `PT → FT` only.**
> The paper provides **zero evidence** for any `MT → FT` path that we
> could reproduce — every public `FT-*` checkpoint branches directly
> from `RLDX-1-PT`, and the per-task FTs on top of `MT-*` were never
> released (see [§7.2](#72-released-checkpoints--what-each-one-actually-validates)).
> Mid-train also requires multi-modal in-house data (tactile / torque /
> long-horizon memory) that no PAS user has. We therefore drop the
> entire memory / motion / physics module set + the alignment-warmup
> trainer plumbing from v1, and ship the same single-stage recipe the
> released `FT-SIMPLER-WIDOWX` checkpoint uses. MT support is tracked
> as a phase-2 follow-up. Sections referring to MT, memory, motion,
> and physics below are kept as **context for that future work**, not
> as v1 deliverables.

---

## 0. Scope summary

| Capability | Ship in v1? | Notes |
|---|---|---|
| `RLDX-1-PT` (6.9 B, video PT, no add-ons) as base | ✅ | Sole reproducible entry point. |
| MSAT action head + Qwen3-VL-8B backbone | ✅ | Pure-torch, exportable. |
| Flow-matching `PT → FT` post-train loop | ✅ | The one and only training path in v1. |
| LoRA PEFT (paper App. D) | ✅ | Action-LoRA default; backbone-LoRA optional. See §5.3. |
| SimplerEnv WidowX as the alignment benchmark | ✅ | Primary parity target — matches `RLDX-1-FT-SIMPLER-WIDOWX`. |
| Motion module (STSS) | ❌ phase 2 | MT-only; no released FT uses it. |
| Memory module (n_mem queue) | ❌ phase 2 | MT-only; stateful runtime work deferred. |
| Physics stream (tactile / torque) | ❌ phase 2 | MT-only; no upstream public sensor data. |
| `new_param_warmup_steps` / alignment-warmup callback | ❌ phase 2 | Only useful when new modality streams exist. |
| `MT-ALLEX` / `MT-DROID` checkpoint support | ❌ phase 2 | Skipped — no reproducible downstream path. |
| Real-Time Chunking — training (`rtc_training_max_delay`) | ⚠️ optional | Pure forward pass; defaults to 0. Keep config field, do not validate in v1. |
| RTC inference `trained` mode | ⚠️ optional | Hard-inpaint, static-graph safe. No released checkpoint has RTC. |
| RTC inference `guided` mode | ❌ | Jacobian VJP through DiT — incompatible with OV / fullgraph compile. |
| RECAP post-training (RL) | ❌ phase 2+ | **Not in upstream repo.** Paper-only. |
| Triton kernel chain | ❌ for inference | CUDA-only. We rely on OV/ONNX fusion. |
| CUDA Graph + Static Graph Conversion | ❌ for inference | Replaced by OV `compile_model` cache. |

---

## 1. Pretrained starting point — `RLWRLD/RLDX-1-PT`

[Hub: `RLWRLD/RLDX-1-PT`](https://huggingface.co/RLWRLD/RLDX-1-PT)

- **6.9 B params, no functional add-ons** (no motion / memory / physics
  parameters in the state dict). The 8.1 B `MT-*` and `FT-*` variants are
  built by mid-training new modules on top of PT.
- Confirmed by every shipped script: [`finetune.sh`](../../RLDX-1/run_scripts/train/examples/finetune.sh),
  [`midtrain_rldx1_droid.sh`](../../RLDX-1/run_scripts/train/examples/midtrain_rldx1_droid.sh),
  and all `run_scripts/train/benchmarks/*` default to `BASE_MODEL_PATH=RLWRLD/RLDX-1-PT`.
- **Implication for Studio (v1)**: our `Rldx1.from_pretrained("RLWRLD/RLDX-1-PT")`
  must load the PT state dict cleanly with `use_motion = use_memory = use_physics = False`
  hard-coded. The conditional add-on instantiation + `new_param_warmup_steps`
  callback land in phase 2 together with MT support.
- **Security** ([`lib.security`](../../.github/instructions/lib.security.instructions.md) rules 6, 9, 10): pretrained-load path must
  - Use a **pinned `revision=` SHA**, not `"main"`.
  - Set `weights_only=True` on every `torch.load` (Qwen3-VL upstream uses
    `safetensors` already, so this is mostly defensive).
  - Refuse `trust_remote_code=True` unless the call site documents the
    Qwen3-VL repo + pins revision. Qwen3-VL has first-party
    `transformers` support → we should not need `trust_remote_code` at all.

---

## 2. Triton in inference — avoid; use only at training

### What's CUDA-only

All Triton lives under [`rldx/inference/`](../../RLDX-1/rldx/inference/):

| Path | Role |
|---|---|
| `engine/kernels/fused_add2_rmsnorm.py`, `fused_memory_attention.py`, … | Triton kernels |
| `engine/cuda_graph.py`, `engine/torch_inductor.py` | CUDA Graph + `torch.compile` wrappers |
| `model/graph_safe_vla.py`, `action_model/model/graph_safe_*.py`, `backbone/model/graph_safe_*.py` | Static-graph rewrites of the modules with config-dependent ops factored out |
| `_rtc_dispatch.py`, `serve_optimization.py`, `benchmark_vla.py` | Path selectors (A=Eager, B=Compile, C=CUDA-Graph+Compile, D=Custom-Triton) |

**Training uses none of this.** Training entrypoint is
[`rldx/experiment/launch_train.py`](../../RLDX-1/rldx/experiment/launch_train.py),
which calls `RLDXForVisionLanguageAction` from
[`rldx/model/core/rldx.py`](../../RLDX-1/rldx/model/core/rldx.py) — pure PyTorch +
Flash-Attention-2 (an `attn_implementation` choice on the Qwen3-VL backbone,
not a custom kernel).

### Plan
- **Training side (Studio)**: keep Flash-Attention 2 / SDPA as the attention
  backend — same as `Groot` policy. No triton.
- **Inference side (`physicalai` runtime)**: export the model to OV / ONNX
  from the **eager** forward pass. We will not depend on any file under
  `rldx/inference/*` — the runtime sees only the manifest + the exported
  graph. The CUDA-Graph / Triton path is irrelevant once the graph is owned
  by OpenVINO.
- Document this explicitly in the policy README: "the upstream
  `rldx/inference/` tree is GPU-only and is replaced by the OV adapter."

---

## 3. Memory module — deferred to phase 2

> **Out of v1 scope.** The memory module only exists on the released
> `MT-*` checkpoints, which we are not loading in v1. The notes below
> are kept verbatim from the original integration plan as a reference
> for the phase-2 MT support — `delta_timestamps` wiring, stateful
> runtime preprocessor, and manifest extension all stay roughly the
> same when we revisit.

Spec (paper §2.1 Functionality 2; reproduced for v1):
- Memory queue **Q_t = [h_{t-n_mem·H}, …, h_{t-H}]**, n_mem = 3, stride =
  H + 1 = action chunk horizon.
- Fed alongside current cognition `h_t` into a lightweight 2-layer causal
  Transformer (`TransformerMemory` in [`rldx/model/modules/memory.py`](../../RLDX-1/rldx/model/modules/memory.py),
  hidden 4096, 16 heads, causal RoPE).
- At training time the queue is built **inside the loss step** from past
  cognition snapshots; at inference time it is maintained as session state
  across `predict_action_chunk()` calls.

### Training-time consequence — LeRobot dataset side

`LeRobotDataModule` already uses `delta_timestamps` to fetch past observations,
so the mechanism exists. Two options:

| Option | What changes | Cost |
|---|---|---|
| **A. Multi-frame fetch + re-encode** | Use `delta_timestamps` to fetch frames at offsets `[-3H, -2H, -H, 0]`. Run the VLM `n_mem + 1` times per sample to produce the cognition queue. | Simple. ~4× VLM cost per sample → **prohibitive** for an 8 B backbone. |
| **B. Cached cognition tokens, sampled in-batch** | Mirror upstream: per sample fetch only the current frame, but supplement with a cached cognition tensor from a parallel "memory loader" that streams the same episode at stride H. Treat the cached tokens as a separate column. | Needs a side-table of pre-computed cognition tokens per episode (or use the model's own cognition output from a previous in-batch step — but that breaks DDP determinism). |
| **C. Upstream's approach** | RLDX-1 picks Option A but caps the workload via `--video-length 4` and `motion-insert-layer 9` (cheap motion encoder runs on early-layer features only). Memory snapshots reuse the same VLM forward. | Acceptable for 8 H200 nodes. **Painful on a single Intel dGPU.** |

**v1 recommendation**: ship Option **C** (upstream behaviour) wired
through `delta_timestamps`, gated behind `use_memory=False` so a default
Studio run pays no extra cost. Document Option B as a future optimization.

Concretely in `LeRobotDataModule`:
```python
delta_timestamps = {
    "observation.images.top": [-3 * h, -2 * h, -h, 0],     # if use_memory
    "observation.state": [0],
    "action": list(range(action_horizon)),
}
```
where `h = action_horizon` (= memory_stride). Generate this from the
policy config via [`get_delta_timestamps_from_policy`](../src/physicalai/data/lerobot/utils/delta_timestamps.py) by adding an
`rldx1` branch that reads `memory_length`, `memory_stride`, and
`action_horizon`.

### Inference-time consequence — `physicalai` runtime

Upstream's `SessionRegistry` (see [`rldx/policy/session_registry.py`](../../RLDX-1/rldx/policy/session_registry.py))
is the reference. We do not need its full surface — the runtime's
`InferenceModel` already runs single-session — but we need:
- A **stateful preprocessor** (or a new `runner` type) that holds the
  cognition queue between `predict_action_chunk()` calls.
- A `reset()` hook called on episode boundaries (matches Studio's
  `Gym.reset()` and the runtime's session lifecycle).
- The queue must store **detached, cloned** tensors (upstream does
  `state.memory_tokens.detach().clone()` — copy verbatim).

Manifest extension: a new runner `type: action_chunking_with_memory` whose
`init` block declares `memory_length`, `memory_stride`, and the cognition
token shape. Ship this in the `physicalai` repo alongside the policy.

---

## 4. Studio module decomposition — `policies/rldx1/`

Match the standing four-file pattern (see `policies/pi05/`, `policies/groot/`).
**v1 layout — PT → FT only**:

```
library/src/physicalai/policies/rldx1/
├── __init__.py              # re-export Rldx1, Rldx1Config, Rldx1Model
├── config.py                # @dataclass(frozen=True) Rldx1Config — mirror RLDXConfig (PT-shape only)
├── model.py                 # nn.Module — pure torch, exportable
├── policy.py                # Lightning wrapper, subclass of policies.base.Policy
├── preprocessor.py          # image/text/state preprocessors (no memory queue in v1)
├── pretrained_utils.py      # PT weight loader, key fix-ups, dataset_stats extraction
└── components/
    ├── msat.py              # Multi-Stream Action Transformer (port from rldx/model/modules/action_model/)
    ├── lora.py              # PEFT helper — mirror of pi0/components/lora.py (see §5.3)
    └── backbone.py          # Qwen3-VL-8B adapter (selects layer 18, freezes top-4 etc.)
```

Phase-2 additions (deferred — pulled in when MT support lands):

```
    ├── memory.py            # TransformerMemory (verbatim port)
    ├── motion.py            # STSS encoder (when use_motion)
    └── physics.py           # PhysicsHead + p-stream (when use_physics)
```

Mapping vs upstream:

| Upstream path | Studio target | Notes |
|---|---|---|
| [`rldx/configs/model/rldx.py:RLDXConfig`](../../RLDX-1/rldx/configs/model/rldx.py) | `rldx1/config.py:Rldx1Config` | Strip GR00T-isms; keep the ~80 fields that actually drive behaviour. |
| [`rldx/model/core/rldx.py:RLDXForVisionLanguageAction`](../../RLDX-1/rldx/model/core/rldx.py) | `rldx1/model.py:Rldx1Model` | Drop the Lightning glue; keep the forward(noisy_action, τ, h, m, s, p) signature. |
| [`rldx/model/modules/backbone/adapter.py:VTCQwen3VLBackbone`](../../RLDX-1/rldx/model/modules/backbone/adapter.py) | `rldx1/components/backbone.py` | Wraps `transformers.Qwen3VLModel`, extracts layer 18, applies motion-residual hook. |
| [`rldx/model/modules/action_model/msat.py`](../../RLDX-1/rldx/model/modules/action_model/msat.py) + `blocks.py` | `rldx1/components/msat.py` | The double-stream + single-stream blocks, joint self-attention. |
| [`rldx/model/modules/memory.py`](../../RLDX-1/rldx/model/modules/memory.py) | `rldx1/components/memory.py` | Verbatim — already pure torch + transformers `LlamaConfig`. |
| [`rldx/model/modules/action_model/rtc.py`](../../RLDX-1/rldx/model/modules/action_model/rtc.py) | `rldx1/components/rtc.py` | Drop `guided_velocity` (autograd VJP — see §8). Keep `sample_training_prefix`, `build_per_token_time`, `build_noisy_trajectory_rtc`. |
| [`rldx/data/state_action/state_action_processor.py`](../../RLDX-1/rldx/data/state_action/state_action_processor.py) | `rldx1/preprocessor.py` | Normalization (1st/99th percentile), sin/cos state encoding, color jitter. |

`Rldx1.forward(batch)` returns loss in train mode (`self.training`) and
delegates to `predict_action_chunk(batch)` in eval — same contract as
`Pi05` and `Groot`. Action queue (`n_action_steps` execution horizon) is
inherited from `policies.base.Policy`.

---

## 5. `finetune.sh` — the v1 training script

v1 reproduces a single training path: `RLDX-1-PT → FT` on a user's
LeRobot dataset, equivalent to the recipe that produced every released
`RLDX-1-FT-*` checkpoint.

Upstream reference: [`rldx/experiment/launch_train.py`](../../RLDX-1/rldx/experiment/launch_train.py)
driven by [`run_scripts/train/examples/finetune.sh`](../../RLDX-1/run_scripts/train/examples/finetune.sh)
with all `--use-motion / --use-memory / --use-physics` flags off.

Key hparams (from paper §6.1 "Implementation Details" + the per-benchmark
FT configs we audited in §7.2):

| Field | Default | Per-benchmark deltas |
|---|---|---|
| `BASE_MODEL_PATH` | `RLWRLD/RLDX-1-PT` | fixed for every v1 FT |
| Optimizer | AdamW, lr 1e-4, cosine + 5 % linear warmup | fixed |
| `max_steps` | 60 000 | 20 K for SIMPLER Google, 250 K for RC365 |
| `global_batch_size` | 1024 | 256 for LIBERO, 196 for RC365 |
| `state_dropout_prob` | 0.0 | 0.5 for SIMPLER-Google and GR-1 |
| Frozen layers | vision encoder + LLM except top-4 LLM layers | fixed |
| `action_horizon` | 16 | fixed in v1 |
| Action backbone PEFT | LoRA r=64 (paper App. D free-lunch) | see §5.3 |

Three YAML presets ship together with the policy:

```
configs/physicalai/rldx1-ft-default.yaml          # full backbone + LoRA action (paper App. D free-lunch)
configs/physicalai/rldx1-ft-consumer-gpu.yaml     # LoRA both, fits 24 GiB single GPU
configs/physicalai/rldx1-ft-paper-baseline.yaml   # full FT both (parity-run only)
```

**Mid-train (`midtrain_rldx1_*.sh`) is deferred to phase 2.** The
comparison between mid-train and post-train scripts, the
`new_param_warmup_steps` callback, and the modality-mixture data loader
are not implemented in v1 — see the scope banner at the top of this
doc.

### 5.1 Training stages — context only (PT → MT → FT)

> **v1 only ships `PT → FT`.** This subsection is retained as
> background for phase-2 MT work. Skip on first read.

The paper defines three stages but the released checkpoint names only
expose two of them. Quick terminology fix:

| Paper name | Checkpoint prefix | What it produces |
|---|---|---|
| Pre-train | `RLDX-1-PT` | Generalist multi-embodiment base. No memory / motion / physics modules. |
| Mid-train | `RLDX-1-MT-{ALLEX, DROID}` | Embodiment-specialized backbone **with** add-on modules baked in. |
| Post-train (BC) | `RLDX-1-FT-{LIBERO, SIMPLER-*, GR1, ROBOCASA, RC365}` | Task / benchmark-specific BC fine-tune. Add-ons off. |
| Post-train (RL) | _none released_ | RECAP loop on top of a BC post-train. Paper-only. |

`FT-*` in the checkpoint naming corresponds to the paper's **post-train**
stage — pure imitation learning on a single benchmark. Every released
`FT-*` checkpoint is **`PT → FT`**, skipping mid-train.

**Goal comparison:**

| Axis | Mid-train | Post-train (BC) |
|---|---|---|
| Starts from | `PT` weights | `PT` **or** `MT-*` weights |
| What's added to the model | New modality streams (memory, motion, physics) initialized from scratch | Nothing — same architecture as the starting checkpoint |
| Trainable params | First 2 K steps: only new-modality params. Then: all params. | Top-4 LLM layers + action model + projectors (vision encoder + lower LLM stay frozen). |
| Dataset shape | **Mixture** of broad public data + narrow in-house data carrying the new modalities | Single benchmark / single task |
| LR | 5e-5 (small — protects PT priors) | 1e-4 with cosine + 5 % warmup |
| Steps × batch | 25 K × 1024 | 60 K × 1024 (256 for LIBERO, 196 for RoboCasa365) |
| Compute envelope (paper) | 15 h × 64 H200 | varies, single-digit H200-days per benchmark |
| What it bakes in | Embodiment identity + memory/motion/physics weights | Task-specific motor policy |

**Data composition (paper §4.2 / §4.3):**

- **MT-ALLEX**: in-house ALLEX teleop + Robocurate synthetic, 5:5.
  The in-house slice is the only source of tactile / torque supervision.
- **MT-DROID** (FR3 variant): DROID 92 K episodes + in-house FR3 teleop
  with tactile/torque, 8:2. DROID provides manipulation breadth; the
  in-house 20 % carries the new-modality signal.
- **Post-train**: the downstream task's training split — LIBERO subset,
  SIMPLER Bridge or Fractal, GR-1 Tabletop, RoboCasa, RoboCasa365, or
  the user's own teleop set.

**Key consequence — mid-train is gated by data, not by choice.** You
only run mid-train when:

1. You have data with **new modalities** (tactile, torque, or long-horizon
   memory snapshots) that the PT model has never seen, **and**
2. You have enough **task diversity** in that data to avoid catastrophic
   forgetting of the PT priors.

If either condition fails, mid-train hurts more than it helps and the
PT → FT path is correct.

**Can I just deploy `MT-ALLEX` directly and skip post-train?** Maybe — but
**the paper provides no evidence either way**. Every ALLEX / FR3 /
OpenArm number in the paper is run on a post-trained checkpoint:

- §6.2 OpenArm, Figure 14 caption: *"We report the success rates (%) of
  fine-tuned VLAs."*
- §G.3 ALLEX experiment details: *"We fine-tune the RLDX-1 model
  initialized from the mid-training stage for 30 K steps per task..."*
  (lr 1e-4, batch 128, state_dropout 0.8).
- §G.4 FR3 experiment details: *"We post-train the mid-trained RLDX-1
  for 30 K steps per task..."* (lr 1e-4, batch 64, state_dropout 0.3).
- The "zero-shot" wording elsewhere in the paper refers to **sim-to-real
  visual transfer** in SIMPLER — generalization across rendering, on a
  policy that was itself post-trained, not "skip the FT stage."

So `MT-ALLEX` is best understood as **a much better starting point than
`PT` for ALLEX-family post-training**, not as a deployable policy on
its own. Predicted reasons FT-on-top-of-MT still helps:

1. **Distribution shift.** MT is 50/50 real teleop + Robocurate
   synthetic. Real deployment scenes look like neither.
2. **Breadth vs. depth.** 25 K MT steps at lr 5e-5 over a multi-task
   mixture deliberately doesn't overfit any single task.
3. **Per-task hyperparameter knobs.** Paper Table 8 — `state_dropout`,
   `num_denoising_steps`, action exec horizon, and the action-time
   distribution are all tuned per-benchmark at post-train time.

These are plausible but unmeasured. If you want to test MT-as-deployment
empirically, that's an interesting ablation to run during validation
(see §7.3).

### 5.2 SO-101 user — v1 recipe

A typical SO-101 setup is a single-arm 6-DoF parallel gripper, no
tactile, no torque, and one or two cameras. The user records one
LeRobot dataset for one task.

**v1 recipe: `PT → FT`**, the only path we support. Identical shape to
the released `RLDX-1-FT-SIMPLER-WIDOWX` checkpoint (single-arm gripper,
no add-ons, state_dropout 0.0) — only the dataset changes.

Concrete config:

```yaml
# configs/physicalai/rldx1-finetune-so101.yaml
policy:
  _target_: physicalai.policies.rldx1.Rldx1
  base_model_path: RLWRLD/RLDX-1-PT
  revision: <pinned commit SHA>          # lib.security rule 9
  embodiment_tag: NEW_EMBODIMENT         # SO-101 isn't in the canonical list

  use_memory: false
  use_motion: false
  use_physics: false
  rtc_inference_mode: none               # no released RTC checkpoint anyway

  action_horizon: 16                     # PT default
  tune_top_llm_layers: 4
  tune_visual: false
  state_dropout_prob: 0.0                # bump to 0.5 only if overfitting

trainer:
  max_steps: 60000
  global_batch_size: 256                 # LIBERO-style scale
  optimizer:
    lr: 1.0e-4
    scheduler: cosine_with_warmup
    warmup_ratio: 0.05

datamodule:
  _target_: physicalai.data.lerobot.LeRobotDataModule
  repo_id: <user's SO-101 dataset>
  delta_timestamps:
    action: list(range(16))
    observation.state: [0]
    observation.images.front: [0]
```

Phase-2 SO-101 use cases that would justify mid-train (deferred):

1. **Multi-task SO-101 corpus** with 20+ tasks — enable `use_memory=true`,
   alignment-warmup callback, lr 5e-5 on the mixture.
2. **Tactile retrofit** (e.g., AnySkin on the gripper) — enable
   `use_physics=true` + `allow_missing_physics=true`.

Neither is in v1.

### 5.3 PEFT — paper Appendix D applied

Paper Appendix D / Table 6 measures LoRA configurations on RoboCasa Kitchen
against a "Full FT top-4 backbone + full FT action model" baseline of
62.67 % mean SR over 24 tasks × 50 episodes, on a single H200.

| Backbone VLM | Action model | SR | Trainable params | VRAM @ bs=32 | VRAM @ bs=1 |
|---|---|---|---|---|---|
| Full FT top-4 | Full FT | **62.67 %** | 2,376 M | 87.1 GiB | 56.8 GiB |
| Full FT top-4 | LoRA r=64 | **62.67 %** | 1,150 M | 76.6 GiB | 37.2 GiB |
| Full FT top-4 | LoRA r=8 | 60.17 % | 1,134 M | 76.3 GiB | 36.8 GiB |
| LoRA r=64 | LoRA r=64 | 55.33 % | **398 M (5.7 %)** | **35.9 GiB** | **23.7 GiB** |
| LoRA r=8 | LoRA r=8 | 45.75 % | 364 M | 35.4 GiB | 23.1 GiB |
| Frozen | LoRA r=64 | 36.42 % | 378 M | 27.5 GiB | 23.3 GiB |
| Frozen | LoRA r=8 | 21.25 % | 362 M | 27.2 GiB | 22.9 GiB |

Three rules to encode:

1. **LoRA r=64 on the action model is a free lunch** — same SR as full FT
   for ~½ the trainable parameters. This is the recommended default.
2. **LoRA on both backbone top-4 + action model** is the consumer-GPU path
   — 24 GiB at bs=1, 7-pt SR drop. Single 4090 / A6000 / Arc-770 mode.
3. **Don't freeze the VLM entirely.** 26-pt SR gap vs. full FT — the top-4
   layers must train somehow (full or LoRA).

**PAS implementation — mirror [`policies/pi0/components/lora.py`](../src/physicalai/policies/pi0/components/lora.py).**

Add `library/src/physicalai/policies/rldx1/components/lora.py` as a
copy of the pi0 helper (it's a 50-line `peft.get_peft_model` wrapper),
then expose PEFT fields on `Rldx1Config`:

```python
# Backbone (Qwen3-VL top-4 LLM layers)
backbone_peft_mode: Literal["full", "lora", "frozen"] = "full"
backbone_lora_rank: int = 64
backbone_lora_alpha: int = 64
backbone_lora_dropout: float = 0.0
backbone_lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

# Action model (MSAT)
action_peft_mode: Literal["full", "lora"] = "lora"     # paper-recommended default
action_lora_rank: int = 64
action_lora_alpha: int = 64
action_lora_dropout: float = 0.0
action_lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2")
```

Apply in `Rldx1.__init__` **after** `tune_top_llm_layers` has marked
the top-4 trainable (peft only wraps layers that are `requires_grad=True`
at wrap time, so lower frozen layers stay untouched):

```python
from physicalai.policies.rldx1.components.lora import apply_lora

if cfg.action_peft_mode == "lora":
    self.model.action_model = apply_lora(
        self.model.action_model,
        rank=cfg.action_lora_rank,
        alpha=cfg.action_lora_alpha,
        dropout=cfg.action_lora_dropout,
        target_modules=cfg.action_lora_targets,
    )

if cfg.backbone_peft_mode == "lora":
    self.model.vlm = apply_lora(
        self.model.vlm,
        rank=cfg.backbone_lora_rank,
        alpha=cfg.backbone_lora_alpha,
        dropout=cfg.backbone_lora_dropout,
        target_modules=cfg.backbone_lora_targets,
    )
elif cfg.backbone_peft_mode == "frozen":
    for p in self.model.vlm.parameters():
        p.requires_grad_(False)
```

**Three config presets to ship:**

```
configs/physicalai/rldx1-ft-default.yaml          # full VLM + LoRA r=64 action  (free-lunch)
configs/physicalai/rldx1-ft-consumer-gpu.yaml     # LoRA r=64 both              (single-GPU 24 GiB)
configs/physicalai/rldx1-ft-paper-baseline.yaml   # full FT both                (parity-run only)
```

**Per-task adapter swap — the real efficiency win for "20 tasks ≠ 20
checkpoints".** Once both PEFT modes work, the workflow from §5.1
collapses:

1. Train one base FT once with `full` backbone + `lora` action.
2. For each new task, train **just** a fresh action-LoRA adapter
   (~150 MB on disk vs. ~16 GB for a full ckpt) on top of the shared
   base. Reuse the same `trainer.fit()` entrypoint with a
   `peft_adapter_init_from: <base_run_dir>` field on `Rldx1Config`.
3. At inference, load base once + swap `peft` adapters per task via
   `model.load_adapter(path, adapter_name=task)` →
   `model.set_adapter(task)`. Adapter switch is <50 ms.

For 20 ALLEX tasks: 1 base ckpt (16 GB) + 20 × ~150 MB adapters ≈ 19 GB
total, vs. 320 GB of per-task full checkpoints. Same SR ceiling on each
task as a per-task LoRA-FT (paper Table 6, row "Full FT VLM + LoRA r=64").

**OpenVINO export caveat.** `peft.PeftModel` wraps Linear layers with
`LoraLayer`, which `torch.export` doesn't trace cleanly. The Studio-side
exporter must merge adapters into base weights before tracing:

```python
# Studio-side exporter
from peft import PeftModel
if isinstance(model.vlm, PeftModel):
    model.vlm = model.vlm.merge_and_unload()
if isinstance(model.action_model, PeftModel):
    model.action_model = model.action_model.merge_and_unload()
# OV export proceeds as on a non-PEFT model
```

After merge, the OV graph is bit-identical to a non-PEFT export — no
runtime change in `physicalai`. **Implication for per-task adapter
swap: each deployed task needs its own merged OV graph** (`.xml` /
`.bin`). Runtime-level adapter switching is a torch-only feature; it
doesn't survive the OV freeze.

**Security note** (per [`lib.security.instructions.md`](../../.github/instructions/lib.security.instructions.md)):
`peft` adapter checkpoints are safetensors by default — same allowlist
applies. The `apply_lora` helper takes only typed config fields, no
`class_path` injection surface.

---

## 6. RECAP RL — not feasible for v1

- **Upstream status**: zero implementation. Single dead flag
  `add_rl_callback: bool = False` at
  [`rldx/configs/training/training_config.py:116`](../../RLDX-1/rldx/configs/training/training_config.py#L116),
  never read anywhere. No critic module, no rollout loop, no
  advantage-conditioned trainer. Confirmed by grep.
- **What the paper specifies (Appendix C)**:
  - Critic: `gemma3-4b-it` VLM + LoRA rank 128 on **all** trainable parts
    (including vision encoder). Trained for 1 epoch on success-only
    demonstrations at lr 1e-4. Predicts a **distributional value** using
    native number tokens (no new head).
  - Policy: 30 K steps on AdamW lr 1e-4, global batch 128, advantage-
    conditioned supervision derived from the critic, action chunk 40 for
    ALLEX.
  - Loop: alternate (rollout policy → merge into dataset → retrain critic
    on successes only → re-annotate advantages → retrain policy) for N
    iterations.
- **What it would take in Studio**:
  - New trainer mode under `library/src/physicalai/train/` for
    advantage-conditioned policy training (call it `RecapTrainer` or a
    `RecapPolicyMixin`).
  - Critic model package — Gemma-3-4B + LoRA + value head reading native
    number tokens.
  - Real-robot or sim rollout loop — couples to `physicalai.gyms` /
    `physicalai.runtime`.
  - **Big cost item**: requires a working closed-loop sim (LIBERO,
    SIMPLER, or RoboCasa) inside Studio's training pipeline. Today our
    sim integration is eval-only.
- **Recommendation**: punt to a follow-up PR. Track as "RLDX-1 phase 2:
  RECAP post-training", scoped after we have a working BC port + a
  working SimplerEnv-driven rollout loop.

---

## 7. Evaluation harness — tiered validation, PT → FT checkpoints

We want an environment that lets us close the loop *"PAS Rldx1 +
RLDX-1-FT checkpoint → same number ± a few points as the paper"*. v1
validates only checkpoints on the `PT → FT` path (all six released
`FT-*` are eligible); `MT-*` validation is deferred to phase 2.

### 7.1 Sim env candidates

| Candidate | Vendored upstream? | Numerical baseline in our notes | Stability | Recommendation |
|---|---|---|---|---|
| **SimplerEnv WidowX** ([`external_dependencies/SimplerEnv`](../../RLDX-1/external_dependencies/SimplerEnv)) | ✅ uv-managed venv, runs out of the box. | **−2.4 pts** vs paper (69.5 vs 71.9). 4 tasks, 200 episodes each. | High — ManiSkill2 / SAPIEN; no GPU sim, deterministic seeds. | **Primary v1 target** (matches `FT-SIMPLER-WIDOWX`). |
| **LIBERO** ([`external_dependencies/LIBERO`](../../RLDX-1/external_dependencies/LIBERO)) | ✅ | None for RLDX-1; we already have PI0.5 baselines + a working gym. | High — most stable Studio integration target overall. | **Secondary v1 target** (matches `FT-LIBERO`, exercises the full add-on schema loader). |
| SimplerEnv Google-VM | ✅ same venv as WidowX. | −5.25 pts vs paper (76.25 vs 81.5). | Same as WidowX. | Triangulation; larger reproduction gap. |
| GR-1 Tabletop | ✅ [`external_dependencies/robocasa-gr1-tabletop-tasks`](../../RLDX-1/external_dependencies/robocasa-gr1-tabletop-tasks) | None yet. | Heavy — humanoid + dexterous hands, Mujoco-based. | Tertiary (matches `FT-GR1`). Low priority unless WidowX drifts. |
| RoboCasa Kitchen | ✅ [`external_dependencies/robocasa`](../../RLDX-1/external_dependencies/robocasa) + own uv venv. | Not in our notes. | Medium — Mujoco-based, more flaky than SimplerEnv. | Skip for v1. |
| RoboCasa365 | ✅ [`external_dependencies/robocasa365`](../../RLDX-1/external_dependencies/robocasa365) | None. | Largest, longest-horizon. | Skip — composite tasks pollute the alignment signal. |

### 7.2 Released checkpoints — what each one actually validates

All 10 RLWRLD model repos were enumerated via the HF API
(`api/models?author=RLWRLD`) and their `config.json` audited. Branching
structure:

```
                                            ┌── FT-LIBERO
                                            ├── FT-SIMPLER-WIDOWX
       Qwen3-VL 8B                          ├── FT-SIMPLER-GOOGLE   (state_dropout=0.5)
       (= RLDX-1-VLM) ──► PT (6.9B, base) ──┼── FT-GR1              (state_dropout=0.5)
                                            ├── FT-ROBOCASA
                                            ├── FT-RC365            (250K × 196)
                                            │
                                            ├── MT-ALLEX  (8.1B, mem+mot+phys[torque],         ah=40)
                                            └── MT-DROID  (8.1B, mem+mot+phys[tactile,torque], ah=16)
```

Full matrix:

| Checkpoint | Size | Stage | Path | Embodiment / Benchmark | mem / mot / phys | state_dropout | action_horizon | Notes |
|---|---|---|---|---|---|---|---|---|
| [`RLDX-1-VLM`](https://huggingface.co/RLWRLD/RLDX-1-VLM) | 8B | upstream VLM | (none) | Qwen3-VL 8B Instruct | — | — | — | Bare VLM; the input to PT. Not a policy. |
| [`RLDX-1-PT`](https://huggingface.co/RLWRLD/RLDX-1-PT) | 6.9B | **pre-train** | `VLM → PT` | multi-embodiment generalist | off / off / off | 0.0 | 16 | The base. All downstream checkpoints branch from here. |
| [`RLDX-1-MT-ALLEX`](https://huggingface.co/RLWRLD/RLDX-1-MT-ALLEX) | 8.1B | **mid-train** | `PT → MT` | ALLEX humanoid (48 DoF) | **on / on / on** | 0.3 | **40** | `physics_keys=['torque']`, `motion_insert_layer=9`, `memory_video_delta_indices=[-48,-32,-16,0]`. |
| [`RLDX-1-MT-DROID`](https://huggingface.co/RLWRLD/RLDX-1-MT-DROID) | 8.1B | **mid-train** | `PT → MT` | FR3 single-arm + DROID | **on / on / on** | 0.3 | 16 | `physics_keys=['tactile','torque']` (extra tactile vs ALLEX). |
| [`RLDX-1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO) | 6.9B | **post-train** | `PT → FT` | LIBERO (single-arm sim) | off / off / off (explicit) | 0.0 | 16 | Only FT ckpt that ships the full add-on schema with flags explicitly toggled off + both LoRA flags off. |
| [`RLDX-1-FT-SIMPLER-WIDOWX`](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX) | 6.9B | **post-train** | `PT → FT` | SIMPLER WidowX (BridgeData) | off / off / off | 0.0 | 16 | **Primary parity target** for our integration (§7.3). |
| [`RLDX-1-FT-SIMPLER-GOOGLE`](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE) | 6.9B | **post-train** | `PT → FT` | SIMPLER Google-VM / VA (Fractal) | off / off / off | **0.5** | 16 | Paper Table 8 calls for state_dropout 0.5 on Google-VA, 20 K steps. |
| [`RLDX-1-FT-GR1`](https://huggingface.co/RLWRLD/RLDX-1-FT-GR1) | 6.9B | **post-train** | `PT → FT` | GR-1 Tabletop (humanoid sim) | off / off / off | **0.5** | 16 | Same shape as WidowX, only difference is stronger state regularization. |
| [`RLDX-1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA) | 6.9B | **post-train** | `PT → FT` | RoboCasa Kitchen | off / off / off | 0.0 | 16 | The benchmark used in App. D PEFT table. |
| [`RLDX-1-FT-RC365`](https://huggingface.co/RLWRLD/RLDX-1-FT-RC365) | 6.9B | **post-train** | `PT → FT` | RoboCasa365 | off / off / off | 0.0 | 16 | Paper §6.1 calls for 250 K steps × batch 196 instead of the 60 K × 1024 default. |

**Key facts the matrix confirms:**

1. **No `MT → FT` checkpoint is released.** Every paper number in the
   ALLEX / FR3 real-robot tables (§6.3, §6.4, §G.3, §G.4) uses a
   task-specific FT on top of `MT-*`, but **those per-task FTs were
   never uploaded**. The released `MT-*` checkpoints are the pre-FT
   artifacts.
2. **All six `FT-*` are sim benchmarks branching directly from PT.**
   Architecture is identical to PT (6.9 B, no add-on streams) — only
   weights and 2-3 hparams differ.
3. **No checkpoint enables RTC.** Every public weight was trained with
   `rtc_training_max_delay = 0`, so setting
   `rtc_inference_mode = "trained"` at load time wouldn't help — the
   model has no training signal for the prefix τ=1 distribution. RTC
   validation requires retraining a checkpoint ourselves with
   `rtc_training_max_delay > 0` (cheapest target: LIBERO) or waiting
   for an RTC-enabled upstream checkpoint.
4. **Only three knobs change across the FT family:**
   `state_dropout_prob` (0.0 or 0.5), training steps (20 K / 60 K /
   250 K), and batch size (256 / 1024 / 196). Everything else inherits
   from PT.
5. **ALLEX's memory stride is 16, not H+1.** With `action_horizon=40`
   and `memory_video_delta_indices=[-48,-32,-16,0]`, snapshots overlap
   in time (stride 16, not the 41 implied by paper §2.1's "stride =
   H+1"). Our `delta_timestamps` wiring must read
   `memory_video_delta_indices` directly, not derive it from
   `action_horizon`.
6. **`FT-LIBERO` is the schema outlier.** It's the only FT ckpt that
   still carries the full add-on / LoRA schema with fields explicitly
   set, while the other FT configs ship a slimmer schema where those
   fields are absent (default `false`). Same effective behavior,
   different config provenance — likely exported with a different
   config dumper revision. Our loader must tolerate both.

### 7.3 Validation plan, in order

v1 only validates the `PT → FT` path. MT smoke tests and the
MT-as-deployment ablation are deferred to phase 2.

1. **PT load** — `Rldx1.from_pretrained("RLWRLD/RLDX-1-PT")` on CPU,
   verify every weight maps, no orphans, no shape mismatches.
2. **WidowX parity** — primary milestone:
   1. Pull `RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX`.
   2. Reproduce the paper's `eval_simpler.sh widowx` number from inside
      RLDX-1 (we already have this run logged at 69.5%).
   3. Port the same eval to PAS: `physicalai benchmark` against
      `simpler_env_widowx/*` driving our `Rldx1` policy. Same prompts,
      same `n_episodes=200`, same `DENOISE_STEP=10`, same `ACTION_STEPS=2`.
   4. Hit ≥ 67 % mean (within 3 pts of paper) → base-architecture
      integration validated.
3. **LIBERO parity** — secondary milestone, exercises the smallest FT
   schema variant we have to load:
   1. Pull `RLWRLD/RLDX-1-FT-LIBERO`. This is the only FT ckpt that
      ships the full add-on schema fields explicitly toggled off — good
      coverage for the loader.
   2. Run through our existing LIBERO gym (the one used for pi05).
   3. Hit paper Table 1 ± 3 pts.
4. **GR1 parity** (optional, low priority):
   1. Vendor `robocasa-gr1-tabletop-tasks` into PAS following the same
      pattern as SimplerEnv.
   2. Run `RLDX-1-FT-GR1` through PAS; expect the paper's mean ± 3 pts.
   3. Only pursue if WidowX shows unexplained drift — GR1 differs from
      WidowX only by state-dropout, so it's a triangulation, not a new
      capability test.
5. **Train-from-PT smoke** — end-to-end recipe validation:
   1. `Rldx1.from_pretrained("RLWRLD/RLDX-1-PT")`, freeze vision + LLM
      except top-4, attach action-LoRA r=64.
   2. Run 500 `trainer.fit()` steps on a small LeRobot dataset (LIBERO
      goal subset or a SO-101 sample).
   3. Verify: loss decreases, no NaN, checkpoint saves + reloads, single
      `predict_action_chunk` post-train returns sane action shape.
   *This is the only step that actually exercises the new training
   plumbing \u2014 steps 1\u20134 are pure inference.*

Phase-2 validation steps (deferred — kept for reference):

- **ALLEX smoke test** — load `RLWRLD/RLDX-1-MT-ALLEX` with
  `use_memory=use_motion=use_physics=True`, verify state-dict map,
  shapes, and gradient flow.
- **LIBERO RTC retrain** — train with `rtc_training_max_delay=4`,
  validate `rtc_inference_mode="trained"` degradation curve. Optional
  identity-Jacobian (`guided-approx`) check for export.
- **MT-as-deployment ablation** — measure SR gap between
  `RLDX-1-MT-DROID` deployed directly and a `FT-*` of the same task.
  Paper never reports this number.

---

## 8. RTC — training vs inference, gradient analysis

### 8.1 Training-time RTC ([`rtc_training_max_delay`](../../RLDX-1/rldx/configs/model/rldx.py))

- **What it does**: per sample, draws a prefix length `d_i ~ U[0, max_delay]`,
  marks the first `d_i` action positions as "frozen prefix" (set them to
  the ground-truth clean action, τ=1, exclude from the loss). Trains the
  model to predict the postfix conditioned on a clean prefix — i.e. it
  teaches the network to in-paint future actions consistent with an already-
  committed prefix.
- **Gradient mechanics**: standard flow-matching MSE loss with a per-token
  τ tensor and a postfix-only loss mask. **No autograd tricks, no VJP.**
  See [`sample_training_prefix`](../../RLDX-1/rldx/model/modules/action_model/rtc.py#L105) and
  [`build_per_token_time`](../../RLDX-1/rldx/model/modules/action_model/rtc.py#L132) — both pure tensor ops.
- **Same as the original paper?** Same algorithm as the RTC training
  procedure in arXiv 2512.05964 (Algorithm 1). RLDX-1's RTC training is
  **not** the original arXiv 2506.07339 (Black et al., "Real-Time
  Chunking"), which was inference-only.
- **Verdict**: ship it. Zero export risk.

### 8.2 Inference RTC

Three modes plus one simplification trick already in our pi05 port:

| Mode | What it does | Gradient required? | Exportable to OV/ONNX? |
|---|---|---|---|
| `none` | Standard Euler denoising; no inpainting. | No | ✅ trivial. |
| `trained` | Hard-replace the prefix positions at each Euler step (`x_τ[:, :d] = Y[:, :d]`), feed per-token time with `τ=1` on the prefix. Requires a checkpoint trained with `training_max_delay > 0`. | **No** | ✅ — just a `where(prefix_mask, Y, x_τ)` op + per-token time tensor. |
| `guided` (RLDX-1, true VJP) | Jacobian universal-guidance (arXiv 2506.07339 Eq. 2). Computes `v_guided = v + c · VJP[(Y − â¹_t)ᵀ diag(W), ∂â¹_t / ∂x_τ]`, where `â¹_t = x_τ + (1−τ)v`. **The VJP flows through the entire MSAT stack** because [`guided_velocity`](../../RLDX-1/rldx/model/modules/action_model/rtc.py#L230) sets `requires_grad_(True)` *before* calling `velocity_fn`. | **Yes — `torch.autograd.grad` through the full DiT**. | ❌ — needs autograd at runtime. |
| `guided` (PI / LeRobot / our pi05, identity-Jacobian approximation) | Same formula on paper, but the [LeRobot port](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py) calls the velocity head **before** setting `requires_grad_(True)`. Since `v_t` is detached at that point, the autograd graph for `x1_t = x_t - time * v_t` reduces to `∂x1_t/∂x_t = I`. The `autograd.grad` call returns `grad_outputs` unchanged — i.e. `correction = (Y − â¹_t) ⊙ W` with no backward pass through the network. | **Effectively no.** The autograd block is a no-op. | ✅ — once the no-op autograd wrapper is removed, this is pure forward ops. |

**The PI / LeRobot ordering "bug" is the key insight.** Both implementations
*look* like Eq. 2, but in PI's the Jacobian collapses to the identity. So
the algorithm that ships in `pi0.5` is `v_guided = v + c · err` — a
zeroth-order approximation of the original RTC. This is why our pi05
export works without any autograd machinery: we are not running the
paper's algorithm, we are running PI's simplification of it.

RLDX-1 is the **only** one of the implementations we've reviewed that
actually evaluates the paper's true VJP. Whether the extra gradient flow
matters empirically is **not ablated** in the RLDX-1 paper.

#### Export options, in increasing fidelity

| Path | Behaviour | Cost vs paper | Notes |
|---|---|---|---|
| A. Ship `none` | No prefix conditioning. | Big — loses RTC entirely. | Trivial. |
| B. Ship `trained` (recommended) | Hard inpaint + per-token τ at every Euler step. Matches arXiv 2512.05964 algorithm exactly. | Small — paper's preferred mode for inference. | Requires retraining (or fine-tuning) with `rtc_training_max_delay > 0`. |
| C. Ship `guided-approx` | Port `guided_velocity` but mirror the PI ordering (`v` without grad, drop the `autograd.grad` call, set `correction = (Y − â_t) ⊙ W`). | Unknown — identical to what pi05 already ships, so at least no worse than our current RTC behaviour. | Trivially exportable. |
| D. Ship `guided` (true VJP) | Keep RLDX-1's ordering, run in eager torch with autograd. | Zero (matches paper). | **Not exportable to OV/ONNX.** |

**Recommendation**:
- v1 export: ship **B (`trained`)** as the primary mode and **C
  (`guided-approx`)** as a secondary mode that does not require a
  specially-trained checkpoint. Both reuse the existing pi05 RTC
  infrastructure on the runtime side — no new autograd machinery, no
  graph-export blockers.
- Refuse mode `guided` (true VJP) at export time with an error: "RLDX-1's
  guided RTC requires autograd through the DiT and is not exportable.
  Use `trained` (recommended) or `guided-approx`."
- The Studio policy `Rldx1` should hold both code paths so eager-torch
  evaluation can still reproduce the paper's true `guided` mode for
  research comparison.
- Note for the runtime team: the per-Euler-step prefix injection and
  per-token time scheduling are runner-level concerns. Add
  `action_chunking_with_rtc_inpaint` and `action_chunking_with_rtc_guided_approx`
  to [`physicalai/src/physicalai/inference/runners/`](../../physicalai/src/physicalai/inference/runners) — both share
  the same `compute_prefix_weights` and `update_x_tau` helpers, only the
  velocity-correction step differs.

#### Open empirical question

We should run a side-by-side ablation **inside the RLDX-1 repo** on
SimplerEnv WidowX with three configurations:
1. `guided` (true VJP — the upstream default)
2. `guided-approx` (PI-style identity Jacobian)
3. `trained` (hard inpaint, requires a checkpoint with `training_max_delay > 0`)

If `guided-approx` lands within 1 pt of `guided`, we can confidently
recommend it as the OV/ONNX export target. If not, we have to retrain
with `training_max_delay > 0` and ship mode `trained` only.

---

## 9. Operator fusion under OpenVINO

The paper's `fused_*` kernel list (Appendix F, Table 7) maps onto OV
patterns that **OpenVINO's graph compiler already recognises**:

| RLDX-1 fused kernel | OV equivalent | Note |
|---|---|---|
| `fused_vision_attention` (RoPE+RoPE+SDPA) | OV `RoPE` + `ScaledDotProductAttention` fusion pass | Picked up automatically when ONNX export emits the `Attention-V1` pattern. |
| `fused_llm_attention` (RMSNorm + RoPE + SDPA) | OV `RMSNorm` + `ScaledDotProductAttention` fusion | Confirmed for Llama-class models — Qwen3-VL backbone benefits identically. |
| `fused_add2_layernorm`, `fused_add2_rmsnorm`, `fused_add3_rmsnorm` | OV `Add + LayerNorm` / `Add + RMSNorm` epilogue fusion | Standard transformations in `nGraph` post-processing. |
| `fused_memory_attention` | Same as `fused_vision_attention` | Memory module is identical-shape attention. |
| `fused_mlp_swiglu`, `grouped_swiglu` | OV `Swish + Mul + MatMul` fusion / fused FFN pattern | Recognized for Llama-style MLPs. |

**Plan**:
- Do **not** port any Triton. Export the eager torch module via the
  Studio's `ExportablePolicyMixin` path (see [`pi05/policy.py`](../src/physicalai/policies/pi05/policy.py)) using
  `OpenVINOExportParameters`.
- Trust OV's `ov.Core().compile_model(..., config={"PERFORMANCE_HINT": "LATENCY"})`
  to do the fusion. Verify post-export by dumping the compiled graph and
  confirming the `Attention` + `RMSNorm` nodes have been merged.
- Add a smoke test under `library/tests/` that compares per-step latency
  with and without OV `LATENCY` hint on a representative batch — analogue
  to RLDX-1's `inference/benchmark_vla.py` but on Intel hardware.
- **No memory-fusion source code lives in Studio.** This is OV's job.

---

## 10. Training-only optimization paths — where we need to branch

| Optimization | Used at training? | Safe at export? | Action |
|---|---|---|---|
| Flash-Attention 2 (Qwen3-VL) | ✅ | ❌ (CUDA-only Triton) | Branch via `attn_implementation`: `flash_attention_2` during train, `sdpa` during export. Mirror Groot's approach ([`policies/groot/config.py:attn_implementation`](../src/physicalai/policies/groot/config.py)). |
| `torch.compile` on the loss step | optional | ❌ for graph export | Wrap training step in `if not self.exporting:` guard via Lightning's `compile_model` config field. |
| Gradient checkpointing on Qwen3-VL | ✅ | n/a | Disable in `model.eval()`; export always runs from `eval()`. |
| `torch.autograd.grad` in RTC `guided` | inference-time only | ❌ | Reject at export, see §8. |
| Mixed precision `bf16` | ✅ | ⚠️ keep weights in fp16 / fp32 for OV | Convert to fp32 before OV export; OV will compress back to fp16 if `INFERENCE_PRECISION_HINT=f16`. |
| Dropout (state / memory / physics) | ✅ | n/a | `model.eval()` zeroes dropout. |
| PEFT LoRA on action model / backbone | optional, training-time | ❌ | Call `peft_model.merge_and_unload()` before export. See Appendix D — paper shows LoRA recovers full-FT performance at half the params; we should support it via a `_merge_lora_before_export()` hook. |

**Branching contract**: `Rldx1.export()` runs in this order:
1. `self.eval()`
2. Disable gradient checkpointing
3. Merge LoRA adapters (if any)
4. Swap `attn_implementation` to `sdpa`
5. Strip RTC `guided` path (raise if `rtc_inference_mode == "guided"`)
6. Dispatch to `OpenVINOExportParameters` / `ONNXExportParameters`

Encapsulate that in `Rldx1Model.prepare_for_export()` so both backends
share the same pre-export normalization.

---

## 11. Section 5 (Inference Strategy) — per-component portability

Quote the paper's structure and annotate each item with our ONNX/OV
verdict.

### 5.1 Graph Capture Optimization

| RLDX-1 mechanism | Why it exists | Need it under OV/ONNX? |
|---|---|---|
| Static Graph Conversion (RoPE caches + attention masks precomputed) | Eliminates configuration-dependent ops that fragment CUDA-Graph capture. | **No.** OV ingests an ONNX graph that is already static — RoPE buffers become weights / constants automatically. We replicate the same effect by exporting from `model.eval()` with fixed `max_seq_len`. |
| CUDA Graph (`torch.cuda.graph`) wrap of the full forward | One launch per inference step on RTX 5090. | **No.** OV's `compile_model` builds a static execution plan that is functionally equivalent on Intel GPU / NPU / CPU. |
| `torch.compile(fullgraph=True)` | Captures torch IR into a single subgraph before CUDA-Graph wrap. | **No.** `torch.compile` is incompatible with `torch.onnx.export` mid-pipeline; we go directly from eager → ONNX → OV. |

### 5.2 Kernel Optimization

| RLDX-1 mechanism | OV / ONNX replacement |
|---|---|
| Hand-written Triton fused kernels (Appendix F, Table 7) | OV's built-in fusion passes (see §9). Verified on Llama-family models — Qwen3-VL backbone matches that template. |
| Manual NVIDIA Nsight Compute profiling loop | Replace with OV's `benchmark_app` + Studio's `InferenceLatencyBenchmark`. |
| Short-prefill optimization (no autoregressive KV growth) | Already a natural fit for OV — short, fixed sequence lengths are OV's strength. |

**Bottom line**: Section 5 is a documentation of *how RLWRLD got 1.63×
on Blackwell GPUs*. None of it has to be ported. The OV inference path
is structurally simpler and gets its speedup from OV's own graph
compiler.

---

## 12. CUDA Graph + Static Graph — what to keep for OV / ONNX

| Idea from RLDX-1 | OV / ONNX analogue | Action |
|---|---|---|
| Convert configuration-dependent ops (RoPE freqs, attention masks) to constants. | OV folds these into `Constant` nodes when ONNX export sees `register_buffer(..., persistent=True)`. | Audit `Rldx1Model` for any `torch.arange(seq_len, ...)` inside `forward`. Hoist into `__init__` and `register_buffer`. **This is the one architectural change we should copy.** |
| Capture the whole forward as one graph. | OV `compile_model` already does this. | Free. |
| Pre-allocate I/O tensors and replay. | OV's infer request buffer caching. | Free. |
| Kernel-level fusion. | OV's transformation passes. | Free. |

So the answer to *"can we just do CUDA Graph + Static Graph Conversion?"*
is: **the Static-Graph-Conversion *idea* is the one to copy** (factor
config-dependent ops out of the forward) — but we copy it as
*register_buffer hygiene*, not as a runtime mechanism. CUDA Graph itself
has no OV/ONNX analogue and does not need one.

---

## Validation checklist (in execution order)

**v1 — PT → FT only:**

1. ✅ License posture documented (this file + [rldx-1.md](rldx-1.md) verdict).
2. ⬜ Reproduce paper SimplerEnv WidowX score (69.5–72%) inside RLDX-1 repo
   to lock the baseline.
3. ⬜ Land `Rldx1` Studio package per §4 (v1 layout — no memory / motion / physics components).
4. ⬜ Load `RLWRLD/RLDX-1-PT` into `Rldx1` and verify per-tensor weight match
   (allow renames; reject shape mismatches).
5. ⬜ Smoke train: BC fine-tune on a 1 K-step LIBERO subset; loss curve sane.
6. ⬜ Smoke eval: `predict_action_chunk()` on a recorded SO-101 sample;
    shape `(1, 16, action_dim)`, no NaN.
7. ⬜ **WidowX parity** (per §7.3 step 2): `Rldx1` + `RLDX-1-FT-SIMPLER-WIDOWX`
    through PAS benchmark on SimplerEnv; expect mean ≥ 67 % across 4
    tasks × 200 episodes. **Locks base-architecture integration.**
8. ⬜ **LIBERO parity** (per §7.3 step 3): `RLDX-1-FT-LIBERO` through our
    LIBERO gym; expect paper Table 1 ± 3 pts. Also exercises the
    add-on-fields-explicitly-off schema variant in the loader.
9. ⬜ **Train-from-PT smoke** (per §7.3 step 5): 500-step fine-tune from
    `RLDX-1-PT` with action-LoRA r=64; loss decreases, checkpoint round-trips.
10. ⬜ Export `Rldx1` to OpenVINO via `ExportablePolicyMixin`
     (`merge_and_unload()` PEFT first per §5.3). Run on Intel iGPU + dGPU;
     measure latency.
11. ⬜ (Optional) **GR1 parity** (per §7.3 step 4): vendor
     `robocasa-gr1-tabletop-tasks`, run `RLDX-1-FT-GR1`, expect paper
     mean ± 3 pts. Only chase if WidowX shows unexplained drift.

**Phase 2 — MT / RTC / add-on streams (deferred):**

12. ⬜ Add `use_motion=True` path, train on small video dataset, validate
     parity with upstream within a fixed-seed run.
13. ⬜ Add `use_memory=True` + the multi-stride `delta_timestamps` config
     (reading from `memory_video_delta_indices`, not derived from
     `action_horizon`); validate the memory queue is filled correctly via
     a unit test on a synthetic episode.
14. ⬜ **ALLEX smoke test**: `RLDX-1-MT-ALLEX` loads into `Rldx1` with all
     three add-ons on; `predict_action_chunk` returns `(B, 40, D)`,
     no NaN; every checkpoint tensor binds; gradient flows through
     memory / motion / physics streams on a dummy loss. **Locks the
     add-on subsystems.**
15. ⬜ Add `new_param_warmup_steps` Lightning callback + retrain a small
     MT run on a public mixture (DROID slice + a small in-house substitute);
     loss curve sane.
16. ⬜ Add RTC training (`rtc_training_max_delay=4`) on LIBERO, verify
     loss decreases similarly to the non-RTC baseline (sanity check, not
     parity).
17. ⬜ Eval RTC-trained LIBERO checkpoint with `rtc_inference_mode ∈
     {none, trained, guided-approx}`; confirm graceful degradation as
     `inference_delay` grows (paper Table 4 pattern).
18. ⬜ Repeat OV export with `use_memory=True` and the new manifest runner
     type (`action_chunking_with_memory`). Validate the session queue
     survives across `select_action()` calls.
19. ⬜ RECAP RL post-training (see §6 — punted).
20. ⬜ Physics stream training data (FR3 AnySkin or ALLEX
     tactile — requires a public dataset path).

---

## Open questions

- **Q1.** Can we get a written research-only-OK from RLWRLD that lets us
  ship `RLDX-1-PT` weights in the PAS catalog? If not, the integration
  is upstream-checkpoint-only (user downloads from HF themselves).
- **Q2.** Do we accept the 4× VLM forward cost during training when
  `use_memory=True`, or do we prototype Option B (cached cognition tokens)
  before shipping? Decision blocker: profile a 256-batch step on Intel
  dGPU with the 8 B Qwen3-VL backbone.
- **Q3.** Is RTC `trained` mode worth the extra ~6 ms/step (paper Table 4
  shows minimal-gain) versus shipping plain `none` mode in v1?
- **Q4.** Which `select_layer` for the cognition tokens — paper uses 18
  (out of 36 in Qwen3-VL-8B); fine-tune ablation in §6.5 (Table 2) shows
  18 beats 8 and 28 by ~5 pts. Hard-code 18 unless we have a reason to
  expose the field.
