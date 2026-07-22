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
> released `FT-ROBOCASA` / `FT-SIMPLER-WIDOWX` / `FT-LIBERO`
> checkpoints all use. **Primary parity target is RoboCasa Kitchen**
> (see [§7.4](#74-robocasa-kitchen--v1-primary-integration-plan)) —
> active upstream, lerobot reference wrapper, and a commercial-friendly
> training dataset. MT support is tracked as a phase-2 follow-up.
> Sections referring to MT, memory, motion, and physics below are kept
> as **context for that future work**, not as v1 deliverables.

> **Correctness carve-out (2026-07): VTC multi-frame video + image
> geometry/augmentation are promoted into phase 1.**
> v1 validation runs the pretrained `RLDX-1-FT-ROBOCASA` weight through
> the PAS `Rldx1` policy and compares success rate **1:1** against the
> paper number. `FT-ROBOCASA` was trained with the VTC video path always
> on (4 frames/step at strides `[-6,-4,-2,0]`) and the upstream
> `AspectAreaResizeAndCrop` geometry. A single-frame or
> different-geometry input changes the backbone's visual tokens and makes
> SR non-comparable, so these are **correctness requirements, not
> deferred work**. This is unrelated to the memory module (still phase-2):
> the 4 frames are a temporal window *inside one observation step*, not
> cross-step history — `memory_length=1` stays the default. Full gap list
> and required changes: [§0.1](#01-phase-1-correctness-additions--vtc-multi-frame-video--image-geometry).

---

## 0. Scope summary

| Capability                                               | Ship in v1?      | Notes                                                                                                                                                                                                       |
| -------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RLDX-1-PT` (6.9 B, video PT, no add-ons) as base        | ✅               | Sole reproducible entry point.                                                                                                                                                                              |
| MSAT action head + Qwen3-VL-8B backbone                  | ✅               | Pure-torch, exportable.                                                                                                                                                                                     |
| Flow-matching `PT → FT` post-train loop                  | ✅               | The one and only training path in v1.                                                                                                                                                                       |
| LoRA PEFT (paper App. D)                                 | ✅               | Action-LoRA default; backbone-LoRA optional. See §5.3.                                                                                                                                                      |
| VTC multi-frame video input (`video_length=4`, `video_stride=2` → strides `[-6,-4,-2,0]`) | ✅ **v1 (correctness)** | FT-ROBOCASA trains 4 video frames/step; single-frame input breaks SR parity. Promoted from phase 2 (2026-07). See §0.1. |
| Image geometry — `AspectAreaResizeAndCrop` (area-budget resize + 32-aligned crop) | ✅ **v1 (correctness)** | Eval geometry must match the checkpoint exactly; eval and train share the single `AspectAreaResizeAndCrop` transform. See §0.1. |
| Replay-consistent augmentation (`ReplayCompose` + `apply_with_replay`: ColorJitter, random crop) | ✅ **v1** | Required for train-from-PT parity — one sampled replay blob shared across the 4 frames. See §0.1. |
| RoboCasa Kitchen as the alignment benchmark              | ✅               | **Primary parity target** — matches `RLDX-1-FT-ROBOCASA`; lerobot reference wrapper exists; commercial-friendly data. See §7.4.                                                                             |
| SimplerEnv WidowX as the alignment benchmark             | ⚠️ secondary     | Useful only as a single-arm-gripper triangulation check. RLDX-1-FT-SIMPLER-WIDOWX matches the SO-101-style embodiment but the upstream `simpler-env` is semi-stale and RLDX-1 ships its own forked adapter. |
| Motion module (STSS)                                     | ❌ phase 2       | MT-only; no released FT uses it.                                                                                                                                                                            |
| Memory module (n_mem queue)                              | ❌ phase 2       | MT-only; stateful runtime work deferred.                                                                                                                                                                    |
| Physics stream (tactile / torque)                        | ❌ phase 2       | MT-only; no upstream public sensor data.                                                                                                                                                                    |
| `new_param_warmup_steps` / alignment-warmup callback     | ❌ phase 2       | Only useful when new modality streams exist.                                                                                                                                                                |
| `MT-ALLEX` / `MT-DROID` checkpoint support               | ❌ phase 2       | Skipped — no reproducible downstream path.                                                                                                                                                                  |
| Real-Time Chunking — training (`rtc_training_max_delay`) | ⚠️ optional      | Pure forward pass; defaults to 0. Keep config field, do not validate in v1.                                                                                                                                 |
| RTC inference `trained` mode                             | ⚠️ optional      | Hard-inpaint, static-graph safe. No released checkpoint has RTC.                                                                                                                                            |
| RTC inference `guided` mode                              | ❌               | Jacobian VJP through DiT — incompatible with OV / fullgraph compile.                                                                                                                                        |
| RECAP post-training (RL)                                 | ❌ phase 2+      | **Not in upstream repo.** Paper-only.                                                                                                                                                                       |
| Triton kernel chain                                      | ❌ for inference | CUDA-only. We rely on OV/ONNX fusion.                                                                                                                                                                       |
| CUDA Graph + Static Graph Conversion                     | ❌ for inference | Replaced by OV `compile_model` cache.                                                                                                                                                                       |

---

## 0.1 Phase-1 correctness additions — VTC multi-frame video + image geometry

**Promoted from phase 2 (2026-07).** Validation runs the pretrained
`RLDX-1-FT-ROBOCASA` weight through the PAS `Rldx1` policy and compares
success rate 1:1 against the paper / upstream number. Any input-pipeline
deviation that changes the model's visual tokens breaks that comparison,
so the following upstream behaviours are **v1 correctness requirements**,
not deferred work.

### Why this is a correctness issue

`RLDX-1-FT-ROBOCASA` was trained with the VTC video path **always on** —
`VideoFeature.is_active` returns `True` unconditionally
([video.py](../../RLDX-1/rldx/experiment/features/video.py#L14-L20)). The
RoboCasa video modality anchors at `delta_indices=[0]`
([robocasa_config.py](../../RLDX-1/rldx/configs/data/robocasa_config.py#L34-L35)),
and the video feature unions the strides `{(i-(L-1))·S : i∈[0,L)}` with
`L=4, S=2` → **`[-6,-4,-2,0]`**
([video.py](../../RLDX-1/rldx/experiment/features/video.py#L34-L39)).
Result: every observation step feeds the backbone **4 temporal frames**,
each tiled to `image_grid_thw = [1,6,6]`. A single-frame input produces a
different visual-token count and different backbone activations — the
success rate is not comparable.

This is distinct from the memory module (still phase-2): the 4 frames are
a temporal window *inside one observation step* (video tokens), not
cross-step history. `memory_length=1` stays the default and never enters
the `memory_length > 1 and self.training` branch.

### What is currently missing in the ported code

The v1 port (`library/src/physicalai/policies/rldx1/`) currently ships a
single-frame, deterministic-geometry pipeline. Gaps to close for phase 1:

| # | Missing behaviour | Upstream reference | Current PAS state |
| - | ----------------- | ------------------ | ----------------- |
| 1 | 4-frame temporal stacking at strides `[-6,-4,-2,0]` | [`extract_step_data`](../../RLDX-1/rldx/data/dataset/sharded_single_step_dataset.py#L31), [video.py](../../RLDX-1/rldx/experiment/features/video.py#L22-L40) | `NUM_FRAMES` hardcoded to `1` — [transforms.py#L363](../src/physicalai/policies/rldx1/preprocessor.py#L363); `video_length`/`video_stride` present but unused — [config_rldx.py#L253](../src/physicalai/policies/rldx1/components/config_rldx.py#L253) |
| 2 | `AspectAreaResizeAndCrop` (area-budget resize → 32-aligned crop) | [augmentations.py#L137](../../RLDX-1/rldx/data/augmentations.py#L137) | `AspectAreaResizeAndCrop` (shared eval+train geometry) — [augmentations.py](../src/physicalai/policies/rldx1/augmentations.py) |
| 3 | Replay-consistent augmentation (`apply_with_replay` + `ReplayCompose`: ColorJitter, random crop) — one sampled param set across all 4 frames | [augmentations.py#L84](../../RLDX-1/rldx/data/augmentations.py#L84) | dropped — [preprocessing.py#L30](../src/physicalai/policies/rldx1/preprocessing.py#L30) |
| 4 | Multi-frame conversation assembly (`num_frames` image tokens per view into the Qwen chat template) | RLDXProcessor `_get_vlm_inputs` — [processing_rldx.py](../../RLDX-1/rldx/model/core/processing_rldx.py#L615) | single-frame conversation — `_build_conversations` in [transforms.py](../src/physicalai/policies/rldx1/preprocessor.py) |
| 5 | **Inference-time** frame stacking — assemble the `[-6,-4,-2,0]` window from a per-env-step history buffer at rollout | `MultiStepWrapper` — [multistep_wrapper.py](../../RLDX-1/rldx/eval/sim/wrapper/multistep_wrapper.py#L175-L270) (deque size `span+1`, reset-filled with the initial frame, sampled at the delta offsets) | not implemented — gyms feed a single frame per step |

The backbone adapter already accepts `num_frames`
([adapter.py#L450](../src/physicalai/policies/rldx1/components/backbone/adapter.py#L450)),
so the **model path is capable** — only the data (train) and rollout
(inference) paths need the multi-frame wiring. Items 1–4 are the training
path; item 5 is the rollout path, and both must produce the identical
`(num_frames, C, H, W)` per-view stack the backbone was trained on.

### Eval-critical vs. train-critical

- **Eval / SR parity (pretrained weight):** items **1 and 2** are
  load-bearing. Inference applies no stochastic augmentation, but it
  *does* consume 4 frames with the exact `AspectAreaResizeAndCrop`
  geometry.
- **Train-from-PT parity:** item **3** (stochastic replay augmentation)
  additionally matters — FT training saw ColorJitter / random-crop with
  per-sample-consistent params across the 4 frames.

Ship all four in phase 1 so both eval parity and train-from-PT parity
hold.

### Required changes (phase 1)

Items 1–6 are **implemented** (2026-07) — items 1–5 (training +
preprocessor) plus item 6 (rollout frame stacking, gym-agnostic in the
policy).

1. ✅ **Data fetch** — `get_rldx1_delta_timestamps` +
   `get_delta_timestamps_from_policy("rldx1", ...)` emit the video window
   `[-6,-4,-2,0]` (from `video_length` / `video_stride`) for each camera
   key, so `LeRobotDataModule` returns 4 frames per step
   ([delta_timestamps.py](../src/physicalai/data/lerobot/utils/delta_timestamps.py)).
   `Rldx1.get_delta_timestamps(dataset_or_repo_id)` reads `video_length` /
   `video_stride` / `chunk_size` off the config and auto-detects the camera
   keys + fps from the dataset metadata (no per-view key lists).
2. ✅ **Preprocessor** — `NUM_FRAMES` now reports the real frame count and
   `_build_conversations` stacks frames **frame-major / view-inner** into
   the Qwen conversation ([transforms.py](../src/physicalai/policies/rldx1/preprocessor.py)).
3. ✅ **Image geometry** — eval and train share a single `AspectAreaResizeAndCrop`
   transform; eval runs it as a deterministic `A.Compose`, train wraps it with
   the stochastic stages in a `ReplayCompose`.
4. ✅ **Augmentation** — `apply_with_replay` + `ReplayCompose` ported to
   [augmentations.py](../src/physicalai/policies/rldx1/augmentations.py);
   train mode shares one replay blob across a sample's frames/views, eval
   bypasses to the deterministic geometry. Off by default (params `None`).
5. ✅ **Parity test** — `test_multiframe_forward_matches_vendored` diffs the
   native multi-frame `forward` against the vendored upstream
   `_get_vlm_inputs` + collator (`pixel_values`, `image_grid_thw`,
   `num_frames`).
6. ✅ **Rollout frame stacking** — the policy now assembles the
   `[-6,-4,-2,0]` window at **env-step cadence** from a per-view frame-history
   buffer, matching upstream `MultiStepWrapper` semantics (deque size
   `span+1`, reset-filled with the oldest frame). `Rldx1.reset()` clears the
   buffer, `select_action` (called every env step) appends the current frame,
   and `predict_action_chunk` gathers the window and hands the preprocessor a
   `(B, num_frames, C, H, W)` per-view stack
   ([policy.py](../src/physicalai/policies/rldx1/policy.py)). A batch that
   already carries a temporal axis (the training / validation
   `delta_timestamps` path, 5-D views) is passed through unchanged, so this is
   gym-agnostic — PushT eval and the gym validation rollout both get the
   4-frame stack without a per-env wrapper. Buffer mechanics covered offline by
   [test_rldx1_video_window.py](../../tests/unit/policies/test_rldx1_video_window.py).

**Still deferred to phase 2** (unchanged): `ShardedMixtureDataset` /
`ShardedSingleStepDataset` sharding + background caching, per-embodiment
statistics merging, and the memory / motion / physics streams. Sharding is
a throughput optimisation only — it does not affect SR parity, so
`LeRobotDataModule` remains the v1 dataset path.

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

### 1.1 Parameter budget by tuning flag

The 6.9 B total splits across five sub-module groups, each gated by one
`tune_*` flag on `Rldx1Config`. Measured on a real
`Rldx1Model.from_pretrained("RLWRLD/RLDX-1-PT")` load (via
[`tmp_scripts/rldx1_param_breakdown.py`](../../tmp_scripts/rldx1_param_breakdown.py)):

| Flag (`Rldx1Config`)   | Module(s)                                                                                  |         Params | % of total |
| ---------------------- | ------------------------------------------------------------------------------------------ | -------------: | ---------: |
| `tune_llm`             | Qwen3-VL LLM backbone — the 18 kept decoder layers (`select_layer=18` truncates 36 → 18) + `embed_tokens` + `lm_head` | 4,732,314,112 |     68.27% |
| `tune_diffusion_model` | MSAT action head (the flow-matching DiT)                                                    | 1,263,763,280 |     18.23% |
| `tune_visual`          | Vision tower                                                                                |   576,388,336 |      8.31% |
| `tune_projector`       | State/action encoders + decoder, position embedding, mask token (CategorySpecific per-embodiment) | 359,303,424 |      5.18% |
| `tune_vlln`            | VLM-output LayerNorm — `nn.Identity` in the PT checkpoint, so 0 params                      |             0 |      0.00% |
| **Total**              |                                                                                            | 6,932,031,296 |    100.00% |

Key consequences for the training recipe:

- **`tune_llm` is the dominant lever (68%).** The backbone truncates at
  `select_layer=18`, so "full LLM" is ~4.7 B, not the ~7 B of a complete
  Qwen3-VL-8B. Flipping `tune_llm` on is what moves a run from the ~2.4 B
  default toward the full 6.9 B.
- **The paper/PAS default trains a 2.4 B slice, not the whole model.**
  With the shipped defaults (`tune_top_llm_layers=4`, `tune_visual=False`,
  `tune_projector=True`, `tune_diffusion_model=True`) the trainable set is
  the top 4 of 18 LLM layers (~0.78 B) + projector (0.36 B) + diffusion
  head (1.26 B) + vlln (0) ≈ **2.4 B**, matching the App. D
  "Full FT top-4 + Full FT action" row (§5.3). The frozen ~4.5 B is the
  other 14 LLM layers + embeddings + `lm_head` + vision tower.
- **`tune_top_llm_layers=N` is the middle ground** — it unfreezes only
  `language_model.layers[-N:]` (a subset of the `tune_llm` group), leaving
  embeddings and `lm_head` frozen. `tune_llm=True` overrides it and
  unfreezes the entire LLM group.
- **The action head is cheap to adapt (5%).** Keep `tune_projector=True`
  for any new embodiment — the CategorySpecific projectors are what map to
  the target robot's action space; freezing them pins the output mapping
  to the pretrained `general_embodiment` values.

---

## 2. Triton in inference — avoid; use only at training

### What's CUDA-only

All Triton lives under [`rldx/inference/`](../../RLDX-1/rldx/inference/):

| Path                                                                                              | Role                                                                        |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `engine/kernels/fused_add2_rmsnorm.py`, `fused_memory_attention.py`, …                            | Triton kernels                                                              |
| `engine/cuda_graph.py`, `engine/torch_inductor.py`                                                | CUDA Graph + `torch.compile` wrappers                                       |
| `model/graph_safe_vla.py`, `action_model/model/graph_safe_*.py`, `backbone/model/graph_safe_*.py` | Static-graph rewrites of the modules with config-dependent ops factored out |
| `_rtc_dispatch.py`, `serve_optimization.py`, `benchmark_vla.py`                                   | Path selectors (A=Eager, B=Compile, C=CUDA-Graph+Compile, D=Custom-Triton)  |

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

- Memory queue **Q*t = [h*{t-n*mem·H}, …, h*{t-H}]**, n_mem = 3, stride =
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

| Option                                           | What changes                                                                                                                                                                                                                     | Cost                                                                                                                                                                       |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Multi-frame fetch + re-encode**             | Use `delta_timestamps` to fetch frames at offsets `[-3H, -2H, -H, 0]`. Run the VLM `n_mem + 1` times per sample to produce the cognition queue.                                                                                  | Simple. ~4× VLM cost per sample → **prohibitive** for an 8 B backbone.                                                                                                     |
| **B. Cached cognition tokens, sampled in-batch** | Mirror upstream: per sample fetch only the current frame, but supplement with a cached cognition tensor from a parallel "memory loader" that streams the same episode at stride H. Treat the cached tokens as a separate column. | Needs a side-table of pre-computed cognition tokens per episode (or use the model's own cognition output from a previous in-batch step — but that breaks DDP determinism). |
| **C. Upstream's approach**                       | RLDX-1 picks Option A but caps the workload via `--video-length 4` and `motion-insert-layer 9` (cheap motion encoder runs on early-layer features only). Memory snapshots reuse the same VLM forward.                            | Acceptable for 8 H200 nodes. **Painful on a single Intel dGPU.**                                                                                                           |

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
├── preprocessor.py          # image/text/state preprocessors — VTC 4-frame stacking + AspectAreaResizeAndCrop + replay aug (§0.1); no memory queue in v1
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

| Upstream path                                                                                                       | Studio target                  | Notes                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| [`rldx/configs/model/rldx.py:RLDXConfig`](../../RLDX-1/rldx/configs/model/rldx.py)                                  | `rldx1/config.py:Rldx1Config`  | Strip GR00T-isms; keep the ~80 fields that actually drive behaviour.                                                                 |
| [`rldx/model/core/rldx.py:RLDXForVisionLanguageAction`](../../RLDX-1/rldx/model/core/rldx.py)                       | `rldx1/model.py:Rldx1Model`    | Drop the Lightning glue; keep the forward(noisy_action, τ, h, m, s, p) signature.                                                    |
| [`rldx/model/modules/backbone/adapter.py:VTCQwen3VLBackbone`](../../RLDX-1/rldx/model/modules/backbone/adapter.py)  | `rldx1/components/backbone.py` | Wraps `transformers.Qwen3VLModel`, extracts layer 18, applies motion-residual hook.                                                  |
| [`rldx/model/modules/action_model/msat.py`](../../RLDX-1/rldx/model/modules/action_model/msat.py) + `blocks.py`     | `rldx1/components/msat.py`     | The double-stream + single-stream blocks, joint self-attention.                                                                      |
| [`rldx/model/modules/memory.py`](../../RLDX-1/rldx/model/modules/memory.py)                                         | `rldx1/components/memory.py`   | Verbatim — already pure torch + transformers `LlamaConfig`.                                                                          |
| [`rldx/model/modules/action_model/rtc.py`](../../RLDX-1/rldx/model/modules/action_model/rtc.py)                     | `rldx1/components/rtc.py`      | Drop `guided_velocity` (autograd VJP — see §8). Keep `sample_training_prefix`, `build_per_token_time`, `build_noisy_trajectory_rtc`. |
| [`rldx/data/state_action/state_action_processor.py`](../../RLDX-1/rldx/data/state_action/state_action_processor.py) | `rldx1/preprocessor.py`        | Normalization (1st/99th percentile), sin/cos state encoding.                                                                         |
| [`rldx/data/augmentations.py`](../../RLDX-1/rldx/data/augmentations.py) (`AspectAreaResizeAndCrop`, `apply_with_replay`, `ReplayCompose`) | `rldx1/preprocessor.py`        | **v1 correctness (§0.1)** — 4-frame VTC geometry + replay-consistent ColorJitter/crop. Same random params across the 4 frames.       |

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

| Field                | Default                                      | Per-benchmark deltas                     |
| -------------------- | -------------------------------------------- | ---------------------------------------- |
| `BASE_MODEL_PATH`    | `RLWRLD/RLDX-1-PT`                           | fixed for every v1 FT                    |
| Optimizer            | AdamW, lr 1e-4, cosine + 5 % linear warmup   | fixed                                    |
| `max_steps`          | 60 000                                       | 20 K for SIMPLER Google, 250 K for RC365 |
| `global_batch_size`  | 1024                                         | 256 for LIBERO, 196 for RC365            |
| `state_dropout_prob` | 0.0                                          | 0.5 for SIMPLER-Google and GR-1          |
| Frozen layers        | vision encoder + LLM except top-4 LLM layers | fixed                                    |
| `action_horizon`     | 16                                           | fixed in v1                              |
| Action backbone PEFT | LoRA r=64 (paper App. D free-lunch)          | see §5.3                                 |

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

| Paper name      | Checkpoint prefix                                     | What it produces                                                        |
| --------------- | ----------------------------------------------------- | ----------------------------------------------------------------------- |
| Pre-train       | `RLDX-1-PT`                                           | Generalist multi-embodiment base. No memory / motion / physics modules. |
| Mid-train       | `RLDX-1-MT-{ALLEX, DROID}`                            | Embodiment-specialized backbone **with** add-on modules baked in.       |
| Post-train (BC) | `RLDX-1-FT-{LIBERO, SIMPLER-*, GR1, ROBOCASA, RC365}` | Task / benchmark-specific BC fine-tune. Add-ons off.                    |
| Post-train (RL) | *none released*                                       | RECAP loop on top of a BC post-train. Paper-only.                       |

`FT-*` in the checkpoint naming corresponds to the paper's **post-train**
stage — pure imitation learning on a single benchmark. Every released
`FT-*` checkpoint is **`PT → FT`**, skipping mid-train.

**Goal comparison:**

| Axis                      | Mid-train                                                                           | Post-train (BC)                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Starts from               | `PT` weights                                                                        | `PT` **or** `MT-*` weights                                                             |
| What's added to the model | New modality streams (memory, motion, physics) initialized from scratch             | Nothing — same architecture as the starting checkpoint                                 |
| Trainable params          | First 2 K steps: only new-modality params. Then: all params.                        | Top-4 LLM layers + action model + projectors (vision encoder + lower LLM stay frozen). |
| Dataset shape             | **Mixture** of broad public data + narrow in-house data carrying the new modalities | Single benchmark / single task                                                         |
| LR                        | 5e-5 (small — protects PT priors)                                                   | 1e-4 with cosine + 5 % warmup                                                          |
| Steps × batch             | 25 K × 1024                                                                         | 60 K × 1024 (256 for LIBERO, 196 for RoboCasa365)                                      |
| Compute envelope (paper)  | 15 h × 64 H200                                                                      | varies, single-digit H200-days per benchmark                                           |
| What it bakes in          | Embodiment identity + memory/motion/physics weights                                 | Task-specific motor policy                                                             |

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
  revision: <pinned commit SHA> # lib.security rule 9
  embodiment_tag: NEW_EMBODIMENT # SO-101 isn't in the canonical list

  use_memory: false
  use_motion: false
  use_physics: false
  rtc_inference_mode: none # no released RTC checkpoint anyway

  action_horizon: 16 # PT default
  tune_top_llm_layers: 4
  tune_visual: false
  state_dropout_prob: 0.0 # bump to 0.5 only if overfitting

trainer:
  max_steps: 60000
  global_batch_size: 256 # LIBERO-style scale
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

| Backbone VLM  | Action model | SR          | Trainable params  | VRAM @ bs=32 | VRAM @ bs=1  |
| ------------- | ------------ | ----------- | ----------------- | ------------ | ------------ |
| Full FT top-4 | Full FT      | **62.67 %** | 2,376 M           | 87.1 GiB     | 56.8 GiB     |
| Full FT top-4 | LoRA r=64    | **62.67 %** | 1,150 M           | 76.6 GiB     | 37.2 GiB     |
| Full FT top-4 | LoRA r=8     | 60.17 %     | 1,134 M           | 76.3 GiB     | 36.8 GiB     |
| LoRA r=64     | LoRA r=64    | 55.33 %     | **398 M (5.7 %)** | **35.9 GiB** | **23.7 GiB** |
| LoRA r=8      | LoRA r=8     | 45.75 %     | 364 M             | 35.4 GiB     | 23.1 GiB     |
| Frozen        | LoRA r=64    | 36.42 %     | 378 M             | 27.5 GiB     | 23.3 GiB     |
| Frozen        | LoRA r=8     | 21.25 %     | 362 M             | 27.2 GiB     | 22.9 GiB     |

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

| Candidate                                                                                                                         | Vendored upstream?                                                                                                       | Numerical baseline in our notes                                     | Stability                                                                                                                                                           | Recommendation                                                                                                                              |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **RoboCasa Kitchen** ([`external_dependencies/robocasa`](../../RLDX-1/external_dependencies/robocasa) + upstream `robocasa@v1.0`) | ✅ vendored fork + own uv venv; upstream is installable via git+`--no-deps`.                                             | None yet — paper target is 70.6 % mean over 24 tasks × 50 episodes. | Medium — Mujoco / robosuite; lerobot reference wrapper exists and absorbs the known flakiness (objaverse-NaN, `split="test"` default, `lerobot==0.3.3` shadow pin). | **Primary v1 target** (matches `FT-ROBOCASA`). See §7.4.                                                                                    |
| **LIBERO** ([`external_dependencies/LIBERO`](../../RLDX-1/external_dependencies/LIBERO))                                          | ✅                                                                                                                       | None for RLDX-1; we already have PI0.5 baselines + a working gym.   | High — most stable Studio integration target overall.                                                                                                               | **Secondary v1 target** (matches `FT-LIBERO`, exercises the full add-on schema loader).                                                     |
| SimplerEnv WidowX ([`external_dependencies/SimplerEnv`](../../RLDX-1/external_dependencies/SimplerEnv))                           | ✅ uv-managed venv, runs out of the box.                                                                                 | **−2.4 pts** vs paper (69.5 vs 71.9). 4 tasks, 200 episodes each.   | High — ManiSkill2 / SAPIEN; no GPU sim, deterministic seeds. **But** upstream `simpler-env/SimplerEnv` is semi-stale; RLDX-1 carries a custom 260-line adapter.     | Tertiary — embodiment-matched (single-arm gripper) but pipeline cost is higher than RoboCasa.                                               |
| SimplerEnv Google-VM                                                                                                              | ✅ same venv as WidowX.                                                                                                  | −5.25 pts vs paper (76.25 vs 81.5).                                 | Same as WidowX.                                                                                                                                                     | Triangulation; larger reproduction gap.                                                                                                     |
| GR-1 Tabletop                                                                                                                     | ✅ [`external_dependencies/robocasa-gr1-tabletop-tasks`](../../RLDX-1/external_dependencies/robocasa-gr1-tabletop-tasks) | None yet.                                                           | Heavy — humanoid + dexterous hands, Mujoco-based.                                                                                                                   | Quaternary (matches `FT-GR1`). Low priority.                                                                                                |
| RoboCasa365                                                                                                                       | ✅ [`external_dependencies/robocasa365`](../../RLDX-1/external_dependencies/robocasa365)                                 | None — paper target 31.5 % mean over 365 tasks.                     | Largest, longest-horizon.                                                                                                                                           | Drop-in once RoboCasa Kitchen wrapper lands (same env class, swap `task="atomic_seen"` → `task="composite_*"`). Defer until Kitchen passes. |

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

| Checkpoint                                                                           | Size | Stage          | Path       | Embodiment / Benchmark           | mem / mot / phys           | state_dropout | action_horizon | Notes                                                                                                   |
| ------------------------------------------------------------------------------------ | ---- | -------------- | ---------- | -------------------------------- | -------------------------- | ------------- | -------------- | ------------------------------------------------------------------------------------------------------- |
| [`RLDX-1-VLM`](https://huggingface.co/RLWRLD/RLDX-1-VLM)                             | 8B   | upstream VLM   | (none)     | Qwen3-VL 8B Instruct             | —                          | —             | —              | Bare VLM; the input to PT. Not a policy.                                                                |
| [`RLDX-1-PT`](https://huggingface.co/RLWRLD/RLDX-1-PT)                               | 6.9B | **pre-train**  | `VLM → PT` | multi-embodiment generalist      | off / off / off            | 0.0           | 16             | The base. All downstream checkpoints branch from here.                                                  |
| [`RLDX-1-MT-ALLEX`](https://huggingface.co/RLWRLD/RLDX-1-MT-ALLEX)                   | 8.1B | **mid-train**  | `PT → MT`  | ALLEX humanoid (48 DoF)          | **on / on / on**           | 0.3           | **40**         | `physics_keys=['torque']`, `motion_insert_layer=9`, `memory_video_delta_indices=[-48,-32,-16,0]`.       |
| [`RLDX-1-MT-DROID`](https://huggingface.co/RLWRLD/RLDX-1-MT-DROID)                   | 8.1B | **mid-train**  | `PT → MT`  | FR3 single-arm + DROID           | **on / on / on**           | 0.3           | 16             | `physics_keys=['tactile','torque']` (extra tactile vs ALLEX).                                           |
| [`RLDX-1-FT-LIBERO`](https://huggingface.co/RLWRLD/RLDX-1-FT-LIBERO)                 | 6.9B | **post-train** | `PT → FT`  | LIBERO (single-arm sim)          | off / off / off (explicit) | 0.0           | 16             | Only FT ckpt that ships the full add-on schema with flags explicitly toggled off + both LoRA flags off. |
| [`RLDX-1-FT-SIMPLER-WIDOWX`](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX) | 6.9B | **post-train** | `PT → FT`  | SIMPLER WidowX (BridgeData)      | off / off / off            | 0.0           | 16             | **Primary parity target** for our integration (§7.3).                                                   |
| [`RLDX-1-FT-SIMPLER-GOOGLE`](https://huggingface.co/RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE) | 6.9B | **post-train** | `PT → FT`  | SIMPLER Google-VM / VA (Fractal) | off / off / off            | **0.5**       | 16             | Paper Table 8 calls for state_dropout 0.5 on Google-VA, 20 K steps.                                     |
| [`RLDX-1-FT-GR1`](https://huggingface.co/RLWRLD/RLDX-1-FT-GR1)                       | 6.9B | **post-train** | `PT → FT`  | GR-1 Tabletop (humanoid sim)     | off / off / off            | **0.5**       | 16             | Same shape as WidowX, only difference is stronger state regularization.                                 |
| [`RLDX-1-FT-ROBOCASA`](https://huggingface.co/RLWRLD/RLDX-1-FT-ROBOCASA)             | 6.9B | **post-train** | `PT → FT`  | RoboCasa Kitchen                 | off / off / off            | 0.0           | 16             | The benchmark used in App. D PEFT table.                                                                |
| [`RLDX-1-FT-RC365`](https://huggingface.co/RLWRLD/RLDX-1-FT-RC365)                   | 6.9B | **post-train** | `PT → FT`  | RoboCasa365                      | off / off / off            | 0.0           | 16             | Paper §6.1 calls for 250 K steps × batch 196 instead of the 60 K × 1024 default.                        |

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

### 7.4 RoboCasa Kitchen — v1 primary integration plan

RoboCasa Kitchen is the v1 alignment benchmark. This subsection covers
the Studio gym wrapper, the data path, the two-checkpoint key-schema
gotcha, and a parallel π₀.₅ baseline that anchors the comparison.

#### Why RoboCasa over SimplerEnv WidowX

| Axis                              | RoboCasa Kitchen                                                                                                                                                                                 | SimplerEnv WidowX                                                                                                                                                                |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream package status           | `robocasa@v1.0` actively maintained ([`robocasa/robocasa`](https://github.com/robocasa/robocasa)); install via git URL with `--no-deps` (its `setup.py` pins `lerobot==0.3.3`).                  | `simpler-env/SimplerEnv` semi-stale; not on PyPI.                                                                                                                                |
| Lerobot reference wrapper         | ✅ [`lerobot/src/lerobot/envs/robocasa.py`](../../lerobot/src/lerobot/envs/robocasa.py) — production-quality, async-vec ready.                                                                   | ❌ no `simpler.py` in `lerobot/src/lerobot/envs/`.                                                                                                                               |
| Glue code in RLDX-1               | Lightweight wrapper around upstream; ~80-line [`gymnasium_basic.py`](../../RLDX-1/external_dependencies/robocasa/robocasa/utils/gym_utils/gymnasium_basic.py) + GR00T-style env-ID registration. | 260-line custom adapter ([`rldx/eval/sim/SimplerEnv/simpler_env.py`](../../RLDX-1/rldx/eval/sim/SimplerEnv/simpler_env.py)) with sticky-gripper FSM and quat↔euler bookkeeping. |
| Embodiment match to SO-101        | Mobile manipulator (PandaOmron) — wrong shape (base-motion + control-mode in the 12-D action).                                                                                                   | Single-arm gripper (WidowX) — closer to SO-101's 6-DoF + gripper.                                                                                                                |
| Commercial-friendly training data | ✅ `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` (CC-BY-4.0), 24-task × 3000-demo Kitchen partition.                                                                                       | ⚠️ BridgeData (research-licensed); no canonical fine-tune dataset.                                                                                                               |
| Paper-target SR                   | 70.6 % mean over 24 tasks × 50 ep.                                                                                                                                                               | 71.9 % mean over 4 tasks × 200 ep.                                                                                                                                               |

**Trade-off**: RoboCasa wins on infrastructure and data; SimplerEnv WidowX
wins on embodiment match. We start with RoboCasa for pipeline validation
and add SimplerEnv WidowX later as an embodiment-matched parity check.

#### Paper-24 task availability in robocasa@v1.0

The paper ([arxiv:2406.02523](https://arxiv.org/pdf/2406.02523), Table
reproduced below) evaluated `RLDX-1-FT-ROBOCASA` on **24 single-stage
atomic tasks** at 50 episodes each, mean SR **70.6 %**. The robocasa
v1.0 refactor (SHA `56e355c`) renamed `kitchen/single_stage/` →
`kitchen/atomic/` and **merged several paper tasks together**.
Consequence: we can compare 19 / 24 paper rows directly, 4 / 24 only as
family aggregates, and 1 / 24 not at all.

This is the canonical eval mapping. **Do not rely on robocasa's own
`atomic_seen` group for paper parity** — it is a different v1.0
curation (18 tasks, 10 of which the paper never tested). Use
`atomic_seen` only for smoke-testing the wrapper and for tracking
v1.0-native benchmark drift.

Legend:

- ✅ — exact paper class name still registered in v1.0; compare 1:1.
- 🟡 — paper class merged into a family; compare against the family
  proxy listed in **v1.0 env class**. Lose paper's per-orientation
  breakdown but recover a comparable aggregate.
- ❌ — class dropped in v1.0 with no equivalent; cannot compare.

| Group        | Paper task              | Paper SR | v1.0 env class                | In `atomic_seen`? |       Status        |
| ------------ | ----------------------- | -------: | ----------------------------- | :---------------: | :-----------------: |
| Pick & place | `PnPCabToCounter`       |     44.0 | `PickPlaceCabinetToCounter`   |        no         |         ✅          |
| Pick & place | `PnPCounterToCab`       |     64.0 | `PickPlaceCounterToCabinet`   |        yes        |         ✅          |
| Pick & place | `PnPCounterToMicrowave` |     38.0 | `PickPlaceCounterToMicrowave` |        no         |         ✅          |
| Pick & place | `PnPCounterToSink`      |     72.0 | `PickPlaceCounterToSink`      |        no         |         ✅          |
| Pick & place | `PnPCounterToStove`     |     66.0 | `PickPlaceCounterToStove`     |        yes        |         ✅          |
| Pick & place | `PnPMicrowaveToCounter` |     26.0 | `PickPlaceMicrowaveToCounter` |        no         |         ✅          |
| Pick & place | `PnPSinkToCounter`      |     76.0 | `PickPlaceSinkToCounter`      |        yes        |         ✅          |
| Pick & place | `PnPStoveToCounter`     |     56.0 | `PickPlaceStoveToCounter`     |        no         |         ✅          |
| Open / close | `OpenSingleDoor`        |     80.0 | `OpenCabinet` (proxy)         |        yes        |         🟡          |
| Open / close | `OpenDoubleDoor`        |     62.0 | `OpenCabinet` (proxy)         |        yes        |         🟡          |
| Open / close | `CloseSingleDoor`       |    100.0 | `CloseCabinet` (proxy)        |        no         |         🟡          |
| Open / close | `CloseDoubleDoor`       |     94.0 | `CloseCabinet` (proxy)        |        no         |         🟡          |
| Open / close | `OpenDrawer`            |     80.0 | `OpenDrawer`                  |        yes        |         ✅          |
| Open / close | `CloseDrawer`           |     98.0 | `CloseDrawer`                 |        no         |         ✅          |
| Others       | `TurnOnStove`           |     24.0 | `TurnOnStove`                 |        no         |         ✅          |
| Others       | `TurnOffStove`          |     22.0 | `TurnOffStove`                |        yes        |         ✅          |
| Others       | `TurnOnSinkFaucet`      |    100.0 | `TurnOnSinkFaucet`            |        yes        |         ✅          |
| Others       | `TurnOffSinkFaucet`     |     96.0 | `TurnOffSinkFaucet`           |        no         |         ✅          |
| Others       | `TurnSinkSpout`         |     90.0 | `TurnSinkSpout`               |        no         |         ✅          |
| Others       | `CoffeeServeMug`        |     82.0 | `CoffeeServeMug`              |        no         |         ✅          |
| Others       | `CoffeeSetupMug`        |     44.0 | `CoffeeSetupMug`              |        yes        |         ✅          |
| Others       | `TurnOnMicrowave`       |     86.0 | `TurnOnMicrowave`             |        yes        |         ✅          |
| Others       | `TurnOffMicrowave`      |     96.0 | `TurnOffMicrowave`            |        no         |         ✅          |
| Others       | `CoffeePressButton`     |     98.0 | —                             |        no         |         ❌          |
| **Total**    | **24 tasks**            | **70.6** | —                             |      8 / 24       | 19 ✅ / 4 🟡 / 1 ❌ |

**Implications for the eval suite**

- Define a constant `RLDX1_PAPER_TASKS_DIRECT` listing the 19 ✅ rows
  by their v1.0 class name. Compare per-task SR to the paper number
  one-to-one. Headline metric: **mean SR over 19 tasks × 50 ep**;
  paper-equivalent reference = mean of the same 19 rows in the paper =
  **66.3 %** (vs. paper's 70.6 % over the full 24).
- For the 4 🟡 door rows, run `OpenCabinet` and `CloseCabinet` once
  each at 50 ep. Report each as an aggregate proxy against the paper's
  two-row mean (open: mean 71.0; close: mean 97.0). Do not average
  proxies into the headline number — log separately.
- Skip `CoffeePressButton`. Note its absence in the report.

**Optional secondary benchmark**: also run robocasa's native
`atomic_seen` (18 tasks). It overlaps the paper-24 on 8 tasks and adds
10 v1.0-native ones (`CloseBlenderLid`, `NavigateKitchen`,
`SlideDishwasherRack`, etc.). Useful for tracking against the current
upstream benchmark, not for paper parity.

#### Gym wrapper — port from lerobot

Add `library/src/physicalai/gyms/robocasa.py`, modelled on
[`physicalai/gyms/libero.py`](../src/physicalai/gyms/libero.py) and ported from
[`lerobot/src/lerobot/envs/robocasa.py`](../../lerobot/src/lerobot/envs/robocasa.py). Concrete deltas:

| Change                                                                                                                                                                                                                          | Why                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Use `physicalai.data.observation.Observation` (not lerobot's `RobotObservation`).                                                                                                                                               | PAS-native type; same `.to_dict()` surface.                                                                                                                                                                                                    |
| Drop `_LazyAsyncVectorEnv`.                                                                                                                                                                                                     | PAS gyms run single-env today; vectorization is a separate concern.                                                                                                                                                                            |
| Keep lerobot's three workarounds verbatim: `obj_registries=("lightwheel",)`, `split` defaulting to `"all"`, `_TASK_GROUP_SPLITS` mapping `atomic_seen → split="target"`.                                                        | These are upstream bugs that bit lerobot; re-discovering them wastes a week.                                                                                                                                                                   |
| Expose `task: str` accepting both task-group keywords (`"atomic_seen"`, `"composite_seen"`, `"composite_unseen"`, `"pretrain50/100/200/300"`) and individual task names (`"PickPlaceCounterToCabinet"`, comma-separated, etc.). | All recognised keywords resolve via `robocasa.utils.dataset_registry`. Single names skip the resolver. Eval-specific subsets (e.g. the paper-parity slices from §7.4) live alongside the benchmark that consumes them, not in the gym wrapper. |
| `obs_type ∈ {"pixels", "pixels_agent_pos"}` with three default cameras (`robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`).                                                                               | These are the v1.0+ upstream camera names; matches `RLDX-1-FT-RC365` and any pi05 fine-tune we train fresh.                                                                                                                                    |
| Action space: 12-D flat `Box`. `step()` calls a `convert_action()` helper that re-splits into RoboCasa's dict action (`base_motion(4) + control_mode(1) + ee_pos(3) + ee_rot(3) + gripper(1)`).                                 | Matches lerobot's contract; downstream eval code stays array-based.                                                                                                                                                                            |

Then add `RoboCasaBenchmark` under `library/src/physicalai/benchmark/gyms/robocasa/`,
mirroring `LiberoBenchmark`. Defaults: `task="atomic_seen"`, `num_episodes=50`,
`max_steps=720`, `n_action_steps=16` — taken from
[`run_scripts/eval/robocasa_kitchen/eval_robocasa.sh`](../../RLDX-1/run_scripts/eval/robocasa_kitchen/eval_robocasa.sh).

#### Step-by-step gym integration plan

Execute these steps in order. Each step is independently testable and
leaves the tree in a green state.

**Step 1 — add an install script** (`library/scripts/install_robocasa.sh`).

The lerobot project deliberately does **not** expose `robocasa` as a
pyproject extra. We tried; the unified uv resolve is unsolvable for
the same reasons:

- `robocasa`'s `setup.py` pins `lerobot==0.3.3`, which collides with
  our `lerobot>=0.5.1` base dep. `override-dependencies` neutralises
  this but only the first half of the problem.
- `robocasa` requires `robosuite` master (>=1.5dev) for
  `HybridMobileBase` / composite controllers. Listing `robosuite @
git+...` in `[robocasa]` shadows hf-libero's `robosuite==1.4.0` pin
  globally, even with a `[tool.uv].conflicts` entry between `[libero]`
  and `[robocasa]` — because `[all]` still pulls `[libero]` into the
  robocasa-active split.
- `robocasa`'s `tianshou==0.4.10` pin transitively requires
  `protobuf<3.20`, which collides with our `onnx`'s `protobuf>=3.20`.
  (tianshou is never imported by robocasa — a dead pin.)

Ship the install procedure as `library/scripts/install_robocasa.sh`
instead. It mirrors lerobot's
[`docker/Dockerfile.benchmark.robocasa`](../../lerobot/docker/Dockerfile.benchmark.robocasa)
recipe:

```bash
# 1. Create a robocasa-dedicated venv (NOT shared with [libero]).
uv venv .venv-robocasa
source .venv-robocasa/bin/activate
uv sync --extra cu128                  # or cpu / xpu

# 2. Install robocasa + robosuite at the SHAs lerobot pins.
bash library/scripts/install_robocasa.sh

# 3. Download kitchen assets (~2GB; lightweight set only).
yes y | python -m robocasa.scripts.download_kitchen_assets \
    --type tex tex_generative fixtures_lw objs_lw

# 4. Headless servers only.
export MUJOCO_GL=egl
```

SHAs in the script are the same ones lerobot pins:
[`robocasa@56e355c`](https://github.com/robocasa/robocasa/tree/56e355ccc64389dfc1b8a61a33b9127b975ba681)
and
[`robosuite@aaa8b9b`](https://github.com/ARISE-Initiative/robosuite/tree/aaa8b9b214ce8e77e82926d677b4d61d55e577ab).
Bump them together when picking up upstream fixes.

**No `pyproject.toml` change ships with v1** — the `[robocasa]` extra
does not exist and `[tool.uv].conflicts` stays unchanged. The script
plus README is the contract.

**Step 2 — verify the upstream env runs standalone**.

Before touching PAS code, confirm the install script produced a
working robocasa in the new venv:

```python
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
env = RoboCasaGymEnv(
    env_name="CloseFridge",  # any atomic_seen task; PnPCounterToCab is v0.x-only
    split="all",
    obj_registries=("lightwheel",),
    camera_widths=256,
    camera_heights=256,
)
obs, _ = env.reset(seed=0)
print({k: type(v).__name__ for k, v in obs.items()})
```

Expected: a dict with `video.robot0_agentview_left/right`,
`video.robot0_eye_in_hand`, plus `state.*` keys (base_position(3),
base_rotation(4), end_effector_position_relative(3),
end_effector_rotation_relative(4), gripper_qpos(2)). If `reset()` raises
`Probabilities contain NaN`, the lightwheel asset pack is missing —
re-run the `download_kitchen_assets --type objs_lw` step from Step 1.

**Step 3 — port the wrapper to PAS** (new file
`library/src/physicalai/gyms/robocasa.py`, ~250 LoC).

Copy [`lerobot/src/lerobot/envs/robocasa.py`](../../lerobot/src/lerobot/envs/robocasa.py)
verbatim, then apply these edits:

1. Replace `from lerobot.types import RobotObservation` with
   `from physicalai.data.observation import Observation` and
   `from physicalai.gyms.base import Gym`.
2. Change the class declaration `class RoboCasaEnv(gym.Env)` → `class
RoboCasaGym(Gym)` to match PAS naming (see `LiberoGym`, `PushTGym`).
   Drop the `super().reset(seed=...)` call — `Gym` is a plain `ABC`,
   not a `gym.Env`, so there's no base to forward to.
3. Wrap the upstream import in the `_LIBERO_AVAILABLE`-style guard from
   [`physicalai/gyms/libero.py`](../src/physicalai/gyms/libero.py#L82-L99) — set
   `_ROBOCASA_AVAILABLE`, expose `_check_robocasa_available()`. The
   install hint **does not** point at a `[robocasa]` extra (that extra
   intentionally does not exist — see Step 1); it points at
   [`library/scripts/install_robocasa.sh`](../scripts/install_robocasa.sh)
   and tells the user to create a dedicated `.venv-robocasa`.
4. Implement the five `Gym` abstract methods. `reset`, `step`, `close`
   already exist in the lerobot wrapper; the new ones are
   `sample_action` and `to_observation`. Keep the LIBERO two-method
   pattern: `_format_raw_obs` returns a plain dict (`{"pixels": ...,
"agent_pos": ...}`) and `to_observation` wraps that dict into an
   `Observation` (HWC→CHW, uint8→float/255, unsqueeze batch, attach
   `task_description`). Splitting them keeps `_format_raw_obs`
   directly testable and mirrors `LiberoGym._format_raw_obs` /
   `LiberoGym.to_observation`.
   - `sample_action()`: `return torch.from_numpy(self.action_space.sample()).float()`.
   - `to_observation(raw_obs)`: see
     [`LiberoGym.to_observation`](../src/physicalai/gyms/libero.py#L505)
     for the exact conversion. Use
     [`physicalai/data/observation.py`](../src/physicalai/data/observation.py)
     as the source of truth for `Observation` field shapes.
5. Delete `_LazyAsyncVectorEnv`, `_make_env_fns`, `create_robocasa_envs`,
   the `AsyncVectorEnv` branch, and the `episode_index` parameter
   itself (it exists only to spread a seed across `AsyncVectorEnv`
   workers; with no vector-env there is nothing to spread). Replace
   the multi-task factory helper with a `create_robocasa_gyms(tasks:
list[str], ...)` function that returns `list[RoboCasaGym]` — one
   per task name. Model on `create_libero_gyms` in
   [`physicalai/gyms/libero.py`](../src/physicalai/gyms/libero.py),
   but accept **only explicit task names** (or a group keyword like
   `"atomic_seen"` resolved via `_resolve_tasks`) — no numeric task
   IDs. RoboCasa tasks are named, not indexed.
6. Drop the in-`step()` auto-reset (`if terminated: ...
self.reset()`). The lerobot wrapper auto-resets to satisfy
   `gym.vector.AsyncVectorEnv`'s autoreset contract. PAS gyms are
   driven by an explicit benchmark loop (see `LiberoBenchmark`) that
   calls `gym.reset()` between episodes, so auto-reset would just
   discard the terminal observation and double the per-episode cost.
7. Keep `_TASK_GROUP_SPLITS`, `_resolve_tasks`, `convert_action`,
   `DEFAULT_CAMERAS`, `DEFAULT_OBJ_REGISTRIES`, `OBS_STATE_DIM`,
   `ACTION_DIM` verbatim. `_TASK_GROUP_SPLITS` covers only the
   upstream `robocasa.utils.dataset_registry` groups (`atomic_seen`,
   `composite_seen`, `composite_unseen`,
   `pretrain50/100/200/300`). The paper-derived subsets from §7.4
   (`paper_direct`, `paper_door_proxies`) ship in a follow-up PR
   alongside `RoboCasaBenchmark` — they are eval metadata, not gym
   metadata, and keeping the gym module paper-agnostic preserves a
   clean boundary.
8. Keep raw RoboCasa camera names (`robot0_agentview_left`,
   `robot0_eye_in_hand`, `robot0_agentview_right`) verbatim in the
   `Observation.images` dict — do **not** introduce a
   `CAMERA_NAME_MAPPING` like `LiberoGym` has. Per-policy renames go
   through the `RLDX_CAMERA_REMAP_KITCHEN` adapter at policy-input
   time (see §7.4 wrapper-deltas table and §8.x). The Studio policy
   keys must match the upstream RoboCasa dataset keys exactly.
9. Convert `step()`'s incoming `action` from `torch.Tensor` to numpy
   before `convert_action()` — PAS `Gym.step` is typed `torch.Tensor`.
   Copy the `LiberoGym.step` pattern
   ([libero.py:401-405](../src/physicalai/gyms/libero.py#L401)).
10. Register in `library/src/physicalai/gyms/__init__.py`:
    `from .robocasa import RoboCasaGym, create_robocasa_gyms`.

**Step 4 — unit test** (new file
`library/tests/unit/gyms/test_robocasa.py`, ~150 LoC).

Mirror [`tests/unit/gyms/test_libero.py`](../tests/unit/gyms/test_libero.py).
Use `pytest.importorskip("robocasa"); pytest.importorskip("robosuite")`
at module top. Cover:

- `test_resolves_task_group_to_split`: `_resolve_tasks("atomic_seen")`
  returns `(list of task names, "target")` against the pinned SHA.
  Assert membership of a few known-stable v1.0 names (`CloseFridge`,
  `OpenCabinet`, `OpenDrawer`) rather than the exact count, so an
  upstream task-mix update doesn't immediately break the test.
- `test_resolves_single_task_keeps_split_none`: `_resolve_tasks("CloseFridge")`
  returns `(["CloseFridge"], None)`.
- `test_rejects_unknown_task_group`: `_resolve_tasks("definitely_not_a_group")`
  splits on comma, treats as a single task — verify it doesn't crash
  the regex path.
- `test_convert_action_shapes`: `convert_action(np.zeros(12))` returns
  a dict with the five expected keys and matching slice widths.
- `test_observation_space_pixels_agent_pos`: instantiate with
  `obs_type="pixels_agent_pos"`, assert `OBS_STATE_DIM == 16` and the
  `agent_pos` Box has shape `(16,)`.
- `test_observation_space_pixels_only`: assert the absence of
  `agent_pos` when `obs_type="pixels"`.

All of these run without spinning up MuJoCo — the constructor only
touches `_env` lazily inside `_ensure_env()`.

**Step 5 — integration smoke test** (new file
`library/tests/integration/gyms/test_robocasa_e2e.py`, ~120 LoC).

Mirror [`tests/integration/gyms/test_libero_e2e.py`](../tests/integration/gyms/test_libero_e2e.py).
Marked `@pytest.mark.integration` and `@pytest.mark.slow` so it runs
only in the heavy CI lane. Cover:

- `test_reset_returns_observation_shape`: `gym.reset(seed=0)` returns
  an `Observation` whose images dict has the three default cameras
  with shape `(256, 256, 3)` uint8, and `state.shape == (16,)`.
- `test_step_random_action`: one `gym.step(gym.sample_action())` does
  not raise; returns `(Observation, float, bool, bool, dict)`.
- `test_episode_termination_on_success_flag`: skip if no easy success
  trigger — at minimum verify `info["is_success"]` key is present.
- `test_act_policy_roundtrip` (optional, expensive): wrap an
  untrained `ACT` policy, do 10 steps, assert no NaNs in actions.
  Mirrors the LIBERO E2E pattern.

Use `task="CloseFridge"` (single atomic task, cheapest scene, verified
in Step 2) for all integration tests. Don't run the full `atomic_seen`
group from a test — that's an eval, not a smoke check.

**Step 6 — benchmark wrapper** (new directory
`library/src/physicalai/benchmark/gyms/robocasa/`).

Create three files mirroring `benchmark/libero/`:

```
benchmark/gyms/robocasa/
├── __init__.py              # re-export RoboCasaBenchmark
├── benchmark.py             # RoboCasaBenchmark class (driver)
└── config.py                # RoboCasaBenchmarkConfig dataclass
```

`RoboCasaBenchmarkConfig` fields (paper / RLDX-1 defaults):

| Field                | Default         | Source                 |
| -------------------- | --------------- | ---------------------- |
| `task`               | `"atomic_seen"` | `eval_robocasa.sh`     |
| `num_episodes`       | 50              | `eval_robocasa.sh`     |
| `max_steps`          | 720             | `eval_robocasa.sh`     |
| `n_action_steps`     | 16              | `eval_robocasa.sh`     |
| `observation_height` | 256             | matches FT checkpoints |
| `observation_width`  | 256             | matches FT checkpoints |
| `seed`               | 0               | reproducibility        |

`RoboCasaBenchmark.run(policy)` loops over the resolved task list, for
each task instantiates a `RoboCasaGym`, runs `num_episodes` rollouts,
and returns a `BenchmarkResult` with per-task and aggregate SR. Reuse
the rollout loop from `LiberoBenchmark` — they have the same shape.

**Step 7 — fixtures and pytest markers**.

Add `robocasa` to the marker list in [`library/pyproject.toml`](../pyproject.toml)
under `[tool.pytest.ini_options].markers` only if a new marker is
needed (`requires_robocasa_assets`). If reusing `slow` + `integration`,
no marker change is required.

Add a `conftest.py` skip clause if needed — the existing one already
handles `robosuite` import-time side effects (see
[`tests/conftest.py`](../tests/conftest.py)).

**Step 8 — docs** (one short README, no separate doc file).

Append a section to `library/src/physicalai/gyms/README.md` (or
create one if it doesn't exist) documenting:

- The `robocasa` extra and the LIBERO conflict.
- The asset download command (`python -m robocasa.scripts.download_kitchen_assets --type objs_lw`).
- The three known upstream gotchas already encoded as workarounds in
  the wrapper (objaverse-NaN, `split="test"` default, `atomic_seen`
  → `target` split).

**Step 9 — validation (per the §7.4 plan + checklist items 2, 8)**.

In order:

1. Dedicated venv ready: `uv venv .venv-robocasa && source
.venv-robocasa/bin/activate && uv sync --active --extra cu128 &&
bash library/scripts/install_robocasa.sh` all succeed; `python -c
"from physicalai.gyms import RoboCasaGym; print(RoboCasaGym)"`
   prints the class.
2. Step 4 unit tests pass.
3. Step 5 integration smoke test passes on a CUDA box with MuJoCo
   assets installed.
4. Reproduce paper RoboCasa Kitchen number (~70 %) inside the RLDX-1
   repo via the upstream eval shell — locks the baseline.
5. RoboCasa Kitchen parity (checklist item 8): `Rldx1` +
   `RLDX-1-FT-ROBOCASA` through `RoboCasaBenchmark(task="paper_direct")`
   (the 19 ✅ rows from the paper-24 table); expect mean SR within 3
   pts of the paper-equivalent reference (**66.3 %** over the 19
   directly comparable rows). Also run `task="paper_door_proxies"`
   and log as aggregates against the paper's door rows.
   `CoffeePressButton` is unavailable; note as gap.

#### Install layout (manual; not a pyproject extra)

We deliberately do **not** add a `[robocasa]` extra to
[`library/pyproject.toml`](../pyproject.toml). The dep graph is
unsolvable in a single uv resolution — see Step 1 above for the three
specific conflicts (`lerobot==0.3.3` shadow pin, `robosuite==1.4.0`
clash with `[libero]`, `tianshou==0.4.10` → `protobuf<3.20` vs.
`onnx>=3.20`).

Install via [`library/scripts/install_robocasa.sh`](../scripts/install_robocasa.sh)
into a dedicated venv. SHAs come from
[`lerobot/docker/Dockerfile.benchmark.robocasa`](../../lerobot/docker/Dockerfile.benchmark.robocasa);
bump together when pulling upstream fixes.

Prefer the upstream `robocasa` SHA over RLDX-1's vendored fork — the
RLDX-1 fork is a pre-v1 snapshot (v0.2) patched for GR00T-style
multi-resolution camera outputs we don't need, and the lerobot-pinned
SHA tracks upstream fixes.

##### Note on LIBERO pinning

LIBERO uses the regular pyproject extra pattern
(`libero = ["hf-libero>=0.1.3,<0.2.0"]`) because it actually ships on
PyPI. The `<0.2.0` cap is load-bearing: `hf-libero` transitively
locks `robosuite==1.4.0`, which the LIBERO success-rate baselines
were calibrated against. A future minor bump could drag in robosuite
1.5+ and drift the numbers. Bump deliberately and re-record baselines
when LIBERO publishes 0.2.

LIBERO and RoboCasa cannot share a venv (PyPI robosuite 1.4 vs.
master). Use separate venvs per benchmark.

#### Two FT checkpoints — different camera-key schemas

`RLDX-1-FT-ROBOCASA` (Kitchen) and `RLDX-1-FT-RC365` ship from the **same
simulator** but with **different observation key configs**:

| Field              | `FT-ROBOCASA` ([`robocasa_config.py`](../../RLDX-1/rldx/configs/data/robocasa_config.py)) | `FT-RC365` ([`robocasa365_config.py`](../../RLDX-1/rldx/configs/data/robocasa365_config.py)) |
| ------------------ | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Camera keys        | `left_view`, `right_view`, `wrist_view`                                                   | `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`                      |
| State keys         | `eef_pos_rel`, `eef_rot_rel`, `gripper_qpos`, `base_position`, `base_rotation` (16-D)     | same                                                                                         |
| Action keys (12-D) | `eef_position(3) + eef_rotation(3) + gripper_close(1) + base_motion(4) + control_mode(1)` | same                                                                                         |
| Action chunk       | 16                                                                                        | 16                                                                                           |

**Lerobot's wrapper emits the v1.0+ names** (`robot0_agentview_*`,
`robot0_eye_in_hand`), so `FT-RC365` is drop-in. `FT-ROBOCASA` needs a
3-key camera rename inside our preprocessor:

```python
# library/src/physicalai/policies/rldx1/preprocessor.py — Kitchen-checkpoint adapter
RLDX_CAMERA_REMAP_KITCHEN = {
    "robot0_agentview_left": "left_view",
    "robot0_agentview_right": "right_view",
    "robot0_eye_in_hand":     "wrist_view",
}
```

Auto-detected from the checkpoint config: `FT-ROBOCASA`'s `modality_keys`
list under `video` contains `left_view`; `FT-RC365`'s contains
`robot0_agentview_left`. Branch on that string.

#### Why was `RLDX-1-FT-ROBOCASA` produced this way?

From [`finetune_rldx1_robocasa.sh`](../../RLDX-1/run_scripts/train/benchmarks/finetune_rldx1_robocasa.sh):

```
Base       : RLWRLD/RLDX-1-PT      (the 6.9 B video-pretrained foundation)
Dataset    : robocasa_mg_gr00t_300 = the "Robot Arm Kitchen Manipulation:
             72K trajectories" partition of nvidia/PhysicalAI-Robotics-
             GR00T-X-Embodiment-Sim, capped at ~300/task ≈ 7.2 K episodes.
             24 tasks under prefix `single_panda_gripper.*`.
Modality   : robocasa_config.py (cameras renamed left/right/wrist_view —
             that's where the schema delta above comes from).
Embodiment : GENERAL_EMBODIMENT (single MLP head, no per-robot specialisation)
Augment    : ColorJitter(brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08)
Optim      : 8 × H100, global bsz 512, grad_accum 2, max_steps 60 K
Add-ons    : NONE (no --use-memory, --use-motion, --use-physics)
             → vanilla MSAT, identical shape to FT-SIMPLER-WIDOWX.
```

RC365 differs only in: 16 GPU × 2 nodes, batch 192, 250 K steps, dataset
= `robocasa365/v1.0/pretrain`. The 365 dataset is not yet on a
single-link public HF mirror — defer until Kitchen lands.

#### Parallel π₀.₅ baseline — the commercial-friendly fallback

The RLDX checkpoints carry the non-commercial RLWRLD model license, so
they're research-only. We anchor the comparison with a **PAS-native
π₀.₅ checkpoint** trained on identical data:

1. **Data**: `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`
   ([CC-BY-4.0, HF dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)),
   subset `Robot Arm Kitchen Manipulation`
   (`single_panda_gripper.*` × 24 Kitchen tasks). Already LeRobot-v2.1
   with three cameras + the 12-D dict action.

   ```bash
   for task in PnPCounterToCab OpenSingleDoor CoffeePressButton ...; do
     huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
       --repo-type dataset --include "single_panda_gripper.${task}/**" \
       --local-dir $HOME/robocasa_kitchen
   done
   ```

2. **Volume**: 300 demos/task (~7.2 K total) matches the RLDX recipe.
   Full 3000/task is overkill for v1.

3. **Training config** (new file
   `library/src/physicalai/configs/datasets/robocasa_kitchen.py`):

   ```python
   RoboCasaKitchenDataConfig(
       repo_id="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
       task_filter=r"single_panda_gripper\..*",
       image_keys=["robot0_agentview_left", "robot0_agentview_right",
                   "robot0_eye_in_hand"],
       state_keys=[                              # 3 + 4 + 3 + 4 + 2 = 16-D
           "state.base_position", "state.base_rotation",
           "state.end_effector_position_relative",
           "state.end_effector_rotation_relative",
           "state.gripper_qpos",
       ],
       action_keys=[                             # 3 + 3 + 1 + 4 + 1 = 12-D
           "action.end_effector_position", "action.end_effector_rotation",
           "action.gripper_close", "action.base_motion", "action.control_mode",
       ],
       chunk_size=16,
   )
   ```

4. **Hyperparameters**: mirror the RLDX Kitchen recipe — AdamW lr 1e-4
   → cosine + 5 % warmup, ColorJitter(0.3/0.4/0.5/0.08), batch 512
   (8-GPU) or as fits, 60 K steps (π₀.₅ typically converges in 30 K —
   start there).

5. **Reference target**: paper Table 1b reports π₀.₅ baseline on
   RoboCasa Kitchen in the 50–60 % range vs. RLDX-1 at 70.6 %. If our
   pi05 lands in that range, the data + eval pipeline are validated;
   any deeper drift points at the wrapper or the pre/post-processor.

#### Why this is the right shape

- **Pipeline first, parity second.** RoboCasa Kitchen gives us a working
  end-to-end loop (data → train → sim eval) with public data and an
  off-the-shelf lerobot reference. Once it's green, swapping the policy
  (RLDX vs π₀.₅ vs ACT vs SmolVLA) becomes a one-line change.
- **No new runtime components for inference parity.** The lerobot port
  gives us obs in `Observation`-compatible shape; the existing pi05
  runner handles 16-step chunked action emission. No new manifest
  preprocessor / postprocessor / runner is required just to evaluate
  RLDX or π₀.₅ on RoboCasa.
- **RoboCasa365 falls out almost for free.** Same wrapper, same key
  schema — just pass `task="composite_seen"` (or any other group from
  [`_TASK_GROUP_SPLITS`](../../lerobot/src/lerobot/envs/robocasa.py#L61)). RLDX-1-FT-RC365 becomes a sanity check, not a
  separate integration.

#### What stays out of v1

- **RTC inference** on RoboCasa: `FT-ROBOCASA` was trained with
  `rtc_training_max_delay=0`, so RTC delivers no win on this
  checkpoint. Defer to phase 2 alongside the LIBERO RTC retrain.
- **Vec-env / parallel rollouts**: RLDX-1's eval shell script spawns 4
  GPU shards manually; for v1 we run sequentially and report wall-clock
  honestly. AsyncVectorEnv support is a phase-2 follow-up.
- **Mid-train data**: NVIDIA's `robocasa_mg_gr00t_300` was used to fine-
  tune `FT-ROBOCASA` on top of `RLDX-1-PT`, not for mid-training. Same
  story applies to our π₀.₅ baseline: PT → FT, no MT.

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

| Mode                                                                | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Gradient required?                                    | Exportable to OV/ONNX?                                                     |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------- |
| `none`                                                              | Standard Euler denoising; no inpainting.                                                                                                                                                                                                                                                                                                                                                                                                                                   | No                                                    | ✅ trivial.                                                                |
| `trained`                                                           | Hard-replace the prefix positions at each Euler step (`x_τ[:, :d] = Y[:, :d]`), feed per-token time with `τ=1` on the prefix. Requires a checkpoint trained with `training_max_delay > 0`.                                                                                                                                                                                                                                                                                 | **No**                                                | ✅ — just a `where(prefix_mask, Y, x_τ)` op + per-token time tensor.       |
| `guided` (RLDX-1, true VJP)                                         | Jacobian universal-guidance (arXiv 2506.07339 Eq. 2). Computes `v_guided = v + c · VJP[(Y − â¹_t)ᵀ diag(W), ∂â¹_t / ∂x_τ]`, where `â¹_t = x_τ + (1−τ)v`. **The VJP flows through the entire MSAT stack** because [`guided_velocity`](../../RLDX-1/rldx/model/modules/action_model/rtc.py#L230) sets `requires_grad_(True)` *before* calling `velocity_fn`.                                                                                                                 | **Yes — `torch.autograd.grad` through the full DiT**. | ❌ — needs autograd at runtime.                                            |
| `guided` (PI / LeRobot / our pi05, identity-Jacobian approximation) | Same formula on paper, but the [LeRobot port](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py) calls the velocity head **before** setting `requires_grad_(True)`. Since `v_t` is detached at that point, the autograd graph for `x1_t = x_t - time * v_t` reduces to `∂x1_t/∂x_t = I`. The `autograd.grad` call returns `grad_outputs` unchanged — i.e. `correction = (Y − â¹_t) ⊙ W` with no backward pass through the network. | **Effectively no.** The autograd block is a no-op.    | ✅ — once the no-op autograd wrapper is removed, this is pure forward ops. |

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

| Path                            | Behaviour                                                                                                                              | Cost vs paper                                                                                        | Notes                                                                   |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| A. Ship `none`                  | No prefix conditioning.                                                                                                                | Big — loses RTC entirely.                                                                            | Trivial.                                                                |
| B. Ship `trained` (recommended) | Hard inpaint + per-token τ at every Euler step. Matches arXiv 2512.05964 algorithm exactly.                                            | Small — paper's preferred mode for inference.                                                        | Requires retraining (or fine-tuning) with `rtc_training_max_delay > 0`. |
| C. Ship `guided-approx`         | Port `guided_velocity` but mirror the PI ordering (`v` without grad, drop the `autograd.grad` call, set `correction = (Y − â_t) ⊙ W`). | Unknown — identical to what pi05 already ships, so at least no worse than our current RTC behaviour. | Trivially exportable.                                                   |
| D. Ship `guided` (true VJP)     | Keep RLDX-1's ordering, run in eager torch with autograd.                                                                              | Zero (matches paper).                                                                                | **Not exportable to OV/ONNX.**                                          |

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

| RLDX-1 fused kernel                                                | OV equivalent                                          | Note                                                                       |
| ------------------------------------------------------------------ | ------------------------------------------------------ | -------------------------------------------------------------------------- |
| `fused_vision_attention` (RoPE+RoPE+SDPA)                          | OV `RoPE` + `ScaledDotProductAttention` fusion pass    | Picked up automatically when ONNX export emits the `Attention-V1` pattern. |
| `fused_llm_attention` (RMSNorm + RoPE + SDPA)                      | OV `RMSNorm` + `ScaledDotProductAttention` fusion      | Confirmed for Llama-class models — Qwen3-VL backbone benefits identically. |
| `fused_add2_layernorm`, `fused_add2_rmsnorm`, `fused_add3_rmsnorm` | OV `Add + LayerNorm` / `Add + RMSNorm` epilogue fusion | Standard transformations in `nGraph` post-processing.                      |
| `fused_memory_attention`                                           | Same as `fused_vision_attention`                       | Memory module is identical-shape attention.                                |
| `fused_mlp_swiglu`, `grouped_swiglu`                               | OV `Swish + Mul + MatMul` fusion / fused FFN pattern   | Recognized for Llama-style MLPs.                                           |

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

| Optimization                          | Used at training?       | Safe at export?                       | Action                                                                                                                                                                                                            |
| ------------------------------------- | ----------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Flash-Attention 2 (Qwen3-VL)          | ✅                      | ❌ (CUDA-only Triton)                 | Branch via `attn_implementation`: `flash_attention_2` during train, `sdpa` during export. Mirror Groot's approach ([`policies/groot/config.py:attn_implementation`](../src/physicalai/policies/groot/config.py)). |
| `torch.compile` on the loss step      | optional                | ❌ for graph export                   | Wrap training step in `if not self.exporting:` guard via Lightning's `compile_model` config field.                                                                                                                |
| Gradient checkpointing on Qwen3-VL    | ✅                      | n/a                                   | Disable in `model.eval()`; export always runs from `eval()`.                                                                                                                                                      |
| `torch.autograd.grad` in RTC `guided` | inference-time only     | ❌                                    | Reject at export, see §8.                                                                                                                                                                                         |
| Mixed precision `bf16`                | ✅                      | ⚠️ keep weights in fp16 / fp32 for OV | Convert to fp32 before OV export; OV will compress back to fp16 if `INFERENCE_PRECISION_HINT=f16`.                                                                                                                |
| Dropout (state / memory / physics)    | ✅                      | n/a                                   | `model.eval()` zeroes dropout.                                                                                                                                                                                    |
| PEFT LoRA on action model / backbone  | optional, training-time | ❌                                    | Call `peft_model.merge_and_unload()` before export. See Appendix D — paper shows LoRA recovers full-FT performance at half the params; we should support it via a `_merge_lora_before_export()` hook.             |

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

| RLDX-1 mechanism                                                    | Why it exists                                                            | Need it under OV/ONNX?                                                                                                                                                                                   |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Static Graph Conversion (RoPE caches + attention masks precomputed) | Eliminates configuration-dependent ops that fragment CUDA-Graph capture. | **No.** OV ingests an ONNX graph that is already static — RoPE buffers become weights / constants automatically. We replicate the same effect by exporting from `model.eval()` with fixed `max_seq_len`. |
| CUDA Graph (`torch.cuda.graph`) wrap of the full forward            | One launch per inference step on RTX 5090.                               | **No.** OV's `compile_model` builds a static execution plan that is functionally equivalent on Intel GPU / NPU / CPU.                                                                                    |
| `torch.compile(fullgraph=True)`                                     | Captures torch IR into a single subgraph before CUDA-Graph wrap.         | **No.** `torch.compile` is incompatible with `torch.onnx.export` mid-pipeline; we go directly from eager → ONNX → OV.                                                                                    |

### 5.2 Kernel Optimization

| RLDX-1 mechanism                                         | OV / ONNX replacement                                                                                            |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Hand-written Triton fused kernels (Appendix F, Table 7)  | OV's built-in fusion passes (see §9). Verified on Llama-family models — Qwen3-VL backbone matches that template. |
| Manual NVIDIA Nsight Compute profiling loop              | Replace with OV's `benchmark_app` + Studio's `InferenceLatencyBenchmark`.                                        |
| Short-prefill optimization (no autoregressive KV growth) | Already a natural fit for OV — short, fixed sequence lengths are OV's strength.                                  |

**Bottom line**: Section 5 is a documentation of *how RLWRLD got 1.63×
on Blackwell GPUs*. None of it has to be ported. The OV inference path
is structurally simpler and gets its speedup from OV's own graph
compiler.

---

## 12. CUDA Graph + Static Graph — what to keep for OV / ONNX

| Idea from RLDX-1                                                                | OV / ONNX analogue                                                                                  | Action                                                                                                                                                                          |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Convert configuration-dependent ops (RoPE freqs, attention masks) to constants. | OV folds these into `Constant` nodes when ONNX export sees `register_buffer(..., persistent=True)`. | Audit `Rldx1Model` for any `torch.arange(seq_len, ...)` inside `forward`. Hoist into `__init__` and `register_buffer`. **This is the one architectural change we should copy.** |
| Capture the whole forward as one graph.                                         | OV `compile_model` already does this.                                                               | Free.                                                                                                                                                                           |
| Pre-allocate I/O tensors and replay.                                            | OV's infer request buffer caching.                                                                  | Free.                                                                                                                                                                           |
| Kernel-level fusion.                                                            | OV's transformation passes.                                                                         | Free.                                                                                                                                                                           |

So the answer to *"can we just do CUDA Graph + Static Graph Conversion?"*
is: **the Static-Graph-Conversion *idea* is the one to copy** (factor
config-dependent ops out of the forward) — but we copy it as
*register_buffer hygiene*, not as a runtime mechanism. CUDA Graph itself
has no OV/ONNX analogue and does not need one.

---

## Validation checklist (in execution order)

**v1 — PT → FT only:**

1. ✅ License posture documented (this file + [rldx-1.md](rldx-1.md) verdict).
2. ⬜ Land `physicalai.gyms.robocasa.RoboCasaGym` + `RoboCasaBenchmark` (port
   from lerobot per §7.4); add `robocasa` extra to `library/pyproject.toml`
   pinned to a `robocasa` SHA + `robosuite` SHA (per lerobot Dockerfile).
3. ⬜ Reproduce paper RoboCasa Kitchen score (~70 %) inside RLDX-1 repo
   via the upstream eval shell to lock the baseline.
4. ⬜ Land `Rldx1` Studio package per §4 (v1 layout — no memory / motion / physics components).
5. ⬜ Load `RLWRLD/RLDX-1-PT` into `Rldx1` and verify per-tensor weight match
   (allow renames; reject shape mismatches).
6. ⬜ Smoke train: BC fine-tune on a 1 K-step LIBERO subset; loss curve sane.
7. ⬜ Smoke eval: `predict_action_chunk()` on a recorded SO-101 sample;
   shape `(1, 16, action_dim)`, no NaN.
8. ⬜ **RoboCasa Kitchen parity** (per §7.4): `Rldx1` +
   `RLDX-1-FT-ROBOCASA` through `RoboCasaBenchmark(task="paper_direct")`
   (the 19 ✅ paper tasks still 1:1-comparable in v1.0 — see §7.4
   paper-24 table); expect mean SR within 3 pts of the
   paper-equivalent reference (**66.3 %** over those 19 rows). Also
   run `task="paper_door_proxies"` and report door-family aggregates.
   Apply the `RLDX_CAMERA_REMAP_KITCHEN` adapter for the
   `left_view/right_view/wrist_view` schema. **Locks base-architecture
   integration.**
9. ⬜ **RoboCasa365 parity** (drop-in extension of step 8):
   `RLDX-1-FT-RC365` against the same wrapper with
   `task="composite_seen"` (no key remap needed). Expect ~30 % mean —
   paper target is 31.5 %.
10. ⬜ **LIBERO parity** (per §7.3 step 3): `RLDX-1-FT-LIBERO` through our
    LIBERO gym; expect paper Table 1 ± 3 pts. Also exercises the
    add-on-fields-explicitly-off schema variant in the loader.
11. ⬜ **π₀.₅ baseline on RoboCasa Kitchen** (per §7.4): train `Pi05`
    on `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`
    (`single_panda_gripper.*` × 300/task), 30–60 K steps. Eval through
    the same `RoboCasaBenchmark`. Target paper-π₀.₅ range (50–60 %).
    **Validates the commercial-friendly path end-to-end.**
12. ⬜ **Train-from-PT smoke** (per §7.3 step 5): 500-step fine-tune from
    `RLDX-1-PT` with action-LoRA r=64; loss decreases, checkpoint round-trips.
13. ⬜ Export `Rldx1` to OpenVINO via `ExportablePolicyMixin`
    (`merge_and_unload()` PEFT first per §5.3). Run on Intel iGPU + dGPU;
    measure latency.
14. ⬜ (Optional) **SimplerEnv WidowX triangulation** (per §7.3 step 2):
    `RLDX-1-FT-SIMPLER-WIDOWX` through a Studio SimplerEnv wrapper;
    expect mean ≥ 67 % across 4 tasks × 200 episodes. Embodiment-matched
    (single-arm gripper) sanity check. Only chase if RoboCasa shows
    unexplained drift.
15. ⬜ (Optional) **GR1 parity** (per §7.3 step 4): vendor
    `robocasa-gr1-tabletop-tasks`, run `RLDX-1-FT-GR1`, expect paper
    mean ± 3 pts.

**Phase 2 — MT / RTC / add-on streams (deferred):**

16. ⬜ Add `use_motion=True` path, train on small video dataset, validate
    parity with upstream within a fixed-seed run.
17. ⬜ Add `use_memory=True` + the multi-stride `delta_timestamps` config
    (reading from `memory_video_delta_indices`, not derived from
    `action_horizon`); validate the memory queue is filled correctly via
    a unit test on a synthetic episode.
18. ⬜ **ALLEX smoke test**: `RLDX-1-MT-ALLEX` loads into `Rldx1` with all
    three add-ons on; `predict_action_chunk` returns `(B, 40, D)`,
    no NaN; every checkpoint tensor binds; gradient flows through
    memory / motion / physics streams on a dummy loss. **Locks the
    add-on subsystems.**
19. ⬜ Add `new_param_warmup_steps` Lightning callback + retrain a small
    MT run on a public mixture (DROID slice + a small in-house substitute);
    loss curve sane.
20. ⬜ Add RTC training (`rtc_training_max_delay=4`) on LIBERO, verify
    loss decreases similarly to the non-RTC baseline (sanity check, not
    parity).
21. ⬜ Eval RTC-trained LIBERO checkpoint with `rtc_inference_mode ∈
{none, trained, guided-approx}`; confirm graceful degradation as
    `inference_delay` grows (paper Table 4 pattern).
22. ⬜ Repeat OV export with `use_memory=True` and the new manifest runner
    type (`action_chunking_with_memory`). Validate the session queue
    survives across `select_action()` calls.
23. ⬜ RECAP RL post-training (see §6 — punted).
24. ⬜ Physics stream training data (FR3 AnySkin or ALLEX

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
