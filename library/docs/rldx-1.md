# RLDX-1

Upstream: `/home/yuchunli/git/RLDX-1` — RLWRLD/RLDX-1. Paper: [arXiv 2605.03269](https://arxiv.org/abs/2605.03269) (markdown copy at [../papers/rldx-1.md](../papers/rldx-1.md)).

## Architecture
- Backbone: **Qwen3-VL-8B-Instruct** (frozen by default; `tune_top_llm_layers`, `backbone_use_lora` optional).
- Action head: **Multi-Stream Action Transformer (MSAT)** — MM-DiT extension for action modeling. Three streams (cognition / state-action / physics) coupled by joint self-attention. 4 multi-stream + 8 single-stream blocks, 24 heads, hidden 1024, action horizon 16, flow-matching (Beta(1.5, 1.0) noise, 4 inference steps).
- Optional add-ons (all flow-matching conditioned through MSAT):
  - **Motion module** (`use_motion`) — multi-frame video + cosine-correlation motion encoder injected at VLM layer 9; defaults `video_length=4`, `video_stride=2`.
  - **Memory module** (`use_memory`) — 2-layer causal transformer over a queue of past cognition tokens (`memory_length=4`, snapshots at `memory_stride=16` steps); supports concat or replace fusion.
  - **Physics stream** (`use_physics`) — tactile + torque tokens (e.g. `[30, 7]`); decoder co-predicts future physical signals (`physics_loss_weight=0.1`).
- Real-Time Chunking (RTC) integrated: training-time `rtc_training_max_delay`, inference modes `none`/`guided`/`trained` (latter pairs with `--compile fullgraph`).
- Sizes: **PT 6.9B (video pre-training), MT/FT 8.1B (all add-ons)**.

## License
- **Code: Apache-2.0** (built on **NVIDIA Isaac-GR00T N1.7** Apache codebase; per-file SPDX headers preserved).
- **Weights: RLWRLD Model License v1.0 — NON-COMMERCIAL only**, attribution + share-alike, use restrictions (no military / surveillance). Applies to every `RLWRLD/RLDX-1-*` checkpoint.
- Repo policy: **no external PRs accepted**.

## Weights
HF org `RLWRLD/`:
- `RLDX-1-PT` (6.9B, video pre-trained)
- `RLDX-1-MT-DROID`, `RLDX-1-MT-ALLEX` (8.1B mid-trained with memory + motion + physics)
- `RLDX-1-FT-{LIBERO, SIMPLER-GOOGLE, SIMPLER-WIDOWX, GR1, ROBOCASA, RC365}` (benchmark-specific fine-tunes)

All checkpoints carry the non-commercial license above.

## Fine-tuning
```bash
uv run python rldx/experiment/launch_train.py \
    --base-model-path RLWRLD/RLDX-1-PT \
    --dataset-path /path/to/lerobot_v2.1_dataset \
    --embodiment-tag GENERAL_EMBODIMENT \
    --video-length 4 --use-memory --use-motion \
    --use-physics --physics-keys tactile torque --physics-dims 30 7 \
    --n-cog-tokens 64 --global-batch-size 64 --max-steps 60000
```
DeepSpeed Zero-2/3 configs shipped. PEFT LoRA on both action model (MSAT) and Qwen3 backbone supported. Three-stage recipe: pre-train → mid-train (functional add-ons) → post-train (task adaptation, optional RECAP RL).

**Stage availability in this release** (paper says 3 stages; only the BC ones ship):

| Stage | Shipped? | Where / why not |
|---|---|---|
| Pre-train | ❌ | No script under [run_scripts/train/](../../RLDX-1/run_scripts/train); the multi-embodiment OXE/AGIBOT/HUMANOID mix is not redistributed. Public corpora consumable via `RLWRLD/RLDX-1-PT` weights only. Paper cost: ~195h × 64 H200s. |
| Mid-train DROID (BC) | ✅ | [midtrain_rldx1_droid.sh](../../RLDX-1/run_scripts/train/examples/midtrain_rldx1_droid.sh), runs on public DROID + in-house split. |
| Mid-train ALLEX (BC) | ⚠️ script only | [midtrain_rldx1_allex.sh](../../RLDX-1/run_scripts/train/examples/midtrain_rldx1_allex.sh) — but `real_allex` + `robocurate_*` mix in [dataset_mix.py:27-49](../../RLDX-1/rldx/configs/data/dataset_mix.py) is not published. Data-blocked. |
| Post-train, BC fine-tune | ✅ | Six benchmark scripts in [run_scripts/train/benchmarks/](../../RLDX-1/run_scripts/train/benchmarks) reproduce each `RLDX-1-FT-*`; plus [examples/finetune.sh](../../RLDX-1/run_scripts/train/examples/finetune.sh) template. **This is the only end-to-end reproducible path.** |
| Post-train, RECAP RL | ❌ | Single dead flag `add_rl_callback: bool = False` at [training_config.py:116](../../RLDX-1/rldx/configs/training/training_config.py#L116) — never read. No critic, rollout loop, or advantage-conditioned trainer. Text-prediction critic from §4.3 / App. C is paper-only. |

## Cross-embodiment / generalization
- `EmbodimentTag` enum directly inherited from **GR00T N1.7** (`OXE_*`, `AGIBOT_*`, `GR1`, `LIBERO_PANDA`, `ROBOCASA_PANDA_OMRON`, `NEW_EMBODIMENT`, etc.). Same per-embodiment MLP-head pattern.
- Pre-train mix spans single-arm + dual-arm + humanoid public datasets; mid-train adds in-house ALLEX humanoid + sensor-augmented Franka FR3 demos + Cosmos-Predict2/world-model synthetic trajectories filtered by motion-consistency.
- Zero-shot held-out-embodiment claims: **none** — benchmark numbers (LIBERO 97.8 / LIBERO-Plus 86.7 / SIMPLER-Google-VM 81.5 / WidowX 71.9 / RoboCasa 70.6 / GR-1 Tabletop 58.7 / RoboCasa365 32.1) all use per-benchmark `FT` checkpoints.

## Long-horizon
- 16-step flow-matching chunks (action horizon), no explicit hierarchical planner.
- Long-term context handled inside the model via the optional **memory module** (snapshots-of-past-cognition queue) rather than via planner decomposition.

## Custom CUDA / kernel surface
- **Flash-Attention 2 (2.7.4.post1)** pinned as a required runtime dep for training.
- Heavy **custom Triton kernel chain** under `rldx/inference/`:
  - `op_fused_attention.py` — RMSNorm + RoPE (Triton) + F.sdpa
  - `op_fused_attention_3way.py` — 3-stream variant for ExpandedSingleStreamBlock (V-L / state-action / physics)
  - `op_fused_mlp_swiglu.py` — GEMM + bias + SwiGLU
  - `custom_msat_chain.py`, `custom_expanded_msat_chain.py`, `custom_backbone_chain.py` — `torch.compile(fullgraph)` + CUDA-Graph capture chains, tuned for **RTX 5090 / sm_120 (Blackwell)**.
- Inference server (`run_rldx_server.py`) is the primary deployment surface; ZeroMQ wire protocol, `--compile {none, submodule, fullgraph}` × `--rtc-inference-mode {none, guided, trained}` knobs.

## OpenVINO blockers
- License: weights are **non-commercial**, fails the integration filter outright.
- Even setting that aside: full custom Triton chain is the deployment story, not a fallback — porting it loses the 1.63× speedup that motivates the architecture in the first place.
- Qwen3-VL-8B + 3-stream MM-DiT export work would land via the InternVLA-M1 / Xiaomi-Robotics-0 / ABot-M0 pipelines first.

## Synthetic-data pipeline (paper §3.3, App. B.3 — **not in the repo**)

The repo only ships the **policy**; the synthetic-data flow that produces the `robocurate_*` mid-train datasets referenced in [`rldx/configs/data/dataset_mix.py`](../../RLDX-1/rldx/configs/data/dataset_mix.py) is **entirely external**. Documented here as a reference design to potentially re-implement.

### Pipeline

1. **Scene I2I editing of initial frame** — `FLUX.2-dev` + Canny-edge conditioning, varies {table appearance, target-object identity/appearance, lighting, background}. Canny preserves scene structure so the I2V model still has a plausible starting state.
2. **Task augmentation** — VLM (Qwen3-VL 8B / 30B-A3B) re-captions / generates novel task instructions per source video; short / medium / long variants randomly sampled.
3. **Image-to-Video generation** — per-embodiment Cosmos checkpoint:
   - **GR-1**: LoRA rank-32 fine-tune of `Cosmos-Predict2-14B` on 3027 ActionNet + 92 NVIDIA-GR-1 videos, 93-frame clips @ 432×768, 10K steps, batch 4.
   - **ALLEX**: full fine-tune of `Cosmos-Predict2.5-2B` on ALLEX + OpenArm + ActionNet (2:1:1), 93-frame @ 432×768, 20K steps, batch 8. Uniform + random temporal sampling during training.
4. **(Optional) V2V transfer** — `Cosmos-Transfer2.5-2B` + Canny, diversifies appearance while preserving motion → keeps the IDM-labelled actions valid.
5. **Action labelling via IDM** — 0.1B Diffusion-Transformer with SigLIP-2 vision encoder, flow-matching objective, predicts the action sequence between two observations. GR-1 uses public ckpt [`seonghyeonye/IDM_gr1`](https://huggingface.co/seonghyeonye/IDM_gr1); ALLEX IDM trained from in-house teleop, horizon `H+1=20`, batch 256, 60K steps.
6. **Filter A — video quality (Gemini API)** — two scores per clip: 16 uniformly sampled frames for *instruction following*, 8 frames for *trajectory plausibility*. Drops visually implausible or off-instruction generations before action labelling matters.
7. **Filter B — motion-consistency probe** (the key "does the IDM trajectory make sense" check):
   - Replay IDM-predicted actions in the matching simulator (vendored under `external_dependencies/` — GR-1 Tabletop / RoboCasa / LIBERO / BEHAVIOR / CALVIN sims are present); render a rollout video.
   - Feed `(rollout video, generated video)` into an **attentive probe**: frozen **V-JEPA2** encoder → single cross-attention layer with one learnable query token attending to the concatenated embeddings → linear → alignment logit.
   - Probe trained with **BCE on positive / negative pairs from real demos** (positive = matched real-video + real-actions replayed; negative = shuffled / mismatched). Training detail: 16-frame clips @ 256×256, temporal stride 4, AdamW lr `1e-4`, batch 32.
   - Verification is **visual motion matching**, NOT task success — no per-prompt reward function is needed, which is why this scales to arbitrary new tasks. If the IDM hallucinated the action, the deterministic sim replay won't visually match the generated video and the probe rejects.

### What's in this repo vs. external

| Stage | In `RLDX-1/`? |
|---|---|
| I2I (FLUX.2-dev) | ❌ |
| I2V (Cosmos-Predict2 / 2.5 fine-tune) | ❌ |
| V2V (Cosmos-Transfer2.5) | ❌ |
| IDM training + inference | ❌ (GR-1 ckpt linked above; ALLEX ckpt unreleased) |
| Gemini video-quality scorer | ❌ (external API) |
| V-JEPA2 motion-consistency probe | ❌ (cited as `kim2026robocurate`, unreleased) |
| Simulators used for IDM replay | ✅ vendored in `external_dependencies/` but only wired into `rldx/eval/` for policy eval |
| `robocurate_*` datasets (mix entries) | ✅ referenced in [`dataset_mix.py`](../../RLDX-1/rldx/configs/data/dataset_mix.py), but the LeRobot v2.1 artefacts themselves are not published |

### If we re-implement
- Closest open-sourced precedent: **DreamGen** (`jang2025dreamgen`, NVIDIA). RLDX-1 paper explicitly says "Following jang2025dreamgen, kim2026robocurate" for both the IDM and the filter.
- Upstream artefacts available now: `nvidia/Cosmos-Predict2(.5)`, `nvidia/Cosmos-Transfer2.5`, `seonghyeonye/IDM_gr1`, V-JEPA2 weights, FLUX.2-dev (license-check needed).
- Cheapest first cut: skip the V-JEPA2 probe and lean on (a) Gemini judging for video-quality and (b) per-task sim-success scoring for the subset of tasks where reward is cheap (GR-1 Tabletop, RoboCasa). The V-JEPA2 probe is the contribution that lets this scale to *arbitrary* prompts with no reward function — pick it up second.
- Risk: the pipeline is data-engineering-heavy and end-to-end gains in the paper are modest (`+9.1%` on GR-1 Tabletop over real-data-only); evaluate on one task before scaling.

## Reproduction notes

### SimplerEnv WidowX — `RLWRLD/RLDX-1-FT-SIMPLER-WIDOWX`
Local run via [run_scripts/eval/simpler/eval_simpler.sh widowx](../../RLDX-1/run_scripts/eval/simpler/eval_simpler.sh), 200 episodes/task. Paper per-task numbers from Table 12 in the appendix:

| Task | Local | Paper | Δ |
|---|---|---|---|
| widowx_spoon_on_towel | 88.0% | 88.5% | −0.5 |
| widowx_carrot_on_plate | 83.0% | 83.0% | 0.0 |
| widowx_stack_cube | 61.0% | 64.0% | −3.0 |
| widowx_put_eggplant_in_basket | 46.0% | 52.0% | −6.0 |
| **Mean** | **69.5%** | **71.9%** | **−2.4** |

Within ~2.4 pts of paper. Gap is concentrated on `put_eggplant_in_basket` (the dragging task); the other three tasks reproduce within 3 pts.

### SimplerEnv Google VM — `RLWRLD/RLDX-1-FT-SIMPLER-GOOGLE`
Local run via [run_scripts/eval/simpler/eval_simpler.sh google](../../RLDX-1/run_scripts/eval/simpler/eval_simpler.sh), 200 episodes/task. Paper per-task numbers from Table 12 in the appendix:

| Task | Local (succ/200) | Local | Paper | Δ |
|---|---|---|---|---|
| google_robot_pick_coke_can | 185 | 92.5% | 97.0% | −4.5 |
| google_robot_move_near | 173 | 86.5% | 92.0% | −5.5 |
| google_robot_close_drawer | 150 | 75.0% | 78.5% | −3.5 |
| google_robot_open_drawer | 102 | 51.0% | 58.5% | −7.5 |
| **Overall** | **610/800** | **76.25%** | **81.5%** | **−5.25** |

Local reproduction is ~5.25 pts below paper. Gap is spread roughly evenly across all four tasks (−3.5 to −7.5), not concentrated on any single one. `open_drawer` is the hardest task in both runs (lowest absolute rate) and also shows the largest gap, but no task is anomalous.

## Verdict
**❌ Skip** — same disqualifier as InternVLA-A1: **non-commercial model license**, no external contributions accepted, custom-Triton-first inference path. Technically the most interesting MM-DiT-for-action implementation in the catalog (clean three-stream design + memory + motion + physics integrated in one network, GR00T N1.7-style embodiment tags, mature RTC) — keep as an architectural reference for any future MSAT-style work in `physical-ai-studio`, but do not invest integration effort while the weights stay non-commercial.

**Worth revisiting separately**: the synthetic-data pipeline above is architecture-agnostic and could feed *any* Studio policy. Track as its own follow-up rather than as part of an RLDX-1 port.
