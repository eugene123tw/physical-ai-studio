# Implementation Plan — Fast Plans, Faithful Actions (Block-AR + NGM) in Physical AI Studio

Source paper: [fast-plans-faithful-actions.md](fast-plans-faithful-actions.md)

## 0. Decisions

| Decision | Choice | Why |
|---|---|---|
| New policy vs. extend `Pi05` | **New policy `pi05_waypoint`** that reuses Pi0.5 internals by import/subclass | Different data contract (AWE waypoints), loss (CE + λ·FM), inference (plan loop → variable-length segments), and export (3 graphs). Keeps `Pi05` export/parity untouched. |
| Base model | `Pi05Model` subclass (`Pi05WaypointModel`) | Reuse `PaliGemmaWithExpertModel`, `embed_prefix`, `_make_att_2d_masks`, AdaRMS expert, pretrained loader, `PeftModelMixin`. |
| Ablations | Config enums, not separate classes | `planner_mode: token_ar \| block_ar`, `goal_conditioning: suffix \| naive \| ngm` → reproduces every row of Tables 2 and 4. |
| First target | LIBERO (union of 4 suites, 1,693 episodes) | Paper numbers exist; `benchmark/gyms/libero` already present. Real robot (bimanual) later. |
| Runtime | Torch-only inference first; OpenVINO export + Runtime runner in last phase | Validate the science before investing in 3-graph export. |

Existing hooks that make this cheap:

- Expert already uses AdaRMS: `PaliGemmaWithExpertModel(use_adarms=[False, True])` and `embed_suffix` returns `adarms_cond = time_emb`, which feeds every expert layer ([pi05/model.py](library/src/physicalai/policies/pi05/model.py)). **NGM = `adarms_cond = time_emb + g(t) * r`.**
- `_make_att_2d_masks` cumsum scheme = block-causal attention (same group id → bidirectional, later group → causal). Block-AR mask is just `ar_mask=1` at each block start.
- Time convention: `x_t = t·noise + (1−t)·actions` (t=1 is noise), so the paper's gate "open for t ≥ 0.5, closed for t ≤ 0.2" applies directly.
- Pi0.5 preprocessor already puts discretized state (256 bins) into the prompt: `"Task: ..., State: ...;\nAction: "` — waypoint tokens are appended after this.

## 1. Target layout

```
library/src/physicalai/
  data/waypoints/
    __init__.py
    awe.py              # AWE DP + forced gripper keyframes + max segment length D
    extract.py          # offline extraction over a LeRobot dataset → sidecar parquet + stats (σ_Δ)
    dataset.py          # WaypointDataset wrapper: planner targets + expert segment per sample
  policies/pi05_waypoint/
    __init__.py
    config.py           # Pi05WaypointConfig(Pi05Config-like, PeftConfigMixin)
    tokenizer.py        # WaypointTokenizer: 338 reserved ids, bins, slot families, encode/decode
    planner.py          # block/token-AR input construction, masks, learned query, constrained decoding
    ngm.py              # GoalModulation: Δ̂ scaling, gripper emb, zero-init MLP, phase gate, noise, null
    model.py            # Pi05WaypointModel(Pi05Model)
    preprocessor.py     # extends pi05 preprocessor with waypoint fields
    policy.py           # Pi05Waypoint(Policy): select_action state machine, Lightning hooks
    diagnostics.py      # S / B / U / end-of-plan metrics
library/configs/physicalai/pi05_waypoint/libero/
    token_ar_suffix.yaml  block_ar_suffix.yaml  token_ar_naive.yaml  block_ar_ngm.yaml
library/tests/unit/policies/pi05_waypoint/
    test_awe.py  test_tokenizer.py  test_planner_masks.py  test_ngm.py  test_parity.py  test_select_action.py
library/tests/unit/data/test_waypoint_dataset.py
tmp_scripts/fast_plans/          # viz + diagnostics drivers (matplotlib not a library dep)
```

Register in `policies/__init__.py` and `get_physicalai_policy_class()`.

---

## Phase 1 — Waypoint extraction & data (no model)

### Tasks

1. **`awe.py`** — AWE dynamic programming (Shi et al. 2023):
   - Input: per-episode config trajectory `x[0..T)` (LIBERO: 6-D EEF pos + axis-angle from `observation.state[:6]`, normalized per-dim), gripper binary `g[0..T)` (from action sign), threshold η (LIBERO 0.008, robot 0.015 rad).
   - Constraints: every gripper transition frame is a forced keyframe; every gap ∈ [1, D], D = 32; last frame is a keyframe.
   - Objective: minimum number of keyframes such that max piecewise-linear reconstruction error per segment ≤ η.
   - Output: keyframe indices.
2. **`extract.py`** — run over a LeRobot dataset (`physical-intelligence/libero`, verify 1,693 episodes), emit sidecar parquet per episode: `(frame_idx, q[6], g, d)`, plus terminal marker. Compute and save:
   - per-dim quantile ranges for 300-bin quantization of `q`,
   - `σ_Δ,j = max(1.4826·MAD_j, 0.25·std_j, 1e-3)` over normalized displacements `q_i − s_i`.
   Stats saved alongside dataset stats so they flow into checkpoints and export.
3. **`dataset.py`** — wraps `LeRobotDataModule` dataset. One item = one segment start `t` (with ±1 step jitter):
   - **Planner target:** next ≤ M = 7 waypoints from `t`, first duration recomputed from the jittered `t` (may become 33 → code 33), terminal block with `d = 0` if the episode ends within M.
   - **Expert segment:** `(s_t, q_i, g_i, d_i)` + action chunk `a[t : t+D]` with `action_is_pad` beyond episode end.
   - Share one observation between planner and expert (one prefix forward); see Phase 2 layout.

### Validation

- Unit: AWE error ≤ η on every segment; all gripper transitions present; all `d ∈ [1, 32]`; synthetic piecewise-linear trajectory recovers its exact corners.
- **Dataset statistics must match paper Table 3 / Fig. 3 (LIBERO-Long):** 13,815 segments, median 7, p95 13, max 29, 0% at cap. If off, tune η / state space before going further.

### Visualization

- `tmp_scripts/fast_plans/plot_waypoints.py`: per-episode curves of the 6 config dims with vertical waypoint cuts, duration blocks (width ∝ d), gripper rings, terminal block (paper Fig. 3a).
- Duration histograms per suite (Fig. 3b).
- Sanity overlay: reconstruct trajectory from waypoints (linear interp) on top of raw trajectory.

---

## Phase 2 — Token-AR + suffix baseline (Waypoint Pi0.5)

### Tokenizer (`tokenizer.py`)

- Reserve 338 ids at the tail of the PaliGemma vocab (verify unused range; skip the trailing special tokens as Pi0-FAST does): 300 value bins (shared by all config dims), 2 gripper, 34 duration codes (0 = end, 1..32, 33 = current-state/jitter), `<wp>`, `<dur>`.
- LIBERO block: `<wp> q1..q6 g <dur> d` = 10 tokens, 8 value slots. Robot: 19 tokens, 17 value slots.
- Slot families → logit masks. Decode clamps 33 → 32.
- Round-trip encode/decode test with quantization error ≤ half-bin.

### Model (`Pi05WaypointModel(Pi05Model)`)

Sequence layout (single backbone forward in training):

```
[ images | "Task..., State...; Action: " ] [ plan tokens ] [ expert suffix ]
          prefix (bidirectional)             causal (Token-AR)  expert attends to prefix ONLY
```

- **Planner loss:** CE on value slots via `paligemma.lm_head`, masked to slot family. For the terminal block, supervise only `d = 0`.
- **Expert suffix (suffix-conditioning route):** 3 condition tokens + D = 32 action tokens:
  - `state_proj(s_i)`, `goal_proj([q_i, e(g_i)])`, `dur_emb(d_i)`; att_masks `[1] + [0]*(3 + D − 1)`.
  - Expert must **not** attend to plan tokens (otherwise the label leaks through the plan tokens).
- **Loss:** `L = L_CE + λ·L_FM`, λ = 1.7, no stop-gradient between expert and backbone (unlike knowledge insulation).
- `chunk_size = D = 32`; FM loss over full D with `action_is_pad` masking (config switch `fm_loss_steps: full | executed` to test first-`d_i` only).
- Override `train_expert_only=False` (Pi0.5 default is `True`; the planner needs backbone gradients).
- LoRA rank 16 on backbone + vision + expert via `PeftModelMixin`. **Gotcha:** `inject_lora()` freezes all params — re-enable `requires_grad` on all new modules (projections, gripper emb, learned query, NGM MLP, null embeddings) after injection, or the zero-init NGM output stays zero forever. Add a test.

### Inference (`policy.py` state machine, B = 1 first)

```
select_action(obs):
  if segment_queue: return pop
  if plan is empty or (replan_mode == "receding" and segments_done >= 1):
      plan = planner.decode(obs)            # ≤ M blocks, stop at d=0
      if plan is empty: plan = [NULL_GOAL]  # model trained with null goal (Phase 4) — fallback
  wp = plan.pop(0)
  actions = expert.sample(obs, s=obs.state, wp)   # (1, D, A)
  segment_queue.extend(actions[:, :wp.d])
  return pop
```

- `replan_mode: full_plan` (LIBERO protocol) | `receding` (real robot).
- `predict_action_chunk` returns the next segment `(B, D, A)` to keep the base contract; `select_action` is overridden.
- Token-AR decoding: prefill + one pass per value slot with KV cache; structural tokens are fed together with the next pass. Expose `last_plan_num_passes` for tests/telemetry.

### Validation

- **Parity test:** with waypoint tokens absent and condition tokens removed, `Pi05WaypointModel` expert output equals `Pi05Model` output for the same weights/noise.
- Mask test: expert rows have zero attention to plan columns.
- Overfit one batch with `gemma_300m` variants: CE → ~0, FM decreasing.
- Pass count: `num_passes == 1 + 8·K` for a K-block plan (57 at K = 7).
- **LIBERO:** target ≈ Pi0.5 (paper: 96.70 avg; LIBERO-Long 92.2). Initially run LIBERO-Long only, N = 500 (10 tasks × 50 init states). Note `configs/benchmark/libero.yaml` defaults to 20 episodes — override to 50.

---

## Phase 3 — Block-AR planner

### Tasks (`planner.py`)

- **Training input:** shift by one block. Input block k = embedded tokens of waypoint k−1; input block 0 = learned query (structural template embeddings + zero-init learnable residual, shape `(L_block, width)`). Targets are **not** shifted: block k position predicts waypoint k's slots.
- **Mask:** within-block bidirectional, across-block causal → set `att_masks` to 1 at each block start, 0 inside.
- **Readout:** gather hidden states at q / g / d slot positions → 3 `lm_head` calls with family masks.
- **Decoding:** prefill + one pass per waypoint; stop on `d = 0` or M blocks. Greedy argmax per slot.

### Validation

- Mask property tests (block-diagonal + lower block-triangular).
- `num_passes == 1 + K` (≤ 8).
- Teacher-forced Block-AR on a 1-block plan == Token-AR plan when both are overfit on the same sample (sanity).
- End-of-plan recall/precision on held-out windows (paper robot: Token-AR 0.00 recall, Block-AR 0.61 / 0.85).
- **LIBERO:** within ~1.4 pts of Token-AR per suite (paper: 95.85 avg, Long 91.0).
- Latency: planner ms/plan on XPU (torch) for Token-AR vs Block-AR; expect ~7× reduction in passes.

---

## Phase 4 — Normalized Goal Modulation (NGM) + faithfulness diagnostics

### Tasks (`ngm.py`)

```
Δ̂   = (q_i − s_i) / σ_Δ                      # normalized space, σ_Δ from Phase 1
z   = concat(Δ̂, e(g_i))
r   = MLP(z)                                  # last Linear zero-initialized
g(t)= clamp((t − 0.2) / 0.3, 0, 1)            # closed t ≤ 0.2, open t ≥ 0.5 (linear between: assumption)
adarms_cond = time_emb + g(t) · r
```

- Training-only regularizers applied to **both** routes (suffix goal token and NGM input) with the same sample:
  - condition noise: `Δ̂ ← Δ̂ + N(0, 0.7²)`; suffix `q̃ = s + σ_Δ·Δ̂_noisy`.
  - null dropout p = 0.15: replace suffix goal embedding and NGM input with learned null embeddings.
- Inference: clean goal, gate on, no CFG.
- `goal_conditioning=naive`: gate ≡ 1, no noise, no dropout (reproduces shortcut failure).

### Diagnostics (`diagnostics.py`)

Frozen set: 64 LIBERO-Long segments (held-out), fixed flow noise. **Gotcha:** Pi0.5 `sample_noise` returns zeros unless `use_random_input_noise=True` — must enable for B and all interventions, with a fixed seed shared across intervention pairs.

| Metric | Definition |
|---|---|
| B | RMS action change between two noise draws (executed steps, valid dims), median over segments |
| S_endpoint | median RMS change with `q_i := s_i`, divided by B |
| S_duration | `d_i + 4` (clamp 32), divided by B |
| S_image | absolute RMS change when main camera is swapped with another segment's |
| Retention | S_image / S_image(suffix baseline) |
| U | SR(intact) − SR(q̃_i := s_i) on the same 500 init states — via a `plan_transform` hook on the policy |

### Validation

- **Zero-init parity:** fresh NGM model output == Phase 3 model output (r = 0).
- Gate unit test; noise/dropout applied identically to both routes (test by capturing both inputs).
- Trainability test: after LoRA injection, NGM params have `requires_grad=True` and non-zero grad after one step.
- **Paper Table 4 targets (LIBERO-Long):**

| Variant | S_endpoint | Retention | SR | U |
|---|---|---|---|---|
| Block-AR + suffix | ~0.6% | 100% | ~91 | ~0 |
| Token-AR + naive | ~348% | ~27% | ~15 | +14 |
| Block-AR + NGM | ~42.6% | ~82% | ~96.2 | +7.4 |

  The naive row failing is a required result — it validates the diagnostics.
- Then run all 4 suites for Block-AR + NGM (paper: 98.45 avg).

### Visualization

- Bar charts: S_endpoint / S_duration / S_image / retention per variant; U table.
- Per-segment action overlays: intact vs endpoint-erased vs noise-resampled (one subplot per action dim).

---

## Phase 5 — Rollout visualization & telemetry

- **Plan overlay video:** hook into the benchmark video recorder; project each waypoint's EEF xyz into the LIBERO agentview camera (MuJoCo camera intrinsics/extrinsics), draw polyline + points colored by gripper, highlight current target, annotate `d_i` and "END" block.
- **Telemetry:** policy emits events (`plan_start/end`, `expert_start/end`, `num_passes`) → timeline plot (paper Fig. 4a) and duty cycle.
- **Attention mask heatmap** for Token-AR vs Block-AR (debug aid, also good for docs).
- Optional later: Studio UI overlay of live plan during inference.

---

## Phase 6 — Export & Runtime

### Studio export (3 graphs)

1. `planner_prefill`: images + prompt tokens → prefix KV cache + masks.
2. `planner_block_step`: KV cache + input block ids (or learned-query flag) → slot logits + updated KV (static block length 10 for LIBERO).
3. `expert`: prefix KV + `s, q, g, d` (+ noise) → actions `(1, D, A)` (denoise loop unrolled as in Pi0.5 export).

Manifest adds `σ_Δ`, bin ranges, tokenizer id map, `M`, `D`, `replan_mode`.

### Runtime (`physicalai` repo — ship first)

- `runners/waypoint_hierarchical.py`: plan loop + per-segment expert calls + variable-length segments.
- `postprocessors/waypoint_detokenizer.py`: slot logits → `(q, g, d)` with family masks and clamp.
- Check that `runtime/action_sources` handles variable-length chunks.

### Validation

- Numerical parity torch vs OV per graph (logits argmax identical; expert actions within tolerance).
- `InferenceModel.load(...)` → `select_action` on CPU; LIBERO-Long via `InferenceModel` matches torch SR within noise.
- `InferenceLatencyBenchmark` CPU / Intel GPU: passes/plan, ms/plan, ms/segment.

---

## Open questions (paper underspecified)

1. Gate shape between t = 0.2 and 0.5 (assume linear).
2. Exact role of duration code 33 "current-state input" in the planner input sequence.
3. Learned query construction for the first Block-AR block (assume template embeddings + zero-init residual).
4. FM loss over full D vs first `d_i` steps.
5. Planner windows vs expert segments: paper samples 80 + 80 separately; plan shares one prefix per sample. Switch to separate windows if SR lags.
6. Training steps / LR schedule not given — start from the Pi0.5 LIBERO fine-tune recipe.
7. Terminal block `q, g` targets — assume unsupervised.
8. No reference code released — all numbers above are the acceptance targets.

## Milestone checklist

- [ ] P1: AWE + dataset; LIBERO-Long stats match Table 3; waypoint plots.
- [ ] P2: Token-AR + suffix; parity + overfit tests; LIBERO-Long ≈ 92.
- [ ] P3: Block-AR; ≤ 8 passes; LIBERO-Long ≈ 91; end-of-plan metrics.
- [ ] P4: NGM + diagnostics; Table 4 reproduced (incl. naive failure); 4-suite ≈ 98.45.
- [ ] P5: Plan-overlay videos, latency timeline.
- [ ] P6: OV export + Runtime runner; parity + latency on Intel CPU/GPU.
