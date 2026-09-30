# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Pi0.5 waypoint hierarchy (tokenizer, NGM, planner/expert model, policy loop).

Model tests shrink the Gemma backbone/expert and stub the SigLIP tower so a forward pass runs
on CPU in seconds.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from physicalai.data import Observation
from physicalai.data.constants import IMAGE_MASKS, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.data.observation import ACTION, IMAGES, STATE
from physicalai.policies.pi05_waypoint import Pi05Waypoint, Pi05WaypointConfig, Pi05WaypointModel
from physicalai.policies.pi05_waypoint.model import (
    KEY_SEG_D,
    KEY_SEG_G,
    KEY_SEG_Q,
    KEY_WP_D,
    KEY_WP_G,
    KEY_WP_Q,
    KEY_WP_VALID,
)
from physicalai.policies.pi05_waypoint.ngm import GoalModulation, phase_gate
from physicalai.policies.pi05_waypoint.tokenizer import SlotKind, WaypointTokenizer

NUM_IMG_TOKENS = 4
STATE_DIM = 8
ACTION_DIM = 7
M = 3
D = 8


def _stats() -> dict:
    return {
        "observation.state": {
            "name": "state",
            "shape": (STATE_DIM,),
            "mean": [0.0] * STATE_DIM,
            "std": [1.0] * STATE_DIM,
            "q01": [-1.0] * STATE_DIM,
            "q99": [1.0] * STATE_DIM,
        },
        "action": {
            "name": "action",
            "shape": (ACTION_DIM,),
            "mean": [0.0] * ACTION_DIM,
            "std": [1.0] * ACTION_DIM,
            "q01": [-1.0] * ACTION_DIM,
            "q99": [1.0] * ACTION_DIM,
        },
        "waypoint": {
            "name": "waypoint",
            "sigma_delta": [0.1] * 6,
            "config_dims": [0, 1, 2, 3, 4, 5],
            "num_arms": 1,
            "max_gap": D,
            "normalization_mode": "QUANTILES",
        },
    }


@pytest.fixture
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiny Gemma (8 heads are hardcoded in the joint attention) and a stubbed vision tower."""
    import physicalai.policies.pi05.model as pi05_model_mod

    tiny = pi05_model_mod.GemmaVariantConfig(width=64, depth=2, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=8)
    monkeypatch.setattr(pi05_model_mod, "get_gemma_config", lambda variant: tiny)  # noqa: ARG005

    orig_getitem = pi05_model_mod.CONFIG_MAPPING.__getitem__

    def _tiny_paligemma() -> object:
        cfg = orig_getitem("paligemma")()
        cfg.vision_config.hidden_size = 32
        cfg.vision_config.intermediate_size = 64
        cfg.vision_config.num_hidden_layers = 1
        cfg.vision_config.num_attention_heads = 1
        return cfg

    class _TinyMapping(dict):
        def __getitem__(self, key: str) -> object:
            return _tiny_paligemma if key == "paligemma" else orig_getitem(key)

    monkeypatch.setattr(pi05_model_mod, "CONFIG_MAPPING", _TinyMapping())

    def _embed_image(self, image: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        width = self.paligemma.config.text_config.hidden_size
        return image.flatten(1)[:, : NUM_IMG_TOKENS * width].reshape(image.shape[0], NUM_IMG_TOKENS, width)

    monkeypatch.setattr(pi05_model_mod.PaliGemmaWithExpertModel, "embed_image", _embed_image)


def _model(planner_mode: str = "block_ar", goal_conditioning: str = "ngm", seed: int = 0) -> Pi05WaypointModel:
    torch.manual_seed(seed)
    return Pi05WaypointModel(
        _stats(),
        planner_mode=planner_mode,  # type: ignore[arg-type]
        goal_conditioning=goal_conditioning,  # type: ignore[arg-type]
        max_blocks=M,
        num_bins=300,
        paligemma_variant="gemma_300m",
        action_expert_variant="gemma_300m",
        dtype="float32",
        chunk_size=D,
        num_inference_steps=2,
        use_random_input_noise=True,
    )


def _batch(bsize: int = 2, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    seq = 6
    wp_d = torch.tensor([[3, 5, 0], [2, 4, 6]])[:bsize]
    return {
        IMAGES: torch.rand(1, bsize, 3, 16, 16, generator=g),
        IMAGE_MASKS: torch.ones(1, bsize, dtype=torch.bool),
        TOKENIZED_PROMPT: torch.randint(0, 1000, (bsize, seq), generator=g),
        TOKENIZED_PROMPT_MASK: torch.tensor([[True] * seq, [True] * (seq - 2) + [False] * 2])[:bsize],
        STATE: torch.rand(bsize, STATE_DIM, generator=g) * 2 - 1,
        ACTION: torch.randn(bsize, D, 32, generator=g),
        KEY_WP_Q: torch.rand(bsize, M, 6, generator=g) * 2 - 1,
        KEY_WP_G: torch.randint(0, 2, (bsize, M, 1), generator=g),
        KEY_WP_D: wp_d,
        KEY_WP_VALID: torch.ones(bsize, M, dtype=torch.bool),
        KEY_SEG_Q: torch.rand(bsize, 6, generator=g) * 2 - 1,
        KEY_SEG_G: torch.randint(0, 2, (bsize, 1), generator=g),
        KEY_SEG_D: wp_d[:, 0],
    }


class TestTokenizer:
    def test_reserved_layout_matches_paper(self) -> None:
        tok = WaypointTokenizer()
        assert tok.num_reserved == 338
        assert tok.block_len == 10
        assert tok.num_values == 8
        assert tok.base_id + tok.num_reserved == 257_152 - 128
        assert tok.dur_id == tok.base_id + tok.num_reserved - 1

    def test_bimanual_block(self) -> None:
        tok = WaypointTokenizer(num_cfg=14, num_arms=2)
        assert tok.block_len == 19

    def test_roundtrip(self) -> None:
        tok = WaypointTokenizer()
        q = torch.rand(4, 3, 6) * 2 - 1
        g = torch.randint(0, 2, (4, 3, 1))
        d = torch.randint(0, 33, (4, 3))
        ids = tok.encode(q, g, d)
        assert ids.shape == (4, 3, tok.block_len)
        assert (ids[..., 0] == tok.wp_id).all()
        assert (ids[..., -2] == tok.dur_id).all()
        q2, g2, d2 = tok.values_from_reserved(tok.reserved_targets(q, g, d))
        assert torch.allclose(q2, q, atol=1.0 / tok.num_bins + 1e-6)
        assert torch.equal(g2, g)
        assert torch.equal(d2, d)

    def test_state_code_clamped_on_decode(self) -> None:
        tok = WaypointTokenizer()
        target = tok.reserved_targets(torch.zeros(1, 6), torch.zeros(1, 1, dtype=torch.long), torch.tensor([33]))
        assert int(tok.values_from_reserved(target)[2]) == 32

    def test_family_mask(self) -> None:
        tok = WaypointTokenizer()
        mask = tok.value_family_mask()
        assert mask.shape == (8, 338)
        assert mask[:6].sum(-1).eq(300).all()
        assert int(mask[6].sum()) == 2
        assert int(mask[7].sum()) == 34
        ranges = tok.family_ranges()
        assert ranges[SlotKind.D] == (302, 336)


class TestNGM:
    def test_gate(self) -> None:
        t = torch.tensor([0.0, 0.2, 0.35, 0.5, 1.0])
        assert torch.allclose(phase_gate(t), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))

    def test_zero_init_and_null(self) -> None:
        ngm = GoalModulation(6, 1, 32)
        out = ngm(torch.randn(3, 6), torch.tensor([[0], [1], [1]]), torch.ones(3))
        assert torch.equal(out, torch.zeros_like(out))
        nn_init = torch.nn.init
        nn_init.normal_(ngm.mlp_out.weight)
        drop = torch.tensor([True, True, False])
        out = ngm(torch.randn(3, 6), torch.tensor([[0], [1], [1]]), torch.ones(3), drop)
        assert torch.allclose(out[0], out[1])
        closed = ngm(torch.randn(3, 6), torch.zeros(3, 1, dtype=torch.long), torch.full((3,), 0.1))
        assert torch.equal(closed, torch.zeros_like(closed))

    def test_ungated_ignores_time(self) -> None:
        ngm = GoalModulation(6, 1, 32, gated=False)
        torch.nn.init.normal_(ngm.mlp_out.weight)
        x, g = torch.randn(2, 6), torch.zeros(2, 1, dtype=torch.long)
        assert torch.allclose(ngm(x, g, torch.zeros(2)), ngm(x, g, torch.ones(2)))


class TestConfig:
    def test_defaults(self) -> None:
        cfg = Pi05WaypointConfig()
        assert cfg.chunk_size == 32
        assert (cfg.planner_mode, cfg.goal_conditioning) == ("block_ar", "ngm")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"planner_mode": "x"},
            {"goal_conditioning": "x"},
            {"max_blocks": 0},
            {"gate_lo": 0.6, "gate_hi": 0.5},
            {"train_expert_only": True},
        ],
    )
    def test_invalid(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            Pi05WaypointConfig(**kwargs)


@pytest.mark.usefixtures("tiny_backbone")
class TestModel:
    @pytest.mark.parametrize(("planner_mode", "goal_conditioning"), list(itertools.product(["token_ar", "block_ar"], ["suffix", "naive", "ngm"])))
    def test_loss_and_backward(self, planner_mode: str, goal_conditioning: str) -> None:
        model = _model(planner_mode, goal_conditioning)
        model.train()
        loss, stats = model.compute_loss(_batch())
        assert torch.isfinite(loss)
        assert {"fm_loss", "planner_ce", "planner_acc"} <= set(stats)
        loss.backward()
        assert model.dur_emb.weight.grad is not None
        if model.goal_route is not None:
            assert model.goal_route.mlp_out.weight.grad is not None
        if model.block_query is not None:
            assert model.block_query.grad is not None
            assert model.block_query.grad.abs().sum() > 0

    def test_requires_waypoint_stats(self) -> None:
        stats = _stats()
        del stats["waypoint"]
        with pytest.raises(ValueError, match="waypoint"):
            Pi05WaypointModel(stats, paligemma_variant="gemma_300m", dtype="float32")

    def test_expert_does_not_see_plan_tokens(self) -> None:
        model = _model("token_ar", "suffix").eval()
        batch = _batch()
        other = dict(batch)
        other[KEY_WP_Q] = -batch[KEY_WP_Q]
        other[KEY_WP_D] = torch.ones_like(batch[KEY_WP_D])
        with torch.no_grad():
            torch.manual_seed(1)
            _, s1 = model.compute_loss(batch)
            torch.manual_seed(1)
            _, s2 = model.compute_loss(other)
        assert torch.allclose(s1["fm_loss"], s2["fm_loss"])
        assert not torch.allclose(s1["planner_ce"], s2["planner_ce"])

    def test_zero_init_ngm_matches_suffix(self) -> None:
        suffix = _model("block_ar", "suffix").eval()
        ngm = _model("block_ar", "ngm").eval()
        ngm.load_state_dict(suffix.state_dict(), strict=False)
        batch = _batch()
        with torch.no_grad():
            torch.manual_seed(3)
            a = suffix.predict_segment(batch, batch[KEY_SEG_Q], batch[KEY_SEG_G], batch[KEY_SEG_D])
            torch.manual_seed(3)
            b = ngm.predict_segment(batch, batch[KEY_SEG_Q], batch[KEY_SEG_G], batch[KEY_SEG_D])
        assert torch.allclose(a, b, atol=1e-6)
        assert a.shape == (2, D, ACTION_DIM)

    def test_endpoint_changes_actions_only_through_goal(self) -> None:
        model = _model("block_ar", "suffix").eval()
        batch = _batch()
        noise = torch.randn(2, D, 32)
        with torch.no_grad():
            a = model.predict_segment(batch, batch[KEY_SEG_Q], batch[KEY_SEG_G], batch[KEY_SEG_D], noise=noise.clone())
            b = model.predict_segment(batch, batch[KEY_SEG_Q], batch[KEY_SEG_G], batch[KEY_SEG_D], noise=noise.clone())
            c = model.predict_segment(batch, -batch[KEY_SEG_Q], batch[KEY_SEG_G], batch[KEY_SEG_D], noise=noise.clone())
        assert torch.allclose(a, b)
        assert not torch.allclose(a, c)

    @pytest.mark.parametrize("planner_mode", ["token_ar", "block_ar"])
    def test_decode_pass_count(self, planner_mode: str) -> None:
        model = _model(planner_mode).eval()
        batch = _batch(bsize=1)
        plan, actions = model.plan_and_segment(batch)
        decoded = int(plan.valid[0].sum())
        per_block = 1 if planner_mode == "block_ar" else model.wp_tokenizer.num_values
        assert plan.num_passes == 1 + per_block * decoded
        assert 1 <= decoded <= M
        assert int(plan.d[0, 0]) >= 1
        assert (plan.d[0][plan.valid[0]] <= D).all()
        assert actions.shape == (1, D, ACTION_DIM)

    @pytest.mark.parametrize("planner_mode", ["token_ar", "block_ar"])
    def test_greedy_decode_matches_teacher_forcing(self, planner_mode: str) -> None:
        """Feeding the decoded plan back as targets must reproduce it: train/infer masks and positions agree."""
        model = _model(planner_mode, seed=4).eval()
        batch = _batch(bsize=2)
        with torch.no_grad():
            prefix_pad, cache = model.prefix_cache(batch)
            cur_g = torch.zeros(2, 1, dtype=torch.long)
            plan = model.decode_plan(prefix_pad, cache, batch[STATE], cur_g)
            tf = dict(batch)
            tf[KEY_WP_Q], tf[KEY_WP_G], tf[KEY_WP_D], tf[KEY_WP_VALID] = plan.q, plan.g, plan.d, plan.valid
            logits = model.teacher_forced_planner_logits(tf)
        tok = model.wp_tokenizer
        expected = tok.reserved_targets(plan.q, plan.g, plan.d)
        for k in range(M):
            pred = model.decode_constraints(logits[:, k], first_block=k == 0).argmax(-1)
            for b in range(2):
                if not plan.valid[b, k]:
                    continue
                if int(plan.d[b, k]) == 0:
                    assert pred[b, -1] == expected[b, k, -1]
                else:
                    assert torch.equal(pred[b], expected[b, k]), (k, b)

    def test_waypoint_heads_trainable_after_lora(self) -> None:
        from physicalai.policies.mixins.peft import build_lora_config, inject_lora

        model = _model()
        inject_lora(model, build_lora_config(rank=2, alpha=2, dropout=0.0, target_modules=model.get_default_peft_targets()))
        assert not model.goal_route.mlp_out.weight.requires_grad  # type: ignore[union-attr]
        model.enable_waypoint_head_training()
        assert model.goal_route.mlp_out.weight.requires_grad  # type: ignore[union-attr]
        assert model.block_query.requires_grad  # type: ignore[union-attr]
        assert model.dur_emb.weight.requires_grad


class _FakeTokenizer:
    def __call__(self, text: list[str], max_length: int, **_: object) -> dict:
        ids = torch.randint(0, 1000, (len(text), max_length))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.mark.usefixtures("tiny_backbone")
class TestPolicy:
    def _policy(self, **kwargs: object) -> Pi05Waypoint:
        torch.manual_seed(0)
        policy = Pi05Waypoint(
            dataset_stats=_stats(),
            paligemma_variant="gemma_300m",
            action_expert_variant="gemma_300m",
            dtype="float32",
            chunk_size=D,
            n_action_steps=D,
            max_blocks=M,
            num_inference_steps=2,
            tokenizer_max_length=6,
            gradient_checkpointing=False,
            **kwargs,  # type: ignore[arg-type]
        )
        policy._preprocessor._tokenizer = _FakeTokenizer()  # type: ignore[union-attr]  # noqa: SLF001
        return policy.eval()

    @staticmethod
    def _obs() -> Observation:
        return Observation(
            state=torch.rand(1, STATE_DIM) * 2 - 1,
            images={"top": torch.rand(1, 3, 16, 16)},
            task=["pick up the cube"],
        )

    def test_get_policy(self) -> None:
        from physicalai.policies import get_physicalai_policy_class

        assert get_physicalai_policy_class("pi05_waypoint") is Pi05Waypoint

    @pytest.mark.parametrize("replan_mode", ["full_plan", "receding"])
    def test_select_action_executes_plan_segments(self, replan_mode: str) -> None:
        policy = self._policy(replan_mode=replan_mode)
        steps = 0
        while len([e for e in policy.telemetry if e["event"] == "plan"]) < 2 and steps < 200:  # noqa: PLR2004
            action = policy.select_action(self._obs())
            assert action.shape == (1, ACTION_DIM)
            steps += 1
        plans = [e for e in policy.telemetry if e["event"] == "plan"]
        segments = [e for e in policy.telemetry if e["event"] == "segment"]
        assert len(plans) == 2  # noqa: PLR2004
        first_plan_segments = segments[: plans[0]["num_waypoints"]] if replan_mode == "full_plan" else segments[:1]
        assert steps - 1 == sum(s["d"] for s in first_plan_segments)
        policy.reset()
        assert policy.telemetry == []
        assert policy.last_plan is None

    def test_erase_endpoint_uses_current_state(self) -> None:
        policy = self._policy()
        policy.erase_endpoint = True
        assert policy.select_action(self._obs()).shape == (1, ACTION_DIM)

    def test_predict_action_chunk_shape(self) -> None:
        policy = self._policy()
        assert policy.predict_action_chunk(self._obs()).shape == (1, D, ACTION_DIM)

    def test_export_not_supported(self) -> None:
        assert Pi05Waypoint.get_supported_export_backends() == []


@pytest.mark.usefixtures("tiny_backbone")
class TestDiagnostics:
    def test_executed_rms_masks_steps(self) -> None:
        from physicalai.policies.pi05_waypoint.diagnostics import executed_rms

        a = torch.zeros(2, 4, 3)
        b = torch.zeros(2, 4, 3)
        b[:, 2:] = 1.0
        out = executed_rms(a, b, torch.tensor([2, 4]))
        assert torch.allclose(out, torch.tensor([0.0, 2**-0.5]))

    def test_sensitivity_report(self) -> None:
        from physicalai.policies.pi05_waypoint.diagnostics import sensitivity

        model = _model("block_ar", "suffix").eval()
        report = sensitivity(model, [_batch(), _batch(seed=1)])
        assert report.num_segments == 4  # noqa: PLR2004
        assert report.B > 0
        for value in (report.S_endpoint, report.S_duration, report.S_image, report.S_language):
            assert value >= 0

    def test_planning_report(self) -> None:
        from physicalai.policies.pi05_waypoint.diagnostics import planning_metrics

        model = _model("block_ar").eval()
        report = planning_metrics(model, [_batch()])
        assert report.num_windows == 2  # noqa: PLR2004
        assert report.true_endings == 1
        assert 0 <= report.first_wp_rmse
        assert 2 <= report.mean_passes <= 1 + M  # noqa: PLR2004
