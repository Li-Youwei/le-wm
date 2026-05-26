"""Tests for multi-token visual projection contracts."""

from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from jepa import JEPA
from module import ACTION_HEAD_SIZE, ARPredictor


class FakeEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 4, n_patches: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor, interpolate_pos_encoding: bool = True):
        batch = x.size(0)
        n_tokens = 1 + self.n_patches
        hidden = torch.arange(
            batch * n_tokens * self.hidden_dim,
            dtype=x.dtype,
            device=x.device,
        ).reshape(batch, n_tokens, self.hidden_dim)
        return types.SimpleNamespace(last_hidden_state=hidden)


class RecordingProjector(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.calls: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(x.shape))
        return x + self.offset


class VisualTokenProjectionTest(unittest.TestCase):
    def test_cls_and_patch_projectors_are_not_mixed(self) -> None:
        cls_projector = RecordingProjector(offset=100.0)
        patch_projector = RecordingProjector(offset=1000.0)
        model = JEPA(
            encoder=FakeEncoder(hidden_dim=4, n_patches=16),
            predictor=nn.Identity(),
            projector=cls_projector,
            patch_projector=patch_projector,
            visual_pool_grid=2,
        )

        pixels_agent = torch.zeros(2, 3, 224, 224)
        pixels_hand = torch.zeros(2, 3, 224, 224)
        z_agent, z_hand, _, _ = model.encode(pixels_agent, pixels_hand)

        self.assertEqual(tuple(z_agent.shape), (2, 5, 4))
        self.assertEqual(tuple(z_hand.shape), (2, 5, 4))
        self.assertEqual(cls_projector.calls, [(4, 4)])
        self.assertEqual(patch_projector.calls, [(16, 4)])
        self.assertTrue(torch.all(z_agent[:, 0] < z_agent[:, 1:].amin(dim=1)))

        future_agent, future_hand = model.encode_future_visual(
            pixels_agent,
            pixels_hand,
        )
        self.assertEqual(tuple(future_agent.shape), (2, 4))
        self.assertEqual(tuple(future_hand.shape), (2, 4))
        self.assertEqual(cls_projector.calls, [(4, 4), (4, 4)])
        self.assertEqual(patch_projector.calls, [(16, 4)])

        future_agent_tokens, future_hand_tokens = model.encode_future_visual(
            pixels_agent,
            pixels_hand,
            return_all_tokens=True,
        )
        self.assertEqual(tuple(future_agent_tokens.shape), (2, 5, 4))
        self.assertEqual(tuple(future_hand_tokens.shape), (2, 5, 4))
        self.assertEqual(cls_projector.calls, [(4, 4), (4, 4), (4, 4)])
        self.assertEqual(patch_projector.calls, [(16, 4), (16, 4)])

    def test_patch_level_state_prediction_outputs_visual_token_set(self) -> None:
        torch.manual_seed(0)
        B, D, N = 2, 16, 5
        predictor = ARPredictor(
            embed_dim=D,
            depth=1,
            heads=2,
            dim_head=8,
            mlp_dim=32,
            max_action_tokens=6,
            max_lang_tokens=4,
            proprio_dim=9,
            dropout=0.0,
            emb_dropout=0.0,
            use_state_prediction=True,
            visual_pool_grid=2,
            state_pred_visual_tokens=True,
        )

        z_agent = torch.randn(B, N, D)
        z_hand = torch.randn(B, N, D)
        proprio = torch.randn(B, 9)
        lang = torch.randn(B, 4, D)
        lang_lengths = torch.tensor([4, 2])
        actions = torch.randint(0, 100, (B, 6))
        action_lengths = torch.tensor([4, 5])

        logits, pred_ag, pred_hd, pred_pr = predictor(
            z_agent,
            z_hand,
            proprio,
            lang,
            lang_lengths,
            actions,
            action_lengths,
        )

        self.assertEqual(tuple(logits.shape), (B, 7, ACTION_HEAD_SIZE))
        self.assertEqual(tuple(pred_ag.shape), (B, 4, N, D))
        self.assertEqual(tuple(pred_hd.shape), (B, 4, N, D))
        self.assertEqual(tuple(pred_pr.shape), (B, 4, 9))

    def test_multi_horizon_future_visual_encoding_preserves_horizon_axis(self) -> None:
        cls_projector = RecordingProjector(offset=100.0)
        patch_projector = RecordingProjector(offset=1000.0)
        model = JEPA(
            encoder=FakeEncoder(hidden_dim=4, n_patches=16),
            predictor=nn.Identity(),
            projector=cls_projector,
            patch_projector=patch_projector,
            visual_pool_grid=2,
        )

        pixels_agent = torch.zeros(2, 4, 3, 224, 224)
        pixels_hand = torch.zeros(2, 4, 3, 224, 224)

        future_agent, future_hand = model.encode_future_visual(
            pixels_agent,
            pixels_hand,
        )
        self.assertEqual(tuple(future_agent.shape), (2, 4, 4))
        self.assertEqual(tuple(future_hand.shape), (2, 4, 4))

        future_agent_tokens, future_hand_tokens = model.encode_future_visual(
            pixels_agent,
            pixels_hand,
            return_all_tokens=True,
        )
        self.assertEqual(tuple(future_agent_tokens.shape), (2, 4, 5, 4))
        self.assertEqual(tuple(future_hand_tokens.shape), (2, 4, 5, 4))

    def test_visual257_positional_grids_and_state_queries_match_pool_grid(self) -> None:
        predictor = ARPredictor(
            embed_dim=192,
            depth=1,
            heads=2,
            dim_head=8,
            mlp_dim=256,
            max_action_tokens=80,
            max_lang_tokens=25,
            proprio_dim=9,
            visual_pool_grid=16,
            use_state_prediction=True,
        )

        self.assertEqual(predictor.n_visual_per_view, 257)
        self.assertEqual(tuple(predictor.agent_patch_2d_pos.shape), (16, 16, 192))
        self.assertEqual(tuple(predictor.hand_patch_2d_pos.shape), (16, 16, 192))
        self.assertEqual(tuple(predictor.state_query_embeddings.shape), (4, 3, 192))
        self.assertEqual(predictor.n_state_query, 12)
        self.assertEqual(predictor.max_seq_len, 633)

    def test_multi_visual_token_predictor_requires_grid_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "visual_pool_grid > 0"):
            ARPredictor(
                embed_dim=16,
                depth=1,
                heads=2,
                dim_head=8,
                mlp_dim=32,
                max_action_tokens=6,
                max_lang_tokens=4,
                proprio_dim=9,
                n_visual_tokens_per_view=5,
                visual_pool_grid=0,
            )

    def test_visual_pair_shape_mismatch_fails_fast(self) -> None:
        predictor = ARPredictor(
            embed_dim=16,
            depth=1,
            heads=2,
            dim_head=8,
            mlp_dim=32,
            max_action_tokens=6,
            max_lang_tokens=4,
            proprio_dim=9,
            dropout=0.0,
            emb_dropout=0.0,
            visual_pool_grid=2,
        )

        B, D = 2, 16
        z_agent = torch.randn(B, 5, D)
        z_hand = torch.randn(B, 4, D)
        proprio = torch.randn(B, 9)
        lang = torch.randn(B, 4, D)
        lang_lengths = torch.tensor([4, 2])
        actions = torch.randint(0, 100, (B, 6))
        action_lengths = torch.tensor([4, 5])

        with self.assertRaisesRegex(ValueError, "matching visual tensors"):
            predictor(
                z_agent,
                z_hand,
                proprio,
                lang,
                lang_lengths,
                actions,
                action_lengths,
            )


if __name__ == "__main__":
    unittest.main()
