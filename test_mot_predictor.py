"""CPU tests for the MoT state-prediction predictor variant."""

from __future__ import annotations

import unittest

import torch

from module import ACTION_HEAD_SIZE, ARPredictor, PAD_TOKEN_ID


class MoTPredictorTest(unittest.TestCase):
    def _make_predictor(self) -> ARPredictor:
        torch.manual_seed(0)
        return ARPredictor(
            embed_dim=16,
            depth=1,
            heads=2,
            dim_head=8,
            mlp_dim=32,
            max_action_tokens=5,
            max_lang_tokens=4,
            proprio_dim=9,
            dropout=0.0,
            emb_dropout=0.0,
            use_state_prediction=True,
            state_prediction_arch="mot",
        )

    def _make_inputs(self) -> tuple[torch.Tensor, ...]:
        B, D = 2, 16
        z_agent = torch.randn(B, D)
        z_hand = torch.randn(B, D)
        proprio = torch.randn(B, 9)
        lang = torch.randn(B, 4, D)
        lang_lengths = torch.tensor([4, 2])
        actions = torch.tensor(
            [
                [1, 2, 3, PAD_TOKEN_ID, PAD_TOKEN_ID],
                [4, 5, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID],
            ],
            dtype=torch.long,
        )
        action_lengths = torch.tensor([3, 2])
        return z_agent, z_hand, proprio, lang, lang_lengths, actions, action_lengths

    def test_mot_forward_shapes_match_shared_contract(self) -> None:
        predictor = self._make_predictor()
        out = predictor(*self._make_inputs())
        self.assertEqual(len(out), 4)
        logits, pred_ag, pred_hd, pred_pr = out
        self.assertEqual(logits.shape, (2, 6, ACTION_HEAD_SIZE))
        self.assertEqual(pred_ag.shape, (2, 16))
        self.assertEqual(pred_hd.shape, (2, 16))
        self.assertEqual(pred_pr.shape, (2, 9))

    def test_mot_action_and_state_blocks_are_gradient_isolated(self) -> None:
        predictor = self._make_predictor()

        logits, pred_ag, pred_hd, pred_pr = predictor(*self._make_inputs())
        logits.sum().backward()
        self.assertTrue(any(p.grad is not None for p in predictor.blocks.parameters()))
        self.assertFalse(
            any(p.grad is not None for p in predictor.state_blocks.parameters())
        )

        predictor.zero_grad(set_to_none=True)
        logits, pred_ag, pred_hd, pred_pr = predictor(*self._make_inputs())
        (pred_ag.sum() + pred_hd.sum() + pred_pr.sum()).backward()
        self.assertFalse(any(p.grad is not None for p in predictor.blocks.parameters()))
        self.assertTrue(
            any(p.grad is not None for p in predictor.state_blocks.parameters())
        )

    def test_mot_requires_state_prediction(self) -> None:
        with self.assertRaises(ValueError):
            ARPredictor(
                embed_dim=16,
                depth=1,
                heads=2,
                dim_head=8,
                mlp_dim=32,
                max_action_tokens=5,
                max_lang_tokens=4,
                use_state_prediction=False,
                state_prediction_arch="mot",
            )


if __name__ == "__main__":
    unittest.main()
