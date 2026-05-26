"""CPU tests for the Meta MoT-style predictor variant."""

from __future__ import annotations

import unittest

import torch

from module import ACTION_HEAD_SIZE, ARPredictor, MoTBlock, PAD_TOKEN_ID


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
        self.assertEqual(pred_ag.shape, (2, 4, 16))
        self.assertEqual(pred_hd.shape, (2, 4, 16))
        self.assertEqual(pred_pr.shape, (2, 4, 9))
        self.assertEqual(predictor.n_state_query, 12)
        self.assertEqual(predictor.state_query_tokens.shape, (3, 16))
        self.assertEqual(predictor.state_horizon_embeddings.shape, (4, 16))
        self.assertEqual(predictor.state_modality_embeddings.shape, (3, 16))
        self.assertEqual(predictor._compose_state_query_embeddings().shape, (4, 3, 16))

    def test_mot_uses_modality_specific_transformer_blocks(self) -> None:
        predictor = self._make_predictor()
        self.assertTrue(predictor.use_mot_transformer)
        self.assertIsInstance(predictor.blocks[0], MoTBlock)
        self.assertEqual(len(predictor.blocks[0].attn.to_qkv), 4)
        self.assertEqual(len(predictor.blocks[0].mlp), 4)
        self.assertFalse(hasattr(predictor, "state_blocks"))
        modality_ids = predictor._build_train_modality_ids(
            B=2,
            n_lang=4,
            nv=1,
            include_queries=True,
            device=torch.device("cpu"),
        )
        self.assertEqual(modality_ids.shape, (2, predictor.max_seq_len))
        self.assertEqual(
            modality_ids[0, -12:].tolist(),
            [1, 1, 2] * 4,
        )

    def test_mot_keeps_global_cross_modality_attention_path(self) -> None:
        predictor = self._make_predictor()
        logits, pred_ag, pred_hd, pred_pr = predictor(*self._make_inputs())
        logits.sum().backward()
        self.assertTrue(
            any(p.grad is not None for p in predictor.blocks[0].attn.to_qkv[1].parameters())
        )
        self.assertTrue(any(p.grad is not None for p in predictor.blocks[0].mlp[3].parameters()))

    def test_mot_can_run_without_state_prediction(self) -> None:
        predictor = ARPredictor(
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
            use_state_prediction=False,
            state_prediction_arch="mot",
        )
        out = predictor(*self._make_inputs())
        self.assertEqual(out.shape, (2, 6, ACTION_HEAD_SIZE))

    def test_mot_attention_handles_bfloat16_autocast(self) -> None:
        predictor = self._make_predictor()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = predictor(*self._make_inputs())
        self.assertEqual(len(out), 4)

    def test_old_object_checkpoints_without_mot_flag_default_to_shared(self) -> None:
        predictor = ARPredictor(
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
            use_state_prediction=False,
        )
        delattr(predictor, "use_mot_transformer")
        out = predictor(*self._make_inputs())
        self.assertEqual(out.shape, (2, 6, ACTION_HEAD_SIZE))


if __name__ == "__main__":
    unittest.main()
