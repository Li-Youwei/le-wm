"""Unit tests for ARPredictor attention-mask layout."""

from __future__ import annotations

import unittest

import torch

from module import ARPredictor, PAD_TOKEN_ID


class AttentionMaskLayoutTest(unittest.TestCase):
    def _make_predictor(self) -> ARPredictor:
        return ARPredictor(
            embed_dim=32,
            depth=1,
            heads=1,
            dim_head=32,
            mlp_dim=64,
            max_action_tokens=6,
            max_lang_tokens=5,
            proprio_dim=9,
            use_state_prediction=True,
        )

    def test_training_mask_matches_horizon_query_layout(self) -> None:
        pred = self._make_predictor()
        n_lang = 5
        lang_lengths = torch.tensor([3])
        action_tokens = torch.tensor(
            [[10, 20, 30, 40, PAD_TOKEN_ID, PAD_TOKEN_ID]]
        )

        mask = pred._build_attn_mask(
            n_lang,
            lang_lengths,
            action_tokens,
            27,
            torch.device("cpu"),
        )[0, 0]

        real_lang = [0, 1, 2]
        lang_pad = [3, 4]
        visual_proprio = [5, 6, 7]
        context = real_lang + visual_proprio
        bos = 8
        real_actions = [9, 10, 11, 12]
        action_pad = [13, 14]
        query_blocks = [
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
            [24, 25, 26],
        ]
        query_positions = [p for block in query_blocks for p in block]

        for p in lang_pad + action_pad:
            self.assertFalse(mask[p].any(), f"padding row {p} should be empty")
            self.assertFalse(mask[:, p].any(), f"padding col {p} should be empty")

        for i in real_lang:
            self.assertTrue(mask[i, context].all())
            self.assertFalse(mask[i, bos])
            self.assertFalse(mask[i, real_actions + query_positions].any())

        for i in visual_proprio + [bos]:
            self.assertTrue(mask[i, context + [bos]].all())
            self.assertFalse(mask[i, real_actions + query_positions].any())

        for i in real_actions:
            self.assertTrue(mask[i, context + [bos]].all())
            self.assertTrue(mask[i, list(range(9, i + 1))].all())
            self.assertFalse(mask[i, list(range(i + 1, 15))].any())
            self.assertFalse(mask[i, query_positions].any())

        for block_idx, block in enumerate(query_blocks):
            previous_blocks = [
                p for prior in query_blocks[:block_idx] for p in prior
            ]
            future_blocks = [
                p for later in query_blocks[block_idx + 1 :] for p in later
            ]
            for i in block:
                self.assertTrue(mask[i, context + [bos] + real_actions].all())
                self.assertTrue(mask[i, previous_blocks].all())
                same_block_others = [p for p in block if p != i]
                self.assertTrue(mask[i, i])
                self.assertFalse(mask[i, same_block_others].any())
                self.assertFalse(mask[i, future_blocks].any())
                self.assertFalse(mask[i, action_pad].any())

        for i in context + [bos] + real_actions:
            self.assertFalse(mask[i, query_positions].any())

    def test_generate_mask_uses_lang_visual_bos_action_layout(self) -> None:
        pred = ARPredictor(
            embed_dim=32,
            depth=1,
            heads=1,
            dim_head=32,
            mlp_dim=64,
            max_action_tokens=6,
            max_lang_tokens=5,
            proprio_dim=9,
        )
        n_lang = 5
        lang_lengths = torch.tensor([3])
        # lang(5) + visual/proprio(3) + BOS + two generated action tokens
        L = 11
        mask = pred._build_generate_mask(
            n_lang,
            lang_lengths,
            B=1,
            L=L,
            device=torch.device("cpu"),
        )[0, 0]

        context = [0, 1, 2, 5, 6, 7]
        bos = 8
        actions = [9, 10]

        self.assertTrue(mask[0, context].all())
        self.assertFalse(mask[0, bos])
        self.assertFalse(mask[0, actions].any())
        self.assertTrue(mask[5, context + [bos]].all())
        self.assertFalse(mask[5, actions].any())
        self.assertTrue(mask[bos, context + [bos]].all())
        self.assertFalse(mask[bos, actions].any())
        self.assertTrue(mask[10, context + [bos, 9, 10]].all())


if __name__ == "__main__":
    unittest.main()
