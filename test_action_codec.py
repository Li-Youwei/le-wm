"""Tests for configurable action token codecs."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import action_codec
from action_codec import WorldVLABinActionCodec, build_action_codec
from module import ARPredictor


class WorldVLABinActionCodecTest(unittest.TestCase):
    def test_worldvla_bins_encode_decode_fixed_length_centers(self) -> None:
        codec = WorldVLABinActionCodec(num_bins=256, bin_min=-1.0, bin_max=1.0)
        actions = np.array(
            [
                [
                    [-1.0, -0.5, 0.0, 0.5, 1.0, -2.0, 2.0],
                    [-0.9, -0.1, 0.1, 0.9, 0.25, -0.25, 0.75],
                ]
            ],
            dtype=np.float32,
        )

        tokens = codec.encode(actions)

        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].shape, (2 * 7,))
        self.assertGreaterEqual(int(tokens[0].min()), 0)
        self.assertLessEqual(int(tokens[0].max()), 255)
        self.assertEqual(int(tokens[0][0]), 0)
        self.assertEqual(int(tokens[0][4]), 255)
        self.assertEqual(int(tokens[0][5]), 0)
        self.assertEqual(int(tokens[0][6]), 255)

        decoded = codec.decode(
            torch.from_numpy(tokens[0]).unsqueeze(0),
            torch.tensor([tokens[0].shape[0]], dtype=torch.long),
            time_horizon=2,
            action_dim=7,
        )

        clipped = np.clip(actions, -1.0, 1.0)
        half_bin = (codec.bin_max - codec.bin_min) / codec.num_bins / 2.0
        self.assertEqual(decoded.shape, (1, 2, 7))
        self.assertLessEqual(float(np.abs(decoded - clipped).max()), half_bin + 1e-6)

    def test_build_action_codec_accepts_worldvla_bins_config(self) -> None:
        codec = build_action_codec(
            {
                "type": "worldvla_bins",
                "num_bins": 256,
                "bin_min": -1.0,
                "bin_max": 1.0,
            }
        )

        self.assertEqual(codec.name, "worldvla_bins")
        self.assertEqual(codec.vocab_size, 256)
        self.assertEqual(codec.bos_token_id, 256)
        self.assertEqual(codec.eos_token_id, 257)
        self.assertEqual(codec.pad_token_id, 258)
        self.assertEqual(codec.action_head_size, 258)

    def test_worldvla_bins_decode_requires_exact_fixed_length(self) -> None:
        codec = WorldVLABinActionCodec(num_bins=256, bin_min=-1.0, bin_max=1.0)

        with self.assertRaisesRegex(ValueError, "expected exactly"):
            codec.decode(
                torch.tensor([[128, 129]], dtype=torch.long),
                torch.tensor([2], dtype=torch.long),
                time_horizon=2,
                action_dim=7,
            )

    def test_codec_compatibility_rejects_same_vocab_different_codec(self) -> None:
        compatible = getattr(action_codec, "action_codecs_compatible", None)
        self.assertIsNotNone(compatible)
        fast = build_action_codec({"type": "fast"})
        bins_1024 = build_action_codec({"type": "worldvla_bins", "num_bins": 1024})

        self.assertEqual(fast.vocab_size, bins_1024.vocab_size)
        self.assertFalse(compatible(fast, bins_1024))


class PredictorActionVocabularyTest(unittest.TestCase):
    def test_predictor_uses_configured_action_vocabulary(self) -> None:
        torch.manual_seed(0)
        predictor = ARPredictor(
            embed_dim=16,
            depth=1,
            heads=2,
            dim_head=8,
            mlp_dim=32,
            max_action_tokens=4,
            max_lang_tokens=0,
            proprio_dim=9,
            dropout=0.0,
            emb_dropout=0.0,
            action_vocab_size=256,
        )
        z_agent = torch.randn(1, 16)
        z_hand = torch.randn(1, 16)
        proprio = torch.randn(1, 9)
        action_tokens = torch.tensor(
            [[7, 8, predictor.pad_token_id, predictor.pad_token_id]],
            dtype=torch.long,
        )
        action_lengths = torch.tensor([2], dtype=torch.long)

        logits = predictor(
            z_agent,
            z_hand,
            proprio,
            None,
            None,
            action_tokens,
            action_lengths,
        )

        self.assertEqual(predictor.bos_token_id, 256)
        self.assertEqual(predictor.eos_token_id, 257)
        self.assertEqual(predictor.pad_token_id, 258)
        self.assertEqual(predictor.action_head_size, 258)
        self.assertEqual(tuple(logits.shape), (1, 5, 258))

        mask = predictor._build_attn_mask(
            0,
            None,
            action_tokens,
            predictor.max_seq_len,
            torch.device("cpu"),
        )[0, 0]
        self.assertFalse(mask[6].any())
        self.assertFalse(mask[:, 6].any())
        self.assertFalse(mask[7].any())
        self.assertFalse(mask[:, 7].any())

    def test_generate_masks_configured_bos_token(self) -> None:
        torch.manual_seed(0)
        predictor = ARPredictor(
            embed_dim=16,
            depth=1,
            heads=2,
            dim_head=8,
            mlp_dim=32,
            max_action_tokens=4,
            max_lang_tokens=0,
            proprio_dim=9,
            dropout=0.0,
            emb_dropout=0.0,
            action_vocab_size=256,
        )
        with torch.no_grad():
            predictor.action_head.weight.zero_()
            predictor.action_head.bias.zero_()
            predictor.action_head.bias[predictor.bos_token_id] = 100.0
            predictor.action_head.bias[5] = 10.0

        tokens, lengths = predictor.generate(
            torch.randn(1, 16),
            torch.randn(1, 16),
            torch.randn(1, 9),
            None,
            None,
            max_len=1,
            temperature=0.0,
        )

        self.assertEqual(lengths.tolist(), [1])
        self.assertEqual(tokens.tolist(), [[5]])


if __name__ == "__main__":
    unittest.main()
