"""Tests for multi-horizon state-prediction targets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from libero_dataset import LiberoDataset
from preprocess_libero import save_hdf5


class FutureHorizonTargetsTest(unittest.TestCase):
    def test_dataset_reads_multi_horizon_future_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.hdf5"
            with h5py.File(path, "w") as f:
                vlen_dt = h5py.vlen_dtype(np.int32)
                f.create_dataset("image_agent", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
                f.create_dataset("image_hand", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
                f.create_dataset(
                    "image_agent_future_horizons",
                    data=np.zeros((1, 4, 4, 4, 3), dtype=np.uint8),
                )
                f.create_dataset(
                    "image_hand_future_horizons",
                    data=np.ones((1, 4, 4, 4, 3), dtype=np.uint8),
                )
                proprio_targets = np.arange(36, dtype=np.float32).reshape(1, 4, 9)
                f.create_dataset("proprio", data=np.zeros((1, 9), dtype=np.float32))
                f.create_dataset("proprio_future_horizons", data=proprio_targets)
                f.create_dataset("continuous_actions", data=np.zeros((1, 20, 7), dtype=np.float32))
                fast_tokens = f.create_dataset("fast_tokens", shape=(1,), dtype=vlen_dt)
                fast_tokens[0] = np.array([1, 2, 3], dtype=np.int32)
                f.create_dataset("demo_idx", data=np.array([0], dtype=np.int32))
                f.attrs["language_instruction"] = "pick up the block"
                f.attrs["chunk_size"] = 20
                f.attrs["state_prediction_horizons"] = np.array([5, 10, 15, 20])

            dataset = LiberoDataset(
                tmp,
                max_action_tokens=6,
                max_lang_tokens=4,
                img_size=4,
                use_language=False,
                use_state_prediction=True,
            )
            item = dataset[0]

            self.assertEqual(tuple(item["pixels_agent_future"].shape), (4, 3, 4, 4))
            self.assertEqual(tuple(item["pixels_hand_future"].shape), (4, 3, 4, 4))
            self.assertEqual(tuple(item["proprio_future"].shape), (4, 9))
            np.testing.assert_allclose(item["proprio_future"].numpy(), proprio_targets[0])

    def test_save_hdf5_writes_multi_horizon_fields_and_legacy_last_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "out.hdf5"
            img = np.zeros((4, 4, 3), dtype=np.uint8)
            agent_horizons = np.stack(
                [np.full((4, 4, 3), fill_value=i, dtype=np.uint8) for i in range(4)],
                axis=0,
            )
            hand_horizons = np.stack(
                [np.full((4, 4, 3), fill_value=i + 10, dtype=np.uint8) for i in range(4)],
                axis=0,
            )
            proprio_horizons = np.arange(36, dtype=np.float64).reshape(4, 9)
            samples = {
                "image_agent": [img],
                "image_hand": [img],
                "proprio": [np.zeros(9, dtype=np.float64)],
                "image_agent_future_horizons": [agent_horizons],
                "image_hand_future_horizons": [hand_horizons],
                "proprio_future_horizons": [proprio_horizons],
                "continuous_actions": [np.zeros((20, 7), dtype=np.float32)],
                "demo_idx": [0],
                "chunk_idx": [0],
            }

            save_hdf5(
                str(output_path),
                samples,
                [np.array([1, 2, 3], dtype=np.int32)],
                np.zeros(7, dtype=np.float64),
                np.ones(7, dtype=np.float64),
                chunk_size=20,
                chunk_stride=1,
                image_key="agentview_rgb",
                source_file=__file__,
                num_demos=1,
                language_instruction="pick up the block",
            )

            with h5py.File(output_path, "r") as f:
                self.assertEqual(f["image_agent_future_horizons"].shape, (1, 4, 4, 4, 3))
                self.assertEqual(f["image_hand_future_horizons"].shape, (1, 4, 4, 4, 3))
                self.assertEqual(f["proprio_future_horizons"].shape, (1, 4, 9))
                np.testing.assert_array_equal(
                    f["image_agent_future"][0],
                    agent_horizons[-1],
                )
                np.testing.assert_allclose(
                    f["proprio_future"][0],
                    proprio_horizons[-1],
                )
                np.testing.assert_array_equal(
                    f.attrs["state_prediction_horizons"],
                    np.array([5, 10, 15, 20]),
                )


if __name__ == "__main__":
    unittest.main()
