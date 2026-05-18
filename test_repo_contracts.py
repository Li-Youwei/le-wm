"""Lightweight repository contract tests.

These checks cover configuration/script invariants that do not require LIBERO
data, GPU hardware, or external model downloads.
"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class RepositoryContractsTest(unittest.TestCase):
    def test_libero_config_declares_chunk_size(self) -> None:
        config_text = (ROOT / "config/train/data/libero.yaml").read_text()
        self.assertIn("chunk_size:", config_text)

    def test_pretrained_vision_runner_forwards_warmup_steps_to_hydra(self) -> None:
        script_text = (ROOT / "run_all4_pretrained_vision.sh").read_text()
        self.assertIn("scheduler.warmup_steps", script_text)

    def test_saved_fast_tokenizer_patch_copies_processor_module_fallback(self) -> None:
        source = (ROOT / "preprocess_libero.py").read_text()
        self.assertIn("tokenizer: Any | None = None", source)
        self.assertIn("inspect.getfile", source)
        self.assertIn("tokenizer.__class__.__name__", source)
        self.assertIn("processing_action_tokenizer.py", source)

    def test_weights_loader_rejects_visual_prefix_checkpoint(self) -> None:
        source = (ROOT / "eval_libero.py").read_text()
        self.assertIn("view_embedding", source)
        self.assertIn("agent_patch_2d_pos", source)
        self.assertIn("visual-prefix", source)

    def test_predict_actions_defaults_to_predictor_token_budget(self) -> None:
        source = (ROOT / "jepa.py").read_text()
        self.assertIn("max_len=None", source)
        self.assertIn("self.predictor.max_action_tokens", source)

    def test_language_instruction_is_not_silently_empty(self) -> None:
        preprocess_source = (ROOT / "preprocess_libero.py").read_text()
        dataset_source = (ROOT / "libero_dataset.py").read_text()
        self.assertIn("_derive_instruction_from_filename", preprocess_source)
        self.assertIn("language_source", preprocess_source)
        self.assertIn("Empty language_instruction", dataset_source)
        self.assertIn("use_language=False", dataset_source)

    def test_gripper_aux_is_denormalized_before_eval_execution(self) -> None:
        source = (ROOT / "eval_libero.py").read_text()
        self.assertIn("_denormalize_gripper_aux", source)
        self.assertIn("action_low[6]", source)
        self.assertIn("action_high[6]", source)

    def test_denormalize_actions_keeps_zero_range_dims_constant(self) -> None:
        source = (ROOT / "fast_utils.py").read_text()
        self.assertIn("zero_range = half_range < 1e-8", source)
        self.assertIn("np.where(zero_range, mid, restored)", source)

    def test_training_seed_is_used_globally(self) -> None:
        source = (ROOT / "train.py").read_text()
        self.assertIn("seed = int(cfg.seed)", source)
        self.assertIn("pl.seed_everything(seed, workers=True)", source)
        self.assertIn("seed=seed", source)

    def test_eval_scripts_use_run_seed(self) -> None:
        scripts = ["run_all4_pretrained_vision.sh"]
        for script in scripts:
            with self.subTest(script=script):
                source = (ROOT / script).read_text()
                self.assertNotIn("--seed 42", source)
                self.assertIn('--seed "$SEED"', source)

    def test_removed_legacy_entrypoints_stay_removed(self) -> None:
        removed = [
            "eval.py",
            "smoke_test.py",
            "run_all4.sh",
            "run_ablation.sh",
            "run_visual17.sh",
            "slides",
            "config/eval",
        ]
        for relpath in removed:
            with self.subTest(relpath=relpath):
                self.assertFalse((ROOT / relpath).exists())


if __name__ == "__main__":
    unittest.main()
