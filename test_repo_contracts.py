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

    def test_libero_config_keeps_demo_90_10_split_as_default(self) -> None:
        config_text = (ROOT / "config/train/lewm.yaml").read_text()
        self.assertIn("train_split: 0.9", config_text)
        self.assertIn("split_mode: demo_90_10", config_text)

    def test_baseline_protocol_noop_filter_matches_openvla_rule(self) -> None:
        from libero_baseline_protocol import is_noop_action

        first_stationary = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        first_motion = [1e-3, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        prev = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        same_gripper = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        changed_gripper = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        self.assertTrue(is_noop_action(first_stationary, None))
        self.assertFalse(is_noop_action(first_motion, None))
        self.assertTrue(is_noop_action(same_gripper, prev))
        self.assertFalse(is_noop_action(changed_gripper, prev))

    def test_image_rotation_helper_is_shared_by_training_and_eval(self) -> None:
        from libero_baseline_protocol import rotate_image_180

        img = [[1, 2, 3], [4, 5, 6]]
        expected = [[6, 5, 4], [3, 2, 1]]
        rotated = rotate_image_180(img)
        self.assertEqual(rotated, expected)
        self.assertIsNot(rotated, img)

    def test_eval_defaults_match_worldvla_libero_protocol(self) -> None:
        source = (ROOT / "eval_libero.py").read_text()
        self.assertIn("SUITE_MAX_STEPS", source)
        self.assertIn("default=DEFAULT_EVAL_EPISODES", source)
        self.assertIn("default=DEFAULT_CAMERA_SIZE", source)
        self.assertIn("--num-steps-wait", source)
        self.assertIn("default=DEFAULT_NUM_STEPS_WAIT", source)
        self.assertIn("success = bool(done)", source)
        self.assertIn("init_states[ep]", source)

    def test_run_all4_forwards_warmup_steps_to_hydra(self) -> None:
        script_text = (ROOT / "run_all4.sh").read_text()
        self.assertIn("scheduler.warmup_steps", script_text)
        self.assertIn('--num-episodes "$EVAL_EPISODES"', script_text)
        self.assertIn('--camera-size "$CAMERA_SIZE"', script_text)
        self.assertIn(
            'CKPT_DIR="${CKPT_ROOT}/all4_${ARM}${ARCH_SUFFIX}_split${SPLIT_MODE}_seed${SEED}"',
            script_text,
        )

    def test_run_suite_policy_script_exists_and_uses_suite_eval_protocol(self) -> None:
        script_text = (ROOT / "run_suite_policy.sh").read_text()
        self.assertIn('SUITE="${SUITE:-libero_spatial}"', script_text)
        self.assertIn('PROC_DIR="${PROCESSED_ROOT}/${SUITE}"', script_text)
        self.assertIn('--num-episodes "$EVAL_EPISODES"', script_text)
        self.assertIn('--camera-size "$CAMERA_SIZE"', script_text)

    def test_regenerate_records_pre_action_obs_for_kept_actions(self) -> None:
        source = (ROOT / "regenerate_libero_filtered.py").read_text()
        replay = source[
            source.index("def _replay_demo") : source.index("def _write_demo")
        ]
        self.assertIn("pre_step_record = _obs_record(obs, env)", replay)
        self.assertLess(
            replay.index("pre_step_record = _obs_record(obs, env)"),
            replay.index("obs, _reward, done, _info = env.step(action)"),
        )
        self.assertIn("kept_obs.append(pre_step_record)", replay)

    def test_preprocess_all4_validates_processed_source_before_skip(self) -> None:
        source = (ROOT / "preprocess_all4.sh").read_text()
        self.assertIn("processed_matches_filtered()", source)
        self.assertIn("source_file", source)
        self.assertLess(
            source.index('if [[ ! -f "$filtered" ]]'),
            source.index('if [[ -f "$out" ]]'),
        )
        self.assertIn("[stale]", source)

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

    def test_smoke_test_matches_training_dropout_contract(self) -> None:
        source = (ROOT / "smoke_test.py").read_text()
        self.assertIn("dropout=0.2", source)

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
        scripts = [
            "run_all4.sh",
            "run_all4_grip.sh",
            "run_all4_pretrained_vision.sh",
            "run_all4_visual17.sh",
            "run_object_baseline.sh",
            "run_object_pretrained_vision.sh",
            "run_visual17.sh",
            "run_suite_policy.sh",
        ]
        for script in scripts:
            with self.subTest(script=script):
                source = (ROOT / script).read_text()
                self.assertNotIn("--seed 42", source)
                self.assertIn('--seed "$SEED"', source)


if __name__ == "__main__":
    unittest.main()
