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

    def test_train_config_declares_state_prediction_horizons(self) -> None:
        config_text = (ROOT / "config/train/lewm.yaml").read_text()
        self.assertIn("state_prediction_horizons: [5, 10, 15, 20]", config_text)

    def test_train_config_declares_prediction_stream_weights(self) -> None:
        config_text = (ROOT / "config/train/lewm.yaml").read_text()
        self.assertIn("pred_stream_weights:", config_text)
        self.assertIn("ag: 1.0", config_text)
        self.assertIn("hd: 1.0", config_text)
        self.assertIn("pr: 1.0", config_text)

    def test_train_config_uses_full_train_split_by_default(self) -> None:
        config_text = (ROOT / "config/train/lewm.yaml").read_text()
        self.assertIn("train_split: 1.0", config_text)

    def test_pretrained_vision_runner_forwards_warmup_steps_to_hydra(self) -> None:
        script_text = (ROOT / "run_all4_pretrained_vision.sh").read_text()
        self.assertIn("scheduler.warmup_steps", script_text)

    def test_pretrained_vision_runner_declares_visual257_depth_defaults(self) -> None:
        script_text = (ROOT / "run_all4_pretrained_vision.sh").read_text()
        self.assertIn("all4_sp_sigreg_dinov2_frozen_visual257_patch_sp)", script_text)
        self.assertIn("POOL_GRID_DEFAULT=16", script_text)
        self.assertIn("PREDICTOR_DEPTH_DEFAULT=12", script_text)
        self.assertIn('predictor.depth="$PREDICTOR_DEPTH"', script_text)

    def test_pretrained_vision_runner_uses_suitewise_training_protocol(self) -> None:
        script_text = (ROOT / "run_all4_pretrained_vision.sh").read_text()
        self.assertIn('MAX_STEPS="${MAX_STEPS:-60000}"', script_text)
        self.assertIn('TRAIN_SPLIT="${TRAIN_SPLIT:-1.0}"', script_text)
        self.assertIn('FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-50}"', script_text)
        self.assertIn(
            'EVAL_MAX_STEPS_LIBERO_SPATIAL="${EVAL_MAX_STEPS_LIBERO_SPATIAL:-220}"',
            script_text,
        )
        self.assertIn(
            'EVAL_MAX_STEPS_LIBERO_OBJECT="${EVAL_MAX_STEPS_LIBERO_OBJECT:-280}"',
            script_text,
        )
        self.assertIn(
            'EVAL_MAX_STEPS_LIBERO_GOAL="${EVAL_MAX_STEPS_LIBERO_GOAL:-300}"',
            script_text,
        )
        self.assertIn(
            'EVAL_MAX_STEPS_LIBERO_10="${EVAL_MAX_STEPS_LIBERO_10:-520}"',
            script_text,
        )
        self.assertIn(
            'DEFAULT_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")',
            script_text,
        )
        self.assertIn('TRAIN_HDF5_DIR="${PROCESSED_ROOT}/${suite}"', script_text)
        self.assertIn('CKPT_DIR="${CKPT_ROOT}/${ARM}_${suite}_seed${SEED}"', script_text)
        self.assertIn('--suites "$suite"', script_text)
        self.assertNotIn("all4_flat", script_text)

    def test_pretrained_vision_runner_defaults_to_ddp_batch64(self) -> None:
        script_text = (ROOT / "run_all4_pretrained_vision.sh").read_text()
        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"', script_text)
        self.assertIn('DDP_DEVICES="${DDP_DEVICES:-4}"', script_text)
        self.assertIn('GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"', script_text)
        self.assertIn('PER_DEVICE_BATCH_SIZE=16', script_text)
        self.assertIn('PER_DEVICE_BATCH_SIZE="$BATCH_SIZE"', script_text)
        self.assertIn(
            'ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"',
            script_text,
        )
        self.assertIn('TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp}"', script_text)
        self.assertIn('SYNC_BATCHNORM="${SYNC_BATCHNORM:-true}"', script_text)
        self.assertIn('trainer.devices="$DDP_DEVICES"', script_text)
        self.assertIn('trainer.strategy="$TRAINER_STRATEGY"', script_text)
        self.assertIn('trainer.sync_batchnorm="$SYNC_BATCHNORM"', script_text)
        self.assertIn(
            'trainer.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES"',
            script_text,
        )
        self.assertIn('loader.batch_size="$PER_DEVICE_BATCH_SIZE"', script_text)
        self.assertIn('loader.global_batch_size="$GLOBAL_BATCH_SIZE"', script_text)
        self.assertNotIn("trainer.devices=1", script_text)

    def test_train_config_declares_ddp_batch_controls(self) -> None:
        config_text = (ROOT / "config/train/lewm.yaml").read_text()
        self.assertIn("strategy: auto", config_text)
        self.assertIn("sync_batchnorm: false", config_text)
        self.assertIn("accumulate_grad_batches: 1", config_text)
        self.assertIn("use_distributed_sampler: false", config_text)
        self.assertIn("batch_size: 16", config_text)
        self.assertIn("global_batch_size: 64", config_text)

    def test_train_uses_project_ddp_sampler_and_syncbn_guard(self) -> None:
        source = (ROOT / "train.py").read_text()
        self.assertIn("DistributedWeightedSampler", source)
        self.assertIn("DistributedIndexSampler", source)
        self.assertIn("trainer.sync_batchnorm=True", source)
        self.assertIn("effective_global_batch", source)

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
            "CLAUDE.md",
        ]
        for relpath in removed:
            with self.subTest(relpath=relpath):
                self.assertFalse((ROOT / relpath).exists())


if __name__ == "__main__":
    unittest.main()
