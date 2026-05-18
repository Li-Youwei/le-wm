# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10 research/training repo for LIBERO VLA experiments. Core code lives at the repository root: `jepa.py` and `module.py` define the model, `train.py` wires Hydra and Lightning training, `libero_dataset.py` and `preprocess_libero.py` handle data, and `eval_libero.py` runs LIBERO evaluation. Active configs are under `config/train/`, with `config/train/data/libero.yaml` as the maintained dataset config. The active server workflow is `run_all4_pretrained_vision.sh`; older spatial-only, object-only, video, and slide-generation entrypoints have been removed.

## Build, Test, and Development Commands

- `conda activate vla && pip install -r requirements.txt`: install the pinned training dependencies. Keep `transformers>=4.48,<5`.
- `python train.py data=libero`: run Hydra training on preprocessed LIBERO HDF5 data.
- `bash preprocess_all4.sh`: preprocess all four LIBERO suites; override `RAW_ROOT`, `OUT_ROOT`, `TOKENIZER`, or `PARALLEL` as needed.
- `bash run_all4_pretrained_vision.sh`: server workflow for DINOv2-based 4-suite training, checkpoint selection, and per-suite evaluation. Common overrides: `SEED=2024 ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp CUDA_VISIBLE_DEVICES=0`.
- `tensorboard --logdir <run_dir>/tb_logs`: inspect training curves.

## Coding Style & Naming Conventions

Use 4-space indentation, clear type hints where they clarify tensor or path contracts, and import order of standard library, third-party libraries, then local modules. Python names use `snake_case`; constants such as token IDs stay uppercase. Config keys and environment overrides should be descriptive and snake_case or uppercase respectively. Shell scripts should use `set -euo pipefail` when practical and expose paths through environment variables instead of hardcoding new local machine paths.

## Testing Guidelines

There is no pytest suite or CI pipeline. Use the standalone checks before changing model, preprocessing, or token logic:

- `python test_attn_mask.py`: validates predictor attention-mask behavior.
- `python check_fast_roundtrip.py --processed-dir /path/to/processed --tokenizer /path/to/tokenizer`: validates FAST encode/decode preprocessing consistency.
- `python -B -m unittest test_visual_tokens.py test_attn_mask.py test_mot_predictor.py test_repo_contracts.py`: runs the lightweight model and repository contract tests.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit style, for example `feat: ...`, `feat(scope): ...`, `fix(scope): ...`, and `chore: ...`. Keep commits focused and mention the affected training or evaluation path. Pull requests should include a short intent summary, relevant config/data paths, commands run, metrics or logs for training/eval changes, and screenshots or videos only when visual outputs changed.

## Security & Configuration Tips

Keep large artifacts out of git: checkpoints, TensorBoard logs, Hydra `outputs/`, videos, and local agent state are ignored. Use `STABLEWM_HOME` for model/data storage. Treat `/nas_data_new/caz/data_ssd/libero` as read-only; write preprocessed data to a separate directory.
