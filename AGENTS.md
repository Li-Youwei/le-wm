# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10 research/training repo for the LeWM VLA baseline. Core code lives at the repository root: `jepa.py` and `module.py` define the model, `train.py` wires Hydra and Lightning training, `libero_dataset.py` and `preprocess_libero.py` handle data, and `eval_libero.py` runs LIBERO evaluation. Configs are under `config/train/` and `config/eval/`. Shell workflows live in `run_*.sh`, `launch_*.sh`, and `preprocess_*.sh`. Assets and rendered figures are in `assets/` and `slides/`. Detailed architecture notes are in `CLAUDE.md`.

## Build, Test, and Development Commands

- `conda activate vla && pip install -r requirements.txt`: install the pinned training dependencies. Keep `transformers>=4.48,<5`.
- `python train.py data=libero`: run Hydra training on preprocessed LIBERO HDF5 data.
- `bash preprocess_all4.sh`: preprocess all four LIBERO suites; override `RAW_ROOT`, `OUT_ROOT`, `TOKENIZER`, or `PARALLEL` as needed.
- `bash run_all4.sh`: server workflow for train, checkpoint selection, and per-suite evaluation. Common overrides: `SEED=2024 ARM=sp_sigreg CUDA_VISIBLE_DEVICES=0`.
- `tensorboard --logdir <run_dir>/tb_logs`: inspect training curves.

## Coding Style & Naming Conventions

Use 4-space indentation, clear type hints where they clarify tensor or path contracts, and import order of standard library, third-party libraries, then local modules. Python names use `snake_case`; constants such as token IDs stay uppercase. Config keys and environment overrides should be descriptive and snake_case or uppercase respectively. Shell scripts should use `set -euo pipefail` when practical and expose paths through environment variables instead of hardcoding new local machine paths.

## Testing Guidelines

There is no pytest suite or CI pipeline. Use the standalone checks before changing model, preprocessing, or token logic:

- `python test_attn_mask.py`: validates predictor attention-mask behavior.
- `python smoke_test.py --data /path/to/processed.h5`: runs a GPU forward/backward smoke test on real data.
- `python check_fast_roundtrip.py --processed-dir /path/to/processed --tokenizer /path/to/tokenizer`: validates FAST encode/decode preprocessing consistency.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit style, for example `feat: ...`, `feat(scope): ...`, `fix(scope): ...`, and `chore: ...`. Keep commits focused and mention the affected training or evaluation path. Pull requests should include a short intent summary, relevant config/data paths, commands run, metrics or logs for training/eval changes, and screenshots or videos only when visual outputs changed.

## Security & Configuration Tips

Keep large artifacts out of git: checkpoints, TensorBoard logs, Hydra `outputs/`, videos, and local agent state are ignored. Use `STABLEWM_HOME` for model/data storage. Do not write derived data into read-only raw LIBERO roots; use a separate processed-data directory.
