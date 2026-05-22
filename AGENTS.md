# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python 3.10 research codebase for LIBERO VLA experiments.
Core model and training code lives at the repo root: `train.py`, `jepa.py`,
`module.py`, `vision_backbone.py`, `libero_dataset.py`, and `eval_libero.py`.
Hydra training configuration lives under `config/train/`, with dataset schema in
`config/train/data/libero.yaml`. Utility and runner scripts are also root-level,
for example `preprocess_all4.sh`, `run_all4_pretrained_vision.sh`, and
`check_fast_roundtrip.py`. Tests are root-level `test_*.py` files. Large data,
checkpoints, TensorBoard logs, videos, HDF5 outputs, and Hydra run outputs must
stay outside git.

## Build, Test, and Development Commands

- `conda activate vla`: enter the expected Python 3.10 environment.
- `pip install -r requirements.txt`: install runtime, training, and tokenizer
  dependencies.
- `bash preprocess_all4.sh`: preprocess all four LIBERO suites on the server.
- `ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp SEED=3072 CUDA_VISIBLE_DEVICES=1 bash run_all4_pretrained_vision.sh`: run the current 4-suite training/eval workflow.
- `python -m unittest test_visual_tokens.py test_attn_mask.py test_mot_predictor.py test_repo_contracts.py`: run lightweight local tests.
- `bash -n preprocess_all4.sh run_all4_pretrained_vision.sh`: syntax-check shell runners.
- `python -m py_compile train.py jepa.py module.py eval_libero.py`: catch import-time syntax errors.

If `python` is unavailable in a non-interactive shell, check `which python3` and
use the active project environment explicitly.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and keep functions focused on one training,
data, or evaluation concern. Follow existing naming: snake_case for functions,
variables, config keys, and scripts; PascalCase for classes. Keep Hydra override
keys synchronized between `config/train/*` and shell runners. Prefer structured
parsers and tensor APIs over ad hoc string or shape handling. Run `ruff check .`
when available before submitting broad Python edits.

## Testing Guidelines

Add or update `test_*.py` files for repository contracts, visual-token behavior,
attention masks, and predictor changes. Keep CPU tests lightweight; full LIBERO
rollouts and training runs are server workflows. For script edits, run the
relevant `bash -n` checks plus the narrowest affected unit tests.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style prefixes such as `feat:`, `fix:`,
`docs:`, and `chore:`. Keep commits scoped and imperative, for example
`fix: pass checkpoint top k as hydra addition`. Pull requests should describe
the experiment or bug, list validation commands, note required data/checkpoint
paths, and include metric or rollout results when behavior changes.

## Security & Configuration Tips

Treat `/nas_data_new/caz/data_ssd/libero` as read-only. Write derived artifacts
to approved writable locations such as `/Data/lyw/...`. Do not commit local
credentials, model checkpoints, generated HDF5 files, logs, or videos.
