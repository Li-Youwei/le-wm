# LeWM LIBERO VLA Experiments

This repository is a LIBERO-focused fork of LeWorldModel for
vision-language-action experiments. The active model takes two RGB views,
9D proprioception, and a language instruction, then autoregressively predicts
FAST action tokens. Current experiments compare CLS-only and CLS+pooled-patch
visual prefixes, state prediction, SIGReg, frozen DINOv2, and the MoT predictor
variant.

## Setup

Use Python 3.10 and the `vla` conda environment used on the training server:

```bash
conda activate vla
pip install -r requirements.txt
```

`transformers>=4.48,<5` is required for the FAST tokenizer stack. Server runs
expect preprocessed data and checkpoints under writable `/Data/lyw` paths. Treat
the raw LIBERO root `/nas_data_new/caz/data_ssd/libero` as read-only.

## Data Pipeline

Raw LIBERO demos are preprocessed into HDF5 task files. Each sample stores
agentview RGB, eye-in-hand RGB, 9D proprio, FAST action tokens, normalized
continuous actions, language instruction metadata, and, for state prediction,
future-frame visual/proprio targets at `t+H`.

```bash
bash preprocess_all4.sh
python check_fast_roundtrip.py --processed-dir /path/to/processed --tokenizer /path/to/tokenizer
```

For 4-suite training, the runner expects a flat directory with 40 `.h5` files
and a fitted all-suite FAST tokenizer.

## Training And Evaluation

The active end-to-end launcher is:

```bash
ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp \
SEED=3072 \
CUDA_VISIBLE_DEVICES=1 \
bash run_all4_pretrained_vision.sh
```

The script trains on all four LIBERO suites, selects the best checkpoint by
task-balanced validation CE, then evaluates `libero_spatial`, `libero_object`,
`libero_goal`, and `libero_10` with the same run seed. Key overrides include
`MAX_STEPS`, `VAL_INTERVAL`, `BATCH_SIZE`, `CKPT_ROOT`, `FLAT_DIR`, `TOKENIZER`,
`PROCESSED_ROOT`, and `VISION_ENCODER`.

## Main Files

- `jepa.py`, `module.py`: visual/language/proprio encoders and AR predictor.
- `vision_backbone.py`: SPT ViT or HuggingFace vision backbone construction.
- `train.py`: Hydra + Lightning training loop and loss composition.
- `libero_dataset.py`, `preprocess_libero.py`: HDF5 data loading and preprocessing.
- `eval_libero.py`: closed-loop LIBERO rollout evaluation.
- `run_all4_pretrained_vision.sh`: current server training/eval workflow.

## Checks

There is no CI pipeline. Run lightweight local checks before pushing model or
runner changes:

```bash
python -m unittest test_visual_tokens.py test_attn_mask.py test_mot_predictor.py test_repo_contracts.py
bash -n preprocess_all4.sh run_all4_pretrained_vision.sh
python -m py_compile train.py jepa.py module.py eval_libero.py
```

Large artifacts such as checkpoints, TensorBoard logs, videos, preprocessed
HDF5 files, and Hydra outputs should remain outside git.
