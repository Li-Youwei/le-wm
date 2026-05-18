# Repository Guidelines

This is the merged contributor and agent guide for the current repository. It
is synchronized with `CLAUDE.md` and checked against the current worktree rather
than remembered experiment history. When code and this document diverge, treat
the code/config as source of truth and update both guides.

## Current Repository Scope

This is a Python 3.10 research repo for LIBERO VLA experiments. The active
policy takes two RGB views, 9D proprioception, an optional language instruction,
and autoregressively predicts FAST action tokens decoded into a 20-step LIBERO
action chunk.

The removed original-LeWM planning and old experiment artifacts are not part of
the current repo: no `eval.py`, `config/eval/`, `slides/`, `assets/`,
`smoke_test.py`, or old `run_all4.sh` family.

## Active Configs

`config/train` is still required. `train.py` uses:

```python
@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
```

Active config files:

- `config/train/lewm.yaml`: main Hydra training config.
- `config/train/overfit.yaml`: one-demo overfit sanity config.
- `config/train/data/libero.yaml`: dataset schema used by both configs through
  `defaults: - data: libero`.

`run_all4_pretrained_vision.sh` calls `python train.py data=libero` and then
passes Hydra overrides for DINOv2/HF vision, loss weights, predictor arch,
seed, schedule, step budget, and visual-token settings.

Current defaults from `config/train/lewm.yaml` and
`config/train/data/libero.yaml`: `seed=3072`, `img_size=224`,
`patch_size=14`, `max_action_tokens=80`, `max_lang_tokens=25`,
`chunk_size=20`, `proprio_dim=9`, `batch_size=128`, `AdamW lr=5e-5`,
`weight_decay=0.05`, predictor `depth=6`, `heads=16`, `dim_head=64`,
`mlp_dim=2048`, `dropout=0.2`, `projector.norm_type=layer`, and all optional
loss weights default to `0.0`. `overfit.yaml` keeps model shape but uses
`overfit_demo=0`, `batch_size=8`, `lr=1e-3`, `weight_decay=0`, dropout `0`,
`max_epochs=2000`, and `trainer.devices=1`.

## File Inventory

| File | Current role |
|------|--------------|
| `.gitignore` | Excludes generated artifacts such as caches, checkpoints, logs, videos, and outputs. |
| `AGENTS.md` | Contributor/agent guide. Keep synchronized with `CLAUDE.md`. |
| `CLAUDE.md` | Detailed architecture, data, training, and file-role reference. |
| `LICENSE` | Project license. |
| `README.md` | Short user-facing overview and commands for the current LIBERO workflow. |
| `requirements.txt` | Python dependencies; keeps `transformers>=4.48,<5`. |
| `config/train/lewm.yaml` | Main trainer/loader/optimizer/projector/loss/vision/predictor config. |
| `config/train/overfit.yaml` | One-demo overfit config. |
| `config/train/data/libero.yaml` | LIBERO dataset config. |
| `train.py` | Hydra + Lightning training entrypoint, model construction, losses, sampling, logging, callbacks. |
| `jepa.py` | `JEPA` wrapper for visual/language encoding, future visual encoding, action generation, gripper aux. |
| `module.py` | Transformer/MoT modules, SIGReg, token constants, and `ARPredictor`. |
| `vision_backbone.py` | Builds SPT ViT or HuggingFace visual backbone. |
| `libero_dataset.py` | HDF5 dataset, image preprocessing, T5 tokenization, sampler weights. |
| `preprocess_libero.py` | Converts one raw LIBERO task HDF5 into processed chunk HDF5 with FAST tokens and SP targets. |
| `fit_tokenizer_all4.py` | Fits shared FAST tokenizer over all four LIBERO suites. |
| `preprocess_all4.sh` | Server preprocessing driver for all 40 LIBERO tasks; refuses raw-root writes. |
| `fast_utils.py` | FAST processor loading, token decode, action denormalization. |
| `eval_libero.py` | LIBERO rollout evaluation with closed-loop chunk execution. |
| `run_all4_pretrained_vision.sh` | Active 4-suite train/eval runner for frozen HuggingFace vision backbones. |
| `pick_best_ckpt.py` | Selects best object checkpoint from TensorBoard scalars. |
| `aggregate_all4_results.py` | Parses eval logs into markdown summaries. |
| `verify_sampler_balance.py` | Checks the 3-level weighted sampler. |
| `check_fast_roundtrip.py` | Checks FAST encode/decode against recomputed ground-truth chunks. |
| `utils.py` | Lightning callbacks for object checkpoints, task-balanced CE, overfit logging. |
| `test_visual_tokens.py` | Visual-token and patch-SP CPU tests. |
| `test_mot_predictor.py` | MoT predictor CPU tests. |
| `test_attn_mask.py` | Standalone attention-mask diagnostic. |
| `test_repo_contracts.py` | Repository invariant tests. |

## Data Pipeline

Raw LIBERO is expected on the server at `/nas_data_new/caz/data_ssd/libero`.
Treat it as read-only. Write tokenizer, processed HDF5, checkpoints, logs, and
videos under writable `/Data/lyw/...` paths.

`preprocess_libero.py` writes one processed HDF5 per task. Current datasets:
`image_agent`, `image_hand`, `image_agent_future`, `image_hand_future`,
`proprio`, `proprio_future`, `continuous_actions`, `fast_tokens`,
`fast_length`, `demo_idx`, and `chunk_idx`. Current attrs include
`action_low`, `action_high`, `chunk_size`, `chunk_stride`, `action_dim`,
`language_instruction`, and `language_source`.

All sample indices are raw environment steps. With `H=20`, anchor `t` uses
observation/proprio at `t`, future SP targets at `t+H`, and action targets
computed from observed poses at `t+k+1` for `k=0..H-1`. Valid anchors satisfy
`t <= T-H-1`. Default stride is `1`.

Action chunk definition:

```text
chunk[k, 0:3] = obs/ee_pos[t+k+1] - obs/ee_pos[t]
chunk[k, 3:6] = rotvec(R[t+k+1] * inv(R[t]))
chunk[k, 6]   = actions[t+k, 6]
```

Each task computes 1st/99th percentile action bounds and normalizes chunks to
`[-1, 1]` before FAST tokenization.

The proprio contract is fixed by `normalize_proprio()`:

```text
ee_pos(3) + xyzw_quat(4) + raw gripper finger positions(2)
```

Only the quaternion slice `[3:7]` is renormalized. Do not mean/abs the two
gripper dimensions.

`LiberoDataset` loads a directory of sorted `.h5`/`.hdf5` task files, assigns
stable `task_id` by sorted file order, lazy-opens HDF5 per worker, pads FAST
tokens to `max_action_tokens`, tokenizes one language string per file, loads
future targets when SP is enabled, and provides task/demo/chunk-balanced sampler
weights.

## Model Architecture

`train.py` builds the visual encoder through `build_visual_encoder()`:

- `vision_encoder.source=spt_vit`: `stable_pretraining` ViT factory.
- `vision_encoder.source=hf`: `AutoModel.from_pretrained(...)` wrapped by
  `HFVisionBackbone`.
- `vision_encoder.freeze=true`: eval mode plus `requires_grad_(False)`.

`JEPA.encode()` concatenates the two image views along batch dimension for one
shared visual backbone call. Language uses frozen `T5EncoderModel("t5-small")`
when `data.dataset.use_language=true`.

`visual_tokens.pool_grid` controls visual prefix length:

- `0`: CLS-only, one visual token per view.
- `G > 0`: CLS plus `G*G` pooled patch tokens per view.

With `pool_grid=4`, each view contributes `1 CLS + 16 patch tokens = 17`.
CLS uses `projector`; patches use `patch_projector` when present.

Predictor training sequence:

```text
[language tokens,
 agent visual tokens,
 hand visual tokens,
 proprio token,
 BOS,
 fixed-width action token slots,
 optional Q_ag, Q_hd, Q_pr]
```

Only `BOS + action token slots` go through `action_head`.

Token constants in `module.py`:

- FAST IDs `0..1023`
- `BOS_TOKEN_ID=1024`
- `EOS_TOKEN_ID=1025`
- `PAD_TOKEN_ID=1026`
- action embedding size `1027`
- action head size `1026`

Attention masks are explicit boolean `(B,1,L,L)` masks. Prefix is
bidirectional, action zone is causal, PAD rows/columns are blocked, and SP
queries only attend to real prefix/action tokens plus themselves.

`predictor.state_prediction_arch` supports:

- `shared`: standard transformer blocks.
- `mot`: modality-specific LayerNorm, QKV/output projections, and FFNs for
  language/visual/proprio/action tokens, while attention remains global.

## Losses

`lejepa_forward()` always computes teacher-forced action CE. BOS predicts the
first token, each action position predicts the next token, the final real token
position predicts EOS, and PAD targets are `-100`. `label_smoothing` defaults
to `0.1`.

`loss.pred_weight > 0` enables state prediction. Visual SP predicts CLS by
default. With `visual_tokens.patch_sp=true`, visual SP predicts `(B,N,D)` so
CLS and pooled patches get direct future-target MSE. Future visual targets are
encoded by the same shared visual backbone/projector and are not detached in
current code.

`loss.sigreg_weight > 0` enables SIGReg and requires
`projector.norm_type=batch`. SIGReg uses CLS visual streams only: current
agent/hand, plus future agent/hand when SP is enabled.

`loss.gripper_aux_weight > 0` enables `gripper_aux_head`, which reads the BOS
hidden state and regresses the normalized `(B,H)` gripper command sequence.
Eval denormalizes this aux output through `action_low[6]`/`action_high[6]` and
overwrites decoded action dim 6.

Total loss:

```text
L_total = L_CE
        + pred_weight * L_pred
        + sigreg_weight * L_sigreg
        + gripper_aux_weight * L_gripper_aux
```

## Training And Evaluation

`train.py` seeds from `cfg.seed`, splits by `(file_path, demo_idx)`, uses
`WeightedRandomSampler` outside overfit mode, constructs model/projectors/T5,
wraps the model in `stable_pretraining.Module`, logs TensorBoard, writes
`config.yaml`, saves object checkpoints through `ModelObjectCallBack`, and logs
`validate/ce_loss_taskbal` through `TaskBalancedCEMetric`.

`run_all4_pretrained_vision.sh` is the active server runner. Supported `ARM`
values:

- `all4_dinov2_frozen`
- `all4_sp_sigreg_dinov2_frozen`
- `all4_sp_sigreg_dinov2_frozen_visual17`
- `all4_sp_sigreg_dinov2_frozen_visual17_sep_proj`
- `all4_sp_sigreg_dinov2_frozen_visual17_patch_sp`
- `all4_sp_sigreg_dinov2_frozen_mot`

The runner requires a flat processed dir with at least 40 `.h5` files, a FAST
tokenizer dir, a local HF vision model dir, and conda env `vla`. It trains,
uses `pick_best_ckpt.py`, then evaluates `libero_spatial`, `libero_object`,
`libero_goal`, and `libero_10` with the same `SEED`.

`eval_libero.py` supports `_object.ckpt` full-model checkpoints and baseline
weights checkpoints. Weights loading rejects SP, BatchNorm projector,
gripper-aux, and visual-prefix checkpoints unless they are object checkpoints.

Evaluation preprocesses current observations, encodes, generates FAST tokens,
decodes/denormalizes actions, optionally applies gripper aux override, then
executes chunks closed-loop. Closed-loop execution snapshots the anchor pose at
chunk start and sends residual deltas from current robot pose to each predicted
anchor-relative target.

## Commands

Install:

```bash
conda activate vla
pip install -r requirements.txt
```

Preprocess all suites:

```bash
bash preprocess_all4.sh
```

Train/eval current 4-suite patch-SP DINOv2 run:

```bash
ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp \
SEED=3072 \
CUDA_VISIBLE_DEVICES=1 \
bash run_all4_pretrained_vision.sh
```

Local checks:

```bash
conda run -n vla ruff check .
conda run -n vla python -B -m unittest \
  test_visual_tokens.py test_attn_mask.py test_mot_predictor.py test_repo_contracts.py
bash -n preprocess_all4.sh run_all4_pretrained_vision.sh
git diff --check
```

## Coding And PR Guidance

Use 4-space indentation, `snake_case` Python names, uppercase token constants,
and explicit environment overrides in shell scripts. Keep commits focused and
use the existing Conventional Commit style (`feat:`, `fix:`, `chore:`).
Training/eval PRs should include config paths, commands run, checkpoint/result
paths, and metrics or logs.

## Safety Rules

- Never write into `/nas_data_new/caz/data_ssd/libero`.
- Keep checkpoints, TensorBoard logs, videos, HDF5 outputs, and Hydra outputs
  out of git.
- Keep `run_all4_pretrained_vision.sh` and `config/train/*` synchronized.
- For SIGReg runs, keep `trainer.devices=1` unless SyncBatchNorm support is
  explicitly added.
- Use object checkpoints for SP, BatchNorm projector, visual-token, MoT, or
  gripper-aux models.
