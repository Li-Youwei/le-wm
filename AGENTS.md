# Repository Guidelines

This document is the current contributor and agent guide for this repository.
It was checked against the files in this worktree, not against remembered
experiment history. Treat code and config files as the source of truth when a
future change diverges from this document.

## Current Repository Scope

This is a Python 3.10 research repo for LIBERO VLA experiments. The active
policy takes:

- two RGB views: `agentview_rgb` and `eye_in_hand_rgb`
- 9D proprioception: `ee_pos(3) + xyzw_quat(4) + gripper_states(2)`
- an optional language instruction encoded by frozen `t5-small`
- FAST action tokens decoded into a 20-step LIBERO action chunk

The old original-LeWM planning entrypoints and non-LIBERO configs have been
removed. The current repo does not contain `eval.py`, `config/eval/`,
`slides/`, `assets/`, `smoke_test.py`, or the old `run_all4.sh` family.

## Is `config/train` Still Used?

Yes. `train.py` uses Hydra directly:

```python
@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
```

The active configs are:

- `config/train/lewm.yaml`: default train config.
- `config/train/overfit.yaml`: one-demo pipeline sanity config.
- `config/train/data/libero.yaml`: dataset schema included by both configs via
  `defaults: - data: libero`.

`run_all4_pretrained_vision.sh` calls `python train.py data=libero` and passes
Hydra overrides for the frozen HuggingFace vision backbone, loss weights,
predictor architecture, seed, scheduler warmup, step budget, and visual-token
settings. Deleting `config/train` would break normal training.

### Current Config Defaults

From `config/train/lewm.yaml` and `config/train/data/libero.yaml`:

- `seed: 3072`
- `img_size: 224`, `patch_size: 14`, `encoder_scale: tiny`
- dataset `max_action_tokens: 80`, `max_lang_tokens: 25`, `chunk_size: 20`,
  `proprio_dim: 9`, `use_language: true`
- trainer `accelerator: gpu`, `precision: bf16`, `gradient_clip_val: 1.0`,
  `max_epochs: 100`, `devices: auto`
- loader `batch_size: 128`, `num_workers: ${num_workers}`,
  `persistent_workers: true`, `prefetch_factor: 3`, `pin_memory: true`
- optimizer `AdamW`, `lr: 5e-5`, `weight_decay: 0.05`
- scheduler `warmup_steps: null`; `train.py` resolves this to an explicit
  warmup from total step budget unless the runner overrides it
- predictor depth/heads/dim_head/mlp_dim/dropout:
  `6 / 16 / 64 / 2048 / 0.2`
- `projector.norm_type: layer`
- loss defaults: `pred_weight: 0.0`, `sigreg_weight: 0.0`,
  `gripper_aux_weight: 0.0`, SIGReg kwargs `knots=17`, `num_proj=1024`

`config/train/overfit.yaml` keeps the same model shape but sets
`overfit_demo: 0`, `train_split: 1.0`, `batch_size: 8`, `lr: 1e-3`,
`weight_decay: 0.0`, predictor dropout `0.0`, `max_epochs: 2000`, and
`trainer.devices: 1`.

## File Inventory

| File | Current role |
|------|--------------|
| `.gitignore` | Excludes generated artifacts such as caches, checkpoints, logs, videos, and local outputs. |
| `AGENTS.md` | Single contributor/agent guide with architecture, data, training, and file-role reference. |
| `LICENSE` | Project license. |
| `README.md` | Short user-facing overview and commands for the current LIBERO workflow. |
| `requirements.txt` | Python dependencies. Keeps `transformers>=4.48,<5` for FAST/T5 compatibility. |
| `config/train/lewm.yaml` | Main Hydra config: trainer, loader, optimizer, scheduler, projector, loss, vision encoder, predictor. |
| `config/train/overfit.yaml` | Debug Hydra config for one-demo overfit/memorization checks. |
| `config/train/data/libero.yaml` | LIBERO dataset config: HDF5 dir, max token lengths, chunk size, image size, proprio dim, language/SP switches. |
| `train.py` | Hydra + Lightning training entrypoint, data split/sampler, model construction, losses, logging, callbacks. |
| `jepa.py` | `JEPA` container: visual/language encoding, future visual encoding, action generation wrappers, gripper aux wrapper. |
| `module.py` | Core neural modules: transformer blocks, MoT blocks, SIGReg, `MLP`, token constants, `ARPredictor`. |
| `vision_backbone.py` | Builds either the original SPT ViT or a HuggingFace vision model returning `last_hidden_state`. |
| `libero_dataset.py` | Training dataset over preprocessed HDF5 task files, T5 tokenization, image preprocessing, sampler weights. |
| `preprocess_libero.py` | Converts one raw LIBERO HDF5 task into chunk-level HDF5 with images, proprio, actions, FAST tokens, and future SP targets. |
| `fit_tokenizer_all4.py` | Fits one shared FAST tokenizer over all four LIBERO suites using per-task normalized chunks. |
| `preprocess_all4.sh` | Server script to run `preprocess_libero.py` across the 40 LIBERO tasks. Refuses writes inside the raw LIBERO root. |
| `fast_utils.py` | Loads FAST processor, decodes generated tokens, and denormalizes actions with zero-range handling. |
| `eval_libero.py` | LIBERO rollout evaluation with closed-loop chunk execution, optional videos, object checkpoint loading, and baseline weights loading. |
| `run_all4_pretrained_vision.sh` | Active 4-suite train/eval runner for frozen HuggingFace vision backbones. |
| `pick_best_ckpt.py` | Reads TensorBoard scalars, selects best object checkpoint by validation metric, emits JSON. |
| `select_light_eval_ckpt.py` | Reads light rollout-eval logs for CE top-K checkpoints and selects the highest 4-suite success-rate checkpoint. |
| `aggregate_all4_results.py` | Parses per-suite eval logs and renders a markdown summary. |
| `verify_sampler_balance.py` | Checks 3-level weighted sampler behavior over a flat 4-suite HDF5 directory. |
| `check_fast_roundtrip.py` | Recomputes GT chunks from raw/preprocessed data and checks FAST encode/decode roundtrip error. |
| `utils.py` | Lightning callbacks for object checkpoints, task-balanced CE aggregation, and periodic overfit logging. |
| `test_visual_tokens.py` | CPU tests for visual-token projection and patch-level SP contracts. |
| `test_mot_predictor.py` | CPU tests for the MoT-style predictor path. |
| `test_attn_mask.py` | Standalone attention-mask diagnostic script. |
| `test_repo_contracts.py` | Lightweight repository invariant tests. |

## Data Pipeline

### Raw Data Boundaries

Raw LIBERO is expected on the server at `/nas_data_new/caz/data_ssd/libero`.
Treat that path as read-only. The scripts that write derived files default to
`/Data/lyw/...` paths:

- `fit_tokenizer_all4.py` reads raw HDF5 files and writes tokenizer/audit files.
- `preprocess_all4.sh` reads raw HDF5 files and writes processed `.h5` files.
- `preprocess_all4.sh` aborts if `OUT_ROOT` is inside the raw LIBERO root.

### Preprocessing One Task

`preprocess_libero.py` reads one raw LIBERO task HDF5 and writes one processed
HDF5. The saved datasets include:

- `image_agent`: current agentview image at chunk anchor `t`
- `image_hand`: current eye-in-hand image at chunk anchor `t`
- `image_agent_future`: future agentview image at `t + H`
- `image_hand_future`: future eye-in-hand image at `t + H`
- `proprio`: current 9D proprio at `t`
- `proprio_future`: future 9D proprio at `t + H`
- `continuous_actions`: normalized `(N, H, 7)` anchor-relative action chunks
- `fast_tokens`: variable-length FAST token arrays
- `fast_length`: token lengths
- `demo_idx`, `chunk_idx`: demo id and raw-step anchor index

Important attributes include `action_low`, `action_high`, `chunk_size`,
`chunk_stride`, `action_dim`, `language_instruction`, and `language_source`.

### Temporal Convention

All indices are raw environment steps. With chunk length `H=20`, a sample
anchored at raw step `t` uses:

- observation/proprio at `t`
- future SP target at `t + H`
- action chunk steps `k=0..H-1`
- pose targets from observed poses at `t+k+1`

Valid anchors satisfy `t <= T - H - 1`, because preprocessing needs
`obs[t+H]` to compute the last anchor-relative target and future SP target.
The default stride is `1`.

### Action Chunk Definition

For each chunk element `k`:

```text
chunk[k, 0:3] = obs/ee_pos[t+k+1] - obs/ee_pos[t]
chunk[k, 3:6] = rotvec(R[t+k+1] * inv(R[t]))
chunk[k, 6]   = actions[t+k, 6]
```

Position and rotation chunks are anchor-relative physical targets. The gripper
channel is the raw command from the demo action, not an anchor-relative pose.
After extraction, each task computes per-dimension 1st/99th percentile bounds
and normalizes chunks to `[-1, 1]` before FAST tokenization.

### Proprioception Contract

`normalize_proprio()` in `preprocess_libero.py` is the single source of truth.
Both training data and eval observations use the same 9D layout:

```text
ee_pos(3) + xyzw_quat(4) + raw gripper finger positions(2)
```

The quaternion slice `[3:7]` is renormalized to unit length. Positions and the
two gripper dimensions are passed through unchanged. Do not average or take
absolute values over the two gripper finger positions.

### Dataset Loading

`LiberoDataset` accepts a directory of `.h5` or `.hdf5` files. Each file is a
task. Files are sorted, and that sorted order defines stable `task_id` values.
The dataset:

- lazy-opens one HDF5 handle per worker
- resizes images to `img_size` and applies ImageNet normalization
- pads FAST tokens to `max_action_tokens` using `PAD_TOKEN_ID`
- loads `continuous_actions[:, :, 6]` as `gripper_seq`
- tokenizes one language instruction per HDF5 file when language is enabled
- asserts future fields exist when state prediction is enabled
- provides 3-level sampler weights: task, demo within task, and chunk within demo

## Model Architecture

### Encoders

`train.py` constructs the visual encoder via `build_visual_encoder()`:

- `vision_encoder.source=spt_vit`: uses `stable_pretraining` ViT factory.
- `vision_encoder.source=hf`: uses `AutoModel.from_pretrained(...)` wrapped by
  `HFVisionBackbone`.
- `vision_encoder.freeze=true`: keeps the visual encoder in eval mode and
  disables gradients.

`JEPA.encode()` concatenates agentview and hand images along batch dimension,
runs one shared visual backbone call, then splits the result back into two
views. Language uses frozen `T5EncoderModel.from_pretrained("t5-small")` when
`data.dataset.use_language=true`; T5 hidden size is projected to `embed_dim`.

### Visual Tokens

`visual_tokens.pool_grid` is read from Hydra as an optional key:

- `0`: CLS-only, one visual token per view.
- `G > 0`: CLS plus `G*G` adaptive-average-pooled patch tokens per view.

With `pool_grid=4`, each view contributes `1 + 16 = 17` visual tokens. CLS is
projected by `projector`; patch tokens use `patch_projector` when constructed.
This keeps CLS BatchNorm statistics separate from patch-token statistics.

### Predictor Sequence

The training sequence is:

```text
[language tokens,
 agent visual tokens,
 hand visual tokens,
 proprio token,
 BOS,
 fixed-width action token slots,
 optional Q_ag, Q_hd, Q_pr]
```

Action logits are produced only for `BOS + action token slots`; prefix tokens
and SP query tokens do not go through `action_head`.

Special token constants in `module.py`:

- FAST action IDs: `0..1023`
- `BOS_TOKEN_ID = 1024`
- `EOS_TOKEN_ID = 1025`
- `PAD_TOKEN_ID = 1026`
- action embedding size: `1027`
- action head size: `1026`, so PAD is never predicted

### Attention Mask

`ARPredictor._build_attn_mask()` builds a boolean `(B, 1, L, L)` mask:

- real prefix tokens attend bidirectionally to real prefix tokens
- action tokens attend to real prefix tokens
- action tokens attend causally within the action zone
- language PAD and action PAD rows/columns are masked out
- SP queries, when present, attend to real prefix, real action tokens, and
  themselves only
- no non-query token attends to SP query tokens

Generation uses `_build_generate_mask()` without action PAD.

### Shared Transformer vs MoT-Style Transformer

`predictor.state_prediction_arch` supports:

- `shared`: standard transformer blocks in `Block`.
- `mot`: `MoTBlock`, where language, visual, proprio, and action tokens use
  modality-specific LayerNorms, QKV/output projections, and FFNs while attention
  is still computed globally over the whole sequence.

MoT modality IDs are built in `_build_train_modality_ids()` and
`_build_generate_modality_ids()`. SP queries route by target modality:
`Q_ag/Q_hd` as visual and `Q_pr` as proprio.

## Losses

### Action CE

`lejepa_forward()` always computes teacher-forced cross-entropy over action
positions. Targets are shifted:

- BOS position predicts first real FAST token
- each action-token position predicts the next FAST token
- position after the final real token predicts EOS
- PAD targets are `-100`

`cfg.label_smoothing` defaults to `0.1`.

### State Prediction

`loss.pred_weight > 0` enables state prediction. `train.py` auto-enables
`use_state_prediction` and `LiberoDataset` loads future fields. The predictor
returns `pred_ag`, `pred_hd`, and `pred_pr`.

By default visual SP predicts CLS latents. If `visual_tokens.patch_sp=true`,
the visual heads predict `(B, N, D)` so CLS and pooled patches receive direct
future-state supervision. CLS and patch losses are logged separately and
combined with `visual_tokens.patch_sp_weight`.

Future visual targets are produced by `JEPA.encode_future_visual()` through the
same shared visual backbone and projector. The current code does not detach
those targets.

### SIGReg

`loss.sigreg_weight > 0` constructs `SIGReg` and adds it to total loss.
`train.py` enforces `projector.norm_type=batch` when SIGReg is on. In the
current implementation SIGReg uses CLS visual streams only:

- without SP: current agent CLS and current hand CLS
- with SP: current agent/hand CLS plus future agent/hand CLS

Predictor outputs are not passed directly to SIGReg.

### Gripper Aux

`loss.gripper_aux_weight > 0` enables `use_gripper_aux`. The predictor creates
`gripper_aux_head`, reads from the BOS hidden state, and regresses the normalized
gripper command sequence `(B, H)` from `batch["gripper_seq"]`. During eval,
`eval_libero.py` detects `model.predictor.use_gripper_aux`, denormalizes the
aux output through `action_low[6]`/`action_high[6]`, and overwrites decoded
action dim 6.

Total loss is:

```text
L_total = L_CE
        + pred_weight * L_pred
        + sigreg_weight * L_sigreg
        + gripper_aux_weight * L_gripper_aux
```

## Training Flow

`train.py`:

1. seeds Lightning and a Torch generator from `cfg.seed`
2. builds `LiberoDataset`
3. splits by `(file_path, demo_idx)` so chunks from one demo do not cross train/val
4. uses `WeightedRandomSampler` outside overfit mode
5. constructs visual encoder, predictor, projectors, language encoder/projection
6. wraps `JEPA` in `stable_pretraining.Module`
7. uses `AdamW` and `LinearWarmupCosineAnnealingLR` on step interval
8. logs to TensorBoard under `<STABLEWM_HOME>/<subdir>/tb_logs`
9. saves config to `<run_dir>/config.yaml`
10. saves object checkpoints via `ModelObjectCallBack`
11. logs `validate/ce_loss_taskbal` via `TaskBalancedCEMetric`

When `trainer.max_steps` and `trainer.val_check_interval` are set, object
checkpoints are step-based (`lewm_step_<N>_object.ckpt`) and top-K pruning is
handled by `ModelObjectCallBack`. Otherwise it falls back to epoch-based object
checkpointing.

The active server runner is `run_all4_pretrained_vision.sh`. Its supported
`ARM` values are:

- `all4_dinov2_frozen`
- `all4_sp_sigreg_dinov2_frozen`
- `all4_sp_sigreg_dinov2_frozen_visual17`
- `all4_sp_sigreg_dinov2_frozen_visual17_sep_proj`
- `all4_sp_sigreg_dinov2_frozen_visual17_patch_sp`
- `all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot`
- `all4_sp_sigreg_dinov2_frozen_mot`

The runner requires:

- flat processed data dir with at least 40 `.h5` files
- shared FAST tokenizer dir
- local HuggingFace vision model dir
- conda env `vla`

It exports offline HuggingFace env vars by default, trains, selects CE top-K
checkpoints using `pick_best_ckpt.py`, runs light rollout evals for those
checkpoints, selects the best rollout checkpoint with `select_light_eval_ckpt.py`,
then evaluates all four suites with the same `SEED`. Control this with
`CKPT_SELECT_TOP_K`, `LIGHT_EVAL_EPISODES`, and `FINAL_EVAL_EPISODES`.

## Evaluation Flow

`eval_libero.py` supports two checkpoint formats:

- `_object.ckpt`: full pickled `JEPA`, loaded directly with
  `torch.load(..., weights_only=False)`.
- weights checkpoint: loads a baseline architecture built by `build_model()`.

The weights loader rejects checkpoints containing SP heads, BatchNorm projector
running stats, gripper-aux heads, or visual-prefix keys unless they are object
checkpoints, because `build_model()` constructs only the CLS-only baseline.

At runtime, eval:

1. loads FAST processor
2. loads T5 tokenizer unless `--no-language`
3. finds the processed HDF5 matching each LIBERO task
4. reads `action_low`, `action_high`, `chunk_size`, `action_dim`, and language
5. creates an `OffScreenRenderEnv`
6. preprocesses current observation with the same image/proprio path as training
7. encodes, generates FAST tokens, decodes, denormalizes
8. optionally overrides gripper dim with aux prediction
9. executes the action chunk closed-loop

Closed-loop execution snapshots the anchor pose at the start of a chunk. For
each predicted step, it computes the target anchor-relative pose, reads the
current robot pose, sends the residual delta scaled by runtime controller
scales, and clips inputs to `[-1, 1]`.

## Development Commands

Install:

```bash
conda activate vla
pip install -r requirements.txt
```

Preprocess all suites on the server:

```bash
bash preprocess_all4.sh
```

Train/eval current 4-suite DINOv2 patch-SP run:

```bash
ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp \
SEED=3072 \
CUDA_VISIBLE_DEVICES=1 \
bash run_all4_pretrained_vision.sh
```

Run local lightweight checks:

```bash
conda run -n vla ruff check .
conda run -n vla python -B -m unittest \
  test_visual_tokens.py test_attn_mask.py test_mot_predictor.py test_repo_contracts.py
bash -n preprocess_all4.sh run_all4_pretrained_vision.sh
git diff --check
```

## Safety Rules

- Do not write into `/nas_data_new/caz/data_ssd/libero`.
- Keep checkpoints, TensorBoard logs, videos, HDF5 outputs, and Hydra outputs
  out of git.
- Keep `run_all4_pretrained_vision.sh` and `config/train/*` synchronized; the
  runner depends on Hydra override keys existing or being added with `+`.
- For SIGReg runs, keep `trainer.devices=1` unless SyncBatchNorm support is
  explicitly added.
- Use object checkpoints for SP, BatchNorm projector, visual-token, MoT, or
  gripper-aux models.
