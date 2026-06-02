import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from stable_pretraining import data as dt

logger = logging.getLogger(__name__)


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer


class ModelObjectCallBack(Callback):
    """Pickle the model object on disk for downstream eval_libero.py loading.

    Two modes:
      - **Per-epoch (legacy)**: fires on_train_epoch_end every ``epoch_interval``
        epochs. Used by spatial-only runs that train epoch-bounded.
      - **Per-step (4-suite joint)**: when ``step_interval`` is set, fires on
        each validation end (which Lightning triggers every
        ``trainer.val_check_interval`` steps). Maintains the top-K checkpoints
        by ``validate/ce_loss_taskbal`` and deletes worse-than-K files so disk
        usage stays bounded.

    `_object.ckpt` filename is required by eval_libero.py for SP / BN-projector
    architectures (state_dict mode rejects those — see eval_libero.py:131-145).
    """

    def __init__(
        self,
        dirpath,
        filename: str = "model_object",
        epoch_interval: int = 1,
        step_interval: int | None = None,
        top_k: int = 3,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self.top_k = max(1, int(top_k))
        # (score, step, path) — kept sorted by score ascending; lower CE = better.
        self._top_k_heap: list[tuple[float, int, Path]] = []

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        # In step-based mode, skip the per-epoch save (we save on val end instead).
        if self.step_interval is not None:
            return

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def on_validation_end(self, trainer, pl_module):
        if self.step_interval is None:
            return
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        # Skip Lightning's sanity-check validation pass (step==0).
        if step == 0:
            return
        # Use task-balanced CE for ranking; fall back to global validate/ce_loss
        # if the TaskBalancedCEMetric callback hasn't fired yet (e.g., very
        # first val pass).
        metrics = trainer.callback_metrics
        score_val = metrics.get("validate/ce_loss_taskbal")
        if score_val is None:
            score_val = metrics.get("validate/ce_loss_epoch")
        score = float(score_val) if score_val is not None else float("inf")

        path = self.dirpath / f"{self.filename}_step_{step}_object.ckpt"
        self._dump_model(pl_module.model, path)
        self._update_top_k(score, step, path)
        # Re-point `lewm_latest_object.ckpt` at the highest-step file still on
        # disk after eviction. Previously the symlink was always set to the
        # just-saved `path`, which left a dangling symlink whenever top-K
        # evicted that file (e.g., when the latest step had the worst val CE
        # — typical near the end of an overfit run).
        self._refresh_latest_link()

    def _update_top_k(self, score: float, step: int, path: Path) -> None:
        self._top_k_heap.append((score, step, path))
        # Sort ascending — worst (highest CE) at the end.
        self._top_k_heap.sort(key=lambda x: (x[0], x[1]))
        while len(self._top_k_heap) > self.top_k:
            _bad_score, _bad_step, bad_path = self._top_k_heap.pop()
            try:
                bad_path.unlink()
            except FileNotFoundError:
                pass

    def _refresh_latest_link(self) -> None:
        """Point ``_latest_object.ckpt`` at the highest-step ckpt still on disk.

        The "latest" symlink invariant: if it exists, its target must exist.
        After top-K eviction the just-saved file may be gone; in that case the
        symlink falls back to the surviving member with the largest training
        step (i.e., the most-recently-saved file that is still kept by top-K).
        """
        latest_link = self.dirpath / f"{self.filename}_latest_object.ckpt"
        try:
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
        except OSError:
            pass
        if not self._top_k_heap:
            return
        # Pick the survivor with the largest step (most recent in time).
        most_recent = max(self._top_k_heap, key=lambda triple: triple[1])
        target = most_recent[2]
        try:
            latest_link.symlink_to(target.name)
        except OSError:
            pass

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            logger.error("Error saving model object to %s: %s", path, e)


class TaskBalancedCEMetric(Callback):
    """At validation epoch end, mean the per-task val CE scalars into
    ``validate/ce_loss_taskbal``.

    `lejepa_forward` logs ``validate/ce_loss/task_<i>`` per task that appeared
    in the validation set. Here we aggregate those (each task weighted equally)
    so the saved scalar is robust to suite size imbalance — critical under
    4-suite joint training where some tasks have ~3× more val samples than
    others. Used by `pick_best_ckpt.py` for ckpt selection.
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        per_task_vals = [
            v
            for k, v in trainer.callback_metrics.items()
            if k.startswith("validate/ce_loss/task_")
        ]
        if not per_task_vals:
            return
        try:
            tensors = [
                v.float() if isinstance(v, torch.Tensor) else torch.tensor(float(v))
                for v in per_task_vals
            ]
            mean_ce = torch.stack(tensors).mean()
        except Exception as exc:  # noqa: BLE001
            logger.warning("TaskBalancedCEMetric failed to aggregate: %s", exc)
            return
        pl_module.log(
            "validate/ce_loss_taskbal",
            mean_ce,
            on_step=False,
            on_epoch=True,
            # sync_dist=False is REQUIRED: this log runs INSIDE the
            # `if not trainer.is_global_zero: return` guard above, so only rank 0
            # reaches it. sync_dist=True would make ONLY rank 0 issue an
            # all_reduce that the other ranks never join → NCCL collective
            # mismatch (Float value/Long count vs the other ranks' next
            # collective) → DDP hang. The val loader is unsharded
            # (use_distributed_sampler=False) so rank 0 already aggregated the
            # FULL val set's per-task CE; its local value IS the global one.
            sync_dist=False,
        )


class EarlyProbeCallback(Callback):
    """Stage A in-training health probe.

    On selected training steps (default: 20000), runs a subprocess that calls
    ``quick_probe_eval.py`` on the latest saved object checkpoint. The probe
    rolls out 1 episode × 10 tasks per suite (40 rollouts total) and reports
    a per-suite + overall success rate. Logged to TB as
    ``validate/probe_success_rate_overall`` and four
    ``validate/probe_success/{spatial,object,goal,long}`` scalars.

    Does NOT auto-terminate training on poor probe results — caller reviews
    the log and the TB scalar manually. Logging ``validate/probe_health=0``
    flags suspicious runs (overall < 0.05 and CE not decreasing).
    """

    SUITES: tuple[str, ...] = (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    )

    def __init__(
        self,
        trigger_steps: Sequence[int] = (20000,),
        tokenizer_path: str = "",
        processed_root: str = "",
        ckpt_dir: str = "",
        script_path: str = "quick_probe_eval.py",
    ) -> None:
        super().__init__()
        self.trigger_steps = set(int(s) for s in trigger_steps)
        self.tokenizer_path = tokenizer_path
        self.processed_root = processed_root  # /data/lyw/libero_processed_v5
        self.ckpt_dir = Path(ckpt_dir)
        self.script_path = script_path
        self._fired: set[int] = set()
        # Track the most recent CE for the "flat CE" heuristic.
        self._ce_history: list[tuple[int, float]] = []

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        ce_taskbal = trainer.callback_metrics.get("validate/ce_loss_taskbal")
        if ce_taskbal is not None:
            self._ce_history.append((step, float(ce_taskbal)))

        # Trigger on the validation end that crosses each scheduled step.
        for trig in sorted(self.trigger_steps):
            if step >= trig and trig not in self._fired:
                self._fired.add(trig)
                self._run_probe(trainer, pl_module, trig, step)

    def _run_probe(self, trainer, pl_module, trig: int, step: int) -> None:
        latest_ckpt = self.ckpt_dir / "lewm_latest_object.ckpt"
        if not latest_ckpt.exists():
            # Fallback: find newest *_object.ckpt by mtime.
            candidates = sorted(
                self.ckpt_dir.glob("lewm_step_*_object.ckpt"),
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
            )
            if not candidates:
                logger.warning(
                    "EarlyProbe @ step %d: no checkpoint found in %s",
                    step,
                    self.ckpt_dir,
                )
                return
            latest_ckpt = candidates[-1]
        torch.cuda.empty_cache()
        logger.info("EarlyProbe @ trigger=%d step=%d: %s", trig, step, latest_ckpt)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    self.script_path,
                    "--checkpoint",
                    str(latest_ckpt),
                    "--tokenizer",
                    self.tokenizer_path,
                    "--processed-root",
                    self.processed_root,
                    "--stage",
                    "A",
                    "--num-episodes",
                    "1",
                    "--max-steps",
                    "200",
                ],
                capture_output=True,
                text=True,
                timeout=60 * 60,  # 1h cap
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.error("EarlyProbe @ step %d timed out", step)
            return
        if proc.returncode != 0:
            logger.error(
                "EarlyProbe @ step %d exit=%d stderr=%s",
                step,
                proc.returncode,
                proc.stderr[-500:],
            )
            return
        # Parse the last JSON line from probe stdout.
        result_line = ""
        for line in proc.stdout.strip().splitlines()[::-1]:
            if line.startswith("{") and line.endswith("}"):
                result_line = line
                break
        if not result_line:
            logger.error("EarlyProbe @ step %d: no JSON in stdout", step)
            return
        try:
            result = json.loads(result_line)
        except json.JSONDecodeError as exc:
            logger.error("EarlyProbe @ step %d: bad JSON: %s", step, exc)
            return
        overall = float(result.get("overall_success_rate", 0.0))
        # Lightning forbids `self.log()` inside `on_validation_end` (it lives
        # outside the train/val step loop, where the logger-connector state
        # is locked). Write the probe scalars directly via the TB writer.
        tb = (
            trainer.logger.experiment
            if trainer.logger is not None and hasattr(trainer.logger, "experiment")
            else None
        )

        def _tb_scalar(tag: str, val: float) -> None:
            if tb is not None and hasattr(tb, "add_scalar"):
                tb.add_scalar(tag, val, global_step=step)

        _tb_scalar("validate/probe_success_rate_overall", overall)
        for suite in self.SUITES:
            suite_key = suite.replace("libero_", "").replace("10", "long")
            val = float(result.get("per_suite", {}).get(suite, 0.0))
            _tb_scalar(f"validate/probe_success/{suite_key}", val)
        # Health flag: overall < 0.05 AND CE not improving over last 3 vals.
        ce_flat = False
        if len(self._ce_history) >= 4:
            recent = [v for _, v in self._ce_history[-4:]]
            ce_flat = max(recent) - min(recent) < 0.02
        health = 0.0 if (overall < 0.05 and ce_flat) else 1.0
        _tb_scalar("validate/probe_health", health)
        if health == 0.0:
            logger.warning(
                "EarlyProbe @ step %d: probe_health=0 (overall=%.3f, ce flat). "
                "Review training before letting it burn remaining steps.",
                step,
                overall,
            )


class PeriodicPrintCallback(Callback):
    """Print key training metrics to stdout every N epochs.

    Complements the TensorBoard logger during long-running sanity/overfit
    experiments where you want a compact progress line in the terminal at a
    fixed cadence (instead of Lightning's default per-epoch progress bar).

    Args:
        every_n_epochs: print cadence in epochs.
        keys: metric keys to print. Missing keys are silently skipped. The
              default matches the keys logged by `lejepa_forward` in train.py.
        tag: short prefix shown at the start of each line.
    """

    def __init__(
        self,
        every_n_epochs: int = 100,
        keys: tuple[str, ...] = (
            "fit/ce_loss_epoch",
            "fit/token_accuracy_epoch",
            "validate/ce_loss_epoch",
            "validate/token_accuracy_epoch",
        ),
        tag: str = "overfit",
    ) -> None:
        super().__init__()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.keys = tuple(keys)
        self.tag = tag

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        is_last = trainer.max_epochs is not None and epoch == trainer.max_epochs
        if epoch % self.every_n_epochs != 0 and not is_last:
            return

        metrics = trainer.callback_metrics
        parts = [f"epoch={epoch:>5d}"]
        for key in self.keys:
            if key in metrics:
                val = metrics[key]
                try:
                    val = float(val)
                    short = key.split("/")[-1].replace("_epoch", "")
                    parts.append(f"{short}={val:.4f}")
                except (TypeError, ValueError):
                    continue
        print(f"[{self.tag}] " + "  ".join(parts), flush=True)
