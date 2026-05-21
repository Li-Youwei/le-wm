import logging
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)


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
        save_on_train_step_end: bool = False,
        rank_by_step: bool = False,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self.top_k = max(1, int(top_k))
        self.save_on_train_step_end = bool(save_on_train_step_end)
        self.rank_by_step = bool(rank_by_step)
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
        if self.save_on_train_step_end:
            return
        if not trainer.is_global_zero:
            return
        self._save_step_checkpoint(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.step_interval is None or not self.save_on_train_step_end:
            return
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step == 0 or step % int(self.step_interval) != 0:
            return
        self._save_step_checkpoint(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if self.step_interval is None or not self.save_on_train_step_end:
            return
        if not trainer.is_global_zero:
            return
        # Ensure non-multiple max_steps runs still leave an eval-ready final
        # object checkpoint.
        self._save_step_checkpoint(trainer, pl_module)

    def _save_step_checkpoint(self, trainer, pl_module) -> None:
        step = int(trainer.global_step)
        # Skip Lightning's sanity-check validation pass (step==0).
        if step == 0:
            return
        # Use task-balanced CE for ranking; fall back to global validate/ce_loss
        # if the TaskBalancedCEMetric callback hasn't fired yet (e.g., very
        # first val pass).
        metrics = trainer.callback_metrics
        if self.rank_by_step:
            score = -float(step)
        else:
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
        self._top_k_heap = [
            entry for entry in self._top_k_heap if entry[2] != path
        ]
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
            sync_dist=True,
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
