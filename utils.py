import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

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
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

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

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


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