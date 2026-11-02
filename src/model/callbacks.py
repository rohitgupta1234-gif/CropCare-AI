"""
CropCare AI - Training Callbacks
Custom callbacks for training monitoring and control.
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger,
    LearningRateScheduler,
)
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import numpy as np
from datetime import datetime


def get_early_stopping(
    monitor: str = "val_loss",
    patience: int = 10,
    min_delta: float = 1e-4,
    mode: str = "min",
    restore_best_weights: bool = True,
    verbose: int = 1,
) -> EarlyStopping:
    """Get EarlyStopping callback."""
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        restore_best_weights=restore_best_weights,
        verbose=verbose,
    )


def get_reduce_lr_on_plateau(
    monitor: str = "val_loss",
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-7,
    min_delta: float = 1e-4,
    mode: str = "min",
    verbose: int = 1,
) -> ReduceLROnPlateau:
    """Get ReduceLROnPlateau callback."""
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        min_delta=min_delta,
        mode=mode,
        verbose=verbose,
    )


def get_model_checkpoint(
    filepath: Path,
    monitor: str = "val_accuracy",
    mode: str = "max",
    save_best_only: bool = True,
    save_weights_only: bool = False,
    verbose: int = 1,
) -> ModelCheckpoint:
    """Get ModelCheckpoint callback."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        filepath=str(filepath),
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        verbose=verbose,
    )


def get_csv_logger(
    filepath: Path,
    append: bool = False,
    separator: str = ",",
) -> CSVLogger:
    """Get CSVLogger callback."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return CSVLogger(
        filename=str(filepath),
        append=append,
        separator=separator,
    )


def get_tensorboard(
    log_dir: Path,
    histogram_freq: int = 1,
    write_graph: bool = True,
    write_images: bool = False,
    update_freq: str = "epoch",
) -> TensorBoard:
    """Get TensorBoard callback."""
    log_dir.mkdir(parents=True, exist_ok=True)
    return TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=histogram_freq,
        write_graph=write_graph,
        write_images=write_images,
        update_freq=update_freq,
    )


class TrainingLogger(Callback):
    """Custom callback for detailed training logging."""

    def __init__(self, log_dir: Path, log_every_n_batches: int = 10):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_batches = log_every_n_batches
        self.batch_logs = []
        self.epoch_logs = []

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if batch % self.log_every_n_batches == 0:
            log_entry = {"batch": batch, "timestamp": datetime.now().isoformat()}
            if logs:
                log_entry.update({k: float(v) for k, v in logs.items()})
            self.batch_logs.append(log_entry)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        log_entry = {"epoch": epoch, "timestamp": datetime.now().isoformat()}
        if logs:
            log_entry.update({k: float(v) for k, v in logs.items()})
        self.epoch_logs.append(log_entry)

        # Save periodically
        if epoch % 5 == 0:
            self.save_logs()

    def on_train_end(self, logs: Optional[Dict] = None):
        self.save_logs()

    def save_logs(self):
        with open(self.log_dir / "batch_logs.json", "w") as f:
            json.dump(self.batch_logs, f, indent=2)
        with open(self.log_dir / "epoch_logs.json", "w") as f:
            json.dump(self.epoch_logs, f, indent=2)


class ConfusionMatrixCallback(Callback):
    """Callback to compute and log confusion matrix on validation data."""

    def __init__(
        self,
        validation_data,
        class_names: List[str],
        log_dir: Path,
        frequency: int = 5,
    ):
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.frequency = frequency
        self.matrices = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.frequency == 0:
            # Get predictions
            val_x, val_y = self.validation_data
            predictions = self.model.predict(val_x, verbose=0)
            y_pred = np.argmax(predictions, axis=1)
            y_true = np.argmax(val_y, axis=1)

            # Compute confusion matrix
            cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(self.class_names))
            cm_np = cm.numpy()

            # Normalize
            cm_normalized = cm_np.astype(float) / cm_np.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized)

            matrix_data = {
                "epoch": epoch,
                "matrix": cm_np.tolist(),
                "normalized": cm_normalized.tolist(),
                "class_names": self.class_names,
            }
            self.matrices.append(matrix_data)

            # Save
            with open(self.log_dir / f"confusion_matrix_epoch_{epoch}.json", "w") as f:
                json.dump(matrix_data, f, indent=2)


class MetricsHistory(Callback):
    """Track and save detailed metrics history."""

    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if logs:
            for key, value in logs.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(float(value))

        # Save history
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)


class GradientAccumulation(Callback):
    """Gradient accumulation for effective larger batch sizes."""

    def __init__(self, accumulation_steps: int):
        super().__init__()
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
        self.gradients = None

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        self.step_counter += 1

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if self.step_counter % self.accumulation_steps == 0:
            # Apply accumulated gradients
            self.model.optimizer.apply_gradients(
                zip(self.gradients, self.model.trainable_variables)
            )
            self.gradients = None
        else:
            # Accumulate gradients (handled by custom training loop)
            pass


def get_standard_callbacks(
    checkpoint_dir: Path,
    log_dir: Path,
    csv_log_path: Path,
    monitor: str = "val_accuracy",
    mode: str = "max",
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-7,
) -> List[Callback]:
    """
    Get standard callback suite for training.
    """
    callbacks = [
        get_early_stopping(
            monitor=monitor.replace("accuracy", "loss") if "accuracy" in monitor else monitor,
            patience=early_stopping_patience,
            min_delta=1e-4,
            mode="min" if "loss" in monitor else mode,
        ),
        get_reduce_lr_on_plateau(
            monitor=monitor.replace("accuracy", "loss") if "accuracy" in monitor else monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode="min" if "loss" in monitor else mode,
        ),
        get_model_checkpoint(
            filepath=checkpoint_dir / "best_model.keras",
            monitor=monitor,
            mode=mode,
        ),
        get_model_checkpoint(
            filepath=checkpoint_dir / "epoch_{epoch:03d}_valacc_{val_accuracy:.4f}.keras",
            monitor=monitor,
            mode=mode,
            save_best_only=False,
        ),
        get_csv_logger(csv_log_path),
        get_tensorboard(log_dir / "tensorboard"),
        MetricsHistory(log_dir / "metrics_history.json"),
    ]

    return callbacks


def get_fine_tuning_callbacks(
    checkpoint_dir: Path,
    log_dir: Path,
    csv_log_path: Path,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 7,
) -> List[Callback]:
    """Get callbacks for fine-tuning phase (more patience)."""
    return get_standard_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        csv_log_path=csv_log_path,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_factor=0.3,
        min_lr=1e-8,
    )


class CosineAnnealingScheduler(Callback):
    """Cosine annealing learning rate scheduler."""

    def __init__(
        self,
        initial_lr: float,
        min_lr: float,
        total_epochs: int,
        warmup_epochs: int = 0,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)


if __name__ == "__main__":
    # Test callbacks
    from pathlib import Path
    callbacks = get_standard_callbacks(
        checkpoint_dir=Path("./test_checkpoints"),
        log_dir=Path("./test_logs"),
        csv_log_path=Path("./test_logs/training.csv"),
    )
    print(f"Created {len(callbacks)} standard callbacks")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}")
    print("Callbacks test passed!")