"""
CropCare AI - Model Training Script
Trains MobileNetV2-based classifier on PlantVillage dataset.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard
)
from datetime import datetime

from src.config import (
    PROCESSED_DATA_DIR,
    TRAINED_MODELS_DIR,
    CHECKPOINTS_DIR,
    METADATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    TRAINING_CONFIG,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    TRAINING_HISTORY_PATH,
    CLASS_LABELS_PATH,
    PREPROCESSING_CONFIG_PATH,
)
from src.model.architecture import create_model, compile_model, save_model_artifacts
from src.model.callbacks import get_standard_callbacks
from src.data.augmentation import get_training_augmentation, get_validation_augmentation


def create_data_generators(
    data_dir: Path,
    batch_size: int = BATCH_SIZE,
    target_size: tuple = IMAGE_SIZE,
    augmentation_severity: str = "medium",
):
    """Create train/val/test data generators."""

    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x,  # We'll use albumentations separately
    )

    # Validation/test generator (no augmentation)
    val_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: x,
    )

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir / "train",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True,
        seed=TRAINING_CONFIG['random_seed'],
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir / "val",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
        seed=TRAINING_CONFIG['random_seed'],
    )

    test_generator = val_datagen.flow_from_directory(
        data_dir / "test",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
        seed=TRAINING_CONFIG['random_seed'],
    )

    return train_generator, val_generator, test_generator


def create_tf_datasets(
    data_dir: Path,
    batch_size: int = BATCH_SIZE,
    target_size: tuple = IMAGE_SIZE,
    augmentation_severity: str = "medium",
):
    """Create TensorFlow datasets with albumentations augmentation."""

    train_aug = get_training_augmentation(target_size, augmentation_severity)
    val_aug = get_validation_augmentation(target_size)

    def augment_train(image, label):
        # Convert tensor to numpy for albumentations
        image_np = image.numpy()
        augmented = train_aug(image=image_np)
        return augmented['image'], label

    def augment_val(image, label):
        image_np = image.numpy()
        augmented = val_aug(image=image_np)
        return augmented['image'], label

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        batch_size=batch_size,
        image_size=target_size,
        shuffle=True,
        seed=TRAINING_CONFIG['random_seed'],
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False,
        seed=TRAINING_CONFIG['random_seed'],
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False,
        seed=TRAINING_CONFIG['random_seed'],
    )

    # Unbatch, apply augmentation per image, then re-batch
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.unbatch().map(
        lambda x, y: tf.py_function(augment_train, [x, y], [tf.float32, tf.float32]),
        num_parallel_calls=AUTOTUNE
    ).batch(batch_size).prefetch(AUTOTUNE)

    val_ds = val_ds.unbatch().map(
        lambda x, y: tf.py_function(augment_val, [x, y], [tf.float32, tf.float32]),
        num_parallel_calls=AUTOTUNE
    ).batch(batch_size).prefetch(AUTOTUNE)

    test_ds = test_ds.unbatch().map(
        lambda x, y: tf.py_function(augment_val, [x, y], [tf.float32, tf.float32]),
        num_parallel_calls=AUTOTUNE
    ).batch(batch_size).prefetch(AUTOTUNE)

    # Set shapes
    def set_shapes(images, labels):
        images.set_shape([None, *target_size, 3])
        labels.set_shape([None, NUM_CLASSES])
        return images, labels

    train_ds = train_ds.map(set_shapes)
    val_ds = val_ds.map(set_shapes)
    test_ds = test_ds.map(set_shapes)

    return train_ds, val_ds, test_ds


def train_model(
    data_dir: Path = PROCESSED_DATA_DIR,
    epochs: int = None,
    batch_size: int = BATCH_SIZE,
    initial_lr: float = None,
    fine_tune: bool = True,
    fine_tune_epochs: int = 20,
    fine_tune_lr: float = None,
    augmentation_severity: str = "medium",
    checkpoint_dir: Path = CHECKPOINTS_DIR,
    log_dir: Path = None,
):
    """Train the model."""

    if epochs is None:
        epochs = TRAINING_CONFIG['epochs']
    if initial_lr is None:
        initial_lr = TRAINING_CONFIG['initial_lr']
    if fine_tune_lr is None:
        fine_tune_lr = TRAINING_CONFIG['fine_tune_lr']
    if log_dir is None:
        log_dir = METADATA_DIR / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("CropCare AI - Model Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial LR: {initial_lr}")
    print(f"Fine-tune: {fine_tune}")
    print(f"Augmentation: {augmentation_severity}")

    # Verify data exists
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing data split: {split_dir}. Run preprocess_data.py first.")

    # Create datasets
    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_tf_datasets(
        data_dir, batch_size, IMAGE_SIZE, augmentation_severity
    )

    # Create model
    print("\nCreating model...")
    model = create_model()
    model = compile_model(model, learning_rate=initial_lr)

    print(f"Model parameters: {model.count_params():,}")

    # Callbacks
    callbacks = get_standard_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        csv_log_path=log_dir / "training.csv",
    )

    # Phase 1: Train head only (base frozen)
    print("\n" + "=" * 60)
    print("Phase 1: Training Classification Head (Base Frozen)")
    print("=" * 60)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Fine-tuning
    if fine_tune:
        print("\n" + "=" * 60)
        print("Phase 2: Fine-tuning Top Layers")
        print("=" * 60)

        # Unfreeze top layers
        from src.model.architecture import unfreeze_top_layers
        model = unfreeze_top_layers(model, num_layers=30)

        # Recompile with lower learning rate
        model = compile_model(model, learning_rate=fine_tune_lr)

        # Fine-tuning callbacks (more patience)
        from src.model.callbacks import get_fine_tuning_callbacks
        ft_callbacks = get_fine_tuning_callbacks(
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            csv_log_path=log_dir / "fine_tuning.csv",
        )

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=ft_callbacks,
            verbose=1,
        )

        # Combine histories
        combined_history = {}
        for key in history1.history:
            combined_history[key] = history1.history[key] + history2.history.get(key, [])
    else:
        combined_history = history1.history

    # Save final model
    print("\nSaving final model...")
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to: {FINAL_MODEL_PATH}")

    # Save training history
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_HISTORY_PATH, 'w') as f:
        # Convert numpy types to Python types
        serializable_history = {}
        for key, values in combined_history.items():
            serializable_history[key] = [float(v) for v in values]
        json.dump(serializable_history, f, indent=2)
    print(f"Training history saved to: {TRAINING_HISTORY_PATH}")

    # Save model artifacts
    save_model_artifacts(model)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    test_results = model.evaluate(test_ds, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))
    print(f"Test Results: {test_metrics}")

    return model, combined_history, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train CropCare AI model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--no-fine-tune", action="store_true", help="Skip fine-tuning phase")
    parser.add_argument("--ft-epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--ft-lr", type=float, default=None, help="Fine-tuning learning rate")
    parser.add_argument("--augmentation", type=str, default="medium",
                       choices=["light", "medium", "heavy"], help="Augmentation severity")
    parser.add_argument("--data-dir", type=str, default=None, help="Processed data directory")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROCESSED_DATA_DIR

    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, using CPU")

    model, history, test_metrics = train_model(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        fine_tune=not args.no_fine_tune,
        fine_tune_epochs=args.ft_epochs,
        fine_tune_lr=args.ft_lr,
        augmentation_severity=args.augmentation,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Final model: {FINAL_MODEL_PATH}")
    print(f"Test accuracy: {test_metrics.get('accuracy', 0)*100:.2f}%")


if __name__ == "__main__":
    main()