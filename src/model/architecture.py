"""
CropCare AI - Model Architecture
MobileNetV2-based classifier with transfer learning.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import MobileNetV2
from typing import Optional, Tuple, Dict, Any
import json
from pathlib import Path

from src.config import (
    INPUT_SHAPE,
    NUM_CLASSES,
    MOBILENETV2_ALPHA,
    MOBILENETV2_WEIGHTS,
    DROPOUT_RATE_1,
    DROPOUT_RATE_2,
    DENSE_UNITS,
    CLASS_NAMES,
    METADATA_DIR,
    CLASS_LABELS_PATH,
    PREPROCESSING_CONFIG_PATH,
)


def create_base_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    alpha: float = MOBILENETV2_ALPHA,
    weights: str = MOBILENETV2_WEIGHTS,
    trainable: bool = False,
) -> Model:
    """
    Create MobileNetV2 base model (feature extractor).
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = trainable
    return base_model


def create_classification_head(
    base_output: layers.Layer,
    num_classes: int = NUM_CLASSES,
    dense_units: int = DENSE_UNITS,
    dropout_rate_1: float = DROPOUT_RATE_1,
    dropout_rate_2: float = DROPOUT_RATE_2,
) -> layers.Layer:
    """
    Create custom classification head.
    """
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base_output)
    x = layers.Dropout(dropout_rate_1, name="dropout_1")(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_features",
    )(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate_2, name="dropout_2")(x)
    predictions = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)
    return predictions


def create_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    alpha: float = MOBILENETV2_ALPHA,
    weights: str = MOBILENETV2_WEIGHTS,
    dense_units: int = DENSE_UNITS,
    dropout_rate_1: float = DROPOUT_RATE_1,
    dropout_rate_2: float = DROPOUT_RATE_2,
    base_trainable: bool = False,
) -> Model:
    """
    Create complete MobileNetV2-based classifier.
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Base model (frozen initially)
    base_model = create_base_model(
        input_shape=input_shape,
        alpha=alpha,
        weights=weights,
        trainable=base_trainable,
    )

    # Feature extraction
    base_output = base_model(inputs, training=base_trainable)

    # Classification head
    outputs = create_classification_head(
        base_output,
        num_classes=num_classes,
        dense_units=dense_units,
        dropout_rate_1=dropout_rate_1,
        dropout_rate_2=dropout_rate_2,
    )

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="CropCare_MobileNetV2")

    return model


def unfreeze_top_layers(model: Model, num_layers: int = 20) -> Model:
    """
    Unfreeze top N layers of base model for fine-tuning.
    """
    # Find base model (MobileNetV2)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
            base_model = layer
            break

    if base_model is None:
        # Try to find by layer names
        for layer in model.layers:
            if "mobilenetv2" in layer.name.lower():
                base_model = layer
                break

    if base_model is not None:
        # Unfreeze last N layers
        for layer in base_model.layers[-num_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        print(f"Unfroze last {num_layers} layers of base model")
    else:
        print("Warning: Could not find base model to unfreeze")

    return model


def freeze_base_model(model: Model) -> Model:
    """Freeze entire base model."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            layer.trainable = False
            break
    return model


def compile_model(
    model: Model,
    learning_rate: float = 1e-4,
    label_smoothing: float = 0.1,
) -> Model:
    """
    Compile model with optimizer, loss, and metrics.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing,
        name="categorical_crossentropy",
    )

    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def get_model_summary(model: Model) -> Dict[str, Any]:
    """Get detailed model summary as dictionary."""
    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    return {
        "model_name": model.name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "num_layers": len(model.layers),
    }


def save_model_artifacts(
    model: Model,
    metadata_dir: Path = METADATA_DIR,
    class_labels_path: Path = CLASS_LABELS_PATH,
    preprocessing_config_path: Path = PREPROCESSING_CONFIG_PATH,
):
    """Save model and associated metadata."""
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Save model architecture config
    model_config = {
        "model_name": model.name,
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "architecture": {
            "base_model": "MobileNetV2",
            "alpha": MOBILENETV2_ALPHA,
            "weights": MOBILENETV2_WEIGHTS,
            "dense_units": DENSE_UNITS,
            "dropout_rates": [DROPOUT_RATE_1, DROPOUT_RATE_2],
        },
    }

    with open(metadata_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # Save class labels (index -> name mapping)
    class_labels = {str(i): name for i, name in enumerate(CLASS_NAMES)}
    with open(class_labels_path, "w") as f:
        json.dump(class_labels, f, indent=2)

    # Save preprocessing config
    preprocessing_config = {
        "image_size": list(IMAGE_SIZE),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "input_range": [0, 1],
    }
    with open(preprocessing_config_path, "w") as f:
        json.dump(preprocessing_config, f, indent=2)

    print(f"Model artifacts saved to {metadata_dir}")


def load_model_artifacts(
    metadata_dir: Path = METADATA_DIR,
) -> Dict[str, Any]:
    """Load model metadata."""
    artifacts = {}

    # Model config
    config_path = metadata_dir / "model_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            artifacts["model_config"] = json.load(f)

    # Class labels
    labels_path = metadata_dir / "class_labels.json"
    if labels_path.exists():
        with open(labels_path, "r") as f:
            artifacts["class_labels"] = json.load(f)

    # Preprocessing config
    preprocess_path = metadata_dir / "preprocessing_config.json"
    if preprocess_path.exists():
        with open(preprocess_path, "r") as f:
            artifacts["preprocessing_config"] = json.load(f)

    return artifacts


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    model = compile_model(model)

    summary = get_model_summary(model)
    print(f"Model: {summary['model_name']}")
    print(f"Total params: {summary['total_parameters']:,}")
    print(f"Trainable params: {summary['trainable_parameters']:,}")
    print(f"Input shape: {summary['input_shape']}")
    print(f"Output shape: {summary['output_shape']}")

    # Save artifacts
    save_model_artifacts(model)
    print("Model architecture test passed!")