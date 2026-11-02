"""
CropCare AI - Model Evaluation Script
Evaluates trained model on test set with comprehensive metrics.
"""

import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    accuracy_score, top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    PROCESSED_DATA_DIR,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    METADATA_DIR,
    EVALUATION_METRICS_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    CLASS_NAMES,
    NUM_CLASSES,
    CROP_CLASSES,
    HEALTHY_CLASSES,
)
from src.data.augmentation import get_validation_augmentation


def load_model(model_path: Path = None):
    """Load trained model."""
    if model_path is None:
        if BEST_MODEL_PATH.exists():
            model_path = BEST_MODEL_PATH
        elif FINAL_MODEL_PATH.exists():
            model_path = FINAL_MODEL_PATH
        else:
            raise FileNotFoundError("No model found. Train first or provide --model-path")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def create_test_dataset(data_dir: Path, batch_size: int = BATCH_SIZE):
    """Create test dataset."""
    val_aug = get_validation_augmentation(IMAGE_SIZE)

    def augment_val(image, label):
        image_np = image.numpy()
        augmented = val_aug(image=image_np)
        return augmented['image'], label

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False,
        seed=42,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.map(
        lambda x, y: tf.py_function(augment_val, [x, y], [tf.float32, tf.float32]),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    def set_shapes(images, labels):
        images.set_shape([None, *IMAGE_SIZE, 3])
        labels.set_shape([None, NUM_CLASSES])
        return images, labels

    test_ds = test_ds.map(set_shapes)
    return test_ds


def evaluate_model(
    model: tf.keras.Model,
    test_ds,
    save_plots: bool = True,
    output_dir: Path = None,
):
    """Comprehensive model evaluation."""

    if output_dir is None:
        output_dir = METADATA_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Get predictions
    print("\nGenerating predictions...")
    y_true = []
    y_pred = []
    y_pred_proba = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_proba.extend(preds)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Top-k accuracy
    top3_acc = top_k_accuracy_score(y_true, y_pred_proba, k=3)
    top5_acc = top_k_accuracy_score(y_true, y_pred_proba, k=5)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:        {accuracy*100:.2f}%")
    print(f"  Precision (w):   {precision*100:.2f}%")
    print(f"  Recall (w):      {recall*100:.2f}%")
    print(f"  F1-Score (w):    {f1*100:.2f}%")
    print(f"  Precision (m):   {precision_macro*100:.2f}%")
    print(f"  Recall (m):      {recall_macro*100:.2f}%")
    print(f"  F1-Score (m):    {f1_macro*100:.2f}%")
    print(f"  Top-3 Accuracy:  {top3_acc*100:.2f}%")
    print(f"  Top-5 Accuracy:  {top5_acc*100:.2f}%")

    # Per-class metrics
    print("\nPer-Class Metrics:")
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    per_class = []
    for class_name in CLASS_NAMES:
        if class_name in report:
            metrics = report[class_name]
            per_class.append({
                "class": class_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": int(metrics["support"]),
            })
            print(f"  {class_name}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f} (n={int(metrics['support'])})")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-crop metrics
    crop_metrics = {}
    for crop, classes in CROP_CLASSES.items():
        class_indices = [CLASS_NAMES.index(c) for c in classes if c in CLASS_NAMES]
        if class_indices:
            mask = np.isin(y_true, class_indices)
            if mask.sum() > 0:
                crop_acc = accuracy_score(y_true[mask], y_pred[mask])
                crop_metrics[crop] = {
                    "accuracy": float(crop_acc),
                    "samples": int(mask.sum()),
                }
                print(f"\n{crop} Accuracy: {crop_acc*100:.2f}% ({mask.sum()} samples)")

    # Healthy vs Diseased
    healthy_indices = [CLASS_NAMES.index(c) for c in HEALTHY_CLASSES]
    healthy_mask = np.isin(y_true, healthy_indices)
    diseased_mask = ~healthy_mask

    if healthy_mask.sum() > 0:
        healthy_acc = accuracy_score(y_true[healthy_mask], y_pred[healthy_mask])
        print(f"\nHealthy Accuracy: {healthy_acc*100:.2f}% ({healthy_mask.sum()} samples)")

    if diseased_mask.sum() > 0:
        diseased_acc = accuracy_score(y_true[diseased_mask], y_pred[diseased_mask])
        print(f"Diseased Accuracy: {diseased_acc*100:.2f}% ({diseased_mask.sum()} samples)")

    # Save confusion matrix plot
    if save_plots:
        save_confusion_matrix_plot(cm, output_dir)
        save_per_class_plot(per_class, output_dir)

    # Prepare results dict
    results = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "top3_accuracy": float(top3_acc),
        "top5_accuracy": float(top5_acc),
        "test_samples": int(len(y_true)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "crop_metrics": crop_metrics,
        "healthy_accuracy": float(healthy_acc) if healthy_mask.sum() > 0 else None,
        "diseased_accuracy": float(diseased_acc) if diseased_mask.sum() > 0 else None,
    }

    # Save metrics
    with open(EVALUATION_METRICS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {EVALUATION_METRICS_PATH}")

    # Save detailed CSV
    df = pd.DataFrame(per_class)
    csv_path = output_dir / "per_class_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Per-class CSV saved to: {csv_path}")

    return results


def save_confusion_matrix_plot(cm: np.ndarray, output_dir: Path):
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(16, 14))

    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # Plot
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap='Greens',
        cbar_kws={'label': 'Normalized Count'},
    )

    plt.title('Confusion Matrix (Normalized)', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plot_path = output_dir / "confusion_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to: {plot_path}")


def save_per_class_plot(per_class: list, output_dir: Path):
    """Save per-class metrics bar chart."""
    df = pd.DataFrame(per_class)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['precision', 'recall', 'f1_score']
    titles = ['Precision', 'Recall', 'F1-Score']
    colors = ['#1E88E5', '#43A047', '#FB8C00']

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        bars = ax.barh(df['class'], df[metric], color=color, alpha=0.7)
        ax.set_xlabel(title)
        ax.set_xlim(0, 1.05)
        ax.set_title(f'Per-Class {title}')

        # Add value labels
        for bar, val in zip(bars, df[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "per_class_metrics.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CropCare AI model")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model file")
    parser.add_argument("--data-dir", type=str, default=None, help="Processed data directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else PROCESSED_DATA_DIR
    model_path = Path(args.model_path) if args.model_path else None

    # Load model
    model = load_model(model_path)

    # Create test dataset
    print("\nLoading test dataset...")
    test_ds = create_test_dataset(data_dir, args.batch_size)

    # Evaluate
    results = evaluate_model(
        model,
        test_ds,
        save_plots=not args.no_plots,
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()