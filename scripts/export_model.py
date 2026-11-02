"""
CropCare AI - Model Export Script
Exports trained model for inference and deployment.
"""

import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np

from src.config import (
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    METADATA_DIR,
    CLASS_LABELS_PATH,
    PREPROCESSING_CONFIG_PATH,
    IMAGE_SIZE,
    CLASS_NAMES,
    NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.model.architecture import load_model_artifacts, save_model_artifacts


def load_best_model():
    """Load the best available model."""
    if BEST_MODEL_PATH.exists():
        model_path = BEST_MODEL_PATH
    elif FINAL_MODEL_PATH.exists():
        model_path = FINAL_MODEL_PATH
    else:
        raise FileNotFoundError("No trained model found. Run train_model.py first.")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model, model_path


def export_saved_model(model: tf.keras.Model, export_dir: Path):
    """Export model in SavedModel format."""
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save as SavedModel
    tf.saved_model.save(model, str(export_dir / "saved_model"))
    print(f"SavedModel exported to: {export_dir / 'saved_model'}")


def export_tflite(model: tf.keras.Model, export_dir: Path, optimize: bool = True):
    """Export model as TensorFlow Lite."""
    export_dir.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Enable quantization
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    tflite_path = export_dir / "model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model exported to: {tflite_path}")
    print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")

    # Also save quantized version
    if optimize:
        converter_full = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_full.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_full = converter_full.convert()

        tflite_full_path = export_dir / "model_full_int8.tflite"
        with open(tflite_full_path, 'wb') as f:
            f.write(tflite_full)
        print(f"Full INT8 TFLite model: {tflite_full_path} ({len(tflite_full) / 1024 / 1024:.2f} MB)")


def export_onnx(model: tf.keras.Model, export_dir: Path):
    """Export model as ONNX (requires tf2onnx)."""
    try:
        import tf2onnx
        export_dir.mkdir(parents=True, exist_ok=True)

        # Define input signature
        input_signature = [tf.TensorSpec(shape=[None, *IMAGE_SIZE, 3], dtype=tf.float32, name='input')]

        onnx_path = export_dir / "model.onnx"
        tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            output_path=str(onnx_path),
            opset=13,
        )
        print(f"ONNX model exported to: {onnx_path}")
    except ImportError:
        print("tf2onnx not installed. Skipping ONNX export.")
        print("Install with: pip install tf2onnx")


def create_inference_metadata(export_dir: Path):
    """Create metadata files for inference."""
    export_dir.mkdir(parents=True, exist_ok=True)

    # Class labels
    class_labels = {str(i): name for i, name in enumerate(CLASS_NAMES)}
    with open(export_dir / "class_labels.json", 'w') as f:
        json.dump(class_labels, f, indent=2)

    # Preprocessing config
    preprocessing_config = {
        "image_size": list(IMAGE_SIZE),
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "input_range": [0, 1],
        "color_space": "RGB",
    }
    with open(export_dir / "preprocessing_config.json", 'w') as f:
        json.dump(preprocessing_config, f, indent=2)

    # Model info
    model_info = {
        "architecture": "MobileNetV2",
        "input_shape": list(IMAGE_SIZE) + [3],
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "framework": "TensorFlow/Keras",
        "version": "2.x",
    }
    with open(export_dir / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"Inference metadata saved to: {export_dir}")


def verify_exported_model(export_dir: Path):
    """Verify exported models work correctly."""
    print("\nVerifying exported models...")

    # Test TFLite
    tflite_path = export_dir / "model.tflite"
    if tflite_path.exists():
        try:
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            print(f"  TFLite Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
            print(f"  TFLite Output: {output_details[0]['shape']} {output_details[0]['dtype']}")

            # Test inference
            test_input = np.random.random((1, *IMAGE_SIZE, 3)).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            print(f"  TFLite Test Output Shape: {output.shape}")
            print(f"  TFLite Test Output Sum: {output.sum():.4f}")

        except Exception as e:
            print(f"  TFLite verification failed: {e}")

    # Test SavedModel
    saved_model_dir = export_dir / "saved_model"
    if saved_model_dir.exists():
        try:
            loaded = tf.saved_model.load(str(saved_model_dir))
            infer = loaded.signatures['serving_default']

            test_input = tf.constant(np.random.random((1, *IMAGE_SIZE, 3)), dtype=tf.float32)
            output = infer(test_input)
            print(f"  SavedModel Test Output Keys: {list(output.keys())}")
            for key, val in output.items():
                print(f"  SavedModel {key}: {val.shape}")

        except Exception as e:
            print(f"  SavedModel verification failed: {e}")


def create_deployment_package(export_dir: Path, package_name: str = "cropcare_model"):
    """Create a deployment package with all necessary files."""
    import zipfile

    zip_path = export_dir.parent / f"{package_name}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in export_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(export_dir.parent)
                zipf.write(file_path, arcname)

    print(f"\nDeployment package created: {zip_path}")
    print(f"Package size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export CropCare AI model for deployment")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model file")
    parser.add_argument("--export-dir", type=str, default=None, help="Export directory")
    parser.add_argument("--format", type=str, default="all",
                       choices=["all", "savedmodel", "tflite", "onnx"],
                       help="Export format")
    parser.add_argument("--no-optimize", action="store_true", help="Disable TFLite optimization")
    parser.add_argument("--package", action="store_true", help="Create deployment zip package")

    args = parser.parse_args()

    export_dir = Path(args.export_dir) if args.export_dir else METADATA_DIR / "export"

    # Load model
    model, model_path = load_best_model()

    # Load artifacts
    artifacts = load_model_artifacts()

    print(f"\nExporting model to: {export_dir}")
    print(f"Format: {args.format}")

    if args.format in ["all", "savedmodel"]:
        print("\n--- Exporting SavedModel ---")
        export_saved_model(model, export_dir)

    if args.format in ["all", "tflite"]:
        print("\n--- Exporting TensorFlow Lite ---")
        export_tflite(model, export_dir, optimize=not args.no_optimize)

    if args.format in ["all", "onnx"]:
        print("\n--- Exporting ONNX ---")
        export_onnx(model, export_dir)

    # Create inference metadata
    print("\n--- Creating Inference Metadata ---")
    create_inference_metadata(export_dir)

    # Copy original model artifacts
    for artifact in ["model_config.json", "class_labels.json", "preprocessing_config.json"]:
        src = METADATA_DIR / artifact
        if src.exists():
            import shutil
            shutil.copy2(src, export_dir / artifact)

    # Verify exports
    verify_exported_model(export_dir)

    # Create package
    if args.package:
        print("\n--- Creating Deployment Package ---")
        create_deployment_package(export_dir)

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"Export directory: {export_dir}")
    print("\nFiles created:")
    for f in sorted(export_dir.rglob('*')):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.2f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.relative_to(export_dir)} ({size_str})")


if __name__ == "__main__":
    main()