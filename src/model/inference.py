"""
CropCare AI - Model Inference Utilities
Handles model loading, preprocessing, prediction, and post-processing.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings

from src.config import (
    IMAGE_SIZE,
    CLASS_NAMES,
    CROP_MAPPING,
    HEALTHY_CLASSES,
    NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    CLASS_LABELS_PATH,
    PREPROCESSING_CONFIG_PATH,
    METADATA_DIR,
)
from src.data.validation import ImageValidator, ModelInputValidator, validate_model_prediction


class ModelInference:
    """Main inference class for CropCare AI model."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_labels_path: Optional[Union[str, Path]] = None,
        preprocessing_config_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to saved model (.keras or .h5)
            class_labels_path: Path to class labels JSON
            preprocessing_config_path: Path to preprocessing config JSON
        """
        self.model = None
        self.class_labels = None
        self.preprocessing_config = None
        self.model_loaded = False

        # Set default paths
        if model_path is None:
            # Try best model first, then final model
            if BEST_MODEL_PATH.exists():
                model_path = BEST_MODEL_PATH
            elif FINAL_MODEL_PATH.exists():
                model_path = FINAL_MODEL_PATH
            else:
                raise FileNotFoundError(
                    f"No model found. Checked: {BEST_MODEL_PATH}, {FINAL_MODEL_PATH}"
                )

        if class_labels_path is None:
            class_labels_path = CLASS_LABELS_PATH

        if preprocessing_config_path is None:
            preprocessing_config_path = PREPROCESSING_CONFIG_PATH

        self.model_path = Path(model_path)
        self.class_labels_path = Path(class_labels_path)
        self.preprocessing_config_path = Path(preprocessing_config_path)

        # Load everything
        self._load_model()
        self._load_metadata()

    def _load_model(self):
        """Load the trained model."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.model_loaded = True
            print(f"Model loaded from {self.model_path}")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _load_metadata(self):
        """Load class labels and preprocessing config."""
        # Load class labels
        if self.class_labels_path.exists():
            with open(self.class_labels_path, "r") as f:
                labels_data = json.load(f)
                # Convert string keys to int
                self.class_labels = {int(k): v for k, v in labels_data.items()}
        else:
            # Fallback to config
            self.class_labels = {i: name for i, name in enumerate(CLASS_NAMES)}
            warnings.warn(f"Class labels file not found, using default: {CLASS_LABELS_PATH}")

        # Verify class labels match model output
        if len(self.class_labels) != NUM_CLASSES:
            warnings.warn(
                f"Class label count ({len(self.class_labels)}) "
                f"doesn't match NUM_CLASSES ({NUM_CLASSES})"
            )

        # Load preprocessing config
        if self.preprocessing_config_path.exists():
            with open(self.preprocessing_config_path, "r") as f:
                self.preprocessing_config = json.load(f)
        else:
            self.preprocessing_config = {
                "image_size": list(IMAGE_SIZE),
                "normalization": {
                    "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD,
                },
            }
            warnings.warn("Preprocessing config not found, using defaults")

    def preprocess_image(self, image: Union[Image.Image, np.ndarray, str, Path]) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: PIL Image, numpy array, or file path

        Returns:
            Preprocessed image array of shape (1, H, W, 3)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize
        target_size = tuple(self.preprocessing_config.get("image_size", list(IMAGE_SIZE)))
        image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to array and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array(self.preprocessing_config.get("normalization", {}).get("mean", IMAGENET_MEAN))
        std = np.array(self.preprocessing_config.get("normalization", {}).get("std", IMAGENET_STD))
        img_array = (img_array - mean) / std

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Validate
        valid, msg = ModelInputValidator.validate_preprocessed_image(img_array[0])
        if not valid:
            warnings.warn(f"Preprocessed image validation warning: {msg}")

        return img_array

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        return_probabilities: bool = True,
        top_k: int = 5,
    ) -> Dict:
        """
        Run inference on a single image.

        Args:
            image: Input image
            return_probabilities: Whether to return all class probabilities
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        # Preprocess
        processed = self.preprocess_image(image)

        # Predict
        predictions = self.model.predict(processed, verbose=0)

        # Validate prediction
        valid, msg = validate_model_prediction(predictions)
        if not valid:
            warnings.warn(f"Prediction validation warning: {msg}")

        # Get top-k predictions
        probs = predictions[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = {
            "predictions": [],
            "top_prediction": None,
            "crop": None,
            "disease": None,
            "confidence": 0.0,
            "health_status": "Unknown",
        }

        for idx in top_indices:
            class_name = self.class_labels.get(idx, f"Class_{idx}")
            confidence = float(probs[idx])

            # Determine crop and health status
            crop = self._get_crop_from_class(class_name)
            is_healthy = class_name in HEALTHY_CLASSES

            pred_info = {
                "class_index": int(idx),
                "class_name": class_name,
                "crop": crop,
                "confidence": confidence,
                "is_healthy": is_healthy,
            }
            results["predictions"].append(pred_info)

        # Set top prediction
        top_pred = results["predictions"][0]
        results["top_prediction"] = top_pred["class_name"]
        results["crop"] = top_pred["crop"]
        results["disease"] = top_pred["class_name"]
        results["confidence"] = top_pred["confidence"]
        results["health_status"] = "Healthy" if top_pred["is_healthy"] else "Diseased"

        # Add all probabilities if requested
        if return_probabilities:
            results["all_probabilities"] = {
                self.class_labels.get(i, f"Class_{i}"): float(probs[i])
                for i in range(len(probs))
            }

        return results

    def predict_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, str, Path]],
        top_k: int = 3,
    ) -> List[Dict]:
        """Run inference on a batch of images."""
        results = []
        for img in images:
            results.append(self.predict(img, top_k=top_k))
        return results

    def _get_crop_from_class(self, class_name: str) -> str:
        """Extract crop name from class name."""
        for crop, classes in CROP_MAPPING.items():
            if class_name in classes:
                return crop
        # Fallback: parse from class name
        if class_name.startswith("Tomato_"):
            return "Tomato"
        elif class_name.startswith("Potato_"):
            return "Potato"
        elif class_name.startswith("Pepper_"):
            return "Pepper"
        return "Unknown"

    def get_model_info(self) -> Dict:
        """Get model information."""
        if not self.model_loaded:
            return {"error": "Model not loaded"}

        return {
            "model_path": str(self.model_path),
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "total_parameters": self.model.count_params(),
            "trainable_parameters": sum(
                tf.keras.backend.count_params(w) for w in self.model.trainable_weights
            ),
        }


# Global inference instance (lazy loaded)
_inference_instance = None


def get_inference_engine() -> ModelInference:
    """Get or create global inference engine instance."""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = ModelInference()
    return _inference_instance


def predict_image(
    image: Union[Image.Image, np.ndarray, str, Path],
    top_k: int = 5,
) -> Dict:
    """Convenience function for single image prediction."""
    engine = get_inference_engine()
    return engine.predict(image, top_k=top_k)


def predict_image_file(filepath: Union[str, Path]) -> Dict:
    """Convenience function for predicting from file path."""
    # Validate first
    valid, msg = ImageValidator.validate_file(filepath)
    if not valid:
        return {
            "error": msg,
            "predictions": [],
            "top_prediction": None,
            "crop": None,
            "disease": None,
            "confidence": 0.0,
            "health_status": "Error",
        }

    return predict_image(filepath)


if __name__ == "__main__":
    # Test inference (will fail if no model exists)
    try:
        engine = ModelInference()
        print("Model info:", engine.get_model_info())
    except FileNotFoundError as e:
        print(f"Expected (no model yet): {e}")
    except Exception as e:
        print(f"Error: {e}")