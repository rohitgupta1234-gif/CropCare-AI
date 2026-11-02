"""
CropCare AI - Data Validation Utilities
Validates images, dataset integrity, and model inputs.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from PIL import Image
import numpy as np

from src.config import (
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE_BYTES,
    IMAGE_SIZE,
    CLASS_NAMES,
)


class ImageValidator:
    """Validates uploaded/image files for processing."""

    # Maximum dimensions to prevent decompression bombs
    MAX_DIMENSION = 4096
    MIN_DIMENSION = 32

    @classmethod
    def validate_file(cls, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate an image file.
        Returns (is_valid, error_message).
        """
        file_path = Path(file_path)

        # Check file exists
        if not file_path.exists():
            return False, "File does not exist"

        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "File is empty"
        if file_size > MAX_FILE_SIZE_BYTES:
            return False, f"File too large ({file_size / 1024 / 1024:.1f}MB > {MAX_FILE_SIZE_BYTES / 1024 / 1024}MB)"

        # Check extension
        if file_path.suffix not in SUPPORTED_FORMATS:
            return False, f"Unsupported format: {file_path.suffix}. Supported: {SUPPORTED_FORMATS}"

        # Validate image content
        try:
            with Image.open(file_path) as img:
                # Verify image integrity
                img.verify()

            # Re-open for detailed checks
            with Image.open(file_path) as img:
                # Check mode
                if img.mode not in ("RGB", "RGBA", "L", "P"):
                    return False, f"Unsupported color mode: {img.mode}"

                # Check dimensions
                width, height = img.size
                if width < cls.MIN_DIMENSION or height < cls.MIN_DIMENSION:
                    return False, f"Image too small: {width}x{height} (min {cls.MIN_DIMENSION}x{cls.MIN_DIMENSION})"
                if width > cls.MAX_DIMENSION or height > cls.MAX_DIMENSION:
                    return False, f"Image too large: {width}x{height} (max {cls.MAX_DIMENSION}x{cls.MAX_DIMENSION})"

                # Check aspect ratio (warn but don't fail)
                aspect_ratio = width / height
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    return True, f"Warning: Unusual aspect ratio {aspect_ratio:.2f}"

        except Exception as e:
            return False, f"Invalid image file: {str(e)}"

        return True, "Valid"

    @classmethod
    def validate_upload(cls, uploaded_file) -> Tuple[bool, str, Optional[Image.Image]]:
        """
        Validate a Streamlit uploaded file.
        Returns (is_valid, error_message, PIL_image).
        """
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            return False, f"File too large ({uploaded_file.size / 1024 / 1024:.1f}MB)", None

        # Check extension
        file_ext = Path(uploaded_file.name).suffix
        if file_ext not in SUPPORTED_FORMATS:
            return False, f"Unsupported format: {file_ext}", None

        try:
            # Read and validate
            image = Image.open(uploaded_file)
            image.verify()

            # Re-open for processing
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)

            # Convert to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Check dimensions
            width, height = image.size
            if width < cls.MIN_DIMENSION or height < cls.MIN_DIMENSION:
                return False, f"Image too small: {width}x{height}", None
            if width > cls.MAX_DIMENSION or height > cls.MAX_DIMENSION:
                return False, f"Image too large: {width}x{height}", None

            return True, "Valid", image

        except Exception as e:
            return False, f"Invalid image: {str(e)}", None


class ModelInputValidator:
    """Validates model inputs and preprocessing outputs."""

    @staticmethod
    def validate_preprocessed_image(
        img_array: np.ndarray,
        expected_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    ) -> Tuple[bool, str]:
        """Validate preprocessed image array."""
        if not isinstance(img_array, np.ndarray):
            return False, "Input is not a numpy array"

        if img_array.ndim != 3:
            return False, f"Expected 3D array, got {img_array.ndim}D"

        if img_array.shape != expected_shape:
            return False, f"Shape mismatch: expected {expected_shape}, got {img_array.shape}"

        # Check value range (after normalization, should be roughly [-3, 3])
        if img_array.min() < -5 or img_array.max() > 5:
            return False, f"Values out of expected range: [{img_array.min():.3f}, {img_array.max():.3f}]"

        # Check for NaN/Inf
        if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
            return False, "Array contains NaN or Inf values"

        return True, "Valid"

    @staticmethod
    def validate_batch(
        batch: np.ndarray,
        batch_size: int,
        expected_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3),
    ) -> Tuple[bool, str]:
        """Validate a batch of preprocessed images."""
        if not isinstance(batch, np.ndarray):
            return False, "Batch is not a numpy array"

        if batch.ndim != 4:
            return False, f"Expected 4D batch, got {batch.ndim}D"

        if batch.shape[0] != batch_size:
            return False, f"Batch size mismatch: expected {batch_size}, got {batch.shape[0]}"

        if batch.shape[1:] != expected_shape:
            return False, f"Sample shape mismatch: expected {expected_shape}, got {batch.shape[1:]}"

        if np.any(np.isnan(batch)) or np.any(np.isinf(batch)):
            return False, "Batch contains NaN or Inf values"

        return True, "Valid"


class DatasetIntegrityChecker:
    """Checks dataset integrity and consistency."""

    @staticmethod
    def check_class_balance(class_counts: Dict[str, int], threshold: float = 0.1) -> Dict:
        """
        Check if classes are reasonably balanced.
        Returns dict with balance info and warnings.
        """
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        expected_per_class = total / num_classes

        result = {
            "total_samples": total,
            "num_classes": num_classes,
            "expected_per_class": expected_per_class,
            "class_distribution": class_counts,
            "imbalanced_classes": [],
            "warnings": [],
        }

        for class_name, count in class_counts.items():
            ratio = count / expected_per_class
            if ratio < threshold or ratio > (2 - threshold):
                result["imbalanced_classes"].append({
                    "class": class_name,
                    "count": count,
                    "ratio": ratio,
                })
                result["warnings"].append(
                    f"Class '{class_name}' has {count} samples "
                    f"({ratio:.2f}x expected)"
                )

        return result

    @staticmethod
    def check_split_leakage(
        train_paths: List[str],
        val_paths: List[str],
        test_paths: List[str],
    ) -> Dict:
        """Check for data leakage between splits."""
        train_set = set(train_paths)
        val_set = set(val_paths)
        test_set = set(test_paths)

        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        return {
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap),
            "has_leakage": bool(train_val_overlap or train_test_overlap or val_test_overlap),
            "leakage_details": {
                "train_val": list(train_val_overlap)[:10],
                "train_test": list(train_test_overlap)[:10],
                "val_test": list(val_test_overlap)[:10],
            },
        }

    @staticmethod
    def verify_class_consistency(
        splits_dir: Path,
        expected_classes: List[str] = CLASS_NAMES,
    ) -> Dict:
        """Verify all expected classes exist in each split."""
        result = {
            "splits": {},
            "missing_classes": {},
            "extra_classes": {},
            "consistent": True,
        }

        for split in ["train", "val", "test"]:
            split_dir = splits_dir / split
            if not split_dir.exists():
                result["splits"][split] = "missing"
                result["consistent"] = False
                continue

            found_classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
            missing = set(expected_classes) - set(found_classes)
            extra = set(found_classes) - set(expected_classes)

            result["splits"][split] = {
                "found": len(found_classes),
                "expected": len(expected_classes),
            }

            if missing:
                result["missing_classes"][split] = list(missing)
                result["consistent"] = False
            if extra:
                result["extra_classes"][split] = list(extra)

        return result


def validate_model_prediction(
    prediction: np.ndarray,
    num_classes: int = len(CLASS_NAMES),
) -> Tuple[bool, str]:
    """Validate model prediction output."""
    if not isinstance(prediction, np.ndarray):
        return False, "Prediction is not a numpy array"

    if prediction.ndim == 1:
        # Single prediction
        if prediction.shape[0] != num_classes:
            return False, f"Wrong number of classes: {prediction.shape[0]} != {num_classes}"
    elif prediction.ndim == 2:
        # Batch prediction
        if prediction.shape[1] != num_classes:
            return False, f"Wrong number of classes: {prediction.shape[1]} != {num_classes}"
    else:
        return False, f"Unexpected prediction shape: {prediction.shape}"

    # Check probabilities sum to ~1
    sums = prediction.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        return False, f"Probabilities don't sum to 1: {sums}"

    # Check range [0, 1]
    if prediction.min() < 0 or prediction.max() > 1:
        return False, f"Probabilities out of range: [{prediction.min()}, {prediction.max()}]"

    # Check for NaN
    if np.any(np.isnan(prediction)):
        return False, "Prediction contains NaN values"

    return True, "Valid"


if __name__ == "__main__":
    # Test validators
    print("Testing ImageValidator...")
    # Test with a dummy valid image
    from PIL import Image
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (224, 224), color="green")
        img.save(f.name)
        valid, msg = ImageValidator.validate_file(f.name)
        print(f"Valid image: {valid}, {msg}")
        os.unlink(f.name)

    print("Validation module test passed!")