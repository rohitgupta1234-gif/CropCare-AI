"""
CropCare AI - Dataset Loading and Preprocessing
Handles PlantVillage dataset loading, validation, splitting, and preprocessing.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATASET_INFO_PATH,
    CLASS_NAMES,
    CROP_MAPPING,
    IMAGE_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_FORMATS,
)


class DatasetValidator:
    """Validates dataset integrity and image files."""

    @staticmethod
    def is_valid_image(filepath: Path) -> bool:
        """Check if file is a valid image."""
        try:
            with Image.open(filepath) as img:
                img.verify()
            # Re-open to check mode and size
            with Image.open(filepath) as img:
                if img.mode not in ("RGB", "RGBA", "L"):
                    return False
                if img.width < 32 or img.height < 32:
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def validate_dataset(raw_dir: Path) -> Dict:
        """Validate entire dataset and return statistics."""
        stats = {
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "classes_found": [],
            "class_counts": {},
            "errors": [],
        }

        for class_name in CLASS_NAMES:
            class_dir = raw_dir / class_name
            if not class_dir.exists():
                stats["errors"].append(f"Class directory not found: {class_name}")
                continue

            stats["classes_found"].append(class_name)
            class_count = 0

            for img_file in class_dir.iterdir():
                stats["total_images"] += 1
                if img_file.suffix.lower() in SUPPORTED_FORMATS:
                    if DatasetValidator.is_valid_image(img_file):
                        stats["valid_images"] += 1
                        class_count += 1
                    else:
                        stats["invalid_images"] += 1
                        stats["errors"].append(f"Invalid image: {img_file}")
                else:
                    stats["invalid_images"] += 1
                    stats["errors"].append(f"Unsupported format: {img_file}")

            stats["class_counts"][class_name] = class_count

        return stats


class DatasetSplitter:
    """Handles stratified dataset splitting without leakage."""

    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed

    def split_dataset(
        self,
        raw_dir: Path,
        processed_dir: Path,
        train_ratio: float = TRAIN_SPLIT,
        val_ratio: float = VAL_SPLIT,
        test_ratio: float = TEST_SPLIT,
    ) -> Dict:
        """
        Split dataset into train/val/test with stratification.
        Returns split statistics.
        """
        # Collect all valid image paths with labels
        image_paths = []
        labels = []

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = raw_dir / class_name
            if not class_dir.exists():
                continue

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in SUPPORTED_FORMATS:
                    if DatasetValidator.is_valid_image(img_file):
                        image_paths.append(str(img_file.relative_to(raw_dir)))
                        labels.append(class_idx)

        # Convert to arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # First split: train vs (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths,
            labels,
            train_size=train_ratio,
            stratify=labels,
            random_state=self.random_seed,
        )

        # Second split: val vs test from remaining
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths,
            temp_labels,
            train_size=val_ratio_adjusted,
            stratify=temp_labels,
            random_state=self.random_seed,
        )

        # Create directory structure and copy files
        splits = {
            "train": (train_paths, train_labels),
            "val": (val_paths, val_labels),
            "test": (test_paths, test_labels),
        }

        split_stats = {}

        for split_name, (paths, lbls) in splits.items():
            split_dir = processed_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            class_counts = {name: 0 for name in CLASS_NAMES}

            for rel_path, lbl in zip(paths, lbls):
                class_name = CLASS_NAMES[lbl]
                src = raw_dir / rel_path
                dst_dir = split_dir / class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / Path(rel_path).name

                # Handle duplicate filenames
                counter = 1
                original_dst = dst
                while dst.exists():
                    stem = original_dst.stem
                    suffix = original_dst.suffix
                    dst = dst_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                shutil.copy2(src, dst)
                class_counts[class_name] += 1

            split_stats[split_name] = {
                "total": len(paths),
                "class_counts": class_counts,
            }

        return split_stats


class DataPreprocessor:
    """Image preprocessing for training and inference."""

    def __init__(
        self,
        target_size: Tuple[int, int] = IMAGE_SIZE,
        mean: List[float] = IMAGENET_MEAN,
        std: List[float] = IMAGENET_STD,
    ):
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess a single PIL image for model input."""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize with high-quality resampling
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Normalize with ImageNet stats
        img_array = (img_array - self.mean) / self.std

        return img_array

    def preprocess_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Preprocess a batch of images."""
        return np.stack([self.preprocess_image(img) for img in images])

    def get_preprocessing_config(self) -> Dict:
        """Return preprocessing configuration for saving."""
        return {
            "target_size": list(self.target_size),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


def load_class_labels(metadata_dir: Path) -> Dict:
    """Load class label mapping from metadata."""
    labels_path = metadata_dir / "class_labels.json"
    if labels_path.exists():
        with open(labels_path, "r") as f:
            return json.load(f)
    return {i: name for i, name in enumerate(CLASS_NAMES)}


def save_dataset_info(stats: Dict, split_stats: Dict, output_path: Path):
    """Save dataset information to JSON."""
    info = {
        "dataset_name": "PlantVillage",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "crop_mapping": CROP_MAPPING,
        "validation_stats": stats,
        "split_statistics": split_stats,
        "preprocessing": {
            "image_size": list(IMAGE_SIZE),
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        },
        "splits": {
            "train": TRAIN_SPLIT,
            "val": VAL_SPLIT,
            "test": TEST_SPLIT,
        },
        "random_seed": RANDOM_SEED,
    }
    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)


def prepare_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    metadata_dir: Path = None,
) -> Dict:
    """
    Complete dataset preparation pipeline.
    Returns dataset statistics.
    """
    if metadata_dir is None:
        from src.config import METADATA_DIR
        metadata_dir = METADATA_DIR

    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Validating dataset...")
    validator = DatasetValidator()
    stats = validator.validate_dataset(raw_dir)

    print(f"Total images: {stats['total_images']}")
    print(f"Valid images: {stats['valid_images']}")
    print(f"Invalid images: {stats['invalid_images']}")
    print(f"Classes found: {len(stats['classes_found'])}")

    if stats["invalid_images"] > 0:
        print(f"Warning: {stats['invalid_images']} invalid images found")

    print("\nSplitting dataset...")
    splitter = DatasetSplitter()
    split_stats = splitter.split_dataset(raw_dir, processed_dir)

    for split_name, split_data in split_stats.items():
        print(f"{split_name}: {split_data['total']} images")

    print("\nSaving dataset info...")
    save_dataset_info(stats, split_stats, DATASET_INFO_PATH)

    # Save class labels
    class_labels = {i: name for i, name in enumerate(CLASS_NAMES)}
    with open(metadata_dir / "class_labels.json", "w") as f:
        json.dump(class_labels, f, indent=2)

    # Save preprocessing config
    preprocessor = DataPreprocessor()
    with open(metadata_dir / "preprocessing_config.json", "w") as f:
        json.dump(preprocessor.get_preprocessing_config(), f, indent=2)

    print("Dataset preparation complete!")
    return {"validation": stats, "splits": split_stats}


if __name__ == "__main__":
    prepare_dataset()