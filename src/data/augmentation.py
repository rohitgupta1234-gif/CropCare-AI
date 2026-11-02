"""
CropCare AI - Data Augmentation Pipelines
Training-only augmentations using Albumentations.
"""

import albumentations as A
import numpy as np
from typing import Tuple, Optional
import cv2

from src.config import IMAGE_SIZE


def get_training_augmentation(
    image_size: Tuple[int, int] = IMAGE_SIZE,
    severity: str = "medium",
) -> A.Compose:
    """
    Get training augmentation pipeline.
    Only applied during training, never validation/test.
    """
    h, w = image_size

    if severity == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3
            ),
            A.Resize(h, w),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif severity == "medium":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=30,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            ], p=0.2),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                min_holes=1,
                fill_value=0,
                p=0.2
            ),
            A.Resize(h, w),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif severity == "heavy":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=45,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.6
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 80), p=0.5),
                A.GaussianBlur(blur_limit=5, p=0.5),
                A.MotionBlur(blur_limit=5, p=0.5),
                A.MedianBlur(blur_limit=5, p=0.5),
            ], p=0.4),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.2, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                A.Perspective(scale=(0.05, 0.1), p=0.5),
            ], p=0.3),
            A.CoarseDropout(
                max_holes=16,
                max_height=32,
                max_width=32,
                min_holes=1,
                fill_value=0,
                p=0.3
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.Resize(h, w),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    else:
        raise ValueError(f"Unknown severity: {severity}")


def get_validation_augmentation(
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> A.Compose:
    """
    Get validation/test augmentation pipeline.
    Only resize and normalize - no random augmentations.
    """
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_augmentation(
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> A.Compose:
    """
    Get inference augmentation pipeline.
    Identical to validation - deterministic preprocessing only.
    """
    return get_validation_augmentation(image_size)


class AugmentationPipeline:
    """Wrapper for consistent augmentation application."""

    def __init__(
        self,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        severity: str = "medium",
    ):
        self.image_size = image_size
        self.severity = severity
        self.train_transform = get_training_augmentation(image_size, severity)
        self.val_transform = get_validation_augmentation(image_size)
        self.inference_transform = get_inference_augmentation(image_size)

    def __call__(self, image: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Apply appropriate transform based on mode."""
        if is_training:
            transformed = self.train_transform(image=image)
        else:
            transformed = self.val_transform(image=image)
        return transformed["image"]

    def apply_inference(self, image: np.ndarray) -> np.ndarray:
        """Apply inference transform (deterministic)."""
        transformed = self.inference_transform(image=image)
        return transformed["image"]


def visualize_augmentations(
    image: np.ndarray,
    num_samples: int = 4,
    severity: str = "medium",
) -> list:
    """Generate augmented samples for visualization."""
    transform = get_training_augmentation(severity=severity)
    augmented = []
    for _ in range(num_samples):
        transformed = transform(image=image)
        # Denormalize for visualization (CHW -> HWC)
        img = transformed["image"].transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        augmented.append(img)
    return augmented


if __name__ == "__main__":
    # Test augmentation pipeline
    import matplotlib.pyplot as plt

    # Create dummy image
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    pipeline = AugmentationPipeline(severity="medium")

    # Test training augmentation
    train_aug = pipeline(dummy_img, is_training=True)
    print(f"Training output shape: {train_aug.shape}")
    print(f"Training output range: [{train_aug.min():.3f}, {train_aug.max():.3f}]")

    # Test validation augmentation
    val_aug = pipeline(dummy_img, is_training=False)
    print(f"Validation output shape: {val_aug.shape}")
    print(f"Validation output range: [{val_aug.min():.3f}, {val_aug.max():.3f}]")

    print("Augmentation pipeline test passed!")