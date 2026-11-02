"""
CropCare AI - Main Package
AI-based Crop Disease Detection and Prevention System
"""

__version__ = "1.0.0"
__author__ = "B.Tech Project Team"
__description__ = "Crop Disease Detection using MobileNetV2 Transfer Learning"

from src.config import (
    PROJECT_ROOT,
    CLASS_NAMES,
    CROP_MAPPING,
    HEALTHY_CLASSES,
    DISEASE_CLASSES,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    DATABASE_PATH,
    SUPPORTED_LANGUAGES,
    DISCLAIMER,
)

__all__ = [
    "PROJECT_ROOT",
    "CLASS_NAMES",
    "CROP_MAPPING",
    "HEALTHY_CLASSES",
    "DISEASE_CLASSES",
    "NUM_CLASSES",
    "IMAGE_SIZE",
    "BATCH_SIZE",
    "DATABASE_PATH",
    "SUPPORTED_LANGUAGES",
    "DISCLAIMER",
]