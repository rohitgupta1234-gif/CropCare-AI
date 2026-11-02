"""
CropCare AI - Configuration Constants
Central configuration for all project components.
"""

from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_app"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data Paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"

# Model Paths
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
METADATA_DIR = MODELS_DIR / "metadata"

# Model Files
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.keras"
FINAL_MODEL_PATH = TRAINED_MODELS_DIR / "final_model.keras"
CLASS_LABELS_PATH = METADATA_DIR / "class_labels.json"
PREPROCESSING_CONFIG_PATH = METADATA_DIR / "preprocessing_config.json"
TRAINING_HISTORY_PATH = METADATA_DIR / "training_history.json"
EVALUATION_METRICS_PATH = METADATA_DIR / "evaluation_metrics.json"

# Image Configuration
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
INPUT_SHAPE = (*IMAGE_SIZE, IMAGE_CHANNELS)
BATCH_SIZE = 32

# Normalization (ImageNet stats for MobileNetV2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data Splits (stratified, no leakage)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Training Configuration
EPOCHS = 50
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Model Architecture
MOBILENETV2_ALPHA = 1.0
MOBILENETV2_WEIGHTS = "imagenet"
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2
DENSE_UNITS = 128

# Class Names (PlantVillage - verified from dataset)
# Total: 15 classes (Pepper_bell__Leaf_spot not available in dataset)
CLASS_NAMES = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two-spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
]

# Crop mapping for grouping
CROP_MAPPING = {
    "Tomato": [
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_healthy",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two-spotted_spider_mite",
        "Tomato_Target_Spot",
        "Tomato_Tomato_YellowLeaf_Curl_Virus",
        "Tomato_Tomato_mosaic_virus",
    ],
    "Potato": [
        "Potato_Early_blight",
        "Potato_Late_blight",
        "Potato_healthy",
    ],
    "Pepper": [
        "Pepper_bell_Bacterial_spot",
        "Pepper_bell_healthy",
    ],
}

# Healthy class identifiers
HEALTHY_CLASSES = [
    "Tomato_healthy",
    "Potato_healthy",
    "Pepper_bell_healthy",
]

# Disease classes (all non-healthy)
DISEASE_CLASSES = [c for c in CLASS_NAMES if c not in HEALTHY_CLASSES]

# Crop classes (alias for CROP_MAPPING for compatibility)
CROP_CLASSES = CROP_MAPPING

# Number of classes
NUM_CLASSES = len(CLASS_NAMES)

# Training Configuration (dict format for scripts)
TRAINING_CONFIG = {
    "epochs": EPOCHS,
    "initial_lr": INITIAL_LR,
    "fine_tune_lr": FINE_TUNE_LR,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "reduce_lr_patience": REDUCE_LR_PATIENCE,
    "reduce_lr_factor": REDUCE_LR_FACTOR,
    "min_lr": MIN_LR,
    "batch_size": BATCH_SIZE,
    "random_seed": RANDOM_SEED,
    "validation_split": VAL_SPLIT,
    "test_split": TEST_SPLIT,
}

# Model Architecture Config
MODEL_CONFIG = {
    "architecture": "MobileNetV2",
    "input_shape": INPUT_SHAPE,
    "alpha": MOBILENETV2_ALPHA,
    "weights": MOBILENETV2_WEIGHTS,
    "include_top": False,
    "pooling": "avg",
    "dense_units": DENSE_UNITS,
    "dropout_rates": [DROPOUT_RATE_1, DROPOUT_RATE_2],
}

# Grad-CAM Configuration
# MobileNetV2 last Conv2D layers (not BatchNormalization):
# - "block_16_project" (last project conv in block 16)
# - "Conv_1" (final 1x1 conv before global avg pool)
GRADCAM_LAYER_NAME = "block_16_project"  # MobileNetV2 last Conv2D layer
GRADCAM_ALPHA = 0.4  # Heatmap overlay transparency
GRADCAM_COLORMAP = "jet"

# Fallback layers (Conv2D only) for Grad-CAM
GRADCAM_FALLBACK_LAYERS = [
    "block_16_project",
    "Conv_1",
    "block_15_project",
    "block_14_project",
    "block_13_project",
]

# Database
DATABASE_PATH = PROJECT_ROOT / "cropcare.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Languages
SUPPORTED_LANGUAGES = ["en", "hi"]
DEFAULT_LANGUAGE = "en"

# Streamlit Configuration
STREAMLIT_THEME = {
    "primaryColor": "#2E7D32",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#E8F5E9",
    "textColor": "#1B5E20",
    "font": "sans serif",
}

# Page Configuration
PAGE_CONFIG = {
    "page_title": "CropCare AI - Crop Disease Detection",
    "page_icon": "🌿",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Disclaimer
DISCLAIMER = (
    "⚠️ Disclaimer: This AI system provides disease predictions for educational "
    "and informational purposes only. Predictions are not a substitute for "
    "professional agricultural advice. Always consult with local agricultural "
    "extension services or certified plant pathologists for confirmed diagnosis "
    "and treatment recommendations. The developers are not responsible for "
    "any crop loss or damage resulting from reliance on this system."
)

# Dataset Information
DATASET_INFO = {
    "name": "PlantVillage",
    "source": "https://github.com/spMohanty/PlantVillage-Dataset",
    "citation": "Hughes, D., & Salathé, M. (2015). An open access repository of images "
                "on plant health to enable the development of mobile disease diagnostics. "
                "arXiv preprint arXiv:1511.08060.",
    "crops": ["Tomato", "Potato", "Pepper"],
    "total_classes": 16,
    "total_images": "~54,000",
    "license": "CC BY 4.0",
}


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TRAINED_MODELS_DIR,
        CHECKPOINTS_DIR,
        METADATA_DIR,
        STREAMLIT_DIR / "assets" / "css",
        STREAMLIT_DIR / "assets" / "images",
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Directories created successfully")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {CLASS_NAMES}")