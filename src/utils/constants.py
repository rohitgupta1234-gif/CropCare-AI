"""
CropCare AI - Project Constants
Centralized constants used across the project.
"""

from pathlib import Path

# ==================== PROJECT METADATA ====================
PROJECT_NAME = "CropCare AI"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "AI-based Crop Disease Detection and Prevention System"
PROJECT_AUTHOR = "B.Tech Project Team"
PROJECT_LICENSE = "MIT"
PROJECT_URL = "https://github.com/yourusername/CropCare-AI"

# ==================== SUPPORTED CROPS & CLASSES ====================
# These match exactly the PlantVillage dataset class names
SUPPORTED_CROPS = ["Tomato", "Potato", "Pepper"]

# Class names from PlantVillage (15 total - Pepper_bell__Leaf_spot not in dataset)
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

NUM_CLASSES = len(CLASS_NAMES)

# Crop to class mapping
CROP_CLASSES = {
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

DISEASE_CLASSES = [c for c in CLASS_NAMES if c not in HEALTHY_CLASSES]

# ==================== MODEL CONFIGURATION ====================
MODEL_CONFIG = {
    "architecture": "MobileNetV2",
    "input_shape": (224, 224, 3),
    "alpha": 1.0,
    "weights": "imagenet",
    "include_top": False,
    "pooling": "avg",
    "dense_units": 128,
    "dropout_rates": [0.3, 0.2],
    "l2_regularization": 1e-4,
    "label_smoothing": 0.1,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "initial_lr": 1e-4,
    "fine_tune_lr": 1e-5,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-7,
    "validation_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
    "data_splits": {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
    },
}

# Data augmentation severity levels
AUGMENTATION_CONFIG = {
    "light": {
        "horizontal_flip": 0.5,
        "brightness_contrast": 0.1,
        "shift_scale_rotate": 0.05,
    },
    "medium": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.2,
        "random_rotate90": 0.3,
        "brightness_contrast": 0.2,
        "hue_saturation": 0.3,
        "shift_scale_rotate": 0.1,
        "noise_blur": 0.3,
        "distortion": 0.2,
        "coarse_dropout": 0.2,
    },
    "heavy": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.3,
        "random_rotate90": 0.5,
        "transpose": 0.3,
        "brightness_contrast": 0.3,
        "hue_saturation": 0.5,
        "shift_scale_rotate": 0.15,
        "noise_blur": 0.4,
        "distortion": 0.3,
        "coarse_dropout": 0.3,
        "clahe": 0.2,
        "gamma": 0.2,
    },
}

# ImageNet normalization (for MobileNetV2)
IMAGENET_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

# ==================== IMAGE PROCESSING ====================
IMAGE_CONFIG = {
    "target_size": (224, 224),
    "min_dimension": 32,
    "max_dimension": 4096,
    "supported_formats": {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"},
    "max_file_size_mb": 10,
    "display_max_size": (400, 400),
    "thumbnail_size": (200, 200),
}

# ==================== GRAD-CAM ====================
GRADCAM_CONFIG = {
    "target_layer": "block_16_project",  # MobileNetV2 last Conv2D layer (not BN)
    "alpha": 0.4,
    "colormap": "jet",
    "output_size": (224, 224),
}

# Alternative layer names to try if primary not found (Conv2D layers only)
GRADCAM_FALLBACK_LAYERS = [
    "block_16_project",
    "Conv_1",
    "block_15_project",
    "block_14_project",
    "block_13_project",
]

# ==================== DATABASE ====================
DATABASE_CONFIG = {
    "path": "cropcare.db",
    "url": "sqlite:///cropcare.db",
    "tables": {
        "predictions": "predictions",
        "model_metrics": "model_metrics",
    },
}

# ==================== STREAMLIT UI ====================
STREAMLIT_CONFIG = {
    "page_title": "CropCare AI - Crop Disease Detection",
    "page_icon": "🌿",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primaryColor": "#2E7D32",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#E8F5E9",
        "textColor": "#1B5E20",
        "font": "sans serif",
    },
    "menu_items": {
        "Get Help": "https://github.com/yourusername/CropCare-AI/issues",
        "Report a bug": "https://github.com/yourusername/CropCare-AI/issues",
        "About": "CropCare AI v1.0.0 - AI-powered Crop Disease Detection",
    },
}

# Page configuration for st.set_page_config()
PAGE_CONFIG = {
    "page_title": "CropCare AI - Crop Disease Detection",
    "page_icon": "🌿",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "Get Help": "https://github.com/yourusername/CropCare-AI/issues",
        "Report a bug": "https://github.com/yourusername/CropCare-AI/issues",
        "About": "CropCare AI v1.0.0 - AI-powered Crop Disease Detection",
    },
}

# Page configuration (legacy format for reference)
PAGES = [
    {"name": "Home", "icon": "🏠", "file": "1_Home.py"},
    {"name": "Disease Detection", "icon": "🔬", "file": "2_Disease_Detection.py"},
    {"name": "Prediction History", "icon": "📊", "file": "3_Prediction_History.py"},
    {"name": "Disease Guide", "icon": "📚", "file": "4_Disease_Guide.py"},
    {"name": "Model Information", "icon": "🤖", "file": "5_Model_Information.py"},
]

# ==================== LANGUAGES ====================
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English", "code": "en"},
    "hi": {"name": "Hindi", "native": "हिंदी", "code": "hi"},
}

DEFAULT_LANGUAGE = "en"

# ==================== PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_app"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"

# Model paths
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
METADATA_DIR = MODELS_DIR / "metadata"

BEST_MODEL_PATH = TRAINED_MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = TRAINED_MODELS_DIR / "final_model.keras"
CLASS_LABELS_PATH = METADATA_DIR / "class_labels.json"
PREPROCESSING_CONFIG_PATH = METADATA_DIR / "preprocessing_config.json"
TRAINING_HISTORY_PATH = METADATA_DIR / "training_history.json"
EVALUATION_METRICS_PATH = METADATA_DIR / "evaluation_metrics.json"

# ==================== DATASET ====================
DATASET_INFO = {
    "name": "PlantVillage",
    "source_url": "https://github.com/spMohanty/PlantVillage-Dataset",
    "kaggle_url": "https://www.kaggle.com/datasets/emmarex/plantdisease",
    "citation": (
        "Hughes, D., & Salathé, M. (2015). An open access repository of images "
        "on plant health to enable the development of mobile disease diagnostics. "
        "arXiv preprint arXiv:1511.08060."
    ),
    "license": "CC BY 4.0",
    "total_images": "~54,305",
    "crops": ["Tomato", "Potato", "Pepper"],
    "classes": 16,
    "split": "70/15/15 (train/val/test)",
}

# ==================== SEVERITY LEVELS ====================
SEVERITY_LEVELS = {
    "None": {"level": 0, "color": "#4CAF50", "description": "No disease"},
    "Low": {"level": 1, "color": "#8BC34A", "description": "Minimal impact"},
    "Medium": {"level": 2, "color": "#FFC107", "description": "Moderate impact"},
    "High": {"level": 3, "color": "#FF9800", "description": "Significant impact"},
    "Very High": {"level": 4, "color": "#F44336", "description": "Severe impact"},
}

# ==================== SPREAD METHODS ====================
SPREAD_METHODS = [
    "Wind",
    "Rain splash",
    "Water splash",
    "Wind-driven rain",
    "Contaminated tools",
    "Workers/clothing",
    "Infected seeds",
    "Infected transplants",
    "Soil",
    "Equipment",
    "Whitefly vector",
    "Mechanical contact",
    "Root contact",
    "Air currents",
    "Contaminated debris",
    "Volunteer plants",
    "Cull piles",
]

# ==================== FAVORABLE CONDITIONS ====================
FAVORABLE_CONDITIONS = [
    "High humidity",
    "Warm temperatures",
    "Cool temperatures",
    "Wet foliage",
    "Extended leaf wetness",
    "Poor ventilation",
    "Dense plantings",
    "Plant stress",
    "Poor nutrition",
    "Dusty conditions",
    "Year-round host availability",
    "High vector populations",
]

# ==================== TREATMENT CATEGORIES ====================
TREATMENT_CATEGORIES = {
    "fungicide": "Fungicide",
    "bactericide": "Bactericide",
    "miticide": "Miticide",
    "insecticide": "Insecticide",
    "biological": "Biological Control",
    "cultural": "Cultural Practice",
    "resistant_variety": "Resistant Variety",
    "sanitation": "Sanitation",
    "no_cure": "No Cure (Prevention Only)",
}

# ==================== DISCLAIMER ====================
DISCLAIMER_TEXT = (
    "⚠️ Disclaimer: This AI system provides disease predictions for educational "
    "and informational purposes only. Predictions are not a substitute for "
    "professional agricultural advice. Always consult with local agricultural "
    "extension services or certified plant pathologists for confirmed diagnosis "
    "and treatment recommendations. The developers are not responsible for "
    "any crop loss or damage resulting from reliance on this system."
)

DISCLAIMER_HINDI = (
    "⚠️ अस्वीकरण: यह AI प्रणाली केवल शैक्षिक और सूचनात्मक उद्देश्यों के लिए "
    "रोग पूर्वानुमान प्रदान करती है। पूर्वानुमान पेशेवर कृषि सलाह का विकल्प नहीं हैं। "
    "पुष्ट निदान और उपचार सिफारिशों के लिए हमेशा स्थानीय कृषि विस्तार "
    "सेवाओं या प्रमाणित प्लांट पैथोलॉजिस्ट से परामर्श करें। डेवलपर्स इस "
    "प्रणाली पर निर्भरता के परिणामस्वरूप किसी भी फसल हानि या क्षति के लिए "
    "जिम्मेदार नहीं हैं।"
)

# ==================== EXPORT ====================
__all__ = [
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "CROP_CLASSES",
    "HEALTHY_CLASSES",
    "DISEASE_CLASSES",
    "MODEL_CONFIG",
    "TRAINING_CONFIG",
    "AUGMENTATION_CONFIG",
    "IMAGENET_NORMALIZATION",
    "IMAGE_CONFIG",
    "GRADCAM_CONFIG",
    "DATABASE_CONFIG",
    "STREAMLIT_CONFIG",
    "PAGE_CONFIG",
    "PAGES",
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "DATA_DIR",
    "MODELS_DIR",
    "BEST_MODEL_PATH",
    "FINAL_MODEL_PATH",
    "CLASS_LABELS_PATH",
    "DATASET_INFO",
    "SEVERITY_LEVELS",
    "DISCLAIMER_TEXT",
    "DISCLAIMER_HINDI",
]


if __name__ == "__main__":
    print(f"Project: {PROJECT_NAME} v{PROJECT_VERSION}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Crops: {SUPPORTED_CROPS}")
    print(f"Model: {MODEL_CONFIG['architecture']}")