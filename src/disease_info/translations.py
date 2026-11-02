"""
CropCare AI - Translations Module
English and Hindi translations for UI and disease information.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Translation:
    """Translation pair for a single string."""
    en: str
    hi: str

    def get(self, lang: str) -> str:
        """Get translation for language."""
        if lang == "hi":
            return self.hi
        return self.en


# UI Translations
UI_TRANSLATIONS: Dict[str, Translation] = {
    # Navigation
    "nav_home": Translation("Home", "होम"),
    "nav_detection": Translation("Disease Detection", "रोग पहचान"),
    "nav_history": Translation("Prediction History", "पूर्वानुमान इतिहास"),
    "nav_guide": Translation("Disease Guide", "रोग गाइड"),
    "nav_model_info": Translation("Model Information", "मॉडल जानकारी"),

    # Home Page
    "home_title": Translation("CropCare AI", "क्रॉपकेयर एआई"),
    "home_subtitle": Translation(
        "AI-Powered Crop Disease Detection & Prevention System",
        "एआई-संचालित फसल रोग पहचान और रोकथाम प्रणाली"
    ),
    "home_description": Translation(
        "Upload a crop leaf image to instantly detect diseases and get expert prevention advice.",
        "फसल के पत्ते की छवि अपलोड करके तुरंत रोगों का पता लगाएं और विशेषज्ञ रोकथाम सलाह प्राप्त करें।"
    ),
    "home_get_started": Translation("Get Started", "शुरू करें"),
    "home_supported_crops": Translation("Supported Crops", "समर्थित फसलें"),
    "home_how_it_works": Translation("How It Works", "यह कैसे काम करता है"),
    "step1": Translation("Upload Image", "छवि अपलोड करें"),
    "step1_desc": Translation("Take or upload a clear photo of a crop leaf", "फसल के पत्ते की स्पष्ट फोटो लें या अपलोड करें"),
    "step2": Translation("AI Analysis", "एआई विश्लेषण"),
    "step2_desc": Translation("Our deep learning model analyzes the image", "हमारा डीप लर्निंग मॉडल छवि का विश्लेषण करता है"),
    "step3": Translation("Get Results", "परिणाम प्राप्त करें"),
    "step3_desc": Translation("Receive diagnosis, confidence score & prevention tips", "निदान, आत्मविश्वास स्कोर और रोकथाम युक्तियाँ प्राप्त करें"),

    # Disease Detection Page
    "detection_title": Translation("Disease Detection", "रोग पहचान"),
    "detection_subtitle": Translation("Upload a crop leaf image for AI-powered disease analysis", "एआई-संचालित रोग विश्लेषण के लिए फसल के पत्ते की छवि अपलोड करें"),
    "upload_image": Translation("Upload Image", "छवि अपलोड करें"),
    "drag_drop": Translation("Drag and drop or click to browse", "खींचें और छोड़ें या ब्राउज़ करने के लिए क्लिक करें"),
    "supported_formats": Translation("Supported: JPG, JPEG, PNG (Max 10MB)", "समर्थित: JPG, JPEG, PNG (अधिकतम 10MB)"),
    "preview": Translation("Image Preview", "छवि पूर्वावलोकन"),
    "analyze": Translation("Analyze", "विश्लेषण करें"),
    "analyzing": Translation("Analyzing...", "विश्लेषण 중..."),
    "results": Translation("Results", "परिणाम"),
    "crop_detected": Translation("Crop Detected", "पहचानी गई फसल"),
    "disease_detected": Translation("Disease Detected", "पहचाना गया रोग"),
    "confidence": Translation("Confidence", "आत्मविश्वास"),
    "health_status": Translation("Health Status", "स्वास्थ्य स्थिति"),
    "healthy": Translation("Healthy", "स्वस्थ"),
    "diseased": Translation("Diseased", "रोगग्रस्त"),
    "disease_info": Translation("Disease Information", "रोग जानकारी"),
    "symptoms": Translation("Symptoms", "लक्षण"),
    "causes": Translation("Causes", "कारण"),
    "prevention": Translation("Prevention", "रोकथाम"),
    "treatment": Translation("Treatment Guidance", "उपचार मार्गदर्शन"),
    "gradcam_title": Translation("Model Attention (Grad-CAM)", "मॉडल ध्यान (ग्रेड-कैम)"),
    "gradcam_desc": Translation("Regions that influenced the prediction", "वे क्षेत्र जिन्होंने पूर्वानुमान को प्रभावित किया"),
    "disclaimer": Translation(
        "⚠️ Disclaimer: This AI system provides predictions for educational purposes only. "
        "Consult agricultural experts for confirmed diagnosis and treatment.",
        "⚠️ अस्वीकरण: यह AI प्रणाली केवल शैक्षिक उद्देश्यों के लिए पूर्वानुमान प्रदान करती है। "
        "पुष्ट निदान और उपचार के लिए कृषि विशेषज्ञों से परामर्श करें।"
    ),

    # History Page
    "history_title": Translation("Prediction History", "पूर्वानुमान इतिहास"),
    "history_subtitle": Translation("View and manage your past predictions", "अपने पिछले पूर्वानुमान देखें और प्रबंधित करें"),
    "search_placeholder": Translation("Search by crop, disease, or notes...", "फसल, रोग या नोट्स के अनुसार खोजें..."),
    "filter_crop": Translation("Filter by Crop", "फसल के अनुसार फ़िल्टर करें"),
    "filter_status": Translation("Filter by Status", "स्थिति के अनुसार फ़िल्टर करें"),
    "filter_all": Translation("All", "सभी"),
    "filter_healthy": Translation("Healthy", "स्वस्थ"),
    "filter_diseased": Translation("Diseased", "रोगग्रस्त"),
    "date_from": Translation("From Date", "प्रारंभ तिथि"),
    "date_to": Translation("To Date", "अंतिम तिथि"),
    "clear_filters": Translation("Clear Filters", "फ़िल्टर साफ़ करें"),
    "no_predictions": Translation("No predictions found", "कोई पूर्वानुमान नहीं मिला"),
    "delete_record": Translation("Delete", "हटाएं"),
    "view_details": Translation("View Details", "विवरण देखें"),
    "clear_history": Translation("Clear All History", "सारा इतिहास साफ़ करें"),
    "confirm_clear": Translation(
        "Are you sure you want to delete ALL prediction records? This cannot be undone.",
        "क्या आप वाकई सभी पूर्वानुमान रिकॉर्ड हटाना चाहते हैं? यह पूर्ववत नहीं किया जा सकता।"
    ),
    "confirm_delete": Translation(
        "Delete this prediction record?",
        "इस पूर्वानुमान रिकॉर्ड को हटाएं?"
    ),

    # Analytics/Dashboard
    "analytics_title": Translation("Analytics Overview", "विश्लेषण अवलोकन"),
    "health_distribution": Translation("Health Status Distribution", "स्वास्थ्य स्थिति वितरण"),
    "confidence_distribution": Translation("Confidence Distribution", "आत्मविश्वास वितरण"),
    "predictions_by_crop": Translation("Predictions by Crop", "फसल अनुसार पूर्वानुमान"),
    "top_diseases": Translation("Top Diseases", "शीर्ष रोग"),
    "predictions_over_time": Translation("Predictions Over Time", "समय के साथ पूर्वानुमान"),
    "daily_predictions": Translation("Daily Predictions", "दैनिक पूर्वानुमान"),
    "summary_metrics": Translation("Summary Metrics", "सारांश मेट्रिक्स"),
    "diseases_detected": Translation("Diseases Detected", "पहचाने गए रोग"),
    "average_confidence": Translation("Average Confidence", "औसत आत्मविश्वास"),
    "total_predictions": Translation("Total Predictions", "कुल पूर्वानुमान"),

    # Disease Guide Page
    "guide_title": Translation("Disease Guide", "रोग गाइड"),
    "guide_subtitle": Translation("Comprehensive information about crop diseases", "फसल रोगों के बारे में व्यापक जानकारी"),
    "select_crop": Translation("Select Crop", "फसल चुनें"),
    "select_disease": Translation("Select Disease", "रोग चुनें"),
    "healthy_crops": Translation("Healthy Crops", "स्वस्थ फसलें"),
    "diseased_crops": Translation("Diseased Crops", "रोगग्रस्त फसलें"),
    "severity": Translation("Severity", "गंभीरता"),
    "spread_method": Translation("Spread Method", "प्रसार विधि"),
    "favorable_conditions": Translation("Favorable Conditions", "अनुकूल स्थितियां"),
    "low": Translation("Low", "कम"),
    "medium": Translation("Medium", "मध्यम"),
    "medium_high": Translation("Medium-High", "मध्यम-उच्च"),
    "high": Translation("High", "उच्च"),
    "very_high": Translation("Very High", "बहुत उच्च"),
    "none": Translation("None", "कोई नहीं"),

    # Model Info Page
    "model_title": Translation("Model Information", "मॉडल जानकारी"),
    "model_subtitle": Translation("Technical details and performance metrics", "तकनीकी विवरण और प्रदर्शन मेट्रिक्स"),
    "architecture": Translation("Architecture", "आर्किटेक्चर"),
    "base_model": Translation("Base Model", "बेस मॉडल"),
    "input_shape": Translation("Input Shape", "इनपुट आकार"),
    "num_classes": Translation("Number of Classes", "कक्षाओं की संख्या"),
    "total_params": Translation("Total Parameters", "कुल पैरामीटर"),
    "trainable_params": Translation("Trainable Parameters", "प्रशिक्षित पैरामीटर"),
    "training_config": Translation("Training Configuration", "प्रशिक्षण कॉन्फ़िगरेशन"),
    "optimizer": Translation("Optimizer", "ऑप्टिमाइज़र"),
    "loss_function": Translation("Loss Function", "लॉस फंक्शन"),
    "batch_size": Translation("Batch Size", "बैच आकार"),
    "epochs": Translation("Epochs", "एपॉक्स"),
    "early_stopping": Translation("Early Stopping", "प्रारंभिक रोक"),
    "performance_metrics": Translation("Performance Metrics", "प्रदर्शन मेट्रिक्स"),
    "accuracy": Translation("Accuracy", "सटीकता"),
    "precision": Translation("Precision", "सटीकता (Precision)"),
    "recall": Translation("Recall", "रिकॉल"),
    "f1_score": Translation("F1-Score", "F1-स्कोर"),
    "test_samples": Translation("Test Samples", "टेस्ट नमूने"),
    "confusion_matrix": Translation("Confusion Matrix", "कन्फ्यूज़न मैट्रिक्स"),
    "training_history": Translation("Training History", "प्रशिक्षण इतिहास"),
    "loss_curve": Translation("Loss Curve", "लॉस कर्व"),
    "accuracy_curve": Translation("Accuracy Curve", "सटीकता कर्व"),
    "no_metrics": Translation("No evaluation metrics available. Train the model first.", "कोई मूल्यांकन मेट्रिक्स उपलब्ध नहीं। पहले मॉडल को प्रशिक्षित करें।"),

    # Common
    "loading": Translation("Loading...", "लोड हो रहा है..."),
    "error": Translation("Error", "त्रुटि"),
    "success": Translation("Success", "सफलता"),
    "warning": Translation("Warning", "चेतावनी"),
    "info": Translation("Information", "जानकारी"),
    "close": Translation("Close", "बंद करें"),
    "save": Translation("Save", "सहेजें"),
    "cancel": Translation("Cancel", "रद्द करें"),
    "confirm": Translation("Confirm", "पुष्टि करें"),
    "yes": Translation("Yes", "हाँ"),
    "no": Translation("No", "नहीं"),
    "language": Translation("Language", "भाषा"),
    "english": Translation("English", "अंग्रेजी"),
    "hindi": Translation("Hindi", "हिंदी"),
    "settings": Translation("Settings", "सेटिंग्स"),
    "about": Translation("About", "बारे में"),
    "version": Translation("Version", "संस्करण"),
    "github": Translation("GitHub", "गिटहब"),
    "documentation": Translation("Documentation", "दस्तावेज़ीकरण"),

    # Error Messages
    "err_no_model": Translation(
        "Model not found. Please train or download the model first.",
        "मॉडल नहीं मिला। कृपया पहले मॉडल को प्रशिक्षित या डाउनलोड करें।"
    ),
    "err_invalid_image": Translation(
        "Invalid image file. Please upload a valid JPG, JPEG, or PNG image.",
        "अमान्य छवि फ़ाइल। कृपया एक वैध JPG, JPEG, या PNG छवि अपलोड करें।"
    ),
    "err_file_too_large": Translation(
        "File too large. Maximum size is 10MB.",
        "फ़ाइल बहुत बड़ी है। अधिकतम आकार 10MB है।"
    ),
    "err_prediction_failed": Translation(
        "Prediction failed. Please try again.",
        "पूर्वानुमान विफल। कृपया पुनः प्रयास करें।"
    ),
    "err_database": Translation(
        "Database error. Please try again later.",
        "डेटाबेस त्रुटि। कृपया बाद में पुनः प्रयास करें।"
    ),
    "err_no_history": Translation(
        "No prediction history found.",
        "कोई पूर्वानुमान इतिहास नहीं मिला।"
    ),
}


# Disease-specific translations (for disease names and common terms)
DISEASE_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "Tomato_healthy": {"en": "Tomato - Healthy", "hi": "टमाटर - स्वस्थ"},
    "Tomato_Bacterial_spot": {"en": "Tomato - Bacterial Spot", "hi": "टमाटर - जीवाणु धब्बा"},
    "Tomato_Early_blight": {"en": "Tomato - Early Blight", "hi": "टमाटर - अगेती झुलसा"},
    "Tomato_Late_blight": {"en": "Tomato - Late Blight", "hi": "टमाटर - पछेती झुलसा"},
    "Tomato_Leaf_Mold": {"en": "Tomato - Leaf Mold", "hi": "टमाटर - पत्ता फफूंदी"},
    "Tomato_Septoria_leaf_spot": {"en": "Tomato - Septoria Leaf Spot", "hi": "टमाटर - सेप्टोरिया पत्ता धब्बा"},
    "Tomato_Spider_mites_Two-spotted_spider_mite": {
        "en": "Tomato - Spider Mites", "hi": "टमाटर - मकड़ी घुन"
    },
    "Tomato_Target_Spot": {"en": "Tomato - Target Spot", "hi": "टमाटर - लक्ष्य धब्बा"},
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "en": "Tomato - Yellow Leaf Curl Virus", "hi": "टमाटर - पीला पत्ता कर्ल वायरस"
    },
    "Tomato_Tomato_mosaic_virus": {"en": "Tomato - Mosaic Virus", "hi": "टमाटर - मोज़ेक वायरस"},

    "Potato_healthy": {"en": "Potato - Healthy", "hi": "आलू - स्वस्थ"},
    "Potato_Early_blight": {"en": "Potato - Early Blight", "hi": "आलू - अगेती झुलसा"},
    "Potato_Late_blight": {"en": "Potato - Late Blight", "hi": "आलू - पछेती झुलसा"},

    "Pepper_bell_healthy": {"en": "Pepper - Healthy", "hi": "शिमला मिर्च - स्वस्थ"},
    "Pepper_bell_Bacterial_spot": {"en": "Pepper - Bacterial Spot", "hi": "शिमला मिर्च - जीवाणु धब्बा"},
    "Pepper_bell__Leaf_spot": {"en": "Pepper - Leaf Spot", "hi": "शिमला मिर्च - पत्ता धब्बा"},

    # Crops
    "Tomato": {"en": "Tomato", "hi": "टमाटर"},
    "Potato": {"en": "Potato", "hi": "आलू"},
    "Pepper": {"en": "Bell Pepper", "hi": "शिमला मिर्च"},

    # Common terms
    "Healthy": {"en": "Healthy", "hi": "स्वस्थ"},
    "Diseased": {"en": "Diseased", "hi": "रोगग्रस्त"},
    "Bacterial Spot": {"en": "Bacterial Spot", "hi": "जीवाणु धब्बा"},
    "Early Blight": {"en": "Early Blight", "hi": "अगेती झुलसा"},
    "Late Blight": {"en": "Late Blight", "hi": "पछेती झुलसा"},
    "Leaf Mold": {"en": "Leaf Mold", "hi": "पत्ता फफूंदी"},
    "Septoria Leaf Spot": {"en": "Septoria Leaf Spot", "hi": "सेप्टोरिया पत्ता धब्बा"},
    "Spider Mites": {"en": "Spider Mites", "hi": "मकड़ी घुन"},
    "Target Spot": {"en": "Target Spot", "hi": "लक्ष्य धब्बा"},
    "Yellow Leaf Curl Virus": {"en": "Yellow Leaf Curl Virus", "hi": "पीला पत्ता कर्ल वायरस"},
    "Mosaic Virus": {"en": "Mosaic Virus", "hi": "मोज़ेक वायरस"},
    "Cercospora Leaf Spot": {"en": "Cercospora Leaf Spot", "hi": "सर्कोस्पोरा पत्ता धब्बा"},

    # Severity
    "Low": {"en": "Low", "hi": "कम"},
    "Medium": {"en": "Medium", "hi": "मध्यम"},
    "High": {"en": "High", "hi": "उच्च"},
    "Very High": {"en": "Very High", "hi": "बहुत उच्च"},
    "None": {"en": "None", "hi": "कोई नहीं"},

    # Symptoms/Causes/Prevention/Treatment labels
    "Symptoms": {"en": "Symptoms", "hi": "लक्षण"},
    "Causes": {"en": "Causes", "hi": "कारण"},
    "Prevention": {"en": "Prevention", "hi": "रोकथाम"},
    "Treatment": {"en": "Treatment", "hi": "उपचार"},
    "Spread Method": {"en": "Spread Method", "hi": "प्रसार विधि"},
    "Favorable Conditions": {"en": "Favorable Conditions", "hi": "अनुकूल स्थितियां"},
}


def get_ui_translation(key: str, lang: str = "en") -> str:
    """Get UI translation for a key."""
    if key in UI_TRANSLATIONS:
        return UI_TRANSLATIONS[key].get(lang)
    return key


def get_disease_translation(key: str, lang: str = "en") -> str:
    """Get disease-specific translation."""
    if key in DISEASE_TRANSLATIONS:
        return DISEASE_TRANSLATIONS[key].get(lang, DISEASE_TRANSLATIONS[key]["en"])
    return key


def translate_disease_name(class_name: str, lang: str = "en") -> str:
    """Translate disease class name to display name."""
    return get_disease_translation(class_name, lang)


def translate_crop_name(crop: str, lang: str = "en") -> str:
    """Translate crop name."""
    return get_disease_translation(crop, lang)


def translate_severity(severity: str, lang: str = "en") -> str:
    """Translate severity level."""
    return get_disease_translation(severity, lang)


def translate_list(items: list, lang: str = "en") -> list:
    """Translate a list of strings (for symptoms, causes, etc.)."""
    # For dynamic content, we can't pre-translate everything
    # Return as-is for now, but structure is ready for future expansion
    return items


class Translator:
    """Translation manager for the application."""

    def __init__(self, default_lang: str = "en"):
        self.current_lang = default_lang

    def set_language(self, lang: str):
        """Set current language."""
        if lang in ["en", "hi"]:
            self.current_lang = lang

    def get_lang(self) -> str:
        """Get current language."""
        return self.current_lang

    def t(self, key: str) -> str:
        """Translate UI key."""
        return get_ui_translation(key, self.current_lang)

    def td(self, key: str) -> str:
        """Translate disease key."""
        return get_disease_translation(key, self.current_lang)

    def tc(self, crop: str) -> str:
        """Translate crop."""
        return translate_crop_name(crop, self.current_lang)


# Global translator instance
_translator = Translator()


def get_translator() -> Translator:
    """Get global translator instance."""
    return _translator


def set_language(lang: str):
    """Set global language."""
    _translator.set_language(lang)


def get_language() -> str:
    """Get current global language."""
    return _translator.get_lang()


if __name__ == "__main__":
    # Test translations
    print("Testing translations...")
    for key in ["nav_home", "nav_detection", "detection_title", "healthy", "diseased"]:
        print(f"  {key}: EN='{get_ui_translation(key, 'en')}' HI='{get_ui_translation(key, 'hi')}'")

    for disease in ["Tomato_Early_blight", "Potato_Late_blight", "Pepper_bell_Bacterial_spot"]:
        print(f"  {disease}: EN='{get_disease_translation(disease, 'en')}' HI='{get_disease_translation(disease, 'hi')}'")

    print("Translation test passed!")