# CropCare AI 🌿

**AI-Powered Crop Disease Detection & Prevention System**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Overview

CropCare AI is an intelligent web application that helps farmers and agricultural professionals detect crop diseases from leaf images using deep learning. Built with TensorFlow/Keras (MobileNetV2 transfer learning) and deployed via Streamlit, the system provides real-time disease diagnosis, expert prevention advice, and comprehensive analytics.

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **🔬 Disease Detection** | Upload leaf images (JPG/PNG) for instant AI-powered disease classification |
| **🌱 Multi-Crop Support** | Detects 12 diseases + healthy states across **Tomato, Potato, and Pepper** |
| **🎯 Grad-CAM Visualization** | See which image regions influenced the model's decision |
| **📚 Disease Guide** | Comprehensive database with symptoms, causes, prevention & treatment |
| **📊 Prediction History** | Track all predictions with analytics dashboard & charts |
| **🤖 Model Information** | View architecture, training config, and performance metrics |
| **🌐 Bilingual Support** | English & Hindi interface with full disease translations |
| **💾 SQLite Storage** | Local persistence of predictions and metadata |

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Deep Learning** | TensorFlow 2.16+, Keras, MobileNetV2 (Transfer Learning) |
| **Web Framework** | Streamlit 1.29+ (Multi-page with `st.navigation`) |
| **Data Processing** | NumPy, Pandas, OpenCV, Pillow, Albumentations |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Database** | SQLAlchemy 2.0 + SQLite |
| **Explainability** | Custom Grad-CAM / Grad-CAM++ Implementation |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
CropCare-AI/
├── streamlit_app/
│   ├── main.py                 # Application entry point with st.navigation
│   ├── pages/                  # 5 Streamlit pages
│   │   ├── 1_Home.py           # Landing page with navigation
│   │   ├── 2_Disease_Detection.py  # Image upload & prediction
│   │   ├── 3_Prediction_History.py # History & analytics dashboard
│   │   ├── 4_Disease_Guide.py  # Disease information browser
│   │   └── 5_Model_Information.py # Model architecture & metrics
│   └── components/             # Reusable UI components
│       ├── sidebar.py          # Language selector, quick stats
│       ├── image_uploader.py   # File upload with validation
│       ├── result_card.py      # Prediction results display
│       ├── gradcam_display.py  # Grad-CAM visualization
│       └── history_table.py    # Filterable history table
├── src/
│   ├── config.py               # Central configuration (paths, constants)
│   ├── model/
│   │   ├── inference.py        # Model loading, preprocessing, prediction
│   │   └── architecture.py     # Model building utilities
│   ├── gradcam/
│   │   └── gradcam.py          # Grad-CAM & Grad-CAM++ implementations
│   ├── database/
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   └── crud.py             # Database operations
│   ├── disease_info/
│   │   ├── diseases.py         # Disease metadata (16 classes)
│   │   └── translations.py     # English/Hindi translations
│   ├── data/
│   │   └── validation.py       # Input validation utilities
│   └── utils/
│       ├── constants.py        # UI constants, page config
│       ├── helpers.py          # Utility functions
│       └── image_utils.py      # Image processing utilities
├── models/
│   ├── checkpoints/
│   │   └── best_model.keras    # Trained MobileNetV2 model (15 classes)
│   └── metadata/
│       ├── class_labels.json   # Class index → label mapping
│       └── preprocessing_config.json  # ImageNet normalization params
├── data/                       # Training data (excluded from repo)
├── cropcare.db                 # SQLite database (auto-created)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- 4GB+ RAM (for model loading)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/rohitgupta1234-gif/CropCare-AI.git
cd CropCare-AI

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python -m streamlit run streamlit_app/main.py
```

The app will open at **http://localhost:8501**

---

## 📱 Application Pages

### 1. Home 🏠
- Project overview and quick navigation
- Supported crops summary
- System statistics (after predictions)
- Model overview

### 2. Disease Detection 🔬
- Upload leaf image (JPG/PNG, max 10MB)
- Camera capture support
- Real-time AI prediction with confidence scores
- Grad-CAM heatmap visualization
- Save predictions to history

### 3. Prediction History 📊
- Searchable & filterable prediction table
- Analytics dashboard with charts:
  - Health status distribution (pie)
  - Confidence distribution (histogram)
  - Predictions by crop (bar)
  - Top diseases (horizontal bar)
  - Daily predictions over time (line)
- Summary metrics
- Bulk delete / export

### 4. Disease Guide 📚
- Browse diseases by crop (Tomato, Potato, Pepper)
- Search by name or symptoms
- Detailed cards with:
  - Symptoms, Causes, Prevention, Treatment
  - Severity level (translated)
  - Spread methods & favorable conditions
- Bilingual: English / Hindi

### 5. Model Information 🤖
- Architecture: MobileNetV2 + Custom Head
- Training configuration (optimizer, LR, epochs, etc.)
- Performance metrics (accuracy, precision, recall, F1)
- Confusion matrix (if evaluated)
- Training history charts (loss, accuracy, LR)
- Class distribution

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| **Architecture** | MobileNetV2 (α=1.0) + Custom Classification Head |
| **Base Weights** | ImageNet Pre-trained |
| **Input Shape** | 224 × 224 × 3 (RGB) |
| **Classes** | 15 (12 diseases + 3 healthy) |
| **Training Data** | PlantVillage Dataset (~54,000 images) |
| **Split** | 70% Train / 15% Val / 15% Test (Stratified) |
| **Optimization** | Adam, LR=1e-4 → 1e-5 (fine-tune) |
| **Regularization** | Dropout (0.3, 0.2), L2, Label Smoothing |

### Supported Classes

**Tomato (11):** Bacterial Spot, Early Blight, Healthy, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus

**Potato (3):** Early Blight, Late Blight, Healthy

**Pepper (2):** Bacterial Spot, Healthy

---

## 🌐 Deployment

### Streamlit Community Cloud

1. Fork this repository to your GitHub account
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub and select this repository
4. Set **Main file path**: `streamlit_app/main.py`
5. Deploy!

> **Note:** The SQLite database is ephemeral on Streamlit Cloud. For persistent storage, consider PostgreSQL or cloud storage.

### Local Docker (Optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t cropcare-ai .
docker run -p 8501:8501 cropcare-ai
```

---

## 🔧 Configuration

Key settings in `src/config.py`:

```python
# Model paths (auto-detected)
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.keras"

# Database (auto-created)
DATABASE_PATH = PROJECT_ROOT / "cropcare.db"

# Image limits
MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}

# Languages
SUPPORTED_LANGUAGES = ["en", "hi"]
```

---

## 📊 Database Schema

```sql
-- Predictions table
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    crop TEXT NOT NULL,
    disease TEXT NOT NULL,
    confidence REAL NOT NULL,
    health_status TEXT NOT NULL,  -- 'Healthy' or 'Diseased'
    image_path TEXT,
    gradcam_path TEXT,
    language TEXT DEFAULT 'en',
    notes TEXT
);

-- Model metrics table
CREATE TABLE model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_version TEXT NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    test_samples INTEGER,
    metrics_json TEXT
);
```

---

## 🧪 Development

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Code Formatting
```bash
black src/ streamlit_app/
isort src/ streamlit_app/
flake8 src/ streamlit_app/
```

### Type Checking
```bash
mypy src/ streamlit_app/
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** - Hughes & Salathé (2015) - [GitHub](https://github.com/spMohanty/PlantVillage-Dataset)
- **MobileNetV2** - Sandler et al. (2018) - Google Research
- **Grad-CAM** - Selvaraju et al. (2017) - [Paper](https://arxiv.org/abs/1610.02391)
- **Streamlit** - For the amazing web framework

---

## 📧 Contact

**Rohit Gupta** - B.Tech Project  
GitHub: [@rohitgupta1234-gif](https://github.com/rohitgupta1234-gif)

---

<div align="center">
  <strong>Built with ❤️ for sustainable agriculture</strong>
</div>