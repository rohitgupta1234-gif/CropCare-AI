"""
CropCare AI - Model Information Page
Technical details, performance metrics, and training history.
"""

import sys
from pathlib import Path

# Add project root to Python path for src imports
# Required for Streamlit Cloud deployment
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from src.utils.constants import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    CLASS_NAMES,
    CROP_CLASSES,
    HEALTHY_CLASSES,
    METADATA_DIR,
    EVALUATION_METRICS_PATH,
    TRAINING_HISTORY_PATH,
)
from src.disease_info.translations import get_translator, get_ui_translation
from src.model.architecture import load_model_artifacts
from streamlit_app.components.sidebar import render_sidebar


def main():
    # Render sidebar (page config is set in main.py via st.navigation)
    render_sidebar()

    # Get translator
    translator = get_translator()

    # Page title
    st.markdown(f"# 🤖 {translator.t('model_title')}")
    st.markdown(f"*{translator.t('model_subtitle')}*")

    st.divider()

    # Load model artifacts
    artifacts = load_model_artifacts()

    # Model Architecture
    render_model_architecture(artifacts)

    st.divider()

    # Training Configuration
    render_training_config()

    st.divider()

    # Performance Metrics
    render_performance_metrics()

    st.divider()

    # Training History
    render_training_history()

    st.divider()

    # Class Information
    render_class_info()

    # Disclaimer
    st.divider()
    st.warning(get_ui_translation("disclaimer", translator.get_lang()))


def render_model_architecture(artifacts: dict):
    """Render model architecture details."""
    translator = get_translator()

    st.markdown(f"## 🏗️ {translator.t('architecture')}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {translator.t('base_model')}")
        st.markdown(f"""
        - **Architecture:** {MODEL_CONFIG['architecture']}
        - **Alpha:** {MODEL_CONFIG['alpha']}
        - **Pre-trained weights:** {MODEL_CONFIG['weights']}
        - **Include top:** {MODEL_CONFIG['include_top']}
        - **Pooling:** {MODEL_CONFIG['pooling']}
        - **Input shape:** {MODEL_CONFIG['input_shape']}
        """)

    with col2:
        st.markdown(f"### {translator.t('architecture')} - Custom Head")
        st.markdown(f"""
        - **Dense units:** {MODEL_CONFIG['dense_units']}
        - **Dropout rates:** {MODEL_CONFIG['dropout_rates']}
        - **L2 regularization:** {MODEL_CONFIG['l2_regularization']}
        - **Label smoothing:** {MODEL_CONFIG['label_smoothing']}
        - **Output classes:** {len(CLASS_NAMES)}
        """)

    # Model config from artifacts
    if artifacts.get("model_config"):
        config = artifacts["model_config"]
        st.markdown("### Saved Model Configuration")
        st.json(config)


def render_training_config():
    """Render training configuration."""
    translator = get_translator()

    st.markdown(f"## ⚙️ {translator.t('training_config')}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        - **Optimizer:** Adam
        - **Initial Learning Rate:** {TRAINING_CONFIG['initial_lr']}
        - **Fine-tune Learning Rate:** {TRAINING_CONFIG['fine_tune_lr']}
        - **Loss Function:** Categorical Crossentropy (with label smoothing)
        - **Batch Size:** {TRAINING_CONFIG['batch_size']}
        - **Epochs:** {TRAINING_CONFIG['epochs']}
        """)

    with col2:
        st.markdown(f"""
        - **Early Stopping Patience:** {TRAINING_CONFIG['early_stopping_patience']}
        - **Reduce LR Patience:** {TRAINING_CONFIG['reduce_lr_patience']}
        - **Reduce LR Factor:** {TRAINING_CONFIG['reduce_lr_factor']}
        - **Min Learning Rate:** {TRAINING_CONFIG['min_lr']}
        - **Data Splits:** {TRAINING_CONFIG['data_splits']}
        - **Random Seed:** {TRAINING_CONFIG['random_seed']}
        """)


def render_performance_metrics():
    """Render model performance metrics."""
    translator = get_translator()

    st.markdown(f"## 📊 {translator.t('performance_metrics')}")

    # Try to load evaluation metrics
    metrics = load_evaluation_metrics()

    if metrics:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(translator.t("accuracy"), f"{metrics.get('accuracy', 0)*100:.2f}%")
        with col2:
            st.metric(translator.t("precision"), f"{metrics.get('precision', 0)*100:.2f}%")
        with col3:
            st.metric(translator.t("recall"), f"{metrics.get('recall', 0)*100:.2f}%")
        with col4:
            st.metric(translator.t("f1_score"), f"{metrics.get('f1_score', 0)*100:.2f}%")

        # Per-class metrics
        if "per_class" in metrics:
            st.markdown("### Per-Class Metrics")
            df = pd.DataFrame(metrics["per_class"])
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Confusion Matrix
        if "confusion_matrix" in metrics:
            st.markdown(f"### {translator.t('confusion_matrix')}")
            render_confusion_matrix(metrics["confusion_matrix"])

        # Test samples
        if "test_samples" in metrics:
            st.info(f"{translator.t('test_samples')}: {metrics['test_samples']}")

    else:
        st.info(get_ui_translation("no_metrics", translator.get_lang()))
        st.markdown("""
        **To generate metrics:**
        1. Train the model using the training scripts
        2. Run evaluation on the held-out test set
        3. Metrics will be saved to `models/metadata/evaluation_metrics.json`
        """)


def render_confusion_matrix(cm_data: list):
    """Render confusion matrix as interactive heatmap."""
    cm = np.array(cm_data)

    # Normalize for display
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=CLASS_NAMES,
        y=CLASS_NAMES,
        colorscale='Greens',
        text=np.round(cm_norm, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Normalized Count"),
    ))

    fig.update_layout(
        title="Confusion Matrix (Normalized)",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=600,
        xaxis_tickangle=-45,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_training_history():
    """Render training history charts."""
    translator = get_translator()

    st.markdown(f"## 📈 {translator.t('training_history')}")

    history = load_training_history()

    if history and any(history.values()):
        # Loss curve
        st.markdown(f"### {translator.t('loss_curve')}")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history.get("loss", []),
            name="Training Loss",
            line=dict(color="#E53935"),
        ))
        fig_loss.add_trace(go.Scatter(
            y=history.get("val_loss", []),
            name="Validation Loss",
            line=dict(color="#1E88E5"),
        ))
        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode="x unified",
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # Accuracy curve
        st.markdown(f"### {translator.t('accuracy_curve')}")
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            y=history.get("accuracy", []),
            name="Training Accuracy",
            line=dict(color="#43A047"),
        ))
        fig_acc.add_trace(go.Scatter(
            y=history.get("val_accuracy", []),
            name="Validation Accuracy",
            line=dict(color="#FB8C00"),
        ))
        fig_acc.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            hovermode="x unified",
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # Learning rate curve
        if "lr" in history and history["lr"]:
            st.markdown("### Learning Rate Schedule")
            fig_lr = go.Figure()
            fig_lr.add_trace(go.Scatter(
                y=history["lr"],
                name="Learning Rate",
                line=dict(color="#8E24AA"),
            ))
            fig_lr.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Learning Rate",
                yaxis_type="log",
            )
            st.plotly_chart(fig_lr, use_container_width=True)

    else:
        st.info("Training history not available. Train the model to generate history.")


def render_class_info():
    """Render class distribution and information."""
    translator = get_translator()

    st.markdown(f"## 🏷️ {translator.t('num_classes')} Classes")

    # Class table
    class_data = []
    for i, class_name in enumerate(CLASS_NAMES):
        crop = None
        for c, classes in CROP_CLASSES.items():
            if class_name in classes:
                crop = c
                break

        is_healthy = class_name in HEALTHY_CLASSES

        class_data.append({
            "Index": i,
            "Class Name": class_name,
            "Crop": crop,
            "Status": "✅ Healthy" if is_healthy else "⚠️ Diseased",
        })

    df = pd.DataFrame(class_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Distribution
    st.markdown("### Class Distribution")
    crop_counts = {crop: len(classes) for crop, classes in CROP_CLASSES.items()}

    fig = px.pie(
        values=list(crop_counts.values()),
        names=list(crop_counts.keys()),
        title="Classes per Crop",
        color_discrete_sequence=["#E53935", "#FDD835", "#43A047"],
    )
    st.plotly_chart(fig, use_container_width=True)


def load_evaluation_metrics() -> dict:
    """Load evaluation metrics from file."""
    try:
        if EVALUATION_METRICS_PATH.exists():
            with open(EVALUATION_METRICS_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def load_training_history() -> dict:
    """Load training history from file."""
    try:
        if TRAINING_HISTORY_PATH.exists():
            with open(TRAINING_HISTORY_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


if __name__ == "__main__":
    main()