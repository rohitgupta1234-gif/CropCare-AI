"""
CropCare AI - Home Page
Landing page with project overview and navigation.
Uses native Streamlit components (no unsafe_allow_html).
"""

import sys
from pathlib import Path

# Add project root to Python path for src imports
# Required for Streamlit Cloud deployment
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.utils.constants import SUPPORTED_CROPS, CLASS_NAMES, DISCLAIMER_TEXT
from src.disease_info.translations import get_translator, get_ui_translation
from src.database.crud import get_prediction_stats
from streamlit_app.components.sidebar import render_sidebar


def main():
    # Render sidebar (page config is set in main.py via st.navigation)
    render_sidebar()

    # Get translator
    translator = get_translator()

    # Hero section using native Streamlit
    st.markdown(f"# 🌿 {translator.t('home_title')}")
    st.markdown(f"## {translator.t('home_subtitle')}")
    st.markdown(f"*{translator.t('home_description')}*")

    # Call to action - use page_link with page file path (relative to entrypoint main.py)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.page_link(
            "pages/2_Disease_Detection.py",
            label=f"🚀 {translator.t('home_get_started')}",
            icon="🚀",
            use_container_width=True,
        )

    st.divider()

    # Features grid
    st.markdown(f"## ✨ {translator.t('home_how_it_works')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 📸")
            st.markdown(f"#### {translator.t('step1')}")
            st.markdown(translator.t('step1_desc'))

    with col2:
        with st.container(border=True):
            st.markdown("### 🤖")
            st.markdown(f"#### {translator.t('step2')}")
            st.markdown(translator.t('step2_desc'))

    with col3:
        with st.container(border=True):
            st.markdown("### 📋")
            st.markdown(f"#### {translator.t('step3')}")
            st.markdown(translator.t('step3_desc'))

    st.divider()

    # Supported crops
    st.markdown(f"## 🌱 {translator.t('home_supported_crops')}")

    crop_cols = st.columns(len(SUPPORTED_CROPS))

    crop_info = {
        "Tomato": {"icon": "🍅", "classes": 10, "color": "#E53935"},
        "Potato": {"icon": "🥔", "classes": 3, "color": "#FDD835"},
        "Pepper": {"icon": "🌶️", "classes": 3, "color": "#43A047"},
    }

    for i, crop in enumerate(SUPPORTED_CROPS):
        with crop_cols[i]:
            info = crop_info.get(crop, {"icon": "🌿", "classes": 0, "color": "#2E7D32"})
            with st.container(border=True):
                st.markdown(f"### {info['icon']}")
                st.markdown(f"#### {translator.tc(crop)}")
                st.caption(f"{info['classes']} disease classes")

    st.divider()

    # Quick stats
    st.markdown("## 📈 System Statistics")

    try:
        stats = get_prediction_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Predictions", stats.get("total_predictions", 0))

        with col2:
            avg_conf = stats.get("average_confidence", 0)
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

        with col3:
            healthy = stats.get("by_health_status", {}).get("Healthy", 0)
            st.metric("Healthy Detected", healthy)

        with col4:
            diseased = stats.get("by_health_status", {}).get("Diseased", 0)
            st.metric("Diseases Detected", diseased)

    except Exception:
        st.info("Statistics will appear after you make predictions.")

    st.divider()

    # Model info summary
    st.markdown("## 🤖 Model Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Architecture:** MobileNetV2 + Custom Head
        - Pre-trained on ImageNet
        - Transfer learning for crop diseases
        - 16 disease classes across 3 crops
        - Input: 224×224 RGB images
        """)

    with col2:
        st.markdown("""
        **Training:** PlantVillage Dataset
        - ~54,000 images
        - 70/15/15 train/val/test split
        - Stratified sampling (no leakage)
        - Early stopping & LR scheduling
        """)

    st.divider()

    # Disclaimer
    st.warning(get_ui_translation("disclaimer", translator.get_lang()))

    # Footer
    st.markdown("---")
    st.caption("CropCare AI v1.0.0 | B.Tech Project")
    st.caption("Built with TensorFlow, Streamlit, and ❤️")


if __name__ == "__main__":
    main()