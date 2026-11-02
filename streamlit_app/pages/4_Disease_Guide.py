"""
CropCare AI - Disease Guide Page
Comprehensive disease information browser.
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
from src.utils.constants import CROP_CLASSES, HEALTHY_CLASSES
from src.disease_info.translations import get_translator, get_ui_translation
from src.disease_info.diseases import (
    get_disease_info,
    get_diseases_by_crop,
    get_healthy_classes,
    get_diseased_classes,
    search_diseases,
)
from streamlit_app.components.sidebar import render_sidebar


def main():
    # Render sidebar (page config is set in main.py via st.navigation)
    render_sidebar()

    # Get translator
    translator = get_translator()

    # Page title
    st.markdown(f"# 📚 {translator.t('guide_title')}")
    st.markdown(f"*{translator.t('guide_subtitle')}*")

    st.divider()

    # Search
    search_query = st.text_input(
        f"🔍 {translator.t('search_placeholder')}",
        key="guide_search",
        placeholder=translator.t('search_placeholder'),
    )

    if search_query:
        results = search_diseases(search_query)
        if results:
            st.markdown(f"### 🔍 Search Results for '{search_query}'")
            for disease in results:
                render_disease_card(disease)
        else:
            st.info(f"No diseases found matching '{search_query}'")
        st.divider()

    # Browse by crop
    st.markdown(f"## 🌱 {translator.t('select_crop')}")

    crop_tabs = st.tabs([translator.tc(crop) for crop in CROP_CLASSES.keys()])

    for i, (crop, class_names) in enumerate(CROP_CLASSES.items()):
        with crop_tabs[i]:
            render_crop_diseases(crop, class_names)

    # Disclaimer
    st.divider()
    st.warning(get_ui_translation("disclaimer", translator.get_lang()))


def render_crop_diseases(crop: str, class_names: list):
    """Render diseases for a specific crop."""
    translator = get_translator()

    # Separate healthy and diseased
    healthy_classes = [c for c in class_names if c in HEALTHY_CLASSES]
    diseased_classes = [c for c in class_names if c not in HEALTHY_CLASSES]

    # Healthy section
    if healthy_classes:
        st.markdown(f"### ✅ {translator.t('healthy_crops')}")
        for class_name in healthy_classes:
            disease_info = get_disease_info(class_name)
            if disease_info:
                render_disease_card(disease_info, compact=True)

    # Diseased section
    if diseased_classes:
        st.markdown(f"### ⚠️ {translator.t('diseased_crops')}")
        for class_name in diseased_classes:
            disease_info = get_disease_info(class_name)
            if disease_info:
                render_disease_card(disease_info)


def render_disease_card(disease_info, compact: bool = False):
    """Render a disease information card using native Streamlit components."""
    translator = get_translator()

    # Translated disease name
    display_name = translator.td(disease_info.class_name)
    severity_color = get_severity_color(disease_info.severity)
    # Map severity to translation keys (handle "Very High" -> "very_high", "Medium-High" -> "medium_high")
    severity_key_map = {
        "none": "none",
        "low": "low",
        "medium": "medium",
        "medium-high": "medium_high",
        "high": "high",
        "very high": "very_high",
    }
    severity_key = severity_key_map.get(disease_info.severity.lower(), disease_info.severity.lower())
    severity_display = translator.t(severity_key) if severity_key in ["low", "medium", "high", "very_high", "none"] else disease_info.severity

    if compact:
        # Compact view
        with st.expander(f"{'✅' if disease_info.is_healthy else '⚠️'} {display_name} — {translator.t('severity')}: {severity_display}"):
            render_disease_details(disease_info)
    else:
        # Full card using native Streamlit
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"### {'✅' if disease_info.is_healthy else '⚠️'} {display_name}")
                st.caption(f"🌱 {translator.tc(disease_info.crop)}")
            with col2:
                # Severity badge
                if disease_info.is_healthy:
                    st.success(f"{translator.t('severity')}: {severity_display}")
                else:
                    st.error(f"{translator.t('severity')}: {severity_display}")

            with st.expander(f"📖 View Details — {display_name}", expanded=False):
                render_disease_details(disease_info)


def render_disease_details(disease_info):
    """Render detailed disease information."""
    translator = get_translator()

    if disease_info.is_healthy:
        st.success(f"✅ {translator.t('healthy')}: This crop shows no signs of disease.")

    # Symptoms
    if disease_info.symptoms:
        st.markdown(f"#### 🔍 {translator.t('symptoms')}")
        for symptom in disease_info.symptoms:
            st.markdown(f"• {symptom}")

    # Causes
    if disease_info.causes:
        st.markdown(f"#### 🦠 {translator.t('causes')}")
        for cause in disease_info.causes:
            st.markdown(f"• {cause}")

    # Prevention
    if disease_info.prevention:
        st.markdown(f"#### 🛡️ {translator.t('prevention')}")
        for prev in disease_info.prevention:
            st.markdown(f"• {prev}")

    # Treatment
    if disease_info.treatment:
        st.markdown(f"#### 💊 {translator.t('treatment')}")
        for treat in disease_info.treatment:
            st.markdown(f"• {treat}")

    # Additional info
    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        severity_key_map = {
            "none": "none",
            "low": "low",
            "medium": "medium",
            "medium-high": "medium_high",
            "high": "high",
            "very high": "very_high",
        }
        severity_key = severity_key_map.get(disease_info.severity.lower(), disease_info.severity.lower())
        severity_display = translator.t(severity_key) if severity_key in ["low", "medium", "medium_high", "high", "very_high", "none"] else disease_info.severity
        st.metric(translator.t("severity"), severity_display)

    with col2:
        st.markdown(f"**{translator.t('spread_method')}**")
        for method in disease_info.spread_method:
            st.markdown(f"• {method}")

    with col3:
        st.markdown(f"**{translator.t('favorable_conditions')}**")
        for cond in disease_info.favorable_conditions:
            st.markdown(f"• {cond}")


def get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    colors = {
        "None": "#4CAF50",
        "Low": "#8BC34A",
        "Medium": "#FFC107",
        "High": "#FF9800",
        "Very High": "#F44336",
    }
    return colors.get(severity, "#9E9E9E")


if __name__ == "__main__":
    main()