"""
CropCare AI - Result Card Component
Displays prediction results with confidence, disease info, and Grad-CAM.
Uses native Streamlit components (no unsafe_allow_html).
"""

import streamlit as st
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
from src.disease_info.diseases import get_disease_info
from src.disease_info.translations import get_translator


def render_result_card(
    prediction: Dict[str, Any],
    gradcam_overlay: Optional[Image.Image] = None,
    gradcam_heatmap: Optional[np.ndarray] = None,
    show_details: bool = True,
):
    """
    Render a comprehensive result card for a prediction using native Streamlit components.

    Args:
        prediction: Prediction dictionary from model inference
        gradcam_overlay: Grad-CAM overlay image (PIL)
        gradcam_heatmap: Grad-CAM heatmap array
        show_details: Whether to show detailed disease info
    """
    translator = get_translator()
    lang = translator.get_lang()

    # Extract prediction data
    crop = prediction.get("crop", "Unknown")
    disease = prediction.get("disease", "Unknown")
    confidence = prediction.get("confidence", 0.0)
    health_status = prediction.get("health_status", "Unknown")
    all_probs = prediction.get("all_probabilities", {})
    top_predictions = prediction.get("predictions", [])

    # Health status styling
    is_healthy = health_status == "Healthy"
    status_color = "normal" if is_healthy else "inverse"

    # Main result card using st.container with border
    with st.container(border=True):
        # Header row
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(translator.t("results"))
            st.caption(translator.t("detection_subtitle"))
        with col2:
            status_text = f"{'✅' if is_healthy else '⚠️'} {translator.td(health_status)}"
            if is_healthy:
                st.success(status_text)
            else:
                st.error(status_text)

        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label=translator.t("crop_detected"),
                value=translator.tc(crop),
            )
        with col2:
            st.metric(
                label=translator.t("disease_detected"),
                value=translator.td(disease),
            )
        with col3:
            st.metric(
                label=translator.t("confidence"),
                value=f"{confidence*100:.1f}%",
                delta_color="normal" if is_healthy else "inverse",
            )

        # Status summary
        st.info(f"**{translator.t('health_status')}:** {translator.td(health_status)} | **{translator.t('confidence')}:** {confidence*100:.1f}%")

        # Confidence progress bar
        st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")

    # Top predictions if available
    if top_predictions and len(top_predictions) > 1:
        render_top_predictions(top_predictions, lang)

    # Grad-CAM visualization
    if gradcam_overlay is not None:
        render_gradcam_section(gradcam_overlay, gradcam_heatmap, lang)

    # Detailed disease information
    if show_details:
        if disease in ["Tomato_healthy", "Potato_healthy", "Pepper_bell_healthy"]:
            render_healthy_info(crop, lang)
        else:
            render_disease_details(disease, lang)

    # Disclaimer
    render_disclaimer(lang)


def render_top_predictions(predictions: List[Dict], lang: str):
    """Render top-k predictions as a table."""
    st.markdown("#### 📊 Top Predictions")

    pred_data = []
    for i, pred in enumerate(predictions[:5]):
        pred_data.append({
            "Rank": i + 1,
            "Disease": pred.get("class_name", "Unknown"),
            "Crop": pred.get("crop", "Unknown"),
            "Confidence": f"{pred.get('confidence', 0)*100:.1f}%",
            "Status": "✅ Healthy" if pred.get("is_healthy") else "⚠️ Diseased",
        })

    st.dataframe(pred_data, use_container_width=True, hide_index=True)


def render_gradcam_section(overlay: Image.Image, heatmap: Optional[np.ndarray], lang: str):
    """Render Grad-CAM visualization section."""
    st.markdown("---")
    st.markdown("#### 🎯 Model Attention (Grad-CAM)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

    with col2:
        st.markdown("**Heatmap**")
        if heatmap is not None:
            # Display heatmap
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(heatmap, cmap='jet')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close()
        else:
            st.image(overlay, caption="Heatmap Overlay", use_container_width=True)

    st.caption(
        "Red/yellow regions indicate areas that most influenced the prediction. "
        "This helps verify the model is focusing on relevant leaf features."
    )


def render_disease_details(disease_class: str, lang: str):
    """Render detailed disease information."""
    from src.disease_info.translations import translate_disease_name

    disease_info = get_disease_info(disease_class)
    if not disease_info:
        return

    st.markdown("---")
    st.markdown(f"#### 📋 {translate_disease_name(disease_class, lang)}")

    t = get_translator()

    # Symptoms
    if disease_info.symptoms:
        with st.expander(f"🔍 {t.t('symptoms')}", expanded=True):
            for symptom in disease_info.symptoms:
                st.markdown(f"• {symptom}")

    # Causes
    if disease_info.causes:
        with st.expander(f"🦠 {t.t('causes')}"):
            for cause in disease_info.causes:
                st.markdown(f"• {cause}")

    # Prevention
    if disease_info.prevention:
        with st.expander(f"🛡️ {t.t('prevention')}", expanded=True):
            for prev in disease_info.prevention:
                st.markdown(f"• {prev}")

    # Treatment
    if disease_info.treatment:
        with st.expander(f"💊 {t.t('treatment')}"):
            for treat in disease_info.treatment:
                st.markdown(f"• {treat}")

    # Additional info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t.t("severity"), disease_info.severity)
    with col2:
        st.markdown(f"**{t.t('spread_method')}**")
        for method in disease_info.spread_method:
            st.markdown(f"• {method}")
    with col3:
        st.markdown(f"**{t.t('favorable_conditions')}**")
        for cond in disease_info.favorable_conditions:
            st.markdown(f"• {cond}")


def render_healthy_info(crop: str, lang: str):
    """Render information for healthy crops."""
    from src.disease_info.translations import translate_disease_name

    healthy_class = f"{crop}_healthy"
    disease_info = get_disease_info(healthy_class)

    st.markdown("---")
    st.markdown(f"#### ✅ {translate_disease_name(healthy_class, lang)}")

    t = get_translator()

    if disease_info:
        with st.expander(f"🌿 {t.t('symptoms')}", expanded=True):
            for symptom in disease_info.symptoms:
                st.markdown(f"• {symptom}")

        with st.expander(f"✅ {t.t('prevention')} (Best Practices)"):
            for prev in disease_info.prevention:
                st.markdown(f"• {prev}")


def render_disclaimer(lang: str):
    """Render disclaimer."""
    from src.disease_info.translations import get_ui_translation

    st.markdown("---")
    st.warning(get_ui_translation("disclaimer", lang))


def render_simple_result(prediction: Dict[str, Any]):
    """Render a simple compact result for history view."""
    crop = prediction.get("crop", "Unknown")
    disease = prediction.get("disease", "Unknown")
    confidence = prediction.get("confidence", 0.0)
    health_status = prediction.get("health_status", "Unknown")
    timestamp = prediction.get("timestamp", "")

    is_healthy = health_status == "Healthy"
    status_icon = "✅" if is_healthy else "⚠️"

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{crop}** | {disease}")
            st.caption(timestamp)
        with col2:
            st.metric(label="Confidence", value=f"{confidence*100:.1f}%")
            if is_healthy:
                st.success(f"{status_icon} {health_status}")
            else:
                st.error(f"{status_icon} {health_status}")


if __name__ == "__main__":
    pass