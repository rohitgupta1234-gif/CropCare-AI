"""
CropCare AI - Disease Detection Page
Upload image, get prediction, view results with Grad-CAM.
"""

import sys
from pathlib import Path

# Add project root to Python path for src imports
# Required for Streamlit Cloud deployment
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image
import numpy as np
from src.utils.constants import DISCLAIMER_TEXT
from src.disease_info.translations import get_translator, get_ui_translation
from src.disease_info.diseases import get_disease_info
from src.model.inference import predict_image, get_inference_engine
from src.gradcam.gradcam import GradCAM, generate_gradcam_for_streamlit
from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.image_uploader import (
    render_combined_uploader,
    render_analysis_ready_check,
)
from streamlit_app.components.result_card import render_result_card
from streamlit_app.components.gradcam_display import render_gradcam_section
from src.database.crud import create_prediction
from src.utils.image_utils import save_uploaded_image, preprocess_for_display
from src.config import CLASS_NAMES, DATA_DIR
import io


def main():
    # Render sidebar (page config is set in main.py via st.navigation)
    render_sidebar()

    # Get translator
    translator = get_translator()

    # Page title
    st.markdown(f"# 🔬 {translator.t('detection_title')}")
    st.markdown(f"*{translator.t('detection_subtitle')}*")

    st.divider()

    # Initialize session state
    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None
    if "gradcam_overlay" not in st.session_state:
        st.session_state["gradcam_overlay"] = None
    if "gradcam_heatmap" not in st.session_state:
        st.session_state["gradcam_heatmap"] = None

    # Image upload section
    st.markdown("## 📤 Upload Image")

    def handle_upload(image: Image.Image):
        st.session_state["uploaded_image"] = image
        st.session_state["prediction_result"] = None
        st.session_state["gradcam_overlay"] = None
        st.session_state["gradcam_heatmap"] = None

    uploaded_image = render_combined_uploader(
        key="detection_upload",
        on_upload=handle_upload,
    )

    # Show analysis readiness if image uploaded
    if st.session_state["uploaded_image"] is not None:
        render_analysis_ready_check(st.session_state["uploaded_image"])

    st.divider()

    # Analysis button
    if st.session_state["uploaded_image"] is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                f"🔍 {translator.t('analyze')}",
                type="primary",
                use_container_width=True,
                key="analyze_button",
            ):
                run_analysis()

    # Display results
    if st.session_state["prediction_result"] is not None:
        st.divider()
        st.markdown("## 📋 Results")

        render_result_card(
            prediction=st.session_state["prediction_result"],
            gradcam_overlay=st.session_state["gradcam_overlay"],
            gradcam_heatmap=st.session_state["gradcam_heatmap"],
            show_details=True,
        )

        # Save to database
        if st.button("💾 Save to History", key="save_to_history"):
            save_prediction_to_history()

    # Disclaimer
    st.divider()
    st.warning(get_ui_translation("disclaimer", translator.get_lang()))


def run_analysis():
    """Run disease detection analysis."""
    translator = get_translator()

    with st.spinner(translator.t("analyzing")):
        try:
            # Get prediction
            result = predict_image(st.session_state["uploaded_image"])

            if "error" in result:
                st.error(f"❌ {result['error']}")
                return

            st.session_state["prediction_result"] = result

            # Generate Grad-CAM
            try:
                inference_engine = get_inference_engine()
                model = inference_engine.model

                class_index = result["predictions"][0]["class_index"]

                heatmap, overlay = generate_gradcam_for_streamlit(
                    st.session_state["uploaded_image"], model, class_index
                )

                st.session_state["gradcam_heatmap"] = heatmap
                st.session_state["gradcam_overlay"] = Image.fromarray(overlay)

            except Exception as e:
                st.warning(f"Grad-CAM generation failed: {e}")

            st.success(f"✅ {translator.t('success')}! Disease detected: {result['disease']}")

        except Exception as e:
            st.error(f"❌ {translator.t('err_prediction_failed')}: {e}")


def save_prediction_to_history():
    """Save current prediction to database."""
    translator = get_translator()

    result = st.session_state["prediction_result"]
    if not result:
        return

    try:
        # Save image
        upload_dir = DATA_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        from src.utils.helpers import generate_timestamp_id
        img_filename = f"pred_{generate_timestamp_id()}.png"
        img_path = upload_dir / img_filename

        st.session_state["uploaded_image"].save(img_path)

        # Save Grad-CAM if available
        gradcam_path = None
        if st.session_state["gradcam_overlay"] is not None:
            gradcam_filename = f"gradcam_{generate_timestamp_id()}.png"
            gradcam_path = upload_dir / gradcam_filename
            st.session_state["gradcam_overlay"].save(gradcam_path)

        # Create prediction record
        create_prediction(
            crop=result["crop"],
            disease=result["disease"],
            confidence=result["confidence"],
            health_status=result["health_status"],
            image_path=str(img_path),
            gradcam_path=str(gradcam_path) if gradcam_path else None,
            language=translator.get_lang(),
        )

        st.success("✅ Prediction saved to history!")

    except Exception as e:
        st.error(f"Failed to save: {e}")


if __name__ == "__main__":
    main()