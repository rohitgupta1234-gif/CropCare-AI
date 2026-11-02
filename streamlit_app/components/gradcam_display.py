"""
CropCare AI - Grad-CAM Display Component
Visualizes model attention heatmaps for interpretability.
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple
from src.gradcam.gradcam import GradCAM, generate_gradcam_for_streamlit
from src.model.inference import get_inference_engine
from src.disease_info.translations import get_translator
from src.config import CLASS_NAMES


def render_gradcam_section(
    image: Image.Image,
    predicted_class_index: int,
    model=None,
    show_original: bool = True,
    show_heatmap: bool = True,
    show_overlay: bool = True,
):
    """
    Render Grad-CAM visualization section.

    Args:
        image: Original PIL image
        predicted_class_index: Predicted class index
        model: Keras model (loads default if None)
        show_original: Show original image
        show_heatmap: Show heatmap
        show_overlay: Show overlay
    """
    translator = get_translator()

    if model is None:
        try:
            inference_engine = get_inference_engine()
            model = inference_engine.model
        except Exception as e:
            st.error(f"Could not load model for Grad-CAM: {e}")
            return

    st.markdown("---")
    st.markdown(f"#### 🎯 {translator.t('gradcam_title')}")

    try:
        # Generate Grad-CAM
        with st.spinner(translator.t("analyzing")):
            heatmap, overlay = generate_gradcam_for_streamlit(
                image, model, predicted_class_index
            )

        # Display in columns
        cols = []
        if show_original:
            cols.append("Original")
        if show_heatmap:
            cols.append("Heatmap")
        if show_overlay:
            cols.append("Overlay")

        if not cols:
            return

        st_cols = st.columns(len(cols))

        idx = 0
        if show_original:
            with st_cols[idx]:
                st.markdown("**Original Image**")
                # Resize original to match heatmap size for comparison
                display_img = image.resize((224, 224), Image.Resampling.LANCZOS)
                st.image(display_img, use_container_width=True)
            idx += 1

        if show_heatmap:
            with st_cols[idx]:
                st.markdown("**Attention Heatmap**")
                # Display heatmap using matplotlib for colorbar
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Attention')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            idx += 1

        if show_overlay:
            with st_cols[idx]:
                st.markdown("**Grad-CAM Overlay**")
                st.image(overlay, use_container_width=True)
            idx += 1

        # Explanation
        class_name = CLASS_NAMES[predicted_class_index] if predicted_class_index < len(CLASS_NAMES) else f"Class {predicted_class_index}"
        st.caption(
            f"{translator.t('gradcam_desc')}: **{class_name}**\n\n"
            "Red/yellow regions indicate areas that most influenced the prediction. "
            "This helps verify the model is focusing on relevant leaf features "
            "(lesions, spots, discoloration) rather than background."
        )

    except Exception as e:
        st.error(f"Grad-CAM generation failed: {e}")
        st.info("Grad-CAM requires a trained model with accessible convolutional layers.")


def render_gradcam_comparison(
    image: Image.Image,
    top_predictions: list,
    model=None,
    max_display: int = 3,
):
    """
    Render Grad-CAM for multiple top predictions for comparison.

    Args:
        image: Original PIL image
        top_predictions: List of prediction dicts with class_index
        model: Keras model
        max_display: Maximum number of Grad-CAMs to show
    """
    translator = get_translator()

    if model is None:
        try:
            inference_engine = get_inference_engine()
            model = inference_engine.model
        except Exception:
            return

    st.markdown(f"#### 🔍 {translator.t('gradcam_title')} - Top Predictions Comparison")

    for i, pred in enumerate(top_predictions[:max_display]):
        class_idx = pred.get("class_index", 0)
        class_name = pred.get("class_name", "Unknown")
        confidence = pred.get("confidence", 0)

        with st.expander(f"#{i+1}: {class_name} ({confidence*100:.1f}%)", expanded=(i==0)):
            try:
                heatmap, overlay = generate_gradcam_for_streamlit(
                    image, model, class_idx
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Heatmap**")
                    fig, ax = plt.subplots(figsize=(4, 4))
                    im = ax.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.markdown("**Overlay**")
                    st.image(overlay, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to generate Grad-CAM for {class_name}: {e}")


def create_gradcam_figure(
    heatmap: np.ndarray,
    title: str = "Grad-CAM Heatmap",
    colormap: str = 'jet',
) -> plt.Figure:
    """Create a matplotlib figure for heatmap display."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Attention Weight')
    plt.tight_layout()
    return fig


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet',
) -> Image.Image:
    """
    Create overlay of heatmap on original image.

    Args:
        image: Original PIL image
        heatmap: Heatmap array (0-1)
        alpha: Overlay transparency
        colormap: Matplotlib colormap name

    Returns:
        PIL Image with overlay
    """
    import cv2

    # Resize heatmap to image size
    img_array = np.array(image.resize((224, 224), Image.Resampling.LANCZOS))
    h, w = img_array.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)

    return Image.fromarray(overlay)


def render_gradcam_standalone(
    image: Image.Image,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    class_name: str,
    confidence: float,
):
    """Render standalone Grad-CAM visualization (for history view)."""
    translator = get_translator()

    st.markdown(f"#### 🎯 {translator.t('gradcam_title')}: {class_name} ({confidence*100:.1f}%)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original**")
        display_img = image.resize((224, 224), Image.Resampling.LANCZOS)
        st.image(display_img, use_container_width=True)

    with col2:
        st.markdown("**Heatmap**")
        fig = create_gradcam_figure(heatmap, f"Heatmap - {class_name}")
        st.pyplot(fig)
        plt.close()

    with col3:
        st.markdown("**Overlay**")
        st.image(overlay, use_container_width=True)


if __name__ == "__main__":
    pass