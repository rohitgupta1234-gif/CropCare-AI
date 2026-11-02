"""
CropCare AI - Image Uploader Component
Handles image upload, validation, preview, and preprocessing.
"""

import streamlit as st
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, Callable
from src.utils.image_utils import (
    validate_and_load_image,
    preprocess_for_display,
    image_to_base64,
)
from src.disease_info.translations import get_translator


def render_image_uploader(
    key: str = "image_uploader",
    on_upload: Optional[Callable[[Image.Image], None]] = None,
    show_preview: bool = True,
    max_size_mb: int = 10,
) -> Optional[Image.Image]:
    """
    Render image upload component with validation and preview.

    Args:
        key: Unique key for the uploader
        on_upload: Callback function when image is uploaded
        show_preview: Whether to show image preview
        max_size_mb: Maximum file size in MB

    Returns:
        PIL Image if valid upload, None otherwise
    """
    translator = get_translator()

    st.markdown(f"#### 📤 {translator.t('upload_image')}")

    # File uploader
    uploaded_file = st.file_uploader(
        translator.t("drag_drop"),
        type=["jpg", "jpeg", "png"],
        key=key,
        help=translator.t("supported_formats"),
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Validate and load
        is_valid, message, image = validate_and_load_image(uploaded_file)

        if not is_valid:
            st.error(f"❌ {message}")
            return None

        # Success - show preview
        if show_preview:
            render_image_preview(image, uploaded_file.name)

        # Call callback if provided
        if on_upload:
            on_upload(image)

        return image

    return None


def render_image_preview(image: Image.Image, filename: str = ""):
    """Render image preview with info."""
    translator = get_translator()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"##### 🖼️ {translator.t('preview')}")
        # Display image with reasonable size
        display_img = preprocess_for_display(image)
        st.image(display_img, caption=filename, use_container_width=True)

    with col2:
        st.markdown("##### ℹ️ Image Info")
        # Get format from filename extension since PIL doesn't preserve format when loading from bytes
        file_format = "Unknown"
        if filename:
            ext = Path(filename).suffix.upper().lstrip('.')
            if ext in ('JPG', 'JPEG'):
                file_format = 'JPEG'
            elif ext in ('PNG',):
                file_format = 'PNG'
            else:
                file_format = ext

        info = {
            "Dimensions": f"{image.width} × {image.height}",
            "Aspect Ratio": f"{image.width / image.height:.2f}",
            "Mode": image.mode,
            "Format": file_format,
        }
        for key, value in info.items():
            st.markdown(f"**{key}:** {value}")

        # File size estimation
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        size_kb = len(buf.getvalue()) / 1024
        st.markdown(f"**Est. Size:** {size_kb:.1f} KB")


def render_camera_input(
    key: str = "camera_input",
    on_capture: Optional[Callable[[Image.Image], None]] = None,
) -> Optional[Image.Image]:
    """
    Render camera input for live capture.

    Args:
        key: Unique key for camera input
        on_capture: Callback when image is captured

    Returns:
        PIL Image if captured, None otherwise
    """
    translator = get_translator()

    st.markdown(f"#### 📷 {translator.t('upload_image')} (Camera)")

    camera_file = st.camera_input(
        "Take a picture",
        key=key,
        help="Allow camera access to capture leaf image",
    )

    if camera_file is not None:
        is_valid, message, image = validate_and_load_image(camera_file)

        if not is_valid:
            st.error(f"❌ {message}")
            return None

        render_image_preview(image, "camera_capture.jpg")

        if on_capture:
            on_capture(image)

        return image

    return None


def render_upload_options() -> str:
    """Render upload method selector."""
    translator = get_translator()

    option = st.radio(
        "Upload Method",
        options=["file", "camera"],
        format_func=lambda x: {
            "file": f"📁 {translator.t('upload_image')}",
            "camera": f"📷 {translator.t('upload_image')} (Camera)",
        }[x],
        horizontal=True,
        key="upload_method",
    )

    return option


def render_combined_uploader(
    key: str = "combined_uploader",
    on_upload: Optional[Callable[[Image.Image], None]] = None,
) -> Optional[Image.Image]:
    """
    Render combined file upload and camera input.

    Args:
        key: Base key for components
        on_upload: Callback when image is uploaded/captured

    Returns:
        PIL Image if valid, None otherwise
    """
    upload_method = render_upload_options()

    if upload_method == "file":
        return render_image_uploader(key=f"{key}_file", on_upload=on_upload)
    else:
        return render_camera_input(key=f"{key}_camera", on_capture=on_upload)


def validate_image_for_analysis(image: Image.Image) -> Tuple[bool, str]:
    """
    Additional validation for analysis readiness.

    Args:
        image: PIL Image to validate

    Returns:
        Tuple of (is_valid, message)
    """
    # Check if image is mostly green (leaf-like)
    # Simple heuristic: check green channel dominance
    import numpy as np
    img_array = np.array(image.resize((100, 100)))
    green_mean = img_array[:, :, 1].mean()
    red_mean = img_array[:, :, 0].mean()
    blue_mean = img_array[:, :, 2].mean()

    # Green should be dominant for leaf images
    if green_mean < red_mean * 0.8 or green_mean < blue_mean * 0.8:
        return True, "⚠️ Image may not be a leaf photo. For best results, use a clear leaf image."

    # Check brightness
    brightness = img_array.mean()
    if brightness < 30:
        return True, "⚠️ Image is quite dark. Ensure good lighting for better results."
    if brightness > 220:
        return True, "⚠️ Image is overexposed. Avoid direct sunlight glare."

    return True, "✅ Image looks good for analysis"


def render_analysis_ready_check(image: Image.Image):
    """Show analysis readiness indicator."""
    is_valid, message = validate_image_for_analysis(image)

    if "✅" in message:
        st.success(message)
    elif "⚠️" in message:
        st.warning(message)
    else:
        st.error(message)


if __name__ == "__main__":
    pass