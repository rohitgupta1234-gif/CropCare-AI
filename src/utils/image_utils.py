"""
CropCare AI - Image Utilities
Image processing, validation, and conversion utilities.
"""

import io
import base64
from pathlib import Path
from typing import Tuple, Optional, Union, List
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

from src.config import (
    IMAGE_SIZE,
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE_BYTES,
)
from src.data.validation import ImageValidator


def load_image(file_path: Union[str, Path]) -> Image.Image:
    """Load image from file path."""
    return Image.open(file_path).convert("RGB")


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load image from bytes."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 string."""
    img_bytes = image_to_bytes(image, format)
    return base64.b64encode(img_bytes).decode("utf-8")


def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL image."""
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def resize_image(
    image: Image.Image,
    size: Tuple[int, int] = IMAGE_SIZE,
    maintain_aspect: bool = True,
) -> Image.Image:
    """Resize image with optional aspect ratio preservation."""
    if maintain_aspect:
        # Use ImageOps.fit for smart cropping or thumbnail for fitting
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    else:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def crop_center(image: Image.Image, crop_size: Tuple[int, int]) -> Image.Image:
    """Crop center of image to specified size."""
    w, h = image.size
    cw, ch = crop_size
    left = (w - cw) // 2
    top = (h - ch) // 2
    right = left + cw
    bottom = top + ch
    return image.crop((left, top, right, bottom))


def validate_and_load_image(uploaded_file) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validate and load uploaded file from Streamlit.
    Returns (is_valid, error_message, PIL_image).
    """
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
        return False, f"File too large ({uploaded_file.size / 1024 / 1024:.1f}MB > 10MB)", None

    # Check format
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {file_ext}. Use JPG, JPEG, or PNG.", None

    try:
        # Load and validate
        image = Image.open(uploaded_file)
        image.verify()

        # Re-open for processing
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")

        # Check dimensions
        w, h = image.size
        if w < 32 or h < 32:
            return False, f"Image too small: {w}x{h} (minimum 32x32)", None
        if w > 4096 or h > 4096:
            return False, f"Image too large: {w}x{h} (maximum 4096x4096)", None

        return True, "Valid", image

    except Exception as e:
        return False, f"Invalid image: {str(e)}", None


def preprocess_for_display(image: Image.Image, max_size: Tuple[int, int] = (400, 400)) -> Image.Image:
    """Preprocess image for display (thumbnail)."""
    display_img = image.copy()
    display_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return display_img


def create_image_grid(
    images: List[Image.Image],
    cols: int = 3,
    padding: int = 5,
) -> Image.Image:
    """Create a grid of images for display."""
    if not images:
        return Image.new("RGB", (100, 100), color="white")

    # Resize all to same size
    thumb_size = (200, 200)
    thumbs = [preprocess_for_display(img, thumb_size) for img in images]

    rows = (len(thumbs) + cols - 1) // cols
    thumb_w, thumb_h = thumbs[0].size

    grid_w = cols * thumb_w + (cols + 1) * padding
    grid_h = rows * thumb_h + (rows + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), color="white")

    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = padding + col * (thumb_w + padding)
        y = padding + row * (thumb_h + padding)
        grid.paste(thumb, (x, y))

    return grid


def save_uploaded_image(uploaded_file, save_dir: Path, prefix: str = "upload") -> Path:
    """Save uploaded file to disk with unique name."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    import uuid
    ext = Path(uploaded_file.name).suffix.lower()
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = save_dir / filename

    # Save
    uploaded_file.seek(0)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    return save_path


def enhance_image_for_analysis(image: Image.Image) -> Image.Image:
    """Enhance image for better model analysis."""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Auto-contrast to improve visibility
    image = ImageOps.autocontrast(image, cutoff=1)

    # Slight sharpening
    from PIL import ImageFilter
    image = image.filter(ImageFilter.SHARPEN)

    return image


def get_image_info(image: Image.Image) -> dict:
    """Get image metadata."""
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "aspect_ratio": round(image.width / image.height, 2),
    }


# Streamlit-specific helpers
def display_image_with_caption(image: Image.Image, caption: str = "", max_width: int = 400):
    """Display image in Streamlit with caption."""
    st.image(image, caption=caption, width=max_width)


def create_download_button(
    image: Image.Image,
    filename: str = "cropcare_result.png",
    label: str = "Download Image",
):
    """Create Streamlit download button for image."""
    img_bytes = image_to_bytes(image, "PNG")
    st.download_button(
        label=label,
        data=img_bytes,
        file_name=filename,
        mime="image/png",
    )


if __name__ == "__main__":
    # Test
    print("Testing image utilities...")

    # Create test image
    test_img = Image.new("RGB", (500, 400), color="green")
    print(f"Original size: {test_img.size}")

    # Resize
    resized = resize_image(test_img, (224, 224))
    print(f"Resized: {resized.size}")

    # Base64 conversion
    b64 = image_to_base64(test_img)
    restored = base64_to_image(b64)
    print(f"Base64 roundtrip: {restored.size}")

    print("Image utilities test passed!")