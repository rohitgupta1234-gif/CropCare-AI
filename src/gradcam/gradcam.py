"""
CropCare AI - Grad-CAM Implementation
Gradient-weighted Class Activation Mapping for model interpretability.
"""

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.config import (
    IMAGE_SIZE,
    GRADCAM_LAYER_NAME,
    GRADCAM_ALPHA,
    GRADCAM_COLORMAP,
    CLASS_NAMES,
    GRADCAM_FALLBACK_LAYERS,
)
from src.model.inference import ModelInference, get_inference_engine


class GradCAM:
    """Grad-CAM implementation for MobileNetV2-based models."""

    def __init__(
        self,
        model: tf.keras.Model,
        layer_name: str = GRADCAM_LAYER_NAME,
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: Trained Keras model
            layer_name: Name of the target convolutional layer
        """
        self.model = model
        self.layer_name = layer_name
        self.parent_layer = None  # Nested model containing target layer (e.g., mobilenetv2)
        self.target_layer = None  # The actual target Conv2D layer
        self._validate_and_find_layers()
        self.grad_model = self._build_grad_model()

    def _validate_and_find_layers(self):
        """Validate that the target layer exists and find it (including in nested models)."""
        # First check top-level layers
        layer_names = [layer.name for layer in self.model.layers]
        
        if self.layer_name in layer_names:
            target_layer = self.model.get_layer(self.layer_name)
            if isinstance(target_layer, tf.keras.layers.Conv2D):
                self.target_layer = target_layer
                return
            else:
                print(f"Layer '{self.layer_name}' exists but is {type(target_layer).__name__}, not Conv2D. Searching for Conv2D...")
        else:
            print(f"Layer '{self.layer_name}' not found at top level. Searching in nested models...")
        
        # Search in nested models for Conv2D layers
        conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append((layer.name, layer, None))
            elif hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        conv_layers.append((sublayer.name, sublayer, layer))
        
        if conv_layers:
            # Try to find the requested layer name among Conv2D layers
            for name, layer, parent in conv_layers:
                if name == self.layer_name:
                    self.target_layer = layer
                    self.parent_layer = parent
                    self.layer_name = name
                    print(f"Found Conv2D layer: '{name}' in parent: {parent.name if parent else 'top-level'}")
                    return
            
            # If requested layer not found, use the last Conv2D layer
            last_conv_name, last_conv_layer, last_parent = conv_layers[-1]
            self.target_layer = last_conv_layer
            self.parent_layer = last_parent
            self.layer_name = last_conv_name
            print(f"Requested layer not found as Conv2D. Using last Conv2D layer: '{last_conv_name}' in parent: {last_parent.name if last_parent else 'top-level'}")
            return
        
        # Fallback to GRADCAM_FALLBACK_LAYERS
        for fallback_name in GRADCAM_FALLBACK_LAYERS:
            for name, layer, parent in conv_layers:
                if name == fallback_name:
                    self.target_layer = layer
                    self.parent_layer = parent
                    self.layer_name = name
                    print(f"Using fallback Conv2D layer: '{name}' in parent: {parent.name if parent else 'top-level'}")
                    return
        
        # Last resort: use any Conv2D layer
        if conv_layers:
            last_conv_name, last_conv_layer, last_parent = conv_layers[-1]
            self.target_layer = last_conv_layer
            self.parent_layer = last_parent
            self.layer_name = last_conv_name
            print(f"Using last available Conv2D layer: '{last_conv_name}' in parent: {last_parent.name if last_parent else 'top-level'}")
            return
            
        raise ValueError(f"No Conv2D layers found in model. Available layers: {layer_names}")

    def _build_grad_model(self) -> tf.keras.Model:
        """Build model that outputs both target layer activations and predictions."""
        # If target_layer is at top level, use standard approach
        if self.parent_layer is None:
            return tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[self.target_layer.output, self.model.output],
            )

        # Target layer is in a nested model (parent_layer)
        # Create a wrapper model for parent_layer that outputs both its final output
        # and the target_layer's intermediate output
        parent_input_name = self.parent_layer.inputs[0].name.split(':')[0]

        # Create a model that takes parent_layer's input and outputs both
        # target_layer output and parent_layer's final output
        parent_wrapper = tf.keras.Model(
            inputs=self.parent_layer.inputs,
            outputs=[self.target_layer.output, self.parent_layer.output],
        )

        # Now trace through main model with raw input
        input_shape = self.model.inputs[0].shape[1:]
        raw_input = tf.keras.Input(shape=input_shape, name="gradcam_input")

        x = raw_input
        target_output = None

        for layer in self.model.layers[1:]:  # Skip InputLayer
            if layer is self.parent_layer:
                # Use parent_wrapper to get both target_output and parent_layer output
                parent_input = {parent_input_name: x}
                target_output, parent_layer_output = parent_wrapper(parent_input)
                x = parent_layer_output
            else:
                x = layer(x)

        predictions = x

        return tf.keras.Model(
            inputs=raw_input,
            outputs=[target_output, predictions],
        )
    def generate_heatmap(
        self,
        image: np.ndarray,
        class_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a given image and class.

        Args:
            image: Preprocessed image array of shape (1, H, W, 3)
            class_index: Target class index (None for predicted class)

        Returns:
            Heatmap array of shape (H, W) with values in [0, 1]
        """
        image_tensor = tf.convert_to_tensor(image)

        with tf.GradientTape() as tape:
            tape.watch(image_tensor)

            # Forward pass - grad_model outputs target_layer activations and predictions
            conv_outputs, predictions = self.grad_model(image_tensor)

            # Get target class
            if class_index is None:
                class_index = tf.argmax(predictions[0])

            # Get loss for target class
            loss = predictions[:, class_index]

        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # ReLU to keep only positive contributions
        heatmap = tf.maximum(heatmap, 0)

        # Normalize
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    def overlay_heatmap(
        self,
        original_image: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        alpha: float = GRADCAM_ALPHA,
        colormap: str = GRADCAM_COLORMAP,
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            original_image: Original image (PIL or numpy)
            heatmap: Grad-CAM heatmap (H, W) in [0, 1]
            alpha: Overlay transparency
            colormap: Matplotlib colormap name

        Returns:
            Overlayed image as numpy array (H, W, 3) in [0, 255]
        """
        # Convert to numpy if PIL
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image.convert("RGB"))

        # Resize heatmap to match image
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap (matplotlib >= 3.7 compatible)
        try:
            cmap = cm.get_cmap(colormap)
        except AttributeError:
            # Fallback for newer matplotlib
            import matplotlib
            cmap = matplotlib.colormaps.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Drop alpha
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Convert original to uint8 if needed
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)

        # Blend
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay

    def generate_gradcam(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        class_index: Optional[int] = None,
        return_heatmap_only: bool = False,
    ) -> Dict:
        """
        Complete Grad-CAM pipeline.

        Args:
            image: Input image
            class_index: Target class index
            return_heatmap_only: If True, return only heatmap without overlay

        Returns:
            Dictionary with heatmap, overlay, and metadata
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        # Store original for overlay
        original_array = np.array(pil_image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS))

        # Preprocess for model
        inference_engine = get_inference_engine()
        processed = inference_engine.preprocess_image(pil_image)

        # If no class specified, use model prediction
        if class_index is None:
            predictions = self.model.predict(processed, verbose=0)
            class_index = int(np.argmax(predictions[0]))

        # Generate heatmap
        heatmap = self.generate_heatmap(processed, class_index)

        result = {
            "heatmap": heatmap,
            "class_index": class_index,
            "class_name": CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else f"Class_{class_index}",
        }

        if not return_heatmap_only:
            # Generate overlay
            overlay = self.overlay_heatmap(original_array, heatmap)
            result["overlay"] = overlay
            result["original"] = original_array

        return result


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation for better localization.

    This is a simplified implementation that falls back to standard Grad-CAM
    if higher-order gradient computation fails.
    """

    def generate_heatmap(
        self,
        image: np.ndarray,
        class_index: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap with fallback to standard Grad-CAM."""
        try:
            image_tensor = tf.convert_to_tensor(image)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(image_tensor)
                conv_outputs, predictions = self.grad_model(image_tensor)

                if class_index is None:
                    class_index = tf.argmax(predictions[0])

                loss = predictions[:, class_index]

                # First-order gradients
                grads = tape.gradient(loss, conv_outputs)

            # Second-order gradients (persistent for two gradient computations)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(conv_outputs)
                first_grads = tape.gradient(loss, conv_outputs)
            second_grads = tape2.gradient(first_grads, conv_outputs)

            # Third-order gradients
            with tf.GradientTape() as tape3:
                tape3.watch(conv_outputs)
                second_grads_val = tape2.gradient(first_grads, conv_outputs)
            third_grads = tape3.gradient(second_grads_val, conv_outputs)

            # Grad-CAM++ weights
            # alpha = second_grads / (2 * second_grads + third_grads * conv_outputs)
            denom = 2 * second_grads + third_grads * conv_outputs
            denom = tf.where(denom != 0, denom, tf.ones_like(denom))
            alphas = second_grads / denom

            # Weighted combination
            weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1, 2))

            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
            heatmap = tf.maximum(heatmap, 0)

            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val

            return heatmap.numpy()
        except Exception as e:
            # Fallback to standard Grad-CAM
            print(f"GradCAM++ failed, falling back to Grad-CAM: {e}")
            return super().generate_heatmap(image, class_index)


def create_gradcam_visualization(
    image_path: Union[str, Path],
    model: Optional[tf.keras.Model] = None,
    class_index: Optional[int] = None,
    save_path: Optional[Path] = None,
    use_plus_plus: bool = False,
) -> Dict:
    """
    Create complete Grad-CAM visualization and optionally save.

    Args:
        image_path: Path to input image
        model: Model to use (loads default if None)
        class_index: Target class index
        save_path: Path to save visualization
        use_plus_plus: Use Grad-CAM++ instead of Grad-CAM

    Returns:
        Dictionary with results
    """
    if model is None:
        inference_engine = get_inference_engine()
        model = inference_engine.model

    # Create Grad-CAM instance
    if use_plus_plus:
        gradcam = GradCAMPlusPlus(model)
    else:
        gradcam = GradCAM(model)

    # Generate visualization
    result = gradcam.generate_gradcam(image_path, class_index)

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(result["original"])
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(result["heatmap"], cmap=GRADCAM_COLORMAP)
        axes[1].set_title(f"Heatmap (Class: {result['class_name']})")
        axes[1].axis("off")

        axes[2].imshow(result["overlay"])
        axes[2].set_title(f"Grad-CAM Overlay (α={GRADCAM_ALPHA})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        result["save_path"] = str(save_path)

    return result


def generate_gradcam_for_streamlit(
    image: Union[Image.Image, np.ndarray],
    model: tf.keras.Model,
    class_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap and overlay for Streamlit display.

    Returns:
        Tuple of (heatmap, overlay) as numpy arrays
    """
    gradcam = GradCAM(model)
    result = gradcam.generate_gradcam(image, class_index)
    return result["heatmap"], result["overlay"]


if __name__ == "__main__":
    # Test Grad-CAM (requires model)
    try:
        inference_engine = get_inference_engine()
        model = inference_engine.model

        gradcam = GradCAM(model)
        print(f"Grad-CAM initialized with layer: {gradcam.layer_name}")
        print("Grad-CAM module test passed!")
    except Exception as e:
        print(f"Expected (no model yet): {e}")