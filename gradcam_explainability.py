"""
Grad-CAM Implementation for Brain Tumor Classification Explainability
Generates heatmaps showing which regions the model focuses on for predictions.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    """
    
    def __init__(self, model_path=None, model=None, layer_name=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model_path: Path to saved model file
            model: Pre-loaded model (alternative to model_path)
            layer_name: Name of the convolutional layer for Grad-CAM
        """
        if model is not None:
            self.model = model
        elif model_path and os.path.exists(model_path):
            self.model = load_model(model_path, compile=False)
            logger.info(f"Model loaded from: {model_path}")
        else:
            raise ValueError("Either model_path or model must be provided")
        
        # Auto-detect the last convolutional layer if not specified
        if layer_name is None:
            self.layer_name = self._find_last_conv_layer()
        else:
            self.layer_name = layer_name
        
        logger.info(f"Using layer '{self.layer_name}' for Grad-CAM")
        
        # Create directories
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/gradcam").mkdir(exist_ok=True)
    
    def _find_last_conv_layer(self):
        """
        Automatically find the last convolutional layer in the model.
        """
        for layer in reversed(self.model.layers):
            # Check if layer has 4D output (batch, height, width, channels)
            if len(layer.output_shape) == 4:
                return layer.name
        
        # Fallback: look for common conv layer patterns
        conv_layers = []
        for layer in self.model.layers:
            if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                conv_layers.append(layer.name)
        
        if conv_layers:
            return conv_layers[-1]
        
        raise ValueError("No suitable convolutional layer found for Grad-CAM")
    
    def _resolve_feature_tensor(self):
        """
        Return a 4D feature tensor suitable for Grad-CAM that belongs to the
        same computation graph as self.model.inputs. Tries, in order:
        - layer_name on the top-level model
        - layer_name inside any nested submodel (e.g., MobileNetV2)
        - the first nested submodel's 4D output
        - the last top-level 4D tensor
        """
        # Try direct lookup on top-level model
        try:
            lyr = self.model.get_layer(self.layer_name)
            out = getattr(lyr, "output", None)
            if out is not None and len(out.shape) == 4:
                return out
        except Exception:
            pass

        # Try nested submodels
        for lyr in self.model.layers:
            if hasattr(lyr, "get_layer"):
                try:
                    sub = lyr.get_layer(self.layer_name)
                    out = getattr(sub, "output", None)
                    if out is not None and len(out.shape) == 4:
                        return out
                except Exception:
                    continue

        # Try a nested backbone's output (e.g., mobilenetv2_1.00_224)
        for lyr in self.model.layers:
            if isinstance(lyr, tf.keras.Model):
                try:
                    out = getattr(lyr, "output", None)
                    if out is not None and len(out.shape) == 4:
                        return out
                except Exception:
                    continue

        # Fallback: last 4D tensor at top level
        for lyr in reversed(self.model.layers):
            try:
                out = getattr(lyr, "output", None)
                if out is not None and len(out.shape) == 4:
                    return out
            except Exception:
                continue

        raise ValueError("Could not resolve a valid 4D feature tensor for Grad-CAM")

    def make_gradcam_heatmap(self, img_array, class_index, eps=1e-8):
        """
        Generate Grad-CAM heatmap for a specific class.
        
        Args:
            img_array: Input image array (1, H, W, C)
            class_index: Index of the class to generate heatmap for
            eps: Small epsilon to avoid division by zero
            
        Returns:
            Heatmap array normalized to [0, 1]
        """
        img_array = tf.cast(img_array, tf.float32)
        feature_tensor = self._resolve_feature_tensor()

        # Create a model that maps model inputs → [feature_tensor, predictions]
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [feature_tensor, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        max_val = tf.math.reduce_max(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (max_val + eps)
        return heatmap.numpy()
    
    def generate_gradcam_overlay(self, img_array, heatmap, alpha=0.4):
        """
        Generate Grad-CAM overlay on original image.
        
        Args:
            img_array: Original image array (H, W, C)
            heatmap: Grad-CAM heatmap
            alpha: Transparency of heatmap overlay
            
        Returns:
            Overlayed image
        """
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
        
        # Normalize original image to 0-255 if needed
        if img_array.max() <= 1.0:
            img_array = img_array * 255
        
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img_array
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
        
        return superimposed_img
    
    def explain_prediction(self, image, predicted_class_idx, true_class_idx=None, 
                          class_names=None, save_path=None):
        """
        Generate complete Grad-CAM explanation for a prediction.
        
        Args:
            image: Input image (H, W, C) normalized to [0, 1]
            predicted_class_idx: Index of predicted class
            true_class_idx: Index of true class (optional)
            class_names: List of class names
            save_path: Path to save the explanation image
            
        Returns:
            Dictionary with explanation components
        """
        # Prepare image for model
        img_array = np.expand_dims(image, axis=0)
        
        # Get model prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_prob = predictions[0][predicted_class_idx]
        
        # Generate heatmap
        heatmap = self.make_gradcam_heatmap(img_array, predicted_class_idx)
        
        # Create overlay
        overlay_img = self.generate_gradcam_overlay(image, heatmap)
        
        # Prepare class names
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(predictions[0]))]
        
        predicted_class_name = class_names[predicted_class_idx]
        true_class_name = class_names[true_class_idx] if true_class_idx is not None else "Unknown"
        
        # Create visualization
        if save_path:
            self._save_gradcam_visualization(
                image, heatmap, overlay_img, predictions[0],
                predicted_class_name, true_class_name, predicted_prob,
                class_names, save_path
            )
        
        return {
            'heatmap': heatmap,
            'overlay': overlay_img,
            'predicted_class': predicted_class_name,
            'predicted_prob': predicted_prob,
            'true_class': true_class_name,
            'all_predictions': predictions[0]
        }
    
    def _save_gradcam_visualization(self, original_img, heatmap, overlay_img, 
                                   predictions, predicted_class, true_class, 
                                   predicted_prob, class_names, save_path):
        """
        Save comprehensive Grad-CAM visualization.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Heatmap
        im1 = axes[0, 1].imshow(heatmap, cmap='jet')
        axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[0, 2].imshow(overlay_img)
        axes[0, 2].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Prediction probabilities bar chart
        y_pos = np.arange(len(class_names))
        bars = axes[1, 0].barh(y_pos, predictions)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(class_names)
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_title('Class Probabilities', fontsize=14, fontweight='bold')
        
        # Highlight predicted class
        bars[np.argmax(predictions)].set_color('red')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Prediction summary text
        axes[1, 1].axis('off')
        summary_text = f"""
Prediction Summary:

Predicted Class: {predicted_class}
Confidence: {predicted_prob:.3f}

True Class: {true_class}
Correct: {'✓' if predicted_class == true_class else '✗'}

Top 3 Predictions:
"""
        # Add top 3 predictions
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        for i, idx in enumerate(top_3_idx):
            summary_text += f"{i+1}. {class_names[idx]}: {predictions[idx]:.3f}\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Heatmap statistics
        axes[1, 2].axis('off')
        heatmap_stats = f"""
Heatmap Statistics:

Max Activation: {heatmap.max():.3f}
Mean Activation: {heatmap.mean():.3f}
Std Activation: {heatmap.std():.3f}

Focus Area:
Top 10% pixels cover
{(heatmap > np.percentile(heatmap, 90)).sum() / heatmap.size * 100:.1f}% of image

Activation Distribution:
- High (>0.7): {(heatmap > 0.7).sum() / heatmap.size * 100:.1f}%
- Medium (0.3-0.7): {((heatmap > 0.3) & (heatmap <= 0.7)).sum() / heatmap.size * 100:.1f}%
- Low (<0.3): {(heatmap <= 0.3).sum() / heatmap.size * 100:.1f}%
"""
        
        axes[1, 2].text(0.1, 0.9, heatmap_stats, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Grad-CAM visualization saved to: {save_path}")
    
    def batch_generate_gradcam(self, images, predictions, true_labels, 
                              filenames, class_names, output_dir="outputs/gradcam"):
        """
        Generate Grad-CAM explanations for a batch of images.
        
        Args:
            images: Array of images
            predictions: Array of predicted class indices
            true_labels: Array of true class indices
            filenames: Array of filenames
            class_names: List of class names
            output_dir: Directory to save explanations
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating Grad-CAM explanations for {len(images)} images...")
        
        explanations = []
        
        for i, (image, pred_idx, true_idx, filename) in enumerate(
            zip(images, predictions, true_labels, filenames)
        ):
            try:
                # Create save path
                name_parts = filename.split('.')
                save_path = output_dir / f"{name_parts[0]}_gradcam.png"
                
                # Generate explanation
                explanation = self.explain_prediction(
                    image=image,
                    predicted_class_idx=pred_idx,
                    true_class_idx=true_idx,
                    class_names=class_names,
                    save_path=save_path
                )
                
                explanation['filename'] = filename
                explanations.append(explanation)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{len(images)} explanations")
                    
            except Exception as e:
                logger.error(f"Error generating Grad-CAM for {filename}: {e}")
                continue
        
        logger.info(f"Grad-CAM generation completed. Saved to: {output_dir}")
        
        return explanations
    
    def analyze_gradcam_patterns(self, explanations, class_names):
        """
        Analyze patterns in Grad-CAM explanations across classes.
        
        Args:
            explanations: List of explanation dictionaries
            class_names: List of class names
            
        Returns:
            Analysis results
        """
        logger.info("Analyzing Grad-CAM patterns...")
        
        # Group by predicted class
        class_heatmaps = {class_name: [] for class_name in class_names}
        class_accuracies = {class_name: [] for class_name in class_names}
        
        for exp in explanations:
            pred_class = exp['predicted_class']
            if pred_class in class_heatmaps:
                class_heatmaps[pred_class].append(exp['heatmap'])
                class_accuracies[pred_class].append(
                    exp['predicted_class'] == exp['true_class']
                )
        
        # Calculate statistics
        analysis = {}
        for class_name in class_names:
            if class_heatmaps[class_name]:
                heatmaps = np.array(class_heatmaps[class_name])
                
                analysis[class_name] = {
                    'count': len(heatmaps),
                    'accuracy': np.mean(class_accuracies[class_name]),
                    'avg_max_activation': np.mean([h.max() for h in heatmaps]),
                    'avg_mean_activation': np.mean([h.mean() for h in heatmaps]),
                    'avg_focus_area': np.mean([
                        (h > np.percentile(h, 90)).sum() / h.size 
                        for h in heatmaps
                    ])
                }
        
        # Print analysis
        logger.info("\nGrad-CAM Pattern Analysis:")
        for class_name, stats in analysis.items():
            logger.info(f"\n{class_name}:")
            logger.info(f"  Samples: {stats['count']}")
            logger.info(f"  Accuracy: {stats['accuracy']:.3f}")
            logger.info(f"  Avg Max Activation: {stats['avg_max_activation']:.3f}")
            logger.info(f"  Avg Mean Activation: {stats['avg_mean_activation']:.3f}")
            logger.info(f"  Avg Focus Area: {stats['avg_focus_area']:.3f}")
        
        return analysis

def main():
    """
    Example usage of Grad-CAM explainability.
    """
    logger.info("Grad-CAM Explainability Module")
    logger.info("This module provides explainability for brain tumor classification.")
    logger.info("Run the complete pipeline with evaluate.py to see it in action.")
    
    # Example of how to use Grad-CAM
    try:
        # This would typically load a trained model
        model_path = "models/brain_tumor_classifier.h5"
        if os.path.exists(model_path):
            gradcam = GradCAM(model_path=model_path)
            logger.info("Grad-CAM initialized successfully")
            logger.info(f"Target layer: {gradcam.layer_name}")
        else:
            logger.info(f"Model not found at {model_path}")
            logger.info("Train the classification model first using train_classifier.py")
    
    except Exception as e:
        logger.error(f"Error initializing Grad-CAM: {e}")

if __name__ == "__main__":
    main()
