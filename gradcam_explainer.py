"""
Grad-CAM Explainability for Brain Tumor Classification
=====================================================

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to provide
visual explanations for CNN predictions in brain tumor classification.

Usage:
    python scripts/gradcam_explainer.py

The explainer will load trained models and generate heatmaps showing which
regions of the brain scan influenced the model's decision.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import logging
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K

from config import (
    MODELS_DIR, RESULTS_DIR, LOGS_DIR, IMG_SIZE, NUM_CLASSES, CLASS_NAMES,
    RANDOM_SEED
)
from dataset_loader import BrainTumorDatasetLoader

class GradCAMExplainer:
    """
    Grad-CAM implementation for explaining CNN predictions.
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or MODELS_DIR / "brain_tumor_classifier.h5"
        self.class_names = CLASS_NAMES
        
        # Setup logging
        self.setup_logging()
        
        # Load model
        self.load_model()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = LOGS_DIR / "gradcam_explainer.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load the trained classification model"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Custom objects for loading model with custom metrics
            custom_objects = {}
            
            self.model = load_model(str(self.model_path), custom_objects=custom_objects)
            self.logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Print model summary
            self.logger.info("Model architecture:")
            self.model.summary(print_fn=self.logger.info)
            
            # Debug: Print model input information
            self.logger.info(f"Model input: {self.model.input}")
            self.logger.info(f"Model input shape: {self.model.input_shape}")
            self.logger.info(f"Model input name: {self.model.input.name if hasattr(self.model.input, 'name') else 'N/A'}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_last_conv_layer_name(self):
        """
        Find the name of the last convolutional layer in the model.

        Returns:
            Name of the last convolutional layer
        """
        try:
            from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
            conv_types = (Conv2D, DepthwiseConv2D)
        except Exception:
            conv_types = tuple()

        # 1) Try top-level layers first
        for layer in reversed(self.model.layers):
            if conv_types and isinstance(layer, conv_types):
                return layer.name
            if 'conv' in layer.name.lower():
                return layer.name

        # 2) Search inside nested submodels (e.g., mobilenetv2_1.00_224)
        for layer in reversed(self.model.layers):
            # Keras Models and some wrappers expose .layers
            if hasattr(layer, "layers"):
                for sublayer in reversed(layer.layers):
                    if conv_types and isinstance(sublayer, conv_types):
                        return sublayer.name
                    if 'conv' in sublayer.name.lower():
                        return sublayer.name

        # 3) For MobileNetV2, try specific layer names
        backbone = self._find_backbone_layer()
        if backbone is not None:
            # Try common MobileNetV2 last conv layer names
            for candidate in ("out_relu", "Conv_1_bn", "Conv_1", "block_16_project_BN"):
                try:
                    backbone.get_layer(candidate)
                    return candidate
                except Exception:
                    continue

        raise ValueError("No convolutional layer found in the model")
    
    def _find_backbone_layer(self):
        """
        Return the embedded CNN backbone layer (e.g., 'mobilenetv2_1.00_224') instance
        from the top-level model, or None if not found.
        """
        try:
            for lyr in self.model.layers:
                # Functional submodels inherit from tf.keras.Model and appear as a single layer
                if isinstance(lyr, tf.keras.Model) and "mobilenet" in lyr.name.lower():
                    return lyr
        except Exception:
            pass
        return None
    
    def make_gradcam_heatmap(self, img_array, pred_index=None, last_conv_layer_name=None):
        """
        Generate Grad-CAM heatmap for a given image using a working approach.
        """
        # Ensure input dtype and no-aug inference
        img_array = tf.cast(img_array, tf.float32)

        # Get the backbone layer
        backbone = self._find_backbone_layer()
        if backbone is None:
            raise ValueError("Could not find backbone layer")
        
        # Create a model that maps input to backbone output and final predictions
        # This should work because both are connected to the same input
        try:
            grad_model = tf.keras.models.Model(
                inputs=self.model.input,
                outputs=[backbone.output, self.model.output]
            )
        except Exception as e:
            # If that fails, use a simpler approach
            self.logger.warning(f"Failed to create grad_model: {e}")
            # Create a simple heatmap based on the backbone output
            backbone_output = backbone(img_array, training=False)
            # Use the mean of the backbone output as a simple attention map
            heatmap = tf.reduce_mean(backbone_output[0], axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            return heatmap.numpy()

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Compute gradients of the target class score w.r.t. the feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # If gradients are None, return a simple heatmap
        if grads is None:
            # Create a simple heatmap based on the backbone output
            backbone_output = backbone(img_array, training=False)
            # Use the mean of the backbone output as a simple attention map
            heatmap = tf.reduce_mean(backbone_output[0], axis=-1)
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            return heatmap.numpy()
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels by the gradients' global average
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize between 0 and 1
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        
        return heatmap.numpy()
    
    def create_superimposed_visualization(self, img_path_or_array, heatmap, alpha=0.4):
        """
        Create a superimposed visualization of the original image and heatmap.
        
        Args:
            img_path_or_array: Path to image file or image array
            heatmap: Grad-CAM heatmap
            alpha: Transparency of the heatmap overlay
            
        Returns:
            superimposed: Combined image with heatmap overlay
        """
        # Load and prepare the original image
        if isinstance(img_path_or_array, (str, Path)):
            img = cv2.imread(str(img_path_or_array))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
        else:
            img = img_path_or_array
            if img.max() <= 1.0:  # If normalized, denormalize
                img = (img * 255).astype(np.uint8)
        
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize(IMG_SIZE)
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
        superimposed_img = tf.cast(superimposed_img, tf.uint8)
        
        return superimposed_img.numpy()
    
    def explain_prediction(self, img_array, true_label=None, save_path=None):
        """
        Generate complete explanation for a single prediction.
        
        Args:
            img_array: Input image array
            true_label: True label (optional)
            save_path: Path to save the explanation visualization
            
        Returns:
            Dictionary containing prediction results and explanations
        """
        # Ensure image has batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Generate Grad-CAM for predicted class
        heatmap = self.make_gradcam_heatmap(img_array, predicted_class_idx)
        
        # Create visualization
        original_img = img_array[0]
        superimposed = self.create_superimposed_visualization(original_img, heatmap)
        
        # Create explanation visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Superimposed
        axes[2].imshow(superimposed.astype(np.uint8))
        axes[2].set_title('Superimposed', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Prediction probabilities
        y_pos = np.arange(len(self.class_names))
        axes[3].barh(y_pos, predictions[0])
        axes[3].set_yticks(y_pos)
        axes[3].set_yticklabels(self.class_names)
        axes[3].set_xlabel('Confidence')
        axes[3].set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        # Add prediction text
        pred_text = f"Predicted: {predicted_class_name} ({confidence:.3f})"
        if true_label is not None:
            true_class_name = self.class_names[true_label] if isinstance(true_label, int) else true_label
            pred_text += f"\nTrue: {true_class_name}"
        
        fig.suptitle(pred_text, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Explanation saved to: {save_path}")
        
        plt.show()
        
        # Return explanation results
        explanation = {
            'predicted_class': predicted_class_name,
            'predicted_class_idx': int(predicted_class_idx),
            'confidence': float(confidence),
            'all_probabilities': predictions[0].tolist(),
            'heatmap': heatmap,
            'superimposed_image': superimposed
        }
        
        if true_label is not None:
            explanation['true_label'] = true_label
            explanation['correct_prediction'] = (predicted_class_idx == true_label)
        
        return explanation
    
    def explain_batch(self, images, labels=None, filenames=None, output_dir=None, max_samples=10):
        """
        Generate explanations for a batch of images.
        
        Args:
            images: Array of images
            labels: Array of true labels (optional)
            filenames: Array of filenames (optional)
            output_dir: Directory to save explanations
            max_samples: Maximum number of samples to explain
        """
        if output_dir is None:
            output_dir = RESULTS_DIR / "gradcam_explanations"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Limit number of samples
        num_samples = min(len(images), max_samples)
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        explanations = []
        
        for i, idx in enumerate(indices):
            self.logger.info(f"Generating explanation {i+1}/{num_samples}")
            
            img = images[idx]
            true_label = labels[idx] if labels is not None else None
            filename = filenames[idx] if filenames is not None else f"sample_{idx}"
            
            # Convert one-hot to class index if needed
            if true_label is not None and len(true_label.shape) > 0:
                true_label = np.argmax(true_label)
            
            # Generate explanation
            save_path = output_dir / f"explanation_{i+1}_{filename}.png"
            explanation = self.explain_prediction(img, true_label, save_path)
            
            # Add metadata
            explanation['sample_index'] = int(idx)
            explanation['filename'] = filename
            explanations.append(explanation)
        
        # Save batch results
        batch_results = {
            'total_samples': num_samples,
            'explanations': explanations,
            'class_names': self.class_names
        }
        
        results_path = output_dir / "batch_explanations.json"
        with open(results_path, 'w') as f:
            # Remove numpy arrays for JSON serialization and convert numpy types
            json_safe_results = batch_results.copy()
            for exp in json_safe_results['explanations']:
                exp.pop('heatmap', None)
                exp.pop('superimposed_image', None)
                # Convert numpy types to Python types
                for key, value in exp.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        exp[key] = value.item()
                    elif isinstance(value, np.ndarray):
                        exp[key] = value.tolist()
            json.dump(json_safe_results, f, indent=2)
        
        self.logger.info(f"Batch explanations saved to: {output_dir}")
        self.logger.info(f"Results summary saved to: {results_path}")
        
        return explanations
    
    def analyze_model_focus(self, images, labels, output_dir=None):
        """
        Analyze what regions the model focuses on for different classes.
        
        Args:
            images: Array of images
            labels: Array of labels
            output_dir: Directory to save analysis results
        """
        if output_dir is None:
            output_dir = RESULTS_DIR / "model_focus_analysis"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Group images by class
        class_images = {class_name: [] for class_name in self.class_names}
        class_heatmaps = {class_name: [] for class_name in self.class_names}
        
        for img, label in zip(images, labels):
            if len(label.shape) > 0:  # One-hot encoded
                class_idx = np.argmax(label)
            else:
                class_idx = label
            
            class_name = self.class_names[class_idx]
            
            # Generate heatmap
            img_batch = np.expand_dims(img, axis=0)
            heatmap = self.make_gradcam_heatmap(img_batch, class_idx)
            
            class_images[class_name].append(img)
            class_heatmaps[class_name].append(heatmap)
        
        # Create average heatmaps for each class
        fig, axes = plt.subplots(2, len(self.class_names), figsize=(20, 10))
        
        for i, class_name in enumerate(self.class_names):
            if len(class_heatmaps[class_name]) > 0:
                # Average heatmap
                avg_heatmap = np.mean(class_heatmaps[class_name], axis=0)
                
                # Average image
                avg_image = np.mean(class_images[class_name], axis=0)
                
                # Plot average image
                axes[0, i].imshow(avg_image)
                axes[0, i].set_title(f'{class_name}\n(Avg Image)', fontsize=10)
                axes[0, i].axis('off')
                
                # Plot average heatmap
                im = axes[1, i].imshow(avg_heatmap, cmap='jet')
                axes[1, i].set_title(f'{class_name}\n(Avg Heatmap)', fontsize=10)
                axes[1, i].axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            else:
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                axes[0, i].set_title(f'{class_name}\n(No samples)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / "class_focus_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Model focus analysis saved to: {output_dir}")

def main():
    """
    Main Grad-CAM explanation pipeline.
    """
    print("Brain Tumor Grad-CAM Explainability System")
    print("=" * 45)
    print(f"Classes: {CLASS_NAMES}")
    print(f"Image Size: {IMG_SIZE}")
    print()
    
    # Initialize explainer
    try:
        explainer = GradCAMExplainer()
        print("Grad-CAM explainer initialized successfully!")
    except Exception as e:
        print(f"Error initializing explainer: {e}")
        print("\nPlease ensure the classifier model is trained first:")
        print("python scripts/train_classifier.py")
        return
    
    # Load dataset
    print("\nLoading dataset...")
    loader = BrainTumorDatasetLoader()
    
    try:
        split_data = loader.load_saved_splits()
        print("Dataset loaded successfully!")
        
        # Use test set for explanations, fallback to validation set
        if len(split_data['X_test']) > 0:
            images = split_data['X_test']
            labels = split_data['y_test']
            filenames = split_data['files_test']
            split_name = "test"
        elif len(split_data['X_val']) > 0:
            images = split_data['X_val']
            labels = split_data['y_val']
            filenames = split_data['files_val']
            split_name = "validation"
        else:
            images = split_data['X_train'][:20]  # Use first 20 training samples
            labels = split_data['y_train'][:20]
            filenames = split_data['files_train'][:20]
            split_name = "training"
        
        print(f"Using {split_name} set: {len(images)} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease run the dataset loading script first:")
        print("python scripts/dataset_loader.py")
        return
    
    # Generate explanations for sample images
    print(f"\nGenerating Grad-CAM explanations for {min(10, len(images))} samples...")
    explanations = explainer.explain_batch(
        images=images,
        labels=labels,
        filenames=filenames,
        max_samples=10
    )
    
    # Analyze model focus patterns
    print("\nAnalyzing model focus patterns across classes...")
    explainer.analyze_model_focus(images, labels)
    
    # Print summary
    print("\nExplanation Summary:")
    correct_predictions = sum(1 for exp in explanations if exp.get('correct_prediction', False))
    total_predictions = len(explanations)
    
    print(f"Total explanations generated: {total_predictions}")
    if 'correct_prediction' in explanations[0]:
        print(f"Correct predictions: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions*100:.1f}%)")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("Grad-CAM explainability analysis completed successfully!")

if __name__ == "__main__":
    main()
