"""
Brain Tumor Detection Inference Script
Loads trained models and performs complete analysis on new MRI scans.
"""

import os
import sys
import numpy as np
import cv2
import argparse
import json
from pathlib import Path
import logging
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tumor_size_estimation import TumorSizeEstimator
from gradcam_explainer import GradCAMExplainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrainTumorPredictor:
    """
    Complete brain tumor detection and analysis system for inference.
    """
    
    def __init__(self, classifier_path="/Users/prajwalpatil/Desktop/brain-tumor-preprocessing (3)/scripts/models/brain_tumor_classifier.h5",
                 unet_path="/Users/prajwalpatil/Desktop/brain-tumor-preprocessing (3)/scripts/models/unet_segmentation.h5",
                 img_size=(224, 224)):
        self.classifier_path = classifier_path
        self.unet_path = unet_path
        self.img_size = img_size
        
        # Class names (should match training)
        self.class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        
        # Models
        self.classifier = None
        self.unet = None
        self.gradcam = None
        self.size_estimator = TumorSizeEstimator(pixel_to_mm_ratio=0.5)
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """
        Load trained models.
        """
        logger.info("Loading trained models...")
        
        # Load classifier
        if os.path.exists(self.classifier_path):
            try:
                self.classifier = load_model(self.classifier_path)
                logger.info(f"Classifier loaded successfully from: {self.classifier_path}")
                
                # Initialize Grad-CAM
                self.gradcam = GradCAMExplainer()
                logger.info("Grad-CAM initialized")
                
            except Exception as e:
                logger.error(f"Error loading classifier: {e}")
                self.classifier = None
        else:
            logger.error(f"Classifier not found at: {self.classifier_path}")
        
        # Load U-Net
        if os.path.exists(self.unet_path):
            try:
                self.unet = load_model(self.unet_path, compile=False)
                logger.info(f"U-Net loaded successfully from: {self.unet_path}")
            except Exception as e:
                logger.error(f"Error loading U-Net: {e}")
                self.unet = None
        else:
            logger.warning(f"U-Net not found at: {self.unet_path}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess input image for model inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image = cv2.resize(image, self.img_size)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def classify_tumor(self, image):
        """
        Classify tumor type.
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Classification results dictionary
        """
        if self.classifier is None:
            logger.error("Classifier not loaded")
            return None
        
        try:
            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)
            
            # Get predictions
            predictions = self.classifier.predict(image_batch, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx]
            
            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.class_names):
                class_probabilities[class_name] = float(predictions[0][i])
            
            return {
                'predicted_class': predicted_class,
                'predicted_class_idx': predicted_class_idx,
                'confidence': float(confidence),
                'all_probabilities': class_probabilities
            }
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return None
    
    def segment_tumor(self, image):
        """
        Segment tumor region.
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Segmentation mask
        """
        if self.unet is None:
            logger.warning("U-Net not loaded, skipping segmentation")
            return None
        
        try:
            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)
            
            # Get segmentation mask
            mask_pred = self.unet.predict(image_batch, verbose=0)
            mask = (mask_pred[0].squeeze() > 0.5).astype(np.uint8)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            return None
    
    def estimate_tumor_size(self, mask, predicted_class):
        """
        Estimate tumor size from segmentation mask.
        
        Args:
            mask: Binary segmentation mask
            predicted_class: Predicted tumor class
            
        Returns:
            Size estimation results
        """
        if mask is None:
            return None
        
        try:
            # Calculate area in pixels
            area_pixels = self.size_estimator.calculate_tumor_area_pixels(mask)
            area_mm2 = self.size_estimator.calculate_tumor_area_mm2(area_pixels)
            
            # Get additional properties
            properties = self.size_estimator.get_tumor_contour_properties(mask)
            
            # Categorize size
            size_category = self.size_estimator.categorize_tumor_size([area_mm2])[0]
            
            return {
                'area_pixels': area_pixels,
                'area_mm2': round(area_mm2, 2),
                'size_category': size_category,
                'num_regions': properties['num_regions'],
                'largest_region_pixels': properties['largest_area_pixels'],
                'circularity': round(properties['circularity'], 3),
                'aspect_ratio': round(properties['aspect_ratio'], 3),
                'solidity': round(properties['solidity'], 3)
            }
            
        except Exception as e:
            logger.error(f"Error in size estimation: {e}")
            return None
    
    def generate_explanation(self, image, predicted_class_idx, save_path=None):
        """
        Generate Grad-CAM explanation.
        
        Args:
            image: Preprocessed image array
            predicted_class_idx: Index of predicted class
            save_path: Path to save explanation image
            
        Returns:
            Explanation results
        """
        if self.gradcam is None:
            logger.warning("Grad-CAM not available, skipping explanation")
            return None
        
        try:
            # Use our working GradCAMExplainer method
            explanation = self.gradcam.explain_prediction(
                img_array=image,
                true_label=predicted_class_idx,
                save_path=save_path
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return None
    
    def predict_single_image(self, image_path, output_dir="outputs/predictions"):
        """
        Complete analysis pipeline for a single image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Analyzing image: {image_path}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        if image is None:
            return None
        
        # Get image filename for outputs
        image_name = Path(image_path).stem
        
        # Classification
        classification_results = self.classify_tumor(image)
        if classification_results is None:
            logger.error("Classification failed")
            return None
        
        # Segmentation
        segmentation_mask = self.segment_tumor(image)
        
        # Size estimation
        size_results = None
        if segmentation_mask is not None:
            size_results = self.estimate_tumor_size(
                segmentation_mask, classification_results['predicted_class']
            )
            
            # Save segmentation mask
            mask_path = output_dir / f"{image_name}_segmentation_mask.png"
            cv2.imwrite(str(mask_path), segmentation_mask * 255)
            logger.info(f"Segmentation mask saved to: {mask_path}")
        
        # Generate explanation
        explanation_path = output_dir / f"{image_name}_gradcam_explanation.png"
        explanation_results = self.generate_explanation(
            image, classification_results['predicted_class_idx'], explanation_path
        )
        
        # Compile complete results
        results = {
            'image_path': str(image_path),
            'analysis_timestamp': datetime.now().isoformat(),
            'classification': classification_results,
            'segmentation': {
                'mask_available': segmentation_mask is not None,
                'mask_path': str(output_dir / f"{image_name}_segmentation_mask.png") if segmentation_mask is not None else None
            },
            'size_estimation': size_results,
            'explanation': {
                'available': explanation_results is not None,
                'explanation_path': str(explanation_path) if explanation_results is not None else None
            }
        }
        
        # Save results as JSON
        results_path = output_dir / f"{image_name}_analysis_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert results to JSON-serializable format
        json_results = convert_numpy_types(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Analysis results saved to: {results_path}")
        
        return results
    
    def print_human_readable_results(self, results):
        """
        Print results in human-readable format.
        
        Args:
            results: Analysis results dictionary
        """
        print("\n" + "="*60)
        print("BRAIN TUMOR ANALYSIS RESULTS")
        print("="*60)
        
        print(f"Image: {results['image_path']}")
        print(f"Analysis Time: {results['analysis_timestamp']}")
        
        # Classification results
        classification = results['classification']
        print(f"\nCLASSIFICATION:")
        print(f"  Predicted Class: {classification['predicted_class']}")
        print(f"  Confidence: {classification['confidence']:.3f} ({classification['confidence']*100:.1f}%)")
        
        print(f"\n  All Class Probabilities:")
        for class_name, prob in classification['all_probabilities'].items():
            print(f"    {class_name}: {prob:.3f} ({prob*100:.1f}%)")
        
        # Size estimation results
        if results['size_estimation']:
            size_est = results['size_estimation']
            print(f"\nTUMOR SIZE ESTIMATION:")
            print(f"  Area: {size_est['area_pixels']} pixels ({size_est['area_mm2']} mm²)")
            print(f"  Size Category: {size_est['size_category']}")
            print(f"  Number of Regions: {size_est['num_regions']}")
            
            if size_est['num_regions'] > 0:
                print(f"  Shape Properties:")
                print(f"    Circularity: {size_est['circularity']}")
                print(f"    Aspect Ratio: {size_est['aspect_ratio']}")
                print(f"    Solidity: {size_est['solidity']}")
        else:
            print(f"\nTUMOR SIZE ESTIMATION: Not available")
        
        # Segmentation info
        segmentation = results['segmentation']
        if segmentation['mask_available']:
            print(f"\nSEGMENTATION: Available")
            print(f"  Mask saved to: {segmentation['mask_path']}")
        else:
            print(f"\nSEGMENTATION: Not available")
        
        # Explanation info
        explanation = results['explanation']
        if explanation['available']:
            print(f"\nEXPLAINABILITY: Available")
            print(f"  Grad-CAM explanation saved to: {explanation['explanation_path']}")
        else:
            print(f"\nEXPLAINABILITY: Not available")
        
        print("="*60)

def main():
    """
    Command-line interface for brain tumor prediction.
    """
    parser = argparse.ArgumentParser(description='Brain Tumor Detection and Analysis')
    parser.add_argument('image_path', help='Path to input MRI image')
    parser.add_argument('--output-dir', default='outputs/predictions',
                       help='Directory to save outputs (default: outputs/predictions)')
    parser.add_argument('--classifier-path', default='/Users/prajwalpatil/Desktop/brain-tumor-preprocessing (3)/scripts/models/brain_tumor_classifier.h5',
                       help='Path to classifier model')
    parser.add_argument('--unet-path', default='/Users/prajwalpatil/Desktop/brain-tumor-preprocessing (3)/scripts/models/unet_segmentation.h5',
                       help='Path to U-Net segmentation model')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        logger.error(f"Image not found: {args.image_path}")
        sys.exit(1)
    
    # Initialize predictor
    try:
        predictor = BrainTumorPredictor(
            classifier_path=args.classifier_path,
            unet_path=args.unet_path
        )
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        sys.exit(1)
    
    # Run prediction
    try:
        results = predictor.predict_single_image(args.image_path, args.output_dir)
        
        if results:
            if not args.quiet:
                predictor.print_human_readable_results(results)
            else:
                # Print minimal output for quiet mode
                classification = results['classification']
                print(f"Prediction: {classification['predicted_class']} "
                     f"(Confidence: {classification['confidence']:.3f})")
        else:
            logger.error("Prediction failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
