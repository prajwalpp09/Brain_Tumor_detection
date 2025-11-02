"""
Comprehensive Evaluation Pipeline for Brain Tumor Detection System
Combines classification, segmentation, size estimation, and explainability.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import json

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

from dataset_loader import BrainTumorDatasetLoader
from train_classifier import BrainTumorClassifier
from train_unet import UNetLite, TumorMaskGenerator
from tumor_size_estimation import TumorSizeEstimator
from gradcam_explainability import GradCAM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrainTumorEvaluator:
    """
    Comprehensive evaluation system for brain tumor detection.
    """
    
    def __init__(self, classifier_path="models/brain_tumor_classifier.h5",
                 unet_path="models/unet_segmentation.h5"):
        self.classifier_path = classifier_path
        self.unet_path = unet_path
        self.classifier = None
        self.unet = None
        self.gradcam = None
        
        # Create output directories
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/evaluation").mkdir(exist_ok=True)
        Path("outputs/plots").mkdir(exist_ok=True)
        Path("outputs/reports").mkdir(exist_ok=True)
        
    def load_models(self):
        """
        Load trained models for evaluation.
        """
        logger.info("Loading trained models...")
        
        # Load classifier
        if os.path.exists(self.classifier_path):
            self.classifier = tf.keras.models.load_model(self.classifier_path)
            logger.info(f"Classifier loaded from: {self.classifier_path}")
            
            # Initialize Grad-CAM
            self.gradcam = GradCAM(model=self.classifier)
            logger.info("Grad-CAM initialized")
        else:
            logger.error(f"Classifier not found at: {self.classifier_path}")
            
        # Load U-Net
        if os.path.exists(self.unet_path):
            self.unet = tf.keras.models.load_model(self.unet_path, compile=False)
            logger.info(f"U-Net loaded from: {self.unet_path}")
        else:
            logger.warning(f"U-Net not found at: {self.unet_path}")
    
    def evaluate_classification(self, X_test, y_test, class_names):
        """
        Evaluate classification performance.
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            class_names: List of class names
            
        Returns:
            Classification results dictionary
        """
        logger.info("Evaluating classification performance...")
        
        if self.classifier is None:
            logger.error("Classifier not loaded")
            return None
        
        # Get predictions
        y_pred_proba = self.classifier.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(class_names))
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_true
        }
        
        # Log results
        logger.info(f"Classification Accuracy: {accuracy:.4f}")
        logger.info("\nPer-class Performance:")
        for i, class_name in enumerate(class_names):
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {precision[i]:.4f}")
            logger.info(f"    Recall: {recall[i]:.4f}")
            logger.info(f"    F1-Score: {f1[i]:.4f}")
            logger.info(f"    Support: {support[i]}")
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names, 
                             save_path="outputs/plots/confusion_matrix.png"):
        """
        Plot confusion matrix.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    def plot_classification_metrics(self, results, class_names,
                                   save_path="outputs/plots/classification_metrics.png"):
        """
        Plot classification metrics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision, Recall, F1-Score bar chart
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[0, 0].bar(x - width, results['precision'], width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x, results['recall'], width, label='Recall', alpha=0.8)
        axes[0, 0].bar(x + width, results['f1_score'], width, label='F1-Score', alpha=0.8)
        
        axes[0, 0].set_xlabel('Classes')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Per-Class Performance Metrics')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(class_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Support (number of samples per class)
        axes[0, 1].bar(class_names, results['support'], alpha=0.8, color='orange')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_title('Test Set Class Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Prediction confidence distribution
        max_probs = np.max(results['probabilities'], axis=1)
        axes[1, 0].hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Maximum Prediction Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Confidence Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correct vs Incorrect predictions confidence
        correct_mask = results['predictions'] == results['true_labels']
        correct_probs = max_probs[correct_mask]
        incorrect_probs = max_probs[~correct_mask]
        
        axes[1, 1].hist(correct_probs, bins=15, alpha=0.7, label='Correct', color='green')
        axes[1, 1].hist(incorrect_probs, bins=15, alpha=0.7, label='Incorrect', color='red')
        axes[1, 1].set_xlabel('Maximum Prediction Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Confidence: Correct vs Incorrect Predictions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Classification metrics plot saved to: {save_path}")
    
    def evaluate_segmentation(self, X_test, class_names):
        """
        Evaluate segmentation performance using synthetic masks.
        """
        logger.info("Evaluating segmentation performance...")
        
        if self.unet is None:
            logger.warning("U-Net not loaded, skipping segmentation evaluation")
            return None
        
        # Generate synthetic ground truth masks
        mask_generator = TumorMaskGenerator()
        
        # For evaluation, we'll use a subset of test images
        eval_subset = min(50, len(X_test))  # Limit for computational efficiency
        X_eval = X_test[:eval_subset]
        
        logger.info(f"Generating synthetic masks for {eval_subset} test images...")
        
        # Generate masks (this is a placeholder - in real scenario you'd have ground truth)
        synthetic_masks = []
        for i, image in enumerate(X_eval):
            # Simulate different tumor classes for mask generation
            class_idx = i % len(class_names)
            class_name = class_names[class_idx]
            mask = mask_generator.generate_tumor_mask(image, class_name)
            synthetic_masks.append(mask)
        
        synthetic_masks = np.array(synthetic_masks)
        
        # Get U-Net predictions
        predicted_masks = self.unet.predict(X_eval, verbose=1)
        predicted_masks = (predicted_masks.squeeze() > 0.5).astype(np.uint8)
        
        # Calculate IoU (Intersection over Union)
        ious = []
        for pred_mask, true_mask in zip(predicted_masks, synthetic_masks):
            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()
            iou = intersection / (union + 1e-8)
            ious.append(iou)
        
        mean_iou = np.mean(ious)
        
        logger.info(f"Segmentation Mean IoU: {mean_iou:.4f}")
        
        return {
            'mean_iou': mean_iou,
            'individual_ious': ious,
            'predicted_masks': predicted_masks,
            'synthetic_masks': synthetic_masks
        }
    
    def comprehensive_evaluation(self, dataset_path="processed"):
        """
        Run comprehensive evaluation pipeline.
        """
        logger.info("Starting comprehensive evaluation pipeline...")
        
        # Load dataset
        loader = BrainTumorDatasetLoader()
        dataset = loader.load_dataset(save_manifest=False)
        
        split_data = dataset['split_data']
        class_names = dataset['classes']
        
        # Load models
        self.load_models()
        
        # Evaluate classification
        classification_results = self.evaluate_classification(
            split_data['X_test'], split_data['y_test'], class_names
        )
        
        if classification_results:
            # Plot confusion matrix
            self.plot_confusion_matrix(
                classification_results['confusion_matrix'], class_names
            )
            
            # Plot classification metrics
            self.plot_classification_metrics(classification_results, class_names)
        
        # Evaluate segmentation
        segmentation_results = self.evaluate_segmentation(
            split_data['X_test'], class_names
        )
        
        # Generate Grad-CAM explanations for correctly classified samples
        if self.gradcam and classification_results:
            logger.info("Generating Grad-CAM explanations...")
            
            # Select correctly classified samples
            correct_mask = (classification_results['predictions'] == 
                          classification_results['true_labels'])
            correct_indices = np.where(correct_mask)[0]
            
            # Limit to first 20 correct predictions for efficiency
            sample_indices = correct_indices[:20]
            
            if len(sample_indices) > 0:
                explanations = self.gradcam.batch_generate_gradcam(
                    images=split_data['X_test'][sample_indices],
                    predictions=classification_results['predictions'][sample_indices],
                    true_labels=classification_results['true_labels'][sample_indices],
                    filenames=split_data['files_test'][sample_indices],
                    class_names=class_names
                )
                
                # Analyze Grad-CAM patterns
                self.gradcam.analyze_gradcam_patterns(explanations, class_names)
        
        # Tumor size estimation
        if segmentation_results:
            logger.info("Performing tumor size estimation...")
            
            size_estimator = TumorSizeEstimator(pixel_to_mm_ratio=0.5)
            
            size_results = size_estimator.estimate_tumor_sizes(
                masks=segmentation_results['predicted_masks'],
                filenames=split_data['files_test'][:len(segmentation_results['predicted_masks'])],
                predicted_classes=classification_results['predictions'][:len(segmentation_results['predicted_masks'])],
                class_names=class_names
            )
            
            # Save size estimation results
            size_estimator.save_size_estimation_results(size_results)
            
            # Visualize size distribution
            size_estimator.visualize_size_distribution(size_results)
        
        # Generate comprehensive report
        self.generate_evaluation_report(
            classification_results, segmentation_results, class_names
        )
        
        logger.info("Comprehensive evaluation completed!")
    
    def generate_evaluation_report(self, classification_results, 
                                 segmentation_results, class_names):
        """
        Generate comprehensive evaluation report.
        """
        logger.info("Generating evaluation report...")
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'dataset_info': {
                'num_classes': len(class_names),
                'class_names': class_names,
                'test_samples': len(classification_results['true_labels']) if classification_results else 0
            }
        }
        
        # Classification metrics
        if classification_results:
            report['classification'] = {
                'accuracy': float(classification_results['accuracy']),
                'per_class_metrics': {}
            }
            
            for i, class_name in enumerate(class_names):
                report['classification']['per_class_metrics'][class_name] = {
                    'precision': float(classification_results['precision'][i]),
                    'recall': float(classification_results['recall'][i]),
                    'f1_score': float(classification_results['f1_score'][i]),
                    'support': int(classification_results['support'][i])
                }
        
        # Segmentation metrics
        if segmentation_results:
            report['segmentation'] = {
                'mean_iou': float(segmentation_results['mean_iou']),
                'evaluated_samples': len(segmentation_results['individual_ious'])
            }
        
        # Save report
        report_path = "outputs/reports/evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*50)
        
        if classification_results:
            logger.info(f"Classification Accuracy: {classification_results['accuracy']:.4f}")
            logger.info(f"Average F1-Score: {np.mean(classification_results['f1_score']):.4f}")
        
        if segmentation_results:
            logger.info(f"Segmentation Mean IoU: {segmentation_results['mean_iou']:.4f}")
        
        logger.info("="*50)

def main():
    """
    Main evaluation pipeline.
    """
    logger.info("Brain Tumor Detection System - Comprehensive Evaluation")
    
    # Initialize evaluator
    evaluator = BrainTumorEvaluator()
    
    # Run comprehensive evaluation
    try:
        evaluator.comprehensive_evaluation()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.info("Please ensure:")
        logger.info("1. Preprocessed data is available in 'processed/' directory")
        logger.info("2. Trained models are available in 'models/' directory")
        logger.info("3. Run train_classifier.py and train_unet.py first")

if __name__ == "__main__":
    main()
