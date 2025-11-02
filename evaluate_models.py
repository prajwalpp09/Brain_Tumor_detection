"""
Comprehensive Model Evaluation System
====================================

Evaluates both classification and segmentation models with detailed metrics,
visualizations, and performance analysis.

Usage:
    python scripts/evaluate_models.py

The evaluation system will load trained models and generate comprehensive
performance reports with metrics, confusion matrices, and visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model

from config import (
    MODELS_DIR, RESULTS_DIR, LOGS_DIR, IMG_SIZE, NUM_CLASSES, CLASS_NAMES,
    RANDOM_SEED
)
from dataset_loader import BrainTumorDatasetLoader
from gradcam_explainer import GradCAMExplainer

class ModelEvaluator:
    """
    Comprehensive evaluation system for brain tumor detection models.
    """
    
    def __init__(self):
        self.classifier_model = None
        self.unet_model = None
        self.class_names = CLASS_NAMES
        self.num_classes = NUM_CLASSES
        
        # Setup logging
        self.setup_logging()
        
        # Set random seeds
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = LOGS_DIR / "model_evaluation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load trained models for evaluation"""
        # Load classifier
        classifier_path = MODELS_DIR / "brain_tumor_classifier.h5"
        if classifier_path.exists():
            try:
                self.classifier_model = load_model(str(classifier_path))
                self.logger.info(f"Classifier model loaded from {classifier_path}")
            except Exception as e:
                self.logger.warning(f"Could not load classifier model: {e}")
        
        # Load U-Net
        unet_path = MODELS_DIR / "unet_segmentation.h5"
        if unet_path.exists():
            try:
                # Custom objects for U-Net
                def dice_coefficient(y_true, y_pred, smooth=1e-6):
                    y_true_f = tf.keras.backend.flatten(y_true)
                    y_pred_f = tf.keras.backend.flatten(y_pred)
                    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
                    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
                
                def dice_loss(y_true, y_pred):
                    return 1 - dice_coefficient(y_true, y_pred)
                
                def combined_loss(y_true, y_pred):
                    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                    dice = dice_loss(y_true, y_pred)
                    return bce + dice
                
                custom_objects = {
                    'dice_coefficient': dice_coefficient,
                    'dice_loss': dice_loss,
                    'combined_loss': combined_loss
                }
                
                self.unet_model = load_model(str(unet_path), custom_objects=custom_objects)
                self.logger.info(f"U-Net model loaded from {unet_path}")
            except Exception as e:
                self.logger.warning(f"Could not load U-Net model: {e}")
    
    def evaluate_classifier(self, X_test, y_test, output_dir=None):
        """
        Comprehensive evaluation of the classification model.
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.classifier_model is None:
            self.logger.error("Classifier model not loaded")
            return None
        
        if output_dir is None:
            output_dir = RESULTS_DIR / "classifier_evaluation"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self.logger.info("Evaluating classification model...")
        
        # Get predictions
        y_pred_proba = self.classifier_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Basic metrics
        test_loss, test_accuracy = self.classifier_model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(y_pred[class_mask] == y_true[class_mask])
                per_class_metrics[class_name] = {
                    'accuracy': float(class_accuracy),
                    'samples': int(np.sum(class_mask))
                }
        
        # ROC curves and AUC (for multi-class)
        y_test_bin = label_binarize(y_true, classes=range(self.num_classes))
        roc_auc = {}
        
        if self.num_classes > 2:
            for i, class_name in enumerate(self.class_names):
                if np.sum(y_test_bin[:, i]) > 0:  # Check if class exists in test set
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc[class_name] = auc(fpr, tpr)
        
        # Create visualizations
        self._plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
        self._plot_roc_curves(y_test_bin, y_pred_proba, output_dir / "roc_curves.png")
        self._plot_class_distribution(y_true, y_pred, output_dir / "class_distribution.png")
        self._plot_prediction_confidence(y_pred_proba, y_true, output_dir / "prediction_confidence.png")
        
        # Compile results
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'roc_auc': roc_auc,
            'total_samples': len(X_test),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = output_dir / "classification_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.logger.info(f"Classification evaluation completed. Results saved to: {output_dir}")
        
        return evaluation_results
    
    def evaluate_segmentation(self, X_test, y_test_masks, output_dir=None):
        """
        Comprehensive evaluation of the segmentation model.
        
        Args:
            X_test: Test images
            y_test_masks: Test masks
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.unet_model is None:
            self.logger.error("U-Net model not loaded")
            return None
        
        if output_dir is None:
            output_dir = RESULTS_DIR / "segmentation_evaluation"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        self.logger.info("Evaluating segmentation model...")
        
        # Prepare masks
        if len(y_test_masks.shape) == 3:
            y_test_masks = np.expand_dims(y_test_masks, axis=-1)
        
        # Get predictions
        y_pred_masks = self.unet_model.predict(X_test, verbose=0)
        y_pred_binary = (y_pred_masks > 0.5).astype(np.uint8)
        
        # Basic metrics
        test_loss, test_dice, test_accuracy = self.unet_model.evaluate(X_test, y_test_masks, verbose=0)
        
        # Calculate additional metrics
        iou_scores = []
        dice_scores = []
        pixel_accuracies = []
        
        for i in range(len(y_test_masks)):
            true_mask = y_test_masks[i].squeeze()
            pred_mask = y_pred_binary[i].squeeze()
            
            # IoU (Intersection over Union)
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            iou = intersection / union if union > 0 else 1.0
            iou_scores.append(iou)
            
            # Dice coefficient
            dice = (2 * intersection) / (true_mask.sum() + pred_mask.sum()) if (true_mask.sum() + pred_mask.sum()) > 0 else 1.0
            dice_scores.append(dice)
            
            # Pixel accuracy
            pixel_acc = np.mean(true_mask == pred_mask)
            pixel_accuracies.append(pixel_acc)
        
        # Create visualizations
        self._plot_segmentation_samples(X_test, y_test_masks, y_pred_masks, output_dir / "segmentation_samples.png")
        self._plot_segmentation_metrics(iou_scores, dice_scores, pixel_accuracies, output_dir / "segmentation_metrics.png")
        
        # Compile results
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_dice_coefficient': float(test_dice),
            'test_binary_accuracy': float(test_accuracy),
            'mean_iou': float(np.mean(iou_scores)),
            'std_iou': float(np.std(iou_scores)),
            'mean_dice': float(np.mean(dice_scores)),
            'std_dice': float(np.std(dice_scores)),
            'mean_pixel_accuracy': float(np.mean(pixel_accuracies)),
            'std_pixel_accuracy': float(np.std(pixel_accuracies)),
            'total_samples': len(X_test),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = output_dir / "segmentation_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.logger.info(f"Segmentation evaluation completed. Results saved to: {output_dir}")
        
        return evaluation_results
    
    def _plot_confusion_matrix(self, cm, save_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, y_test_bin, y_pred_proba, save_path):
        """Plot ROC curves for multi-class classification"""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if i < y_test_bin.shape[1] and np.sum(y_test_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distribution(self, y_true, y_pred, save_path):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        true_counts = np.bincount(y_true, minlength=self.num_classes)
        ax1.bar(self.class_names, true_counts, color='skyblue', alpha=0.7)
        ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Predicted distribution
        pred_counts = np.bincount(y_pred, minlength=self.num_classes)
        ax2.bar(self.class_names, pred_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_confidence(self, y_pred_proba, y_true, save_path):
        """Plot prediction confidence distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            if i < len(axes):
                # Get confidence scores for this class
                class_confidences = y_pred_proba[:, i]
                correct_mask = y_true == i
                
                # Plot histogram
                axes[i].hist(class_confidences[correct_mask], bins=20, alpha=0.7, 
                           label='Correct', color='green', density=True)
                axes[i].hist(class_confidences[~correct_mask], bins=20, alpha=0.7, 
                           label='Incorrect', color='red', density=True)
                axes[i].set_title(f'{class_name} Confidence', fontweight='bold')
                axes[i].set_xlabel('Confidence Score')
                axes[i].set_ylabel('Density')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_segmentation_samples(self, images, true_masks, pred_masks, save_path, num_samples=6):
        """Plot segmentation sample results"""
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            # Original image
            axes[i, 0].imshow(images[idx])
            axes[i, 0].set_title('Original' if i == 0 else '')
            axes[i, 0].axis('off')
            
            # True mask
            axes[i, 1].imshow(true_masks[idx].squeeze(), cmap='gray')
            axes[i, 1].set_title('True Mask' if i == 0 else '')
            axes[i, 1].axis('off')
            
            # Predicted mask
            axes[i, 2].imshow(pred_masks[idx].squeeze(), cmap='gray')
            axes[i, 2].set_title('Predicted Mask' if i == 0 else '')
            axes[i, 2].axis('off')
            
            # Overlay
            overlay = images[idx].copy()
            pred_binary = (pred_masks[idx].squeeze() > 0.5).astype(np.uint8)
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], pred_binary * 0.5)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay' if i == 0 else '')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_segmentation_metrics(self, iou_scores, dice_scores, pixel_accuracies, save_path):
        """Plot segmentation metrics distribution"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # IoU scores
        axes[0].hist(iou_scores, bins=20, alpha=0.7, color='blue')
        axes[0].axvline(np.mean(iou_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(iou_scores):.3f}')
        axes[0].set_title('IoU Score Distribution', fontweight='bold')
        axes[0].set_xlabel('IoU Score')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Dice scores
        axes[1].hist(dice_scores, bins=20, alpha=0.7, color='green')
        axes[1].axvline(np.mean(dice_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(dice_scores):.3f}')
        axes[1].set_title('Dice Score Distribution', fontweight='bold')
        axes[1].set_xlabel('Dice Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Pixel accuracies
        axes[2].hist(pixel_accuracies, bins=20, alpha=0.7, color='orange')
        axes[2].axvline(np.mean(pixel_accuracies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(pixel_accuracies):.3f}')
        axes[2].set_title('Pixel Accuracy Distribution', fontweight='bold')
        axes[2].set_xlabel('Pixel Accuracy')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, classification_results, segmentation_results=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            classification_results: Results from classifier evaluation
            segmentation_results: Results from segmentation evaluation (optional)
        """
        report_path = RESULTS_DIR / "comprehensive_evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Brain Tumor Detection System - Evaluation Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Classification results
            if classification_results:
                f.write("## Classification Model Performance\n\n")
                f.write(f"- **Test Accuracy:** {classification_results['test_accuracy']:.4f}\n")
                f.write(f"- **Test Loss:** {classification_results['test_loss']:.4f}\n")
                f.write(f"- **Total Test Samples:** {classification_results['total_samples']}\n\n")
                
                f.write("### Per-Class Performance\n\n")
                for class_name in self.class_names:
                    if class_name in classification_results['classification_report']:
                        metrics = classification_results['classification_report'][class_name]
                        f.write(f"**{class_name}:**\n")
                        f.write(f"- Precision: {metrics['precision']:.4f}\n")
                        f.write(f"- Recall: {metrics['recall']:.4f}\n")
                        f.write(f"- F1-Score: {metrics['f1-score']:.4f}\n")
                        f.write(f"- Support: {metrics['support']}\n\n")
            
            # Segmentation results
            if segmentation_results:
                f.write("## Segmentation Model Performance\n\n")
                f.write(f"- **Test Dice Coefficient:** {segmentation_results['test_dice_coefficient']:.4f}\n")
                f.write(f"- **Test Binary Accuracy:** {segmentation_results['test_binary_accuracy']:.4f}\n")
                f.write(f"- **Mean IoU:** {segmentation_results['mean_iou']:.4f} ± {segmentation_results['std_iou']:.4f}\n")
                f.write(f"- **Mean Pixel Accuracy:** {segmentation_results['mean_pixel_accuracy']:.4f} ± {segmentation_results['std_pixel_accuracy']:.4f}\n")
                f.write(f"- **Total Test Samples:** {segmentation_results['total_samples']}\n\n")
            
            f.write("## Model Files\n\n")
            f.write("- Classification Model: `models/brain_tumor_classifier.h5`\n")
            f.write("- Segmentation Model: `models/unet_segmentation.h5`\n")
            f.write("- Evaluation Results: `results/`\n\n")
            
            f.write("## Visualizations Generated\n\n")
            f.write("- Confusion Matrix\n")
            f.write("- ROC Curves\n")
            f.write("- Class Distribution Comparison\n")
            f.write("- Prediction Confidence Analysis\n")
            if segmentation_results:
                f.write("- Segmentation Sample Results\n")
                f.write("- Segmentation Metrics Distribution\n")
        
        self.logger.info(f"Comprehensive evaluation report saved to: {report_path}")

def main():
    """
    Main evaluation pipeline.
    """
    print("Brain Tumor Detection System - Model Evaluation")
    print("=" * 55)
    print(f"Classes: {CLASS_NAMES}")
    print(f"Image Size: {IMG_SIZE}")
    print()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    print("Loading trained models...")
    evaluator.load_models()
    
    # Load test data
    print("Loading test dataset...")
    loader = BrainTumorDatasetLoader()
    
    try:
        split_data = loader.load_saved_splits()
        
        if len(split_data['X_test']) == 0:
            print("No test data found. Using validation data for evaluation.")
            X_test = split_data['X_val']
            y_test = split_data['y_val']
        else:
            X_test = split_data['X_test']
            y_test = split_data['y_test']
        
        print(f"Test set: {X_test.shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease run the dataset loading script first:")
        print("python scripts/dataset_loader.py")
        return
    
    classification_results = None
    segmentation_results = None
    
    # Evaluate classification model
    if evaluator.classifier_model is not None:
        print("\nEvaluating classification model...")
        classification_results = evaluator.evaluate_classifier(X_test, y_test)
        
        if classification_results:
            print(f"Classification Accuracy: {classification_results['test_accuracy']:.4f}")
            print(f"Classification Loss: {classification_results['test_loss']:.4f}")
    else:
        print("Classification model not available for evaluation.")
    
    # Evaluate segmentation model
    if evaluator.unet_model is not None:
        print("\nGenerating synthetic masks for segmentation evaluation...")
        from train_unet import TumorMaskGenerator
        
        mask_generator = TumorMaskGenerator()
        test_masks = mask_generator.generate_masks_for_dataset(X_test, y_test, CLASS_NAMES)
        
        print("Evaluating segmentation model...")
        segmentation_results = evaluator.evaluate_segmentation(X_test, test_masks)
        
        if segmentation_results:
            print(f"Segmentation Dice Coefficient: {segmentation_results['test_dice_coefficient']:.4f}")
            print(f"Segmentation IoU: {segmentation_results['mean_iou']:.4f}")
    else:
        print("Segmentation model not available for evaluation.")
    
    # Generate comprehensive report
    if classification_results or segmentation_results:
        print("\nGenerating comprehensive evaluation report...")
        evaluator.generate_comprehensive_report(classification_results, segmentation_results)
    
    print(f"\nEvaluation completed! Results saved to: {RESULTS_DIR}")
    
    # Summary
    print("\nEvaluation Summary:")
    if classification_results:
        print(f"✓ Classification model evaluated - Accuracy: {classification_results['test_accuracy']:.4f}")
    if segmentation_results:
        print(f"✓ Segmentation model evaluated - Dice: {segmentation_results['test_dice_coefficient']:.4f}")
    
    print("\nAll evaluation results and visualizations have been saved.")

if __name__ == "__main__":
    main()
