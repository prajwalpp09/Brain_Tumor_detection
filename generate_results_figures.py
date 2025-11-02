#!/usr/bin/env python3
"""
Generate Results Figures for Brain Tumor Detection Paper
Creates training curves and statistical analysis from evaluation results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Setup paths
RESULTS_DIR = Path("scripts/results")
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Load data
print("Loading evaluation results...")
with open(RESULTS_DIR / "classifier_evaluation.json") as f:
    classifier_results = json.load(f)

with open(RESULTS_DIR / "unet_evaluation.json") as f:
    unet_results = json.load(f)

with open(RESULTS_DIR / "classifier_training_history.json") as f:
    classifier_history = json.load(f)

with open(RESULTS_DIR / "unet_training_history.json") as f:
    unet_history = json.load(f)

with open(RESULTS_DIR / "segmentation_evaluation/segmentation_evaluation.json") as f:
    seg_eval = json.load(f)

print("Creating training curves...")

# Figure 1: Training Curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Classifier accuracy and loss
epochs = range(1, len(classifier_history['accuracy']) + 1)
axes[0, 0].plot(epochs, classifier_history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
axes[0, 0].plot(epochs, classifier_history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2)
axes[0, 0].set_title('(a) MobileNetV2 Classification Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1])

axes[0, 1].plot(epochs, classifier_history['loss'], 'b-', label='Training Loss', linewidth=2)
axes[0, 1].plot(epochs, classifier_history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
axes[0, 1].set_title('(b) MobileNetV2 Classification Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# U-Net Dice and loss
epochs_unet = range(1, len(unet_history['dice_coefficient']) + 1)
axes[1, 0].plot(epochs_unet, unet_history['dice_coefficient'], 'g-', label='Training Dice', linewidth=2)
axes[1, 0].plot(epochs_unet, unet_history['val_dice_coefficient'], 'm--', label='Validation Dice', linewidth=2)
axes[1, 0].set_title('(c) U-Net Segmentation Dice Coefficient', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Dice Coefficient')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(epochs_unet, unet_history['loss'], 'g-', label='Training Loss', linewidth=2)
axes[1, 1].plot(epochs_unet, unet_history['val_loss'], 'm--', label='Validation Loss', linewidth=2)
axes[1, 1].set_title('(d) U-Net Segmentation Loss', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=300, bbox_inches='tight')
print(f"Saved training curves to {OUTPUT_DIR / 'training_curves.png'}")

# Print summary tables
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Classification metrics
print("\n--- Classification Results ---")
print(f"Test Accuracy: {classifier_results['test_accuracy']:.4f} ({classifier_results['test_accuracy']*100:.2f}%)")
print(f"Test Loss: {classifier_results['test_loss']:.4f}")
print(f"Total Test Samples: {classifier_results['total_samples']}")

print("\nPer-Class Accuracies:")
for class_name, acc in classifier_results['class_accuracies'].items():
    print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")

# Segmentation metrics
print("\n--- Segmentation Results ---")
print(f"Test Dice Coefficient: {unet_results['test_dice_coefficient']:.4f}")
print(f"Test Binary Accuracy: {unet_results['test_binary_accuracy']:.4f} ({unet_results['test_binary_accuracy']*100:.2f}%)")
print(f"Test Loss: {unet_results['test_loss']:.4f}")
print(f"Mean IoU: {seg_eval['mean_iou']:.4f} ± {seg_eval['std_iou']:.4f}")
print(f"Mean Dice: {seg_eval['mean_dice']:.4f} ± {seg_eval['std_dice']:.4f}")
print(f"Mean Pixel Accuracy: {seg_eval['mean_pixel_accuracy']:.4f} ± {seg_eval['std_pixel_accuracy']:.4f}")

# Size estimation from predictions
print("\n--- Size Estimation Summary ---")
predictions_dir = Path("outputs/predictions")
prediction_files = list(predictions_dir.glob("*_analysis_results.json"))

if prediction_files:
    areas = []
    for pred_file in prediction_files:
        with open(pred_file) as f:
            data = json.load(f)
            if 'size_estimation' in data:
                areas.append(data['size_estimation']['area_mm2'])
    
    if areas:
        areas = np.array(areas)
        print(f"Number of predictions: {len(areas)}")
        print(f"Mean tumor area: {areas.mean():.2f} mm²")
        print(f"Std tumor area: {areas.std():.2f} mm²")
        print(f"Min area: {areas.min():.2f} mm²")
        print(f"Max area: {areas.max():.2f} mm²")

print("\n" + "="*80)
print("Analysis complete!")

