# Brain Tumor Detection System - Evaluation Report

**Generated on:** 2025-10-05 10:50:19

## Classification Model Performance

- **Test Accuracy:** 0.9085
- **Test Loss:** 0.4760
- **Total Test Samples:** 317

### Per-Class Performance

**glioma_tumor:**
- Precision: 0.9072
- Recall: 0.9462
- F1-Score: 0.9263
- Support: 93.0

**meningioma_tumor:**
- Precision: 0.9000
- Recall: 0.8617
- F1-Score: 0.8804
- Support: 94.0

**no_tumor:**
- Precision: 0.8718
- Recall: 0.8500
- F1-Score: 0.8608
- Support: 40.0

**pituitary_tumor:**
- Precision: 0.9341
- Recall: 0.9444
- F1-Score: 0.9392
- Support: 90.0

## Segmentation Model Performance

- **Test Dice Coefficient:** 0.5786
- **Test Binary Accuracy:** 0.9619
- **Mean IoU:** 0.4599 ± 0.2391
- **Mean Pixel Accuracy:** 0.9619 ± 0.0235
- **Total Test Samples:** 317

## Model Files

- Classification Model: `models/brain_tumor_classifier.h5`
- Segmentation Model: `models/unet_segmentation.h5`
- Evaluation Results: `results/`

## Visualizations Generated

- Confusion Matrix
- ROC Curves
- Class Distribution Comparison
- Prediction Confidence Analysis
- Segmentation Sample Results
- Segmentation Metrics Distribution
