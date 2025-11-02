"""
Brain Tumor U-Net Segmentation Model
===================================

Implements U-Net Lite for tumor segmentation with synthetic mask generation.
Integrates with the centralized configuration system.

Usage:
    python scripts/train_unet.py

The U-Net model will automatically use parameters from config.py and load data from the dataset loader.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, 
    BatchNormalization, Activation, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import MeanIoU
import tensorflow.keras.backend as K

from config import (
    MODELS_DIR, RESULTS_DIR, LOGS_DIR, IMG_SIZE, NUM_CLASSES, CLASS_NAMES,
    BATCH_SIZE, UNET_EPOCHS, UNET_LEARNING_RATE, UNET_PATIENCE,
    RANDOM_SEED
)
from dataset_loader import BrainTumorDatasetLoader

class TumorMaskGenerator:
    """
    Generates synthetic tumor masks using image processing techniques.
    Used when ground truth segmentation masks are not available.
    """
    
    def __init__(self):
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = LOGS_DIR / "mask_generation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_tumor_mask(self, image, tumor_class=None):
        """
        Generate tumor mask using thresholding and morphological operations.
        
        Args:
            image: Input RGB image (0-1 normalized)
            tumor_class: Class of tumor (for class-specific processing)
            
        Returns:
            Binary mask of tumor region
        """
        # Convert to uint8 for OpenCV operations
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding to find potential tumor regions
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and keep only significant ones
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(gray)
        
        if contours:
            # Sort contours by area and keep the largest ones
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # For tumor images, create mask from largest contours
            if tumor_class and tumor_class != 'no_tumor':
                # Keep top 1-3 contours depending on their size
                total_area = gray.shape[0] * gray.shape[1]
                for contour in contours[:3]:
                    area = cv2.contourArea(contour)
                    # Only include contours that are significant but not too large
                    if 100 < area < total_area * 0.3:
                        cv2.fillPoly(mask, [contour], 255)
            else:
                # For no_tumor class, create minimal or no mask
                pass
        
        # Normalize mask to 0-1
        mask = mask.astype(np.float32) / 255.0
        
        return mask
    
    def generate_masks_for_dataset(self, images, labels, classes):
        """
        Generate masks for entire dataset.
        
        Args:
            images: Array of images
            labels: Array of one-hot encoded labels
            classes: List of class names
            
        Returns:
            Array of binary masks
        """
        masks = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            # Get class name
            class_idx = np.argmax(label)
            class_name = classes[class_idx]
            
            # Generate mask
            mask = self.generate_tumor_mask(image, class_name)
            masks.append(mask)
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"Generated masks for {i + 1}/{len(images)} images")
        
        return np.array(masks)
    
    def save_sample_masks(self, images, masks, filenames, output_dir=None, num_samples=10):
        """
        Save sample masks for visualization.
        
        Args:
            images: Array of images
            masks: Array of masks
            filenames: Array of filenames
            output_dir: Directory to save samples
            num_samples: Number of samples to save
        """
        if output_dir is None:
            output_dir = RESULTS_DIR / "sample_masks"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Select random samples
        indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
        
        for i, idx in enumerate(indices):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(images[idx])
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Generated mask
            axes[1].imshow(masks[idx], cmap='gray')
            axes[1].set_title('Generated Mask')
            axes[1].axis('off')
            
            # Overlay
            overlay = images[idx].copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = masks[idx]  # Red channel for mask
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"sample_mask_{i+1}_{filenames[idx]}", 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Sample masks saved to: {output_dir}")

class UNetLite:
    """
    Lightweight U-Net implementation for tumor segmentation.
    """
    
    def __init__(self, num_filters=32):
        self.input_shape = (*IMG_SIZE, 3)
        self.num_filters = num_filters
        self.model = None
        self.history = None
        
        # Setup logging
        self.setup_logging()
        
        # Set random seeds for reproducibility
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = LOGS_DIR / "unet_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def conv_block(self, inputs, num_filters, dropout_rate=0.1):
        """
        Convolutional block with BatchNorm and Dropout.
        """
        x = Conv2D(num_filters, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x
    
    def encoder_block(self, inputs, num_filters, dropout_rate=0.1):
        """
        Encoder block with conv block and max pooling.
        """
        x = self.conv_block(inputs, num_filters, dropout_rate)
        p = MaxPooling2D((2, 2))(x)
        return x, p
    
    def decoder_block(self, inputs, skip_features, num_filters, dropout_rate=0.1):
        """
        Decoder block with upsampling and skip connections.
        """
        x = UpSampling2D((2, 2))(inputs)
        x = concatenate([x, skip_features])
        x = self.conv_block(x, num_filters, dropout_rate)
        return x
    
    def build_model(self):
        """
        Build U-Net Lite architecture.
        """
        self.logger.info("Building U-Net Lite model...")
        
        inputs = Input(self.input_shape)
        
        # Encoder
        s1, p1 = self.encoder_block(inputs, self.num_filters)
        s2, p2 = self.encoder_block(p1, self.num_filters * 2)
        s3, p3 = self.encoder_block(p2, self.num_filters * 4)
        s4, p4 = self.encoder_block(p3, self.num_filters * 8)
        
        # Bridge
        b1 = self.conv_block(p4, self.num_filters * 16, dropout_rate=0.2)
        
        # Decoder
        d1 = self.decoder_block(b1, s4, self.num_filters * 8)
        d2 = self.decoder_block(d1, s3, self.num_filters * 4)
        d3 = self.decoder_block(d2, s2, self.num_filters * 2)
        d4 = self.decoder_block(d3, s1, self.num_filters)
        
        # Output
        outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)
        
        self.model = Model(inputs, outputs, name='UNet_Lite')
        
        self.logger.info(f"U-Net Lite built with {self.model.count_params():,} parameters")
        
        return self.model
    
    def dice_coefficient(self, y_true, y_pred, smooth=1e-6):
        """
        Dice coefficient for segmentation evaluation.
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def dice_loss(self, y_true, y_pred):
        """
        Dice loss for segmentation.
        """
        return 1 - self.dice_coefficient(y_true, y_pred)
    
    def combined_loss(self, y_true, y_pred):
        """
        Combined binary crossentropy and dice loss.
        """
        bce = binary_crossentropy(y_true, y_pred)
        dice = self.dice_loss(y_true, y_pred)
        return bce + dice
    
    def compile_model(self):
        """
        Compile the U-Net model using config parameters.
        """
        self.model.compile(
            optimizer=Adam(learning_rate=UNET_LEARNING_RATE),
            loss=self.combined_loss,
            metrics=[self.dice_coefficient, 'binary_accuracy']
        )
        
        self.logger.info("U-Net model compiled successfully")
    
    def get_callbacks(self, model_path=None):
        """
        Get training callbacks using config parameters.
        """
        if model_path is None:
            model_path = MODELS_DIR / "unet_segmentation.h5"
        
        callbacks = [
            ModelCheckpoint(
                filepath=str(model_path),
                monitor='val_dice_coefficient',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=UNET_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=UNET_PATIENCE // 2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, model_path=None):
        """
        Train the U-Net segmentation model using config parameters.
        """
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        if model_path is None:
            model_path = MODELS_DIR / "unet_segmentation.h5"
        
        self.logger.info("Starting U-Net training...")
        self.logger.info(f"Training samples: {len(X_train)}")
        self.logger.info(f"Validation samples: {len(X_val)}")
        self.logger.info(f"Epochs: {UNET_EPOCHS}, Batch size: {BATCH_SIZE}")
        
        # Reshape masks to have channel dimension
        y_train = np.expand_dims(y_train, axis=-1)
        y_val = np.expand_dims(y_val, axis=-1)
        
        callbacks = self.get_callbacks(model_path)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=UNET_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("U-Net training completed!")
        
        # Save training history
        history_path = RESULTS_DIR / "unet_training_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {}
            for key, values in self.history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        self.logger.info(f"Training history saved to: {history_path}")
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained U-Net model on test data.
        
        Args:
            X_test: Test images
            y_test: Test masks
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            self.logger.error("No model to evaluate. Train the model first.")
            return None
        
        self.logger.info("Evaluating U-Net model on test data...")
        
        # Reshape masks to have channel dimension
        y_test = np.expand_dims(y_test, axis=-1)
        
        # Evaluate model
        test_loss, test_dice, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_dice_coefficient': float(test_dice),
            'test_binary_accuracy': float(test_accuracy),
            'total_samples': len(X_test)
        }
        
        # Save evaluation results
        eval_path = RESULTS_DIR / "unet_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.logger.info(f"Test Dice coefficient: {test_dice:.4f}")
        self.logger.info(f"Test binary accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Test loss: {test_loss:.4f}")
        self.logger.info(f"Evaluation results saved to: {eval_path}")
        
        return evaluation_results
    
    def predict_masks(self, images, batch_size=None):
        """
        Predict segmentation masks for images.
        """
        if batch_size is None:
            batch_size = BATCH_SIZE
            
        if self.model is None:
            self.logger.error("Model not built. Build and train the model first.")
            return None
        
        predictions = self.model.predict(images, batch_size=batch_size)
        
        # Remove channel dimension and threshold
        masks = (predictions.squeeze() > 0.5).astype(np.uint8)
        
        return masks
    
    def save_predicted_masks(self, images, filenames, output_dir=None):
        """
        Save predicted masks as images.
        """
        if output_dir is None:
            output_dir = RESULTS_DIR / "predicted_masks"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        masks = self.predict_masks(images)
        
        for i, (mask, filename) in enumerate(zip(masks, filenames)):
            # Create filename for mask
            name_parts = filename.split('.')
            mask_filename = f"{name_parts[0]}_mask.png"
            mask_path = output_dir / mask_filename
            
            # Save mask
            cv2.imwrite(str(mask_path), mask * 255)
            
            if (i + 1) % 50 == 0:
                self.logger.info(f"Saved {i + 1}/{len(masks)} masks")
        
        self.logger.info(f"All masks saved to: {output_dir}")
    
    def plot_training_curves(self, save_path=None):
        """
        Plot U-Net training curves.
        """
        if save_path is None:
            save_path = RESULTS_DIR / "unet_training_curves.png"
            
        if self.history is None:
            self.logger.error("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot dice coefficient
        axes[0, 1].plot(self.history.history['dice_coefficient'], label='Training Dice')
        axes[0, 1].plot(self.history.history['val_dice_coefficient'], label='Validation Dice')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot binary accuracy
        axes[1, 0].plot(self.history.history['binary_accuracy'], label='Training Accuracy')
        axes[1, 0].plot(self.history.history['val_binary_accuracy'], label='Validation Accuracy')
        axes[1, 0].set_title('Binary Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"U-Net training curves saved to: {save_path}")
    
    def save_model_for_deployment(self, model_path=None, savedmodel_path=None):
        """
        Save U-Net model in multiple formats for deployment.
        """
        if self.model is None:
            self.logger.error("No model to save. Build and train the model first.")
            return
        
        if model_path is None:
            model_path = MODELS_DIR / "unet_segmentation.h5"
        if savedmodel_path is None:
            savedmodel_path = MODELS_DIR / "unet_savedmodel"
        
        # Save as H5 format
        self.model.save(str(model_path))
        self.logger.info(f"U-Net model saved as H5 format: {model_path}")
        
        # Save as SavedModel format
        self.model.save(str(savedmodel_path))
        self.logger.info(f"U-Net model saved as SavedModel format: {savedmodel_path}")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(MODELS_DIR / "unet_architecture.json", "w") as json_file:
            json_file.write(model_json)
        
        # Save model summary
        with open(MODELS_DIR / "unet_summary.txt", "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        self.logger.info("U-Net model architecture and summary saved")

def main():
    """
    Main U-Net training pipeline.
    """
    print("Brain Tumor U-Net Segmentation Training Pipeline")
    print("=" * 55)
    print(f"Model: U-Net Lite")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {UNET_EPOCHS}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    loader = BrainTumorDatasetLoader()
    
    try:
        # Try to load saved splits first
        try:
            split_data = loader.load_saved_splits()
            print("Loaded saved dataset splits!")
        except FileNotFoundError:
            print("No saved splits found. Loading and creating new splits...")
            dataset = loader.load_dataset(use_augmented=False)  # Use only original images for segmentation
            split_data = dataset['split_data']
        
        print(f"Training set: {split_data['X_train'].shape}")
        print(f"Validation set: {split_data['X_val'].shape}")
        print(f"Test set: {split_data['X_test'].shape}")
        
        if len(split_data['X_train']) == 0:
            raise ValueError("No training data found!")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease run the preprocessing and dataset loading scripts first:")
        print("1. python scripts/brain_tumor_preprocessing.py")
        print("2. python scripts/dataset_loader.py")
        return
    
    # Generate synthetic masks
    print("\nGenerating synthetic tumor masks...")
    mask_generator = TumorMaskGenerator()
    
    # Generate masks for training and validation sets
    train_masks = mask_generator.generate_masks_for_dataset(
        split_data['X_train'], split_data['y_train'], CLASS_NAMES
    )
    val_masks = mask_generator.generate_masks_for_dataset(
        split_data['X_val'], split_data['y_val'], CLASS_NAMES
    )
    
    print(f"Generated masks - Train: {train_masks.shape}, Val: {val_masks.shape}")
    
    # Save sample masks for visualization
    mask_generator.save_sample_masks(
        split_data['X_train'], train_masks, split_data['files_train']
    )
    
    # Initialize and train U-Net
    print("\nBuilding U-Net model...")
    unet = UNetLite(num_filters=32)
    
    # Build and compile model
    model = unet.build_model()
    unet.compile_model()
    model.summary()
    
    # Train model
    print("\nStarting U-Net training...")
    history = unet.train(
        X_train=split_data['X_train'],
        y_train=train_masks,
        X_val=split_data['X_val'],
        y_val=val_masks
    )
    
    # Plot training curves
    print("\nGenerating training curves...")
    unet.plot_training_curves()
    
    # Evaluate on test set if available
    if len(split_data['X_test']) > 0:
        print("\nGenerating test masks and evaluating...")
        test_masks = mask_generator.generate_masks_for_dataset(
            split_data['X_test'], split_data['y_test'], CLASS_NAMES
        )
        evaluation_results = unet.evaluate_model(split_data['X_test'], test_masks)
        
        # Generate and save test masks
        unet.save_predicted_masks(split_data['X_test'], split_data['files_test'])
    
    # Save model for deployment
    print("\nSaving U-Net model for deployment...")
    unet.save_model_for_deployment()
    
    print("\nU-Net segmentation pipeline completed successfully!")
    print(f"Model saved to: {MODELS_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
