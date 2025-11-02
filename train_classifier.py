"""
Brain Tumor CNN Classification Model Training
============================================

Uses MobileNetV2 with transfer learning for 4-class brain tumor classification.
Integrates with the centralized configuration system.

Usage:
    python scripts/train_classifier.py

The classifier will automatically use parameters from config.py and load data from the dataset loader.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from config import (
    MODELS_DIR, RESULTS_DIR, LOGS_DIR, IMG_SIZE, NUM_CLASSES, CLASS_NAMES,
    BATCH_SIZE, CLASSIFIER_EPOCHS, CLASSIFIER_LEARNING_RATE, CLASSIFIER_PATIENCE,
    RANDOM_SEED
)
from dataset_loader import BrainTumorDatasetLoader

class BrainTumorClassifier:
    """
    CNN classifier for brain tumor detection using MobileNetV2 transfer learning.
    """
    
    def __init__(self, dropout_rate=0.5, l2_reg=0.001):
        self.input_shape = (*IMG_SIZE, 3)
        self.num_classes = NUM_CLASSES
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        self.history = None
        self.classes = CLASS_NAMES
        
        # Setup logging
        self.setup_logging()
        
        # Set random seeds for reproducibility
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = LOGS_DIR / "classifier_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def build_model(self, fine_tune_layers=50):
        """
        Build MobileNetV2-based classification model.
        
        Args:
            fine_tune_layers: Number of top layers to fine-tune
        """
        self.logger.info("Building MobileNetV2 classification model...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            alpha=1.0,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='relu', 
                 kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax',
                       kernel_regularizer=l2(self.l2_reg))(x)
        
        self.model = Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=CLASSIFIER_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info(f"Model built with {self.model.count_params():,} parameters")
        
        # Enable fine-tuning for top layers
        if fine_tune_layers > 0:
            base_model.trainable = True
            # Freeze all layers except the top fine_tune_layers
            for layer in base_model.layers[:-fine_tune_layers]:
                layer.trainable = False
            
            # Recompile with lower learning rate for fine-tuning
            self.model.compile(
                optimizer=Adam(learning_rate=CLASSIFIER_LEARNING_RATE * 0.1),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.logger.info(f"Fine-tuning enabled for top {fine_tune_layers} layers")
        
        return self.model
    
    def get_callbacks(self, model_path=None):
        """
        Get training callbacks using config parameters.
        
        Args:
            model_path: Path to save best model
            
        Returns:
            List of callbacks
        """
        if model_path is None:
            model_path = MODELS_DIR / "brain_tumor_classifier.h5"
        
        callbacks = [
            ModelCheckpoint(
                filepath=str(model_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=CLASSIFIER_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=CLASSIFIER_PATIENCE // 2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, model_path=None):
        """
        Train the classification model using config parameters.
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels (one-hot encoded)
            model_path: Path to save the best model
        """
        if self.model is None:
            self.build_model()
        
        if model_path is None:
            model_path = MODELS_DIR / "brain_tumor_classifier.h5"
        
        self.logger.info("Starting model training...")
        self.logger.info(f"Training samples: {len(X_train)}")
        self.logger.info(f"Validation samples: {len(X_val)}")
        self.logger.info(f"Epochs: {CLASSIFIER_EPOCHS}, Batch size: {BATCH_SIZE}")
        
        # Get callbacks
        callbacks = self.get_callbacks(model_path)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CLASSIFIER_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("Training completed!")
        
        # Save training history
        history_path = RESULTS_DIR / "classifier_training_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {}
            for key, values in self.history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        self.logger.info(f"Training history saved to: {history_path}")
        
        return self.history
    
    def plot_training_curves(self, save_path=None):
        """
        Plot training and validation curves.
        
        Args:
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = RESULTS_DIR / "classifier_training_curves.png"
            
        if self.history is None:
            self.logger.error("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Training curves saved to: {save_path}")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            self.logger.error("No model to evaluate. Train the model first.")
            return None
        
        self.logger.info("Evaluating model on test data...")
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for i, class_name in enumerate(self.classes):
            class_mask = y_true_classes == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(y_pred_classes[class_mask] == y_true_classes[class_mask])
                class_accuracies[class_name] = float(class_acc)
        
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'class_accuracies': class_accuracies,
            'total_samples': len(X_test)
        }
        
        # Save evaluation results
        eval_path = RESULTS_DIR / "classifier_evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        self.logger.info(f"Test accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Test loss: {test_loss:.4f}")
        self.logger.info("Per-class accuracies:")
        for class_name, acc in class_accuracies.items():
            self.logger.info(f"  {class_name}: {acc:.4f}")
        
        self.logger.info(f"Evaluation results saved to: {eval_path}")
        
        return evaluation_results
    
    def save_model_for_deployment(self, model_path=None, savedmodel_path=None):
        """
        Save model in multiple formats for deployment.
        
        Args:
            model_path: Path to H5 model file
            savedmodel_path: Path to SavedModel directory
        """
        if self.model is None:
            self.logger.error("No model to save. Build and train the model first.")
            return
        
        if model_path is None:
            model_path = MODELS_DIR / "brain_tumor_classifier.h5"
        if savedmodel_path is None:
            savedmodel_path = MODELS_DIR / "brain_tumor_savedmodel"
        
        # Save as H5 format
        self.model.save(str(model_path))
        self.logger.info(f"Model saved as H5 format: {model_path}")
        
        # Save as SavedModel format for TensorFlow Serving
        self.model.save(str(savedmodel_path))
        self.logger.info(f"Model saved as SavedModel format: {savedmodel_path}")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(MODELS_DIR / "classifier_architecture.json", "w") as json_file:
            json_file.write(model_json)
        
        # Save model summary
        with open(MODELS_DIR / "classifier_summary.txt", "w") as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Save class names
        with open(MODELS_DIR / "class_names.json", "w") as f:
            json.dump(self.classes, f, indent=2)
        
        self.logger.info("Model architecture, summary, and class names saved")

def main():
    """
    Main training pipeline.
    """
    print("Brain Tumor Classification Training Pipeline")
    print("=" * 50)
    print(f"Model: MobileNetV2 Transfer Learning")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {CLASSIFIER_EPOCHS}")
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
            dataset = loader.load_dataset(use_augmented=False)
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
    
    # Initialize and train classifier
    classifier = BrainTumorClassifier()
    
    # Build model
    print("\nBuilding model...")
    model = classifier.build_model(fine_tune_layers=50)
    model.summary()
    
    # Train model
    print("\nStarting training...")
    history = classifier.train(
        X_train=split_data['X_train'],
        y_train=split_data['y_train'],
        X_val=split_data['X_val'],
        y_val=split_data['y_val']
    )
    
    # Plot training curves
    print("\nGenerating training curves...")
    classifier.plot_training_curves()
    
    # Evaluate on test set if available
    if len(split_data['X_test']) > 0:
        print("\nEvaluating on test set...")
        evaluation_results = classifier.evaluate_model(
            split_data['X_test'], 
            split_data['y_test']
        )
    
    # Save model for deployment
    print("\nSaving model for deployment...")
    classifier.save_model_for_deployment()
    
    print("\nTraining pipeline completed successfully!")
    print(f"Model saved to: {MODELS_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
