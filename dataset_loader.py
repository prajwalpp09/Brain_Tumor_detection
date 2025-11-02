"""
Brain Tumor Dataset Loader
==========================

Handles loading preprocessed images, creating train/val/test splits, and generating CSV manifest.
Integrates with the centralized configuration system.

Usage:
    python scripts/dataset_loader.py

The dataset loader will automatically use paths and parameters from config.py.
"""

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import logging
from pathlib import Path

from config import (
    DATA_DIR, LOGS_DIR, IMG_SIZE, CLASS_NAMES, NUM_CLASSES,
    VALIDATION_SPLIT, TEST_SPLIT, RANDOM_SEED
)

class BrainTumorDatasetLoader:
    """
    Dataset loader for brain tumor classification.
    Handles only original preprocessed images (augmentation removed).
    """
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.img_size = IMG_SIZE
        self.classes = CLASS_NAMES
        self.num_classes = NUM_CLASSES
        self.validation_split = VALIDATION_SPLIT
        self.test_split = TEST_SPLIT
        self.random_seed = RANDOM_SEED
        
        # Setup label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        # Setup logging
        self.setup_logging()
        
        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "manifests").mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = LOGS_DIR / "dataset_loader.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_images_from_folder(self, folder_path, use_augmented=True):
        """
        Load images from a folder structure.
        
        Args:
            folder_path: Path to folder containing class subdirectories
            use_augmented: Ignored (augmentation removed)
            
        Returns:
            images: List of image arrays
            labels: List of corresponding labels
            filenames: List of filenames
        """
        images = []
        labels = []
        filenames = []
        
        folder_path = Path(folder_path)
        
        for class_name in self.classes:
            class_folder = folder_path / class_name
            if not class_folder.exists():
                self.logger.warning(f"Class folder {class_folder} not found, skipping...")
                continue
                
            self.logger.info(f"Loading images from {class_folder}")
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(class_folder.glob(f"*{ext}"))
                image_files.extend(class_folder.glob(f"*{ext.upper()}"))
            
            self.logger.info(f"Found {len(image_files)} images in {class_name}")
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        self.logger.warning(f"Could not load image: {img_path}")
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target size
                    img = cv2.resize(img, self.img_size)
                    
                    # Normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_name)
                    filenames.append(img_path.name)
                    
                except Exception as e:
                    self.logger.error(f"Error loading image {img_path}: {e}")
                    continue
        
        self.logger.info(f"Total images loaded: {len(images)}")
        return np.array(images), np.array(labels), np.array(filenames)
    
    def create_train_val_test_split(self, images, labels, filenames):
        """
        Create train/validation/test splits using config parameters.
        
        Args:
            images: Array of images
            labels: Array of labels
            filenames: Array of filenames
            
        Returns:
            Dictionary containing split data
        """
        train_ratio = 1.0 - self.validation_split - self.test_split
        val_ratio = self.validation_split
        test_ratio = self.test_split
        
        # Encode labels
        encoded_labels = self.label_encoder.transform(labels)
        categorical_labels = to_categorical(encoded_labels, num_classes=self.num_classes)
        
        # First split: separate train from temp (val + test)
        temp_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp, files_train, files_temp = train_test_split(
            images, categorical_labels, filenames,
            test_size=temp_ratio,
            random_state=self.random_seed,
            stratify=encoded_labels
        )
        
        # Second split: separate val from test
        if temp_ratio > 0:
            val_ratio_adjusted = val_ratio / temp_ratio
            X_val, X_test, y_val, y_test, files_val, files_test = train_test_split(
                X_temp, y_temp, files_temp,
                test_size=(1 - val_ratio_adjusted),
                random_state=self.random_seed,
                stratify=np.argmax(y_temp, axis=1)
            )
        else:
            # No validation or test split
            X_val = X_test = np.array([])
            y_val = y_test = np.array([])
            files_val = files_test = np.array([])
        
        # Log split information
        self.logger.info(f"Dataset split completed:")
        self.logger.info(f"  Training: {len(X_train)} samples ({len(X_train)/len(images)*100:.1f}%)")
        self.logger.info(f"  Validation: {len(X_val)} samples ({len(X_val)/len(images)*100:.1f}%)")
        self.logger.info(f"  Test: {len(X_test)} samples ({len(X_test)/len(images)*100:.1f}%)")
        
        return {
            'X_train': X_train, 'y_train': y_train, 'files_train': files_train,
            'X_val': X_val, 'y_val': y_val, 'files_val': files_val,
            'X_test': X_test, 'y_test': y_test, 'files_test': files_test
        }
    
    def save_manifest_csv(self, split_data, manifest_path=None):
        """
        Save dataset manifest as CSV file.
        
        Args:
            split_data: Dictionary containing split data
            manifest_path: Path to save manifest CSV
        """
        if manifest_path is None:
            manifest_path = self.data_dir / "manifests" / "dataset_manifest.csv"
        
        manifest_data = []
        
        # Process each split
        for split_name in ['train', 'val', 'test']:
            files = split_data[f'files_{split_name}']
            labels = split_data[f'y_{split_name}']
            
            if len(files) == 0:
                continue
            
            for filename, label_vec in zip(files, labels):
                # Convert one-hot back to class name
                class_idx = np.argmax(label_vec)
                class_name = self.classes[class_idx]
                
                manifest_data.append({
                    'filename': filename,
                    'class': class_name,
                    'class_idx': class_idx,
                    'split': split_name
                })
        
        # Create DataFrame and save
        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_csv(manifest_path, index=False)
        
        self.logger.info(f"Dataset manifest saved to: {manifest_path}")
        
        # Print summary statistics
        if not manifest_df.empty:
            self.logger.info("\nDataset Summary:")
            summary = manifest_df.groupby(['split', 'class']).size().unstack(fill_value=0)
            self.logger.info(f"\n{summary}")
        
        return manifest_df
    
    def load_dataset(self, use_augmented=True, save_manifest=True):
        """
        Complete dataset loading pipeline using config paths.
        
        Args:
            use_augmented: Ignored (augmentation removed)
            save_manifest: Whether to save CSV manifest
            
        Returns:
            Dictionary containing split data and manifest
        """
        original_folder = self.data_dir / "processed" / "original"
        # augmented_folder = self.data_dir / "processed" / "augmented"
        
        all_images = []
        all_labels = []
        all_filenames = []
        
        # Load original images
        if original_folder.exists():
            self.logger.info("Loading original preprocessed images...")
            orig_images, orig_labels, orig_files = self.load_images_from_folder(
                original_folder, use_augmented=False
            )
            all_images.extend(orig_images)
            all_labels.extend(orig_labels)
            all_filenames.extend(orig_files)
        else:
            self.logger.warning(f"Original images folder not found: {original_folder}")
        
        # Do not attempt to load augmented images
        # if use_augmented and augmented_folder.exists():
        #     self.logger.info("Loading augmented preprocessed images...")
        #     aug_images, aug_labels, aug_files = self.load_images_from_folder(
        #         augmented_folder, use_augmented=True
        #     )
        #     all_images.extend(aug_images)
        #     all_labels.extend(aug_labels)
        #     all_filenames.extend(aug_files)
        # elif use_augmented:
        #     self.logger.warning(f"Augmented images folder not found: {augmented_folder}")
        
        if len(all_images) == 0:
            raise ValueError(
                "No images found! Please run the preprocessing script first:\n"
                "python scripts/brain_tumor_preprocessing.py"
            )
        
        # Convert to numpy arrays
        all_images = np.array(all_images)
        all_labels = np.array(all_labels)
        all_filenames = np.array(all_filenames)
        
        # Create splits
        self.logger.info("Creating train/validation/test splits...")
        split_data = self.create_train_val_test_split(
            all_images, all_labels, all_filenames
        )
        
        # Save manifest
        manifest_df = None
        if save_manifest:
            manifest_df = self.save_manifest_csv(split_data)
        
        splits_dir = self.data_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        for split_name in ['train', 'val', 'test']:
            if len(split_data[f'X_{split_name}']) > 0:
                np.save(splits_dir / f"X_{split_name}.npy", split_data[f'X_{split_name}'])
                np.save(splits_dir / f"y_{split_name}.npy", split_data[f'y_{split_name}'])
                np.save(splits_dir / f"files_{split_name}.npy", split_data[f'files_{split_name}'])
        
        self.logger.info(f"Split data saved to: {splits_dir}")
        
        return {
            'split_data': split_data,
            'manifest': manifest_df,
            'classes': self.classes,
            'label_encoder': self.label_encoder
        }
    
    def load_saved_splits(self):
        """
        Load previously saved split data from numpy files.
        
        Returns:
            Dictionary containing split data
        """
        splits_dir = self.data_dir / "splits"
        
        if not splits_dir.exists():
            raise FileNotFoundError(
                f"No saved splits found at {splits_dir}. "
                "Please run load_dataset() first."
            )
        
        split_data = {}
        
        for split_name in ['train', 'val', 'test']:
            x_file = splits_dir / f"X_{split_name}.npy"
            y_file = splits_dir / f"y_{split_name}.npy"
            files_file = splits_dir / f"files_{split_name}.npy"
            
            if x_file.exists() and y_file.exists() and files_file.exists():
                split_data[f'X_{split_name}'] = np.load(x_file)
                split_data[f'y_{split_name}'] = np.load(y_file)
                split_data[f'files_{split_name}'] = np.load(files_file)
                self.logger.info(f"Loaded {split_name} split: {len(split_data[f'X_{split_name}'])} samples")
            else:
                # Create empty arrays for missing splits
                split_data[f'X_{split_name}'] = np.array([])
                split_data[f'y_{split_name}'] = np.array([])
                split_data[f'files_{split_name}'] = np.array([])
        
        return split_data

def main():
    """
    Example usage of the dataset loader.
    """
    print("Brain Tumor Dataset Loader")
    print("=" * 30)
    
    # Initialize loader
    loader = BrainTumorDatasetLoader()
    
    # Load dataset
    try:
        dataset = loader.load_dataset(
            use_augmented=False,
            save_manifest=True
        )
        
        print("Dataset loading completed successfully!")
        print(f"Classes: {dataset['classes']}")
        
        # Print shapes
        split_data = dataset['split_data']
        print(f"Training set shape: {split_data['X_train'].shape}")
        print(f"Validation set shape: {split_data['X_val'].shape}")
        print(f"Test set shape: {split_data['X_test'].shape}")
        
        # Test loading saved splits
        print("\nTesting saved splits loading...")
        saved_splits = loader.load_saved_splits()
        print("Saved splits loaded successfully!")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease ensure you have run the preprocessing script first:")
        print("python scripts/brain_tumor_preprocessing.py")
        print("\nExpected folder structure:")
        print("  data/processed/original/glioma_tumor/")
        print("  data/processed/original/meningioma_tumor/")
        print("  data/processed/original/no_tumor/")
        print("  data/processed/original/pituitary_tumor/")

if __name__ == "__main__":
    main()
