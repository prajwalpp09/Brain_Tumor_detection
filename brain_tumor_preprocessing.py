"""
Brain Tumor Detection - Data Preprocessing Pipeline
==================================================

This script handles the complete preprocessing pipeline for brain tumor MRI images:
1. Image loading and resizing
2. RGB to grayscale conversion
3. Gaussian blur with 5x5 kernel
4. Binary thresholding at value 45
5. Morphological operations (erosion + dilation)
6. Contour detection and extreme point extraction
7. Brain region cropping

Usage:
    python scripts/brain_tumor_preprocessing.py

Make sure to update the DATASET_PATH in config.py before running.
"""
from tqdm import tqdm
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import logging
from pathlib import Path

from config import (
    DATASET_PATH, DATA_DIR, RESULTS_DIR, LOGS_DIR,
    IMG_SIZE, CLASS_NAMES,
    GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, BINARY_THRESHOLD,
    MORPHOLOGY_ITERATIONS, ADD_PIXELS, validate_dataset_path
)

class BrainTumorPreprocessor:
    def __init__(self):
        """
        Initialize the Brain Tumor Preprocessor using centralized configuration
        """
        self.dataset_path = DATASET_PATH
        self.output_path = DATA_DIR
        self.target_size = IMG_SIZE
        self.add_pixels = ADD_PIXELS
        
        # Validate dataset path
        validate_dataset_path()
        
        # Create output directories
        self.create_output_directories()
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = LOGS_DIR / "preprocessing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_output_directories(self):
        """Create necessary output directories"""
        directories = [
            'processed/original',
            'processed/cropped',
            'processed/preprocessed',
            'processed/intermediate'
        ]
        
        for directory in directories:
            full_path = self.output_path / directory
            full_path.mkdir(parents=True, exist_ok=True)
            
        # Create class subdirectories
        for class_name in CLASS_NAMES:
            class_dir = self.output_path / 'processed' / 'original' / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
                
    def load_and_resize_image(self, image_path):
        """
        Load and resize image to target size
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Resized image or None if failed
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            return resized_image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
            
    def convert_to_grayscale(self, image):
        """Convert RGB image to grayscale"""
        if len(image.shape) == 3:
            # Convert BGR to RGB first (OpenCV loads as BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convert RGB to grayscale
            gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        return gray_image
        
    def apply_gaussian_blur(self, image):
        """Apply Gaussian blur using config parameters"""
        blurred_image = cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
        return blurred_image
        
    def apply_binary_threshold(self, image):
        """Apply binary thresholding using config parameters"""
        _, binary_image = cv2.threshold(image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        return binary_image
        
    def apply_morphological_operations(self, image):
        """Apply morphological operations using config parameters"""
        kernel = np.ones((3, 3), np.uint8)
        
        # Apply erosion
        eroded_image = cv2.erode(image, kernel, iterations=MORPHOLOGY_ITERATIONS)
        
        # Apply dilation
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=MORPHOLOGY_ITERATIONS)
        
        return dilated_image
        
    def find_largest_contour(self, image):
        """Find the largest contour in the image"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
        
    def extract_extreme_points(self, contour):
        """Extract extreme points from contour"""
        if contour is None:
            return None
            
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
        
        return {
            'leftmost': leftmost,
            'rightmost': rightmost,
            'topmost': topmost,
            'bottommost': bottommost
        }
        
    def crop_brain_region(self, original_image, extreme_points):
        """Crop image to brain region using extreme coordinates"""
        if extreme_points is None:
            return original_image
            
        left_x = extreme_points['leftmost'][0]
        right_x = extreme_points['rightmost'][0]
        top_y = extreme_points['topmost'][1]
        bottom_y = extreme_points['bottommost'][1]
        
        # Add padding
        left_x = max(0, left_x - self.add_pixels)
        right_x = min(original_image.shape[1], right_x + self.add_pixels)
        top_y = max(0, top_y - self.add_pixels)
        bottom_y = min(original_image.shape[0], bottom_y + self.add_pixels)
        
        # Crop the image
        cropped_image = original_image[top_y:bottom_y, left_x:right_x]
        
        # Resize back to target size
        if cropped_image.size > 0:
            cropped_image = cv2.resize(cropped_image, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            cropped_image = original_image
            
        return cropped_image
        
    def preprocess_single_image(self, image_path, save_intermediate=False):
        """Complete preprocessing pipeline for a single image"""
        # Step 1: Load and resize image
        image = self.load_and_resize_image(image_path)
        if image is None:
            return None
            
        original_image = image.copy()
        
        # Step 2: Convert to grayscale
        gray_image = self.convert_to_grayscale(image)
        
        # Step 3: Apply Gaussian blur
        blurred_image = self.apply_gaussian_blur(gray_image)
        
        # Step 4: Apply binary thresholding
        binary_image = self.apply_binary_threshold(blurred_image)
        
        # Step 5: Apply morphological operations
        morphed_image = self.apply_morphological_operations(binary_image)
        
        # Step 6: Find largest contour
        largest_contour = self.find_largest_contour(morphed_image)
        
        # Step 7: Extract extreme points
        extreme_points = self.extract_extreme_points(largest_contour)
        
        # Step 8: Crop brain region
        cropped_image = self.crop_brain_region(original_image, extreme_points)
        
        # Save intermediate results if requested
        if save_intermediate:
            base_name = Path(image_path).stem
            self.save_intermediate_results(base_name, {
                'original': original_image,
                'gray': gray_image,
                'blurred': blurred_image,
                'binary': binary_image,
                'morphed': morphed_image,
                'cropped': cropped_image
            })
            
        return cropped_image
        
    def save_intermediate_results(self, base_name, images_dict):
        """Save intermediate processing results"""
        for step_name, image in images_dict.items():
            if image is not None:
                save_path = self.output_path / 'processed' / 'intermediate' / f"{base_name}_{step_name}.jpg"
                cv2.imwrite(str(save_path), image)
                
    def process_dataset(self, save_intermediate=False):
        """Process the entire dataset (without augmentation)"""
        processing_log = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'categories': {}
        }
        
        dataset_path = Path(self.dataset_path)
        
        # Process both Training and Testing directories
        for split in ['Training', 'Testing']:
            split_path = dataset_path / split
            if not split_path.exists():
                self.logger.warning(f"Split directory {split} not found!")
                continue
                
            for category in CLASS_NAMES:
                category_path = split_path / category
                if not category_path.exists():
                    self.logger.warning(f"Category folder {category} not found in {split}!")
                    continue
                    
                image_files = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
                
                category_key = f"{split}_{category}"
                category_stats = {
                    'total': len(image_files),
                    'processed': 0,
                    'failed': 0
                }
                
                self.logger.info(f"Processing {category_key} - {len(image_files)} images")
                
                for image_file in tqdm(image_files, desc=f"Processing {category_key}"):
                    try:
                        # Preprocess the image
                        preprocessed_image = self.preprocess_single_image(image_file, save_intermediate)
                        
                        if preprocessed_image is not None:
                            # Save original preprocessed image
                            base_name = image_file.stem
                            original_save_path = (self.output_path / 'processed' / 'original' / 
                                                  category / f"{base_name}_preprocessed.jpg")
                            cv2.imwrite(str(original_save_path), preprocessed_image)
                            
                            category_stats['processed'] += 1
                            processing_log['successful'] += 1
                        else:
                            category_stats['failed'] += 1
                            processing_log['failed'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error processing {image_file}: {str(e)}")
                        category_stats['failed'] += 1
                        processing_log['failed'] += 1
                        
                    processing_log['total_processed'] += 1
                    
                processing_log['categories'][category_key] = category_stats
                self.logger.info(f"Completed {category_key}: {category_stats['processed']} successful, {category_stats['failed']} failed")
                
        # Save processing log
        log_path = LOGS_DIR / 'preprocessing_log.json'
        with open(log_path, 'w') as f:
            json.dump(processing_log, f, indent=2)
            
        self.logger.info(f"Processing complete!")
        self.logger.info(f"Total processed: {processing_log['total_processed']}")
        self.logger.info(f"Successful: {processing_log['successful']}")
        self.logger.info(f"Failed: {processing_log['failed']}")
        self.logger.info(f"Processing log saved to: {log_path}")
        
    def visualize_preprocessing_steps(self, image_path, save_visualization=True):
        """Visualize the preprocessing steps for a single image"""
        image = self.load_and_resize_image(image_path)
        if image is None:
            self.logger.error("Could not load image for visualization")
            return
            
        original_image = image.copy()
        gray_image = self.convert_to_grayscale(image)
        blurred_image = self.apply_gaussian_blur(gray_image)
        binary_image = self.apply_binary_threshold(blurred_image)
        morphed_image = self.apply_morphological_operations(binary_image)
        
        # Find contour and crop
        largest_contour = self.find_largest_contour(morphed_image)
        extreme_points = self.extract_extreme_points(largest_contour)
        cropped_image = self.crop_brain_region(original_image, extreme_points)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Brain Tumor Preprocessing Pipeline', fontsize=16)
        
        steps = [
            (original_image, 'Original', 'color'),
            (gray_image, 'Grayscale', 'gray'),
            (blurred_image, 'Gaussian Blur', 'gray'),
            (binary_image, 'Binary Threshold', 'gray'),
            (morphed_image, 'Morphological Ops', 'gray'),
            (cropped_image, 'Cropped Brain', 'color'),
        ]
        
        for i, (img, title, cmap) in enumerate(steps):
            row = i // 4
            col = i % 4
            
            if cmap == 'color' and len(img.shape) == 3:
                img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(img_display)
            else:
                axes[row, col].imshow(img, cmap='gray')
                
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
            
        # Hide unused subplots
        for i in range(len(steps), 8):
            row = i // 4
            col = i % 4
            axes[row, col].axis('off')
            
        plt.tight_layout()
        
        if save_visualization:
            base_name = Path(image_path).stem
            viz_path = RESULTS_DIR / f'{base_name}_preprocessing_steps.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to: {viz_path}")
            
        plt.show()

def main():
    """Main function to run the preprocessing pipeline"""
    print("Brain Tumor Detection - Data Preprocessing Pipeline")
    print("=" * 60)
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Target Size: {IMG_SIZE}")
    print()
    
    # Initialize preprocessor
    preprocessor = BrainTumorPreprocessor()
    
    # Find a sample image for visualization
    dataset_path = Path(DATASET_PATH)
    sample_image = None
    
    for split in ['Training', 'Testing']:
        for category in CLASS_NAMES:
            category_path = dataset_path / split / category
            if category_path.exists():
                images = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
                if images:
                    sample_image = images[0]
                    break
        if sample_image:
            break
    
    # Visualize preprocessing steps
    if sample_image:
        print(f"1. Visualizing preprocessing steps using: {sample_image}")
        preprocessor.visualize_preprocessing_steps(sample_image)
    else:
        print("No sample image found for visualization")
    
    # Process the entire dataset
    print("\n2. Processing entire dataset...")
    user_input = input("Do you want to process the entire dataset? (y/n): ")
    
    if user_input.lower() == 'y':
        save_intermediate = input("Save intermediate processing steps? (y/n): ").lower() == 'y'
        preprocessor.process_dataset(save_intermediate=save_intermediate)
    else:
        print("Dataset processing skipped.")
    
    print("\nPreprocessing pipeline completed!")

if __name__ == "__main__":
    main()
