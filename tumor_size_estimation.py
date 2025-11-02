"""
Tumor Size Estimation Module
Computes tumor area from segmentation masks and converts to real-world measurements.
"""

import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TumorSizeEstimator:
    """
    Estimates tumor size from segmentation masks.
    """
    
    def __init__(self, pixel_to_mm_ratio=0.5):
        """
        Initialize tumor size estimator.
        
        Args:
            pixel_to_mm_ratio: Conversion factor from pixels to mm (configurable)
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        
        # Create output directory
        Path("outputs").mkdir(exist_ok=True)
    
    def calculate_tumor_area_pixels(self, mask):
        """
        Calculate tumor area in pixels from binary mask.
        
        Args:
            mask: Binary segmentation mask (0s and 1s)
            
        Returns:
            Tumor area in pixels
        """
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Count white pixels (tumor region)
        tumor_pixels = np.sum(mask > 0)
        
        return tumor_pixels
    
    def calculate_tumor_area_mm2(self, area_pixels):
        """
        Convert pixel area to mm².
        
        Args:
            area_pixels: Area in pixels
            
        Returns:
            Area in mm²
        """
        area_mm2 = area_pixels * (self.pixel_to_mm_ratio ** 2)
        return area_mm2
    
    def get_tumor_contour_properties(self, mask):
        """
        Extract additional tumor properties from contours.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Dictionary with tumor properties
        """
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        properties = {
            'num_regions': len(contours),
            'largest_area_pixels': 0,
            'perimeter_pixels': 0,
            'circularity': 0,
            'aspect_ratio': 0,
            'solidity': 0
        }
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            properties['largest_area_pixels'] = area
            properties['perimeter_pixels'] = perimeter
            
            # Circularity (4π * area / perimeter²)
            if perimeter > 0:
                properties['circularity'] = 4 * np.pi * area / (perimeter ** 2)
            
            # Bounding rectangle for aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            if h > 0:
                properties['aspect_ratio'] = w / h
            
            # Solidity (contour area / convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                properties['solidity'] = area / hull_area
        
        return properties
    
    def estimate_tumor_sizes(self, masks, filenames, predicted_classes, class_names):
        """
        Estimate tumor sizes for a batch of masks.
        
        Args:
            masks: Array of binary masks
            filenames: Array of corresponding filenames
            predicted_classes: Array of predicted class indices
            class_names: List of class names
            
        Returns:
            DataFrame with size estimations
        """
        results = []
        
        logger.info(f"Estimating tumor sizes for {len(masks)} images...")
        
        for i, (mask, filename, class_idx) in enumerate(zip(masks, filenames, predicted_classes)):
            # Get class name
            predicted_class = class_names[class_idx] if class_idx < len(class_names) else 'unknown'
            
            # Calculate basic area
            area_pixels = self.calculate_tumor_area_pixels(mask)
            area_mm2 = self.calculate_tumor_area_mm2(area_pixels)
            
            # Get additional properties
            properties = self.get_tumor_contour_properties(mask)
            
            # Compile results
            result = {
                'filename': filename,
                'predicted_class': predicted_class,
                'tumor_area_pixels': area_pixels,
                'tumor_area_mm2': round(area_mm2, 2),
                'num_tumor_regions': properties['num_regions'],
                'largest_region_pixels': properties['largest_area_pixels'],
                'perimeter_pixels': properties['perimeter_pixels'],
                'circularity': round(properties['circularity'], 3),
                'aspect_ratio': round(properties['aspect_ratio'], 3),
                'solidity': round(properties['solidity'], 3),
                'pixel_to_mm_ratio': self.pixel_to_mm_ratio
            }
            
            results.append(result)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(masks)} images")
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Add size categories
        results_df['size_category'] = self.categorize_tumor_size(results_df['tumor_area_mm2'])
        
        return results_df
    
    def categorize_tumor_size(self, areas_mm2):
        """
        Categorize tumor sizes into small, medium, large.
        
        Args:
            areas_mm2: Array of tumor areas in mm²
            
        Returns:
            Array of size categories
        """
        categories = []
        
        for area in areas_mm2:
            if area == 0:
                categories.append('no_tumor')
            elif area < 50:
                categories.append('small')
            elif area < 200:
                categories.append('medium')
            else:
                categories.append('large')
        
        return categories
    
    def save_size_estimation_results(self, results_df, output_path="outputs/tumor_size_estimation.csv"):
        """
        Save size estimation results to CSV.
        
        Args:
            results_df: DataFrame with size estimation results
            output_path: Path to save CSV file
        """
        results_df.to_csv(output_path, index=False)
        logger.info(f"Size estimation results saved to: {output_path}")
        
        # Print summary statistics
        logger.info("\nTumor Size Estimation Summary:")
        logger.info(f"Total images processed: {len(results_df)}")
        logger.info(f"Images with detected tumors: {len(results_df[results_df['tumor_area_pixels'] > 0])}")
        
        # Size category distribution
        size_dist = results_df['size_category'].value_counts()
        logger.info("\nSize Category Distribution:")
        for category, count in size_dist.items():
            logger.info(f"  {category}: {count} ({count/len(results_df)*100:.1f}%)")
        
        # Class-wise statistics
        logger.info("\nClass-wise Average Tumor Area (mm²):")
        class_stats = results_df.groupby('predicted_class')['tumor_area_mm2'].agg(['count', 'mean', 'std'])
        for class_name, stats in class_stats.iterrows():
            logger.info(f"  {class_name}: {stats['mean']:.2f} ± {stats['std']:.2f} mm² (n={stats['count']})")
    
    def visualize_size_distribution(self, results_df, save_path="outputs/plots/tumor_size_distribution.png"):
        """
        Create visualization of tumor size distribution.
        
        Args:
            results_df: DataFrame with size estimation results
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Size category distribution
        size_counts = results_df['size_category'].value_counts()
        axes[0, 0].pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Tumor Size Category Distribution')
        
        # Area distribution by class
        tumor_data = results_df[results_df['tumor_area_mm2'] > 0]
        if len(tumor_data) > 0:
            sns.boxplot(data=tumor_data, x='predicted_class', y='tumor_area_mm2', ax=axes[0, 1])
            axes[0, 1].set_title('Tumor Area Distribution by Class')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Histogram of tumor areas
        if len(tumor_data) > 0:
            axes[1, 0].hist(tumor_data['tumor_area_mm2'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Tumor Area (mm²)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Tumor Area Distribution')
        
        # Circularity vs Area scatter plot
        if len(tumor_data) > 0:
            scatter = axes[1, 1].scatter(tumor_data['tumor_area_mm2'], tumor_data['circularity'], 
                                       c=tumor_data['predicted_class'].astype('category').cat.codes, 
                                       alpha=0.6)
            axes[1, 1].set_xlabel('Tumor Area (mm²)')
            axes[1, 1].set_ylabel('Circularity')
            axes[1, 1].set_title('Tumor Circularity vs Area')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Size distribution plots saved to: {save_path}")

def main():
    """
    Example usage of tumor size estimation.
    """
    # This would typically be called after segmentation
    logger.info("Tumor Size Estimation Module")
    logger.info("This module is designed to be used after segmentation.")
    logger.info("Run the complete pipeline with evaluate.py to see it in action.")

if __name__ == "__main__":
    main()
