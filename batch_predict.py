"""
Batch Prediction Script for Brain Tumor Detection
Processes multiple images and generates batch analysis reports.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import BrainTumorPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchPredictor:
    """
    Batch processing for brain tumor detection.
    """
    
    def __init__(self, classifier_path="models/brain_tumor_classifier.h5",
                 unet_path="models/unet_segmentation.h5"):
        self.predictor = BrainTumorPredictor(classifier_path, unet_path)
        self.results = []
        self.lock = threading.Lock()
    
    def process_single_image(self, image_path, output_dir):
        """
        Process a single image and return results.
        
        Args:
            image_path: Path to image
            output_dir: Output directory
            
        Returns:
            Processing results
        """
        try:
            results = self.predictor.predict_single_image(image_path, output_dir)
            
            if results:
                # Extract key information for batch summary
                summary = {
                    'filename': Path(image_path).name,
                    'image_path': str(image_path),
                    'predicted_class': results['classification']['predicted_class'],
                    'confidence': results['classification']['confidence'],
                    'analysis_timestamp': results['analysis_timestamp']
                }
                
                # Add size information if available
                if results['size_estimation']:
                    summary.update({
                        'tumor_area_pixels': results['size_estimation']['area_pixels'],
                        'tumor_area_mm2': results['size_estimation']['area_mm2'],
                        'size_category': results['size_estimation']['size_category'],
                        'num_regions': results['size_estimation']['num_regions']
                    })
                else:
                    summary.update({
                        'tumor_area_pixels': 0,
                        'tumor_area_mm2': 0.0,
                        'size_category': 'unknown',
                        'num_regions': 0
                    })
                
                # Thread-safe addition to results
                with self.lock:
                    self.results.append(summary)
                
                return summary
            else:
                logger.error(f"Failed to process: {image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def process_directory(self, input_dir, output_dir="outputs/batch_predictions", 
                         max_workers=4, image_extensions=None):
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_dir: Output directory for results
            max_workers: Number of parallel workers
            image_extensions: List of valid image extensions
            
        Returns:
            Batch processing results
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.error(f"No image files found in: {input_dir}")
            return None
        
        logger.info(f"Found {len(image_files)} images to process")
        logger.info(f"Using {max_workers} parallel workers")
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.process_single_image, img_path, output_dir): img_path
                for img_path in image_files
            }
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    completed += 1
                    
                    if result:
                        logger.info(f"Processed ({completed}/{len(image_files)}): {Path(image_path).name} "
                                  f"-> {result['predicted_class']} ({result['confidence']:.3f})")
                    else:
                        logger.warning(f"Failed ({completed}/{len(image_files)}): {Path(image_path).name}")
                        
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    completed += 1
        
        logger.info(f"Batch processing completed: {len(self.results)}/{len(image_files)} successful")
        
        return self.results
    
    def generate_batch_report(self, output_dir="outputs/batch_predictions"):
        """
        Generate comprehensive batch processing report.
        
        Args:
            output_dir: Directory to save report
        """
        if not self.results:
            logger.warning("No results to generate report from")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save detailed CSV
        csv_path = output_dir / "batch_analysis_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to: {csv_path}")
        
        # Generate summary statistics
        summary_stats = {
            'batch_info': {
                'total_images': len(df),
                'processing_date': datetime.now().isoformat(),
                'successful_predictions': len(df)
            },
            'class_distribution': df['predicted_class'].value_counts().to_dict(),
            'confidence_stats': {
                'mean_confidence': float(df['confidence'].mean()),
                'min_confidence': float(df['confidence'].min()),
                'max_confidence': float(df['confidence'].max()),
                'std_confidence': float(df['confidence'].std())
            }
        }
        
        # Size statistics (if available)
        if 'tumor_area_mm2' in df.columns:
            tumor_data = df[df['tumor_area_mm2'] > 0]
            if len(tumor_data) > 0:
                summary_stats['size_stats'] = {
                    'images_with_tumors': len(tumor_data),
                    'mean_tumor_area_mm2': float(tumor_data['tumor_area_mm2'].mean()),
                    'median_tumor_area_mm2': float(tumor_data['tumor_area_mm2'].median()),
                    'size_category_distribution': tumor_data['size_category'].value_counts().to_dict()
                }
        
        # Save summary JSON
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"Summary statistics saved to: {summary_path}")
        
        # Print summary to console
        self.print_batch_summary(summary_stats)
        
        return summary_stats
    
    def print_batch_summary(self, summary_stats):
        """
        Print batch processing summary to console.
        """
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        
        batch_info = summary_stats['batch_info']
        print(f"Total Images Processed: {batch_info['total_images']}")
        print(f"Successful Predictions: {batch_info['successful_predictions']}")
        print(f"Processing Date: {batch_info['processing_date']}")
        
        print(f"\nCLASS DISTRIBUTION:")
        for class_name, count in summary_stats['class_distribution'].items():
            percentage = (count / batch_info['total_images']) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        conf_stats = summary_stats['confidence_stats']
        print(f"\nCONFIDENCE STATISTICS:")
        print(f"  Mean: {conf_stats['mean_confidence']:.3f}")
        print(f"  Range: {conf_stats['min_confidence']:.3f} - {conf_stats['max_confidence']:.3f}")
        print(f"  Std Dev: {conf_stats['std_confidence']:.3f}")
        
        if 'size_stats' in summary_stats:
            size_stats = summary_stats['size_stats']
            print(f"\nTUMOR SIZE STATISTICS:")
            print(f"  Images with detected tumors: {size_stats['images_with_tumors']}")
            print(f"  Mean tumor area: {size_stats['mean_tumor_area_mm2']:.2f} mm²")
            print(f"  Median tumor area: {size_stats['median_tumor_area_mm2']:.2f} mm²")
            
            print(f"  Size category distribution:")
            for category, count in size_stats['size_category_distribution'].items():
                print(f"    {category}: {count}")
        
        print("="*60)

def main():
    """
    Command-line interface for batch prediction.
    """
    parser = argparse.ArgumentParser(description='Batch Brain Tumor Detection and Analysis')
    parser.add_argument('input_dir', help='Directory containing MRI images')
    parser.add_argument('--output-dir', default='outputs/batch_predictions',
                       help='Directory to save outputs (default: outputs/batch_predictions)')
    parser.add_argument('--classifier-path', default='models/brain_tumor_classifier.h5',
                       help='Path to classifier model')
    parser.add_argument('--unet-path', default='models/unet_segmentation.h5',
                       help='Path to U-Net segmentation model')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
                       help='Image file extensions to process')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Initialize batch predictor
    try:
        batch_predictor = BatchPredictor(
            classifier_path=args.classifier_path,
            unet_path=args.unet_path
        )
    except Exception as e:
        logger.error(f"Error initializing batch predictor: {e}")
        sys.exit(1)
    
    # Process directory
    try:
        results = batch_predictor.process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            image_extensions=args.extensions
        )
        
        if results:
            # Generate batch report
            batch_predictor.generate_batch_report(args.output_dir)
        else:
            logger.error("Batch processing failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
