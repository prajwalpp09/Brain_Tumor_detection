"""
Brain Tumor Detection System Configuration
==========================================

Central configuration file for all project settings.
Update the DATASET_PATH variable below to point to your downloaded Kaggle dataset.
"""

import os
from pathlib import Path


DATASET_PATH = "/Users/prajwalpatil/Desktop/archive (1) 2"  


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# Classification model parameters
CLASSIFIER_EPOCHS = 50
CLASSIFIER_LEARNING_RATE = 0.001
CLASSIFIER_PATIENCE = 10

# Segmentation model parameters
UNET_EPOCHS = 30
UNET_LEARNING_RATE = 0.0001
UNET_PATIENCE = 8

# Preprocessing parameters
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 0
BINARY_THRESHOLD = 45
MORPHOLOGY_ITERATIONS = 2
ADD_PIXELS = 0


CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
NUM_CLASSES = len(CLASS_NAMES)


def validate_dataset_path():
    """Validate that the dataset path exists and has the correct structure."""
    dataset_path = Path(DATASET_PATH)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {DATASET_PATH}\n"
            f"Please download the dataset from: "
            f"https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri\n"
            f"And update the DATASET_PATH in config.py"
        )
    
    # Check for Training and Testing directories
    training_dir = dataset_path / "Training"
    testing_dir = dataset_path / "Testing"
    
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    
    if not testing_dir.exists():
        raise FileNotFoundError(f"Testing directory not found: {testing_dir}")
    
    # Check for class subdirectories
    for class_name in CLASS_NAMES:
        train_class_dir = training_dir / class_name
        test_class_dir = testing_dir / class_name
        
        if not train_class_dir.exists():
            raise FileNotFoundError(f"Training class directory not found: {train_class_dir}")
        
        if not test_class_dir.exists():
            raise FileNotFoundError(f"Testing class directory not found: {test_class_dir}")
    
    print(f"✓ Dataset validation successful: {DATASET_PATH}")
    return True

def get_dataset_info():
    """Get information about the dataset."""
    validate_dataset_path()
    
    dataset_path = Path(DATASET_PATH)
    info = {
        'training_samples': {},
        'testing_samples': {},
        'total_samples': 0
    }
    
    for class_name in CLASS_NAMES:
        train_dir = dataset_path / "Training" / class_name
        test_dir = dataset_path / "Testing" / class_name
        
        train_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
        test_count = len(list(test_dir.glob("*.jpg"))) + len(list(test_dir.glob("*.png")))
        
        info['training_samples'][class_name] = train_count
        info['testing_samples'][class_name] = test_count
        info['total_samples'] += train_count + test_count
    
    return info

if __name__ == "__main__":
    print("Brain Tumor Detection System Configuration")
    print("=" * 50)
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Project Root: {PROJECT_ROOT}")
    print()
    
    try:
        info = get_dataset_info()
        print("Dataset Information:")
        print(f"Total samples: {info['total_samples']}")
        print("\nTraining samples:")
        for class_name, count in info['training_samples'].items():
            print(f"  {class_name}: {count}")
        print("\nTesting samples:")
        for class_name, count in info['testing_samples'].items():
            print(f"  {class_name}: {count}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease update the DATASET_PATH in config.py to point to your dataset location.")
