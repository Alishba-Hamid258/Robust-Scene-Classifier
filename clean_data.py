import os
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_image(file_path):
    """Verify if the image is valid and can be opened."""
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify image integrity
        return True
    except Exception as e:
        logger.error(f"Invalid image {file_path}: {str(e)}")
        return False

def get_image_info(file_path):
    """Get image format and size information."""
    try:
        with Image.open(file_path) as img:
            return img.format, img.size
    except:
        return None, None

def clean_dataset(input_path, output_path, valid_extensions=('.jpg', '.jpeg', '.png')):
    """Clean the dataset by verifying images and creating a new cleaned structure."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize report DataFrame
    report_data = []
    
    # Define dataset splits
    splits = ['seg_train/seg_train', 'seg_test/seg_test', 'seg_pred/seg_pred']
    
    for split in splits:
        split_path = input_path / split
        if not split_path.exists():
            logger.warning(f"Directory {split_path} does not exist")
            continue
            
        output_split_path = output_path / split
        output_split_path.mkdir(parents=True, exist_ok=True)
        
        # Get all subdirectories (classes) for train and test
        subdirs = [d for d in split_path.iterdir() if d.is_dir()] if 'pred' not in split else [split_path]
        
        for subdir in tqdm(subdirs, desc=f"Processing {split}"):
            if 'pred' in split:
                output_subdir = output_split_path
            else:
                output_subdir = output_path / split / subdir.name
                output_subdir.mkdir(exist_ok=True)
            
            # Process each image
            for img_path in subdir.glob('*'):
                if img_path.suffix.lower() not in valid_extensions:
                    logger.warning(f"Skipping invalid extension: {img_path}")
                    report_data.append({
                        'file': str(img_path),
                        'status': 'skipped',
                        'reason': 'invalid_extension',
                        'split': split,
                        'class': subdir.name if 'pred' not in split else 'pred'
                    })
                    continue
                
                if verify_image(img_path):
                    # Copy valid image to output directory
                    output_img_path = output_subdir / img_path.name
                    shutil.copy2(img_path, output_img_path)
                    
                    # Get image info
                    img_format, img_size = get_image_info(img_path)
                    
                    report_data.append({
                        'file': str(img_path),
                        'status': 'valid',
                        'reason': None,
                        'split': split,
                        'class': subdir.name if 'pred' not in split else 'pred',
                        'format': img_format,
                        'size': img_size
                    })
                else:
                    report_data.append({
                        'file': str(img_path),
                        'status': 'invalid',
                        'reason': 'corrupt_image',
                        'split': split,
                        'class': subdir.name if 'pred' not in split else 'pred'
                    })
    
    # Create report
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_path / 'cleaning_report.csv', index=False)
    
    # Generate summary
    summary = {
        'total_images': len(report_data),
        'valid_images': len(report_df[report_df['status'] == 'valid']),
        'invalid_images': len(report_df[report_df['status'] == 'invalid']),
        'skipped_images': len(report_df[report_df['status'] == 'skipped'])
    }
    
    logger.info("Cleaning Summary:")
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    
    # Generate class distribution for train and test
    if 'seg_train/seg_train' in report_df['split'].values:
        train_dist = report_df[report_df['split'] == 'seg_train/seg_train'].groupby('class').size()
        logger.info("\nTrain set class distribution:")
        logger.info(train_dist)
    
    if 'seg_test/seg_test' in report_df['split'].values:
        test_dist = report_df[report_df['split'] == 'seg_test/seg_test'].groupby('class').size()
        logger.info("\nTest set class distribution:")
        logger.info(test_dist)
    
    return summary

from config import BASE_LOCAL_PATH, DATA_DIR

def main():
    # Define paths from config
    input_dataset_path = BASE_LOCAL_PATH # Or wherever the user moved the raw data
    output_dataset_path = os.path.join(BASE_LOCAL_PATH, "cleaned_dataset")
    
    logger.info("Starting dataset cleaning process...")
    # ... (rest of the logic)
    summary = clean_dataset(input_dataset_path, output_dataset_path)
    logger.info("Dataset cleaning completed!")
    
    # Save summary to file
    with open(os.path.join(output_dataset_path, 'cleaning_summary.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main()