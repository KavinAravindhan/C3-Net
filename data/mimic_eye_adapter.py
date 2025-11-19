import os
import pandas as pd
import json
import numpy as np
from pathlib import Path


def prepare_mimic_eye_metadata(data_root, output_dir='data/processed_mimic_eye'):
    """
    Convert MIMIC-Eye dataset into the format expected by our MIMICEyeDataset class.
    
    Creates:
    - metadata.json with train/val/test splits
    - Organized gaze JSON files
    """
    print("Processing MIMIC-Eye dataset...")
    
    # Paths
    master_sheet_path = os.path.join(data_root, 'files', 'master_sheet.csv')
    fixations_dir = os.path.join(data_root, 'files', 'fixations')
    images_dir = os.path.join(data_root, 'files', 'images')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gaze'), exist_ok=True)
    
    # Load master sheet
    print(f"Loading master sheet from {master_sheet_path}")
    df = pd.read_csv(master_sheet_path)
    
    print(f"Found {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create metadata
    metadata = {}
    
    # Process each sample
    for idx, row in df.iterrows():
        # Extract IDs (adjust column names based on actual MIMIC-Eye structure)
        subject_id = row.get('subject_id', row.get('patient_id', f'p{idx}'))
        study_id = row.get('study_id', row.get('dicom_id', f's{idx}'))
        
        sample_id = f"{subject_id}_{study_id}"
        
        # Find image path
        image_path = os.path.join(images_dir, f"{subject_id}", f"{study_id}.jpg")
        if not os.path.exists(image_path):
            # Try alternative structure
            image_path = os.path.join(images_dir, f"{sample_id}.jpg")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {sample_id}, skipping")
            continue
        
        # Find fixation data
        fixation_path = os.path.join(fixations_dir, f"{sample_id}.csv")
        if not os.path.exists(fixation_path):
            print(f"Warning: Fixation data not found for {sample_id}, skipping")
            continue
        
        # Read fixations
        try:
            fixations_df = pd.read_csv(fixation_path)
            fixations = process_fixations(fixations_df)
        except Exception as e:
            print(f"Error processing fixations for {sample_id}: {e}")
            continue
        
        # Save gaze JSON
        gaze_output_path = os.path.join(output_dir, 'gaze', f"{sample_id}.json")
        with open(gaze_output_path, 'w') as f:
            json.dump({'fixations': fixations}, f)
        
        # Copy image (or create symlink)
        import shutil
        image_output_path = os.path.join(output_dir, 'images', f"{sample_id}.jpg")
        if not os.path.exists(image_output_path):
            shutil.copy(image_path, image_output_path)
        
        # Determine label (MIMIC-Eye is for abnormality detection)
        # Adjust based on your specific task
        label = row.get('label', 0)  # Default to 0 if no label
        
        # Assign split (80/10/10)
        rand = np.random.random()
        if rand < 0.8:
            split = 'train'
        elif rand < 0.9:
            split = 'val'
        else:
            split = 'test'
        
        metadata[sample_id] = {
            'split': split,
            'label': int(label),
            'subject_id': str(subject_id),
            'study_id': str(study_id)
        }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Processed {len(metadata)} samples")
    print(f"✓ Saved to {output_dir}")
    print(f"  - Train: {sum(1 for v in metadata.values() if v['split'] == 'train')}")
    print(f"  - Val: {sum(1 for v in metadata.values() if v['split'] == 'val')}")
    print(f"  - Test: {sum(1 for v in metadata.values() if v['split'] == 'test')}")
    
    return output_dir


def process_fixations(fixations_df):
    """
    Convert fixation DataFrame to our expected format.
    
    Expected columns in MIMIC-Eye CSV: x, y, duration (or similar)
    """
    fixations = []
    
    # Get image dimensions (MIMIC-Eye images are typically 1024x1024 or similar)
    # Adjust these based on actual image dimensions
    img_width = 1024
    img_height = 1024
    
    for idx, row in fixations_df.iterrows():
        # Extract coordinates (adjust column names as needed)
        x = row.get('x', row.get('x_position', row.get('gaze_x', 0)))
        y = row.get('y', row.get('y_position', row.get('gaze_y', 0)))
        duration = row.get('duration', row.get('fixation_duration', 200))
        
        # Normalize coordinates to [0, 1]
        x_norm = float(x) / img_width
        y_norm = float(y) / img_height
        
        # Clip to valid range
        x_norm = np.clip(x_norm, 0, 1)
        y_norm = np.clip(y_norm, 0, 1)
        
        fixations.append({
            'x': float(x_norm),
            'y': float(y_norm),
            'duration': float(duration)
        })
    
    return fixations


if __name__ == '__main__':
    # Update this path to your MIMIC-Eye download location
    data_root = 'data/mimic_eye/physionet.org/files/mimic-eye/1.0.0'
    
    output_dir = prepare_mimic_eye_metadata(data_root)
    print(f"\nDataset ready at: {output_dir}")
    print("\nUpdate your test_preprocessing.py to use this path:")
    print(f"  root_dir='{output_dir}'")