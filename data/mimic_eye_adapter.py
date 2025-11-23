import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def prepare_mimic_eye_metadata(data_root, output_dir='data/processed_mimic_eye'):
    """
    Convert MIMIC-Eye EyeGaze dataset into format expected by our dataset class.
    
    Structure:
    - Reads from patient_*/EyeGaze/fixations.csv
    - Matches with patient_*/CXR-JPG/s*/DICOM_ID.jpg
    - Creates metadata.json with train/val/test splits
    """
    print("="*60)
    print("Processing MIMIC-Eye EyeGaze Dataset")
    print("="*60)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gaze'), exist_ok=True)
    
    # Get all patient directories
    patient_dirs = sorted([d for d in os.listdir(data_root) 
                          if d.startswith('patient_')])
    
    print(f"\nFound {len(patient_dirs)} patient directories")
    print("Scanning for EyeGaze data...\n")
    
    metadata = {}
    processed_count = 0
    skipped_count = 0
    
    # Process each patient
    for patient_id in tqdm(patient_dirs, desc="Processing patients"):
        patient_path = os.path.join(data_root, patient_id)
        eyegaze_path = os.path.join(patient_path, 'EyeGaze')
        
        # Check if patient has EyeGaze data
        if not os.path.exists(eyegaze_path):
            continue
        
        fixations_csv = os.path.join(eyegaze_path, 'fixations.csv')
        if not os.path.exists(fixations_csv):
            continue
        
        # Load fixations
        try:
            fix_df = pd.read_csv(fixations_csv)
        except Exception as e:
            print(f"Error reading {patient_id}/EyeGaze/fixations.csv: {e}")
            skipped_count += 1
            continue
        
        # Process each unique DICOM_ID in this patient's fixations
        for dicom_id in fix_df['DICOM_ID'].unique():
            # Find the corresponding image
            image_path = find_image_for_dicom(patient_path, dicom_id)
            
            if image_path is None:
                skipped_count += 1
                continue
            
            # Extract fixations for this image
            image_fixations = fix_df[fix_df['DICOM_ID'] == dicom_id]
            fixations_list = process_fixations(image_fixations)
            
            if len(fixations_list) == 0:
                skipped_count += 1
                continue
            
            # Create sample ID
            sample_id = f"{patient_id}_{dicom_id}"
            
            # Save gaze JSON
            gaze_output_path = os.path.join(output_dir, 'gaze', f"{sample_id}.json")
            with open(gaze_output_path, 'w') as f:
                json.dump({'fixations': fixations_list}, f)
            
            # Copy image
            import shutil
            image_output_path = os.path.join(output_dir, 'images', f"{sample_id}.jpg")
            if not os.path.exists(image_output_path):
                shutil.copy(image_path, image_output_path)
            
            # Assign split (80/10/10 split)
            rand = np.random.random()
            if rand < 0.8:
                split = 'train'
            elif rand < 0.9:
                split = 'val'
            else:
                split = 'test'
            
            # For now, binary classification: normal vs abnormal
            # You can refine this based on master_sheet labels
            label = 0  # Default to normal, update based on your needs
            
            metadata[sample_id] = {
                'split': split,
                'label': int(label),
                'patient_id': patient_id,
                'dicom_id': dicom_id,
                'num_fixations': len(fixations_list)
            }
            
            processed_count += 1
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Successfully processed: {processed_count} images")
    print(f"✗ Skipped: {skipped_count} images")
    print(f"\n✓ Saved to: {output_dir}")
    print(f"\nSplit distribution:")
    print(f"  - Train: {sum(1 for v in metadata.values() if v['split'] == 'train')}")
    print(f"  - Val:   {sum(1 for v in metadata.values() if v['split'] == 'val')}")
    print(f"  - Test:  {sum(1 for v in metadata.values() if v['split'] == 'test')}")
    
    return output_dir


def find_image_for_dicom(patient_path, dicom_id):
    """
    Find the JPG image file for a given DICOM_ID.
    
    Structure: patient_path/CXR-JPG/s{study_id}/{dicom_id}.jpg
    """
    cxr_path = os.path.join(patient_path, 'CXR-JPG')
    
    if not os.path.exists(cxr_path):
        return None
    
    # Search through study directories
    for item in os.listdir(cxr_path):
        if not item.startswith('s'):
            continue
        
        study_path = os.path.join(cxr_path, item)
        if not os.path.isdir(study_path):
            continue
        
        # Look for the image file
        image_file = os.path.join(study_path, f"{dicom_id}.jpg")
        if os.path.exists(image_file):
            return image_file
    
    return None


def process_fixations(fixations_df):
    """
    Convert fixation DataFrame to our expected format.
    
    Uses FPOGX, FPOGY (already normalized 0-1) and FPOGD (duration).
    """
    fixations = []
    
    for idx, row in fixations_df.iterrows():
        # Get normalized coordinates (already in 0-1 range)
        x = row['FPOGX']
        y = row['FPOGY']
        duration = row['FPOGD']
        
        # Skip invalid fixations
        if pd.isna(x) or pd.isna(y) or pd.isna(duration):
            continue
        
        # Ensure values are in valid range
        x = float(np.clip(x, 0, 1))
        y = float(np.clip(y, 0, 1))
        duration = float(max(duration, 0))  # Duration should be positive
        
        fixations.append({
            'x': x,
            'y': y,
            'duration': duration * 1000  # Convert to milliseconds if needed
        })
    
    return fixations


if __name__ == '__main__':
    # Update this path to your MIMIC-Eye location
    data_root = '/media/16TB_Storage/kavin/dataset/physionet.org/files/mimic-eye-multimodal-datasets/1.0.0/mimic-eye'

    original_output_dir = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye'
    
    output_dir = prepare_mimic_eye_metadata(data_root, original_output_dir)
    
    print(f"\n{'='*60}")
    print("Dataset ready!")
    print(f"  root_dir='{output_dir}'")