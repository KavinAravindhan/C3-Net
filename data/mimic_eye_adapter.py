import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def prepare_mimic_eye_metadata(data_root, output_dir='data/processed_mimic_eye'):
    """
    Convert MIMIC-Eye EyeGaze dataset into format expected by our dataset class.
    
    NOW INCLUDES:
    - Images from patient_*/CXR-JPG/s*/DICOM_ID.jpg
    - Gaze from patient_*/EyeGaze/fixations.csv
    - Text from patient_*/CXR-DICOM/s{study_id}.txt
    - Labels from master_sheet.csv
    """

    print("="*60)
    print("Processing MIMIC-Eye: Image + Gaze + Text")
    print("="*60)
    
    # Load master sheet with diagnosis labels
    master_sheet_path = os.path.join(
        data_root, 
        'spreadsheets', 
        'EyeGaze', 
        'master_sheet_with_updated_stayId.csv'
    )
    
    print(f"\nLoading master sheet from:")
    print(f"  {master_sheet_path}")
    
    if not os.path.exists(master_sheet_path):
        raise FileNotFoundError(f"Master sheet not found: {master_sheet_path}")
    
    master_df = pd.read_csv(master_sheet_path)
    print(f"✓ Loaded {len(master_df)} records from master sheet")
    
    # Create DICOM_ID to label mapping
    dicom_to_label = create_label_mapping(master_df)
    print(f"✓ Created label mapping for {len(dicom_to_label)} images")
    
    # Print label distribution
    label_counts = pd.Series(list(dicom_to_label.values())).value_counts()
    print(f"\nLabel distribution:")
    print(f"  Normal (0):   {label_counts.get(0, 0)}")
    print(f"  Abnormal (1): {label_counts.get(1, 0)}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gaze'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)  # NEW
    
    # Get all patient directories
    patient_dirs = sorted([d for d in os.listdir(data_root) 
                          if d.startswith('patient_')])
    
    print(f"\nFound {len(patient_dirs)} patient directories")
    print("Processing patients with gaze data...\n")
    
    metadata = {}
    processed_count = 0
    skipped_no_label = 0
    skipped_no_image = 0
    skipped_no_fixations = 0
    skipped_no_text = 0
    
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
            continue
        
        # Process each unique DICOM_ID
        for dicom_id in fix_df['DICOM_ID'].unique():
            # Check if we have a label for this image
            if dicom_id not in dicom_to_label:
                skipped_no_label += 1
                continue
            
            # Find the corresponding image
            image_path = find_image_for_dicom(patient_path, dicom_id)
            
            if image_path is None:
                skipped_no_image += 1
                continue
            
            # Extract fixations for this image
            image_fixations = fix_df[fix_df['DICOM_ID'] == dicom_id]
            fixations_list = process_fixations(image_fixations)
            
            if len(fixations_list) == 0:
                skipped_no_fixations += 1
                continue
            
            # Extract radiology report (NEW)
            report_text, report_path = extract_report(patient_path, dicom_id)
            
            if report_text is None:
                skipped_no_text += 1
                # Optional: continue or proceed without text
                # For now, we'll skip samples without text to ensure full multimodal
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
            
            # Save cleaned report text (NEW)
            text_output_path = os.path.join(output_dir, 'text', f"{sample_id}.txt")
            with open(text_output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            # Get label from mapping
            label = dicom_to_label[dicom_id]
            
            # Assign split (80/10/10 split)
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
                'patient_id': patient_id,
                'dicom_id': dicom_id,
                'num_fixations': len(fixations_list),
                'has_text': True,  # NEW
                'text_length': len(report_text)  # NEW
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
    print(f"\nSkipped:")
    print(f"  - No label in master sheet: {skipped_no_label}")
    print(f"  - Image not found: {skipped_no_image}")
    print(f"  - No valid fixations: {skipped_no_fixations}")
    print(f"  - No radiology report: {skipped_no_text}")
    
    print(f"\n✓ Saved to: {output_dir}")
    print(f"  - Images: {output_dir}/images/")
    print(f"  - Gaze: {output_dir}/gaze/")
    print(f"  - Text: {output_dir}/text/")
    
    print(f"\nFinal split distribution:")
    
    train_labels = [v['label'] for v in metadata.values() if v['split'] == 'train']
    val_labels = [v['label'] for v in metadata.values() if v['split'] == 'val']
    test_labels = [v['label'] for v in metadata.values() if v['split'] == 'test']
    
    print(f"\nTrain: {len(train_labels)} samples")
    print(f"  - Normal (0):   {train_labels.count(0)}")
    print(f"  - Abnormal (1): {train_labels.count(1)}")
    
    print(f"\nVal: {len(val_labels)} samples")
    print(f"  - Normal (0):   {val_labels.count(0)}")
    print(f"  - Abnormal (1): {val_labels.count(1)}")
    
    print(f"\nTest: {len(test_labels)} samples")
    print(f"  - Normal (0):   {test_labels.count(0)}")
    print(f"  - Abnormal (1): {test_labels.count(1)}")
    
    # Print text statistics
    text_lengths = [v['text_length'] for v in metadata.values()]
    if text_lengths:
        print(f"\nText Statistics:")
        print(f"  - Mean length: {np.mean(text_lengths):.0f} characters")
        print(f"  - Median length: {np.median(text_lengths):.0f} characters")
        print(f"  - Min length: {np.min(text_lengths)} characters")
        print(f"  - Max length: {np.max(text_lengths)} characters")
    
    return output_dir


def extract_report(patient_path, dicom_id):
    """
    Extract and clean radiology report for a given DICOM_ID.
    
    Returns:
        cleaned_text: Cleaned report text (or None if not found)
        report_path: Path to original report file (or None)
    """
    # Find study_id from cxr_meta.csv
    meta_csv_path = os.path.join(patient_path, 'CXR-JPG', 'cxr_meta.csv')
    
    if not os.path.exists(meta_csv_path):
        return None, None
    
    try:
        meta_df = pd.read_csv(meta_csv_path)
        row = meta_df[meta_df['dicom_id'] == dicom_id]
        
        if len(row) == 0:
            return None, None
        
        study_id = row['study_id'].values[0]
        
        # Find report file
        report_path = os.path.join(
            patient_path,
            'CXR-DICOM',
            f's{study_id}.txt'
        )
        
        if not os.path.exists(report_path):
            return None, None
        
        # Read and clean report
        with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
        
        cleaned_text = clean_report_text(raw_text)
        
        return cleaned_text, report_path
        
    except Exception as e:
        return None, None


def clean_report_text(text):
    """
    Clean radiology report text.
    
    - Remove excessive whitespace
    - Normalize line endings
    - Keep essential clinical content
    """
    # Remove common headers (optional - might want to keep structure)
    # For now, keep everything and just clean whitespace
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Basic cleaning
    text = text.strip()
    
    return text


def create_label_mapping(master_df):
    """
    Create DICOM_ID to label mapping.
    
    Binary classification:
    - Label 0 (Normal): normal_reports == 1 OR no abnormalities
    - Label 1 (Abnormal): Any pathology present
    
    Pathology columns: CHF, pneumonia, consolidation, enlarged_cardiac_silhouette, etc.
    """
    dicom_to_label = {}
    
    # Define abnormality columns to check
    abnormality_cols = [
        'CHF', 'pneumonia', 'consolidation', 
        'enlarged_cardiac_silhouette', 'linear__patchy_atelectasis',
        'lobar__segmental_collapse', 
        'not_otherwise_specified_opacity___pleural__parenchymal_opacity__',
        'pleural_effusion_or_thickening', 'pulmonary_edema__hazy_opacity'
    ]
    
    for idx, row in master_df.iterrows():
        dicom_id = row['dicom_id']
        
        # Check if explicitly marked as normal
        is_normal = False
        if 'normal_reports' in row and row['normal_reports'] == 1:
            is_normal = True
        elif 'Normal' in row and row['Normal'] == 1:
            is_normal = True
        
        # Check for any abnormalities
        has_abnormality = False
        for col in abnormality_cols:
            if col in row and row[col] == 1:
                has_abnormality = True
                break
        
        # Assign label
        if is_normal and not has_abnormality:
            label = 0  # Normal
        elif has_abnormality:
            label = 1  # Abnormal
        else:
            # Unclear case - check if mostly normal indicators
            # Default to normal if no clear abnormalities
            label = 0
        
        dicom_to_label[dicom_id] = label
    
    return dicom_to_label


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
    # Path to your MIMIC-Eye dataset
    data_root = '/media/16TB_Storage/kavin/dataset/physionet.org/files/mimic-eye-multimodal-datasets/1.0.0/mimic-eye'
    
    output_dir = prepare_mimic_eye_metadata(
        data_root, 
        output_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye'
    )
    
    print(f"\n{'='*60}")
    print("Full multimodal dataset ready!")
    print("Image + Gaze + Text")
    print(f"{'='*60}")