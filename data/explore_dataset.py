import os
import pandas as pd
from collections import defaultdict

DATASET_ROOT = '/media/16TB_Storage/kavin/dataset/physionet.org/files/mimic-eye-multimodal-datasets/1.0.0/mimic-eye'

print("="*80)
print("EyeGaze Dataset Analysis for C3-Net Training")
print("="*80)

patient_dirs = sorted([d for d in os.listdir(DATASET_ROOT) if d.startswith('patient_')])

stats = {
    'patients_with_eyegaze': 0,
    'total_fixations': 0,
    'images_with_gaze': set(),
    'dicom_ids': set(),
}

print("\nScanning EyeGaze data...")

for patient_id in patient_dirs:
    patient_path = os.path.join(DATASET_ROOT, patient_id)
    eyegaze_path = os.path.join(patient_path, 'EyeGaze')
    
    if not os.path.exists(eyegaze_path):
        continue
    
    fixations_csv = os.path.join(eyegaze_path, 'fixations.csv')
    if not os.path.exists(fixations_csv):
        continue
    
    stats['patients_with_eyegaze'] += 1
    
    # Load fixations
    try:
        df = pd.read_csv(fixations_csv)
        stats['total_fixations'] += len(df)
        
        # Get unique DICOM IDs
        if 'DICOM_ID' in df.columns:
            dicom_ids = df['DICOM_ID'].unique()
            stats['dicom_ids'].update(dicom_ids)
            stats['images_with_gaze'].update([(patient_id, did) for did in dicom_ids])
    except Exception as e:
        print(f"Error reading {patient_id}: {e}")

print(f"\n{'='*80}")
print("RESULTS")
print('='*80)
print(f"Patients with EyeGaze data: {stats['patients_with_eyegaze']}")
print(f"Total fixation points: {stats['total_fixations']:,}")
print(f"Unique images with gaze data: {len(stats['images_with_gaze'])}")
print(f"Unique DICOM IDs: {len(stats['dicom_ids'])}")

if stats['patients_with_eyegaze'] > 0:
    avg_fixations = stats['total_fixations'] / len(stats['images_with_gaze']) if len(stats['images_with_gaze']) > 0 else 0
    print(f"Average fixations per image: {avg_fixations:.1f}")

# Sample one patient's data in detail
print(f"\n{'='*80}")
print("SAMPLE EYEGAZE DATA STRUCTURE")
print('='*80)

for patient_id in patient_dirs:
    eyegaze_path = os.path.join(DATASET_ROOT, patient_id, 'EyeGaze')
    fixations_csv = os.path.join(eyegaze_path, 'fixations.csv')
    
    if os.path.exists(fixations_csv):
        print(f"\nPatient: {patient_id}")
        
        # Load fixations
        df = pd.read_csv(fixations_csv)
        print(f"\nFixations.csv shape: {df.shape}")
        print(f"Columns ({len(df.columns)} total): {df.columns.tolist()[:20]}...")  # Show first 20
        
        # Show key columns with proper type handling
        print(f"\nKey columns for C3-Net:")
        
        # DICOM_ID (string)
        if 'DICOM_ID' in df.columns:
            print(f"  DICOM_ID (identifier):")
            print(f"    Unique images: {df['DICOM_ID'].nunique()}")
            print(f"    Sample: {df['DICOM_ID'].iloc[0]}")
        
        # Numeric columns
        numeric_cols = ['FPOGX', 'FPOGY', 'FPOGD', 'X_ORIGINAL', 'Y_ORIGINAL']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    print(f"  {col}:")
                    print(f"    Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
                    print(f"    Mean: {df[col].mean():.4f}")
                    print(f"    Non-null: {df[col].notna().sum()} / {len(df)}")
                except:
                    print(f"  {col}: Unable to compute statistics")
        
        # Show sample fixations
        key_cols = ['DICOM_ID', 'FPOGX', 'FPOGY', 'FPOGD', 'X_ORIGINAL', 'Y_ORIGINAL']
        available_cols = [c for c in key_cols if c in df.columns]
        
        print(f"\nSample fixations (first 10):")
        print(df[available_cols].head(10).to_string(index=False))
        
        # Check for corresponding image
        if 'DICOM_ID' in df.columns:
            sample_dicom = df['DICOM_ID'].iloc[0]
            print(f"\nLooking for image with DICOM_ID: {sample_dicom}")
            
            cxr_path = os.path.join(DATASET_ROOT, patient_id, 'CXR-JPG')
            if os.path.exists(cxr_path):
                found = False
                for study_dir in os.listdir(cxr_path):
                    study_path = os.path.join(cxr_path, study_dir)
                    if os.path.isdir(study_path):
                        for img_file in os.listdir(study_path):
                            if sample_dicom in img_file:
                                img_path = os.path.join(study_path, img_file)
                                print(f"✓ Image found: {img_path}")
                                
                                # Try to get image dimensions
                                try:
                                    from PIL import Image
                                    img = Image.open(img_path)
                                    print(f"  Image size: {img.size[0]} x {img.size[1]} pixels")
                                    
                                    # Check if X_ORIGINAL/Y_ORIGINAL are in pixel space
                                    if 'X_ORIGINAL' in df.columns and 'Y_ORIGINAL' in df.columns:
                                        print(f"\n  Coordinate Analysis:")
                                        print(f"    X_ORIGINAL range: [{df['X_ORIGINAL'].min():.1f}, {df['X_ORIGINAL'].max():.1f}]")
                                        print(f"    Y_ORIGINAL range: [{df['Y_ORIGINAL'].min():.1f}, {df['Y_ORIGINAL'].max():.1f}]")
                                        print(f"    Image width: {img.size[0]}")
                                        print(f"    Image height: {img.size[1]}")
                                        
                                        if df['X_ORIGINAL'].max() <= img.size[0] and df['Y_ORIGINAL'].max() <= img.size[1]:
                                            print(f"  ✓ X_ORIGINAL/Y_ORIGINAL are in pixel coordinates!")
                                        else:
                                            print(f"  ! Coordinates may need scaling")
                                    
                                    # Check FPOGX/FPOGY (normalized 0-1)
                                    if 'FPOGX' in df.columns and 'FPOGY' in df.columns:
                                        print(f"\n  Normalized Coordinates:")
                                        print(f"    FPOGX range: [{df['FPOGX'].min():.4f}, {df['FPOGX'].max():.4f}]")
                                        print(f"    FPOGY range: [{df['FPOGY'].min():.4f}, {df['FPOGY'].max():.4f}]")
                                        if df['FPOGX'].max() <= 1.0 and df['FPOGY'].max() <= 1.0:
                                            print(f"  ✓ FPOGX/FPOGY are normalized (0-1)!")
                                except ImportError:
                                    print(f"  (Install Pillow to check image dimensions)")
                                except Exception as e:
                                    print(f"  Error checking image: {e}")
                                
                                found = True
                                break
                    if found:
                        break
                if not found:
                    print(f"✗ Image not found for DICOM_ID {sample_dicom}")
        
        # Check master_sheet.csv for additional metadata
        master_sheet = os.path.join(eyegaze_path, 'master_sheet.csv')
        if os.path.exists(master_sheet):
            print(f"\n{'='*80}")
            print(f"Master Sheet Analysis")
            print('='*80)
            ms_df = pd.read_csv(master_sheet)
            print(f"Shape: {ms_df.shape}")
            print(f"Columns: {ms_df.columns.tolist()}")
            print(f"\nSample entry:")
            print(ms_df.head(1).to_string(index=False))
        
        break  # Only show one sample

print(f"\n{'='*80}")
print("C3-NET TRAINING READINESS")
print('='*80)
print("\n✓ Dataset Summary:")
print(f"  • {stats['patients_with_eyegaze']} patients with EyeGaze data")
print(f"  • {len(stats['images_with_gaze'])} images with fixation annotations")
print(f"  • {stats['total_fixations']:,} total fixation points")
print(f"  • ~{stats['total_fixations'] / len(stats['images_with_gaze']):.0f} fixations per image")

print("\n✓ Data Format:")
print("  • Fixation coordinates: FPOGX, FPOGY (normalized 0-1)")
print("  • OR: X_ORIGINAL, Y_ORIGINAL (pixel coordinates)")
print("  • Fixation duration: FPOGD")
print("  • Image identifier: DICOM_ID")
print('='*80)