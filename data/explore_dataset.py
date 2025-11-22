import os
import pandas as pd
from pathlib import Path

# Update this path to your dataset location
DATASET_ROOT = '/media/16TB_Storage/kavin/dataset/physionet.org/files/mimic-eye-multimodal-datasets/1.0.0/mimic-eye'

print("="*60)
print("MIMIC-Eye Deep Gaze Data Exploration")
print("="*60)

# Find all patient directories
patient_dirs = sorted([d for d in os.listdir(DATASET_ROOT) if d.startswith('patient_')])
print(f"\nTotal patients: {len(patient_dirs)}")

# Look at first few patients in detail
for i, patient_id in enumerate(patient_dirs[:3]):
    print(f"\n{'='*60}")
    print(f"Patient {i+1}: {patient_id}")
    print('='*60)
    
    patient_path = os.path.join(DATASET_ROOT, patient_id)
    reflacx_path = os.path.join(patient_path, 'REFLACX')
    
    if not os.path.exists(reflacx_path):
        print(f"  No REFLACX directory for {patient_id}")
        continue
    
    # Load metadata
    metadata_csv = os.path.join(reflacx_path, 'metadata.csv')
    if os.path.exists(metadata_csv):
        metadata_df = pd.read_csv(metadata_csv)
        print(f"\n  Metadata entries: {len(metadata_df)}")
        print(f"  Study IDs: {metadata_df['id'].tolist()}")
        
        # For each study in metadata
        for _, row in metadata_df.iterrows():
            study_id = row['id']
            print(f"\n  Study: {study_id}")
            print(f"    Eye tracking discarded: {row['eye_tracking_data_discarded']}")
            print(f"    Image: {row['image']}")
            print(f"    Split: {row['split']}")
            
            # Check gaze_data folder
            gaze_data_path = os.path.join(reflacx_path, 'gaze_data', study_id)
            if os.path.exists(gaze_data_path):
                contents = os.listdir(gaze_data_path)
                print(f"    Gaze data folder contents: {contents}")
                
                # Look for CSV files
                csv_files = [f for f in contents if f.endswith('.csv')]
                if csv_files:
                    print(f"    CSV files found: {csv_files}")
                    for csv_file in csv_files:
                        csv_path = os.path.join(gaze_data_path, csv_file)
                        df = pd.read_csv(csv_path)
                        print(f"\n    {csv_file}:")
                        print(f"      Shape: {df.shape}")
                        print(f"      Columns: {df.columns.tolist()}")
                        print(f"      First 3 rows:")
                        print(df.head(3).to_string())
                        
                        # Check for key columns
                        if 'x_position' in df.columns or 'X' in df.columns:
                            print(f"      ✓ Has X position column")
                        if 'y_position' in df.columns or 'Y' in df.columns:
                            print(f"      ✓ Has Y position column")
                        if 'duration' in df.columns or 'Duration' in df.columns:
                            print(f"      ✓ Has duration column")
                else:
                    print(f"    No CSV files in gaze_data folder")
            else:
                print(f"    No gaze_data folder for {study_id}")
            
            # Check main_data folder
            main_data_path = os.path.join(reflacx_path, 'main_data', study_id)
            if os.path.exists(main_data_path):
                contents = os.listdir(main_data_path)
                print(f"    Main data folder contents: {contents}")
                
                # Look for CSV files
                csv_files = [f for f in contents if f.endswith('.csv')]
                if csv_files:
                    print(f"    CSV files found: {csv_files}")
                    for csv_file in csv_files:
                        csv_path = os.path.join(main_data_path, csv_file)
                        df = pd.read_csv(csv_path)
                        print(f"\n    {csv_file}:")
                        print(f"      Shape: {df.shape}")
                        print(f"      Columns: {df.columns.tolist()}")
                        print(f"      First 5 rows:")
                        print(df.head(5).to_string())
            else:
                print(f"    No main_data folder for {study_id}")
            
            # Check for image file
            image_path = os.path.join(patient_path, 'CXR-JPG')
            if os.path.exists(image_path):
                study_dirs = os.listdir(image_path)
                for study_dir in study_dirs:
                    study_img_path = os.path.join(image_path, study_dir)
                    if os.path.isdir(study_img_path):
                        images = [f for f in os.listdir(study_img_path) if f.endswith('.jpg')]
                        if images:
                            print(f"    Image found: {images[0]}")
                            print(f"    Image path: {os.path.join(study_img_path, images[0])}")

print("\n" + "="*60)
print("Deep exploration complete!")
print("="*60)

# Summary statistics
print("\n" + "="*60)
print("Dataset Summary")
print("="*60)

total_studies = 0
total_with_gaze = 0
total_with_main_data = 0

for patient_id in patient_dirs[:50]:  # Check first 50 patients
    patient_path = os.path.join(DATASET_ROOT, patient_id)
    reflacx_path = os.path.join(patient_path, 'REFLACX')
    
    if os.path.exists(reflacx_path):
        metadata_csv = os.path.join(reflacx_path, 'metadata.csv')
        if os.path.exists(metadata_csv):
            metadata_df = pd.read_csv(metadata_csv)
            total_studies += len(metadata_df)
            
            for _, row in metadata_df.iterrows():
                study_id = row['id']
                
                # Check gaze_data
                gaze_path = os.path.join(reflacx_path, 'gaze_data', study_id)
                if os.path.exists(gaze_path):
                    csv_files = [f for f in os.listdir(gaze_path) if f.endswith('.csv')]
                    if csv_files:
                        total_with_gaze += 1
                
                # Check main_data
                main_path = os.path.join(reflacx_path, 'main_data', study_id)
                if os.path.exists(main_path):
                    csv_files = [f for f in os.listdir(main_path) if f.endswith('.csv')]
                    if csv_files:
                        total_with_main_data += 1

print(f"\nFirst 50 patients summary:")
print(f"  Total studies: {total_studies}")
print(f"  Studies with gaze CSV: {total_with_gaze}")
print(f"  Studies with main_data CSV: {total_with_main_data}")