import os
import json
import pandas as pd
from pathlib import Path

def check_reports():
    """
    Check radiology report availability for images with gaze data.
    """
    print("="*80)
    print("Checking Radiology Report Availability for C3-Net")
    print("="*80)
    
    # Paths
    processed_dir = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye'
    raw_data_dir = '/media/16TB_Storage/kavin/dataset/physionet.org/files/mimic-eye-multimodal-datasets/1.0.0/mimic-eye'
    
    # Load processed metadata
    metadata_path = os.path.join(processed_dir, 'metadata.json')
    print(f"\nLoading metadata from: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Total samples in dataset: {len(metadata)}")
    
    # Check report availability
    stats = {
        'total': len(metadata),
        'has_report': 0,
        'no_report': 0,
        'no_meta_csv': 0,
        'report_lengths': []
    }
    
    samples_with_reports = []
    samples_without_reports = []
    
    print("\nScanning for radiology reports...")
    
    for sample_id, info in metadata.items():
        patient_id = info['patient_id']
        dicom_id = info['dicom_id']
        
        # Read cxr_meta.csv to get study_id
        meta_csv_path = os.path.join(
            raw_data_dir,
            patient_id,
            'CXR-JPG',
            'cxr_meta.csv'
        )
        
        if not os.path.exists(meta_csv_path):
            stats['no_meta_csv'] += 1
            samples_without_reports.append({
                'sample_id': sample_id,
                'patient_id': patient_id,
                'dicom_id': dicom_id,
                'reason': 'no_meta_csv'
            })
            continue
        
        # Read meta CSV to find study_id
        try:
            meta_df = pd.read_csv(meta_csv_path)
            row = meta_df[meta_df['dicom_id'] == dicom_id]
            
            if len(row) == 0:
                stats['no_report'] += 1
                samples_without_reports.append({
                    'sample_id': sample_id,
                    'patient_id': patient_id,
                    'dicom_id': dicom_id,
                    'reason': 'dicom_not_in_meta'
                })
                continue
            
            study_id = row['study_id'].values[0]
            
            # Check for report file
            report_path = os.path.join(
                raw_data_dir,
                patient_id,
                'CXR-DICOM',
                f's{study_id}.txt'
            )
            
            if os.path.exists(report_path):
                # Read report to get length
                with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                    report_text = f.read()
                
                stats['has_report'] += 1
                stats['report_lengths'].append(len(report_text))
                
                samples_with_reports.append({
                    'sample_id': sample_id,
                    'patient_id': patient_id,
                    'dicom_id': dicom_id,
                    'study_id': study_id,
                    'report_path': report_path,
                    'report_length': len(report_text),
                    'split': info['split']
                })
            else:
                stats['no_report'] += 1
                samples_without_reports.append({
                    'sample_id': sample_id,
                    'patient_id': patient_id,
                    'dicom_id': dicom_id,
                    'study_id': study_id,
                    'reason': 'report_file_missing'
                })
                
        except Exception as e:
            stats['no_report'] += 1
            samples_without_reports.append({
                'sample_id': sample_id,
                'patient_id': patient_id,
                'dicom_id': dicom_id,
                'reason': f'error: {str(e)}'
            })
    
    # Print statistics
    print("\n" + "="*80)
    print("RADIOLOGY REPORT AVAILABILITY STATISTICS")
    print("="*80)
    print(f"\nTotal samples: {stats['total']}")
    print(f"Samples WITH reports: {stats['has_report']} ({100*stats['has_report']/stats['total']:.1f}%)")
    print(f"Samples WITHOUT reports: {stats['no_report']} ({100*stats['no_report']/stats['total']:.1f}%)")
    
    if stats['report_lengths']:
        import numpy as np
        print(f"\nReport length statistics:")
        print(f"  Mean: {np.mean(stats['report_lengths']):.0f} characters")
        print(f"  Median: {np.median(stats['report_lengths']):.0f} characters")
        print(f"  Min: {np.min(stats['report_lengths'])} characters")
        print(f"  Max: {np.max(stats['report_lengths'])} characters")
    
    # Split-wise statistics
    if stats['has_report'] > 0:
        print("\n" + "-"*80)
        print("Split-wise Distribution (WITH reports):")
        print("-"*80)
        
        split_counts = {'train': 0, 'val': 0, 'test': 0}
        for sample in samples_with_reports:
            split_counts[sample['split']] += 1
        
        print(f"Train: {split_counts['train']}")
        print(f"Val:   {split_counts['val']}")
        print(f"Test:  {split_counts['test']}")
    
    # Show sample reports
    if stats['has_report'] > 0:
        print("\n" + "="*80)
        print("SAMPLE REPORTS")
        print("="*80)
        
        # Show 2 samples
        for i, sample in enumerate(samples_with_reports[:2]):
            print(f"\nSample {i+1}:")
            print(f"  Patient ID: {sample['patient_id']}")
            print(f"  DICOM ID: {sample['dicom_id']}")
            print(f"  Study ID: {sample['study_id']}")
            print(f"  Split: {sample['split']}")
            print(f"  Report length: {sample['report_length']} characters")
            
            # Display report
            with open(sample['report_path'], 'r', encoding='utf-8', errors='ignore') as f:
                report_text = f.read()
            
            print(f"\n  Report content:")
            print("  " + "-"*70)
            # Show first 400 characters
            preview = report_text[:400].replace('\n', '\n  ')
            print(f"  {preview}")
            if len(report_text) > 400:
                print("  ...")
            print("  " + "-"*70)
    
    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    coverage_pct = 100*stats['has_report']/stats['total']
    
    if stats['has_report'] == 0:
        print("\n❌ NO REPORTS FOUND")
    elif coverage_pct < 50:
        print(f"\n⚠️  PARTIAL COVERAGE ({coverage_pct:.1f}%)")
        print(f"\nOption: Use only {stats['has_report']} samples with full image+gaze+text")
    else:
        print(f"\n✅ EXCELLENT COVERAGE ({coverage_pct:.1f}%)")
        print("\n🎉 Ready to proceed with FULL multimodal system!")
        print(f"   Image + Gaze + Clinical Text for {stats['has_report']} samples")
    
    print("="*80)
    
    return samples_with_reports, samples_without_reports


if __name__ == '__main__':
    samples_with_text, samples_without_text = check_reports()