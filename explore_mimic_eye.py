import os
import json
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = '/media/16TB_Storage/kavin/dataset/mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye'

print("=" * 70)
print("MIMIC-Eye Dataset Exploration")
print("=" * 70)

patient_dirs = sorted([d for d in os.listdir(DATASET_ROOT) if d.startswith('patient_')])
print(f"\nTotal patient folders found: {len(patient_dirs)}")

# Counters
stats = {
    'has_reflacx':          0,
    'has_eyegaze':          0,
    'has_both':             0,
    'has_neither':          0,
    'reflacx_with_transcription': 0,
    'reflacx_with_fixations':     0,
    'reflacx_with_gaze':          0,
    'eyegaze_with_fixations':     0,
    'eyegaze_with_transcript':    0,
    'has_cxr_jpg':          0,
    'has_cxr_dicom_text':   0,
    'total_images':         0,
    'total_reflacx_ids':    0,
    'empty_transcriptions': 0,
}

sample_transcription = None
sample_transcript_json = None

for patient_id in patient_dirs:
    patient_path = os.path.join(DATASET_ROOT, patient_id)

    has_reflacx  = os.path.exists(os.path.join(patient_path, 'REFLACX'))
    has_eyegaze  = os.path.exists(os.path.join(patient_path, 'EyeGaze'))
    has_cxr_jpg  = os.path.exists(os.path.join(patient_path, 'CXR-JPG'))
    has_cxr_dicom = os.path.exists(os.path.join(patient_path, 'CXR-DICOM'))

    if has_reflacx:  stats['has_reflacx'] += 1
    if has_eyegaze:  stats['has_eyegaze'] += 1
    if has_reflacx and has_eyegaze: stats['has_both'] += 1
    if not has_reflacx and not has_eyegaze: stats['has_neither'] += 1
    if has_cxr_jpg:  stats['has_cxr_jpg'] += 1
    if has_cxr_dicom: stats['has_cxr_dicom_text'] += 1

    # Count CXR images
    if has_cxr_jpg:
        cxr_jpg_path = os.path.join(patient_path, 'CXR-JPG')
        for study_dir in os.listdir(cxr_jpg_path):
            study_path = os.path.join(cxr_jpg_path, study_dir)
            if os.path.isdir(study_path):
                jpgs = [f for f in os.listdir(study_path) if f.endswith('.jpg')]
                stats['total_images'] += len(jpgs)

    # REFLACX checks
    if has_reflacx:
        reflacx_path = os.path.join(patient_path, 'REFLACX')
        main_data    = os.path.join(reflacx_path, 'main_data')
        gaze_data    = os.path.join(reflacx_path, 'gaze_data')

        if os.path.exists(main_data):
            for reflacx_id in os.listdir(main_data):
                reflacx_id_path = os.path.join(main_data, reflacx_id)
                if not os.path.isdir(reflacx_id_path):
                    continue
                stats['total_reflacx_ids'] += 1

                # transcription.txt
                trans_path = os.path.join(reflacx_id_path, 'transcription.txt')
                if os.path.exists(trans_path):
                    content = open(trans_path).read().strip()
                    if content:
                        stats['reflacx_with_transcription'] += 1
                        if sample_transcription is None:
                            sample_transcription = (patient_id, reflacx_id, content[:300])
                    else:
                        stats['empty_transcriptions'] += 1

                # fixations.csv
                if os.path.exists(os.path.join(reflacx_id_path, 'fixations.csv')):
                    stats['reflacx_with_fixations'] += 1

        if os.path.exists(gaze_data):
            for reflacx_id in os.listdir(gaze_data):
                gaze_csv = os.path.join(gaze_data, reflacx_id, 'gaze.csv')
                if os.path.exists(gaze_csv):
                    stats['reflacx_with_gaze'] += 1

    # EyeGaze checks
    if has_eyegaze:
        eyegaze_path = os.path.join(patient_path, 'EyeGaze')

        if os.path.exists(os.path.join(eyegaze_path, 'fixations.csv')):
            stats['eyegaze_with_fixations'] += 1

        # per-dicom transcript.json
        seg_path = os.path.join(eyegaze_path, 'audio_segmentation_transcripts')
        if os.path.exists(seg_path):
            for dicom_id in os.listdir(seg_path):
                tj = os.path.join(seg_path, dicom_id, 'transcript.json')
                if os.path.exists(tj):
                    stats['eyegaze_with_transcript'] += 1
                    if sample_transcript_json is None:
                        try:
                            data = json.load(open(tj))
                            sample_transcript_json = (patient_id, dicom_id, str(data)[:300])
                        except:
                            pass

# ── Print Results ──────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("PATIENT COVERAGE")
print(f"{'─'*70}")
print(f"  REFLACX only     : {stats['has_reflacx'] - stats['has_both']}")
print(f"  EyeGaze only     : {stats['has_eyegaze'] - stats['has_both']}")
print(f"  Both             : {stats['has_both']}")
print(f"  Neither          : {stats['has_neither']}")
print(f"  Has CXR-JPG      : {stats['has_cxr_jpg']}")
print(f"  Has CXR-DICOM txt: {stats['has_cxr_dicom_text']}")
print(f"  Total CXR images : {stats['total_images']}")

print(f"\n{'─'*70}")
print("REFLACX MODALITY COVERAGE")
print(f"{'─'*70}")
print(f"  Total REFLACX IDs (sessions)  : {stats['total_reflacx_ids']}")
print(f"  With transcription.txt        : {stats['reflacx_with_transcription']}")
print(f"  Empty transcription.txt       : {stats['empty_transcriptions']}")
print(f"  With fixations.csv            : {stats['reflacx_with_fixations']}")
print(f"  With gaze.csv                 : {stats['reflacx_with_gaze']}")

print(f"\n{'─'*70}")
print("EYEGAZE MODALITY COVERAGE")
print(f"{'─'*70}")
print(f"  Patients with fixations.csv   : {stats['eyegaze_with_fixations']}")
print(f"  DICOMs with transcript.json   : {stats['eyegaze_with_transcript']}")

print(f"\n{'─'*70}")
print("SAMPLE REFLACX TRANSCRIPTION")
print(f"{'─'*70}")
if sample_transcription:
    pid, rid, text = sample_transcription
    print(f"  Patient : {pid}")
    print(f"  ID      : {rid}")
    print(f"  Text    : {text}")
else:
    print("  None found")

print(f"\n{'─'*70}")
print("SAMPLE EYEGAZE TRANSCRIPT.JSON")
print(f"{'─'*70}")
if sample_transcript_json:
    pid, did, text = sample_transcript_json
    print(f"  Patient : {pid}")
    print(f"  DICOM   : {did}")
    print(f"  Content : {text}")
else:
    print("  None found")

print(f"\n{'='*70}")