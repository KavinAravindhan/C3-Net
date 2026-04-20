import os
import hashlib
import pandas as pd
import json
import numpy as np
from tqdm import tqdm


# ── Deterministic split ───────────────────────────────────────────────────────

def get_patient_split(patient_id):
    """
    Assign split deterministically from patient_id hash.
    Patient-level split ensures no leakage across sessions.
    Ratio: 80 / 10 / 10
    """
    hash_val = int(hashlib.md5(patient_id.encode()).hexdigest(), 16) % 10
    if hash_val < 8:
        return 'train'
    elif hash_val == 8:
        return 'val'
    else:
        return 'test'


# ── Main entry point ──────────────────────────────────────────────────────────

def prepare_mimic_eye_metadata(data_root, output_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye'):
    """
    Build unified metadata for C3-Net from both REFLACX and EyeGaze patients.

    Strategy:
    - REFLACX patients (2,154): gaze from main_data fixations, text from transcription.txt
    - EyeGaze patients (993):   gaze from EyeGaze fixations.csv, text from transcript.json
    - Overlap patients (45):    REFLACX takes priority, EyeGaze folder ignored
    - Split: patient-level 80/10/10 via deterministic hash

    Images are NOT copied — image_path in metadata points to original dataset location.
    Gaze JSONs and text TXTs are written to output_dir/gaze/ and output_dir/text/.

    metadata.json schema per sample:
    {
        "split":          "train" | "val" | "test",
        "label":          0 | 1,
        "patient_id":     str,
        "source":         "reflacx" | "eyegaze",
        "image_path":     str,
        "num_fixations":  int,
        "text_length":    int
    }
    """

    print("=" * 70)
    print("MIMIC-Eye Adapter: REFLACX + EyeGaze → C3-Net")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gaze'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'text'), exist_ok=True)

    # ── Load global label sources ──────────────────────────────────────────
    print("\nLoading label sources...")

    # EyeGaze labels from master_sheet
    eyegaze_master_path = os.path.join(
        data_root, 'spreadsheets', 'EyeGaze',
        'master_sheet_with_updated_stayId.csv'
    )
    eyegaze_label_map = {}
    if os.path.exists(eyegaze_master_path):
        master_df = pd.read_csv(eyegaze_master_path)
        eyegaze_label_map = create_eyegaze_label_mapping(master_df)
        print(f"  EyeGaze label map: {len(eyegaze_label_map)} DICOM IDs")
    else:
        print(f"  WARNING: EyeGaze master sheet not found: {eyegaze_master_path}")

    # REFLACX global metadata: reflacx_id → dicom_id
    reflacx_meta_path = os.path.join(data_root, 'spreadsheets', 'REFLACX', 'metadata.csv')
    reflacx_id_to_dicom = {}
    if os.path.exists(reflacx_meta_path):
        reflacx_meta_df = pd.read_csv(reflacx_meta_path)
        print(f"  REFLACX metadata columns: {reflacx_meta_df.columns.tolist()}")
        # Detect id and dicom columns by name
        id_col    = next((c for c in reflacx_meta_df.columns
                          if c.lower() in ('id', 'reflacx_id', 'session_id')), None)
        dicom_col = next((c for c in reflacx_meta_df.columns
                          if 'dicom' in c.lower()), None)
        if id_col and dicom_col:
            for _, row in reflacx_meta_df.iterrows():
                reflacx_id_to_dicom[str(row[id_col])] = str(row[dicom_col])
            print(f"  REFLACX id→dicom map: {len(reflacx_id_to_dicom)} sessions")
        else:
            print(f"  WARNING: Could not detect id/dicom columns in REFLACX metadata")
    else:
        print(f"  WARNING: REFLACX global metadata not found: {reflacx_meta_path}")

    # ── Scan patients ──────────────────────────────────────────────────────
    patient_dirs = sorted([d for d in os.listdir(data_root) if d.startswith('patient_')])
    print(f"\nFound {len(patient_dirs)} patient directories\n")

    metadata = {}
    counters = {
        'reflacx_processed':   0,
        'eyegaze_processed':   0,
        'skipped_no_image':    0,
        'skipped_no_fixations':0,
        'skipped_no_text':     0,
        'skipped_no_label':    0,
    }

    for patient_id in tqdm(patient_dirs, desc="Processing patients"):
        patient_path = os.path.join(data_root, patient_id)
        split        = get_patient_split(patient_id)

        has_reflacx = os.path.exists(os.path.join(patient_path, 'REFLACX'))
        has_eyegaze = os.path.exists(os.path.join(patient_path, 'EyeGaze'))

        # REFLACX takes priority for overlap patients
        if has_reflacx:
            samples = process_reflacx_patient(
                patient_path, patient_id, split,
                reflacx_id_to_dicom, output_dir, counters
            )
        elif has_eyegaze:
            samples = process_eyegaze_patient(
                patient_path, patient_id, split,
                eyegaze_label_map, output_dir, counters
            )
        else:
            continue

        metadata.update(samples)

    # ── Save metadata ──────────────────────────────────────────────────────
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  REFLACX sessions processed : {counters['reflacx_processed']}")
    print(f"  EyeGaze sessions processed : {counters['eyegaze_processed']}")
    print(f"  Total samples              : {len(metadata)}")
    print(f"\nSkipped:")
    print(f"  No image found    : {counters['skipped_no_image']}")
    print(f"  No fixations      : {counters['skipped_no_fixations']}")
    print(f"  No text           : {counters['skipped_no_text']}")
    print(f"  No label          : {counters['skipped_no_label']}")

    for split_name in ['train', 'val', 'test']:
        split_samples = [v for v in metadata.values() if v['split'] == split_name]
        normal   = sum(1 for v in split_samples if v['label'] == 0)
        abnormal = sum(1 for v in split_samples if v['label'] == 1)
        print(f"\n{split_name.capitalize()}: {len(split_samples)} "
              f"(Normal: {normal}, Abnormal: {abnormal})")

    print(f"\nOutput: {output_dir}")
    return output_dir


# ── REFLACX processing ────────────────────────────────────────────────────────

def process_reflacx_patient(patient_path, patient_id, split,
                             reflacx_id_to_dicom, output_dir, counters):
    """
    Process all REFLACX sessions for a patient.
    Each reflacx_id under REFLACX/main_data/ is one sample.
    """
    samples       = {}
    main_data_path = os.path.join(patient_path, 'REFLACX', 'main_data')

    if not os.path.exists(main_data_path):
        return samples

    for reflacx_id in os.listdir(main_data_path):
        session_path = os.path.join(main_data_path, reflacx_id)
        if not os.path.isdir(session_path):
            continue

        # ── Resolve DICOM ID ──
        dicom_id = reflacx_id_to_dicom.get(reflacx_id)
        if dicom_id is None:
            # Fallback: per-patient REFLACX/metadata.csv
            per_patient_meta = os.path.join(patient_path, 'REFLACX', 'metadata.csv')
            if os.path.exists(per_patient_meta):
                try:
                    df        = pd.read_csv(per_patient_meta)
                    id_col    = next((c for c in df.columns
                                      if c.lower() in ('id', 'reflacx_id', 'session_id')), None)
                    dicom_col = next((c for c in df.columns if 'dicom' in c.lower()), None)
                    if id_col and dicom_col:
                        row = df[df[id_col].astype(str) == reflacx_id]
                        if len(row) > 0:
                            dicom_id = str(row[dicom_col].values[0])
                except Exception:
                    pass

        if dicom_id is None:
            counters['skipped_no_image'] += 1
            continue

        # ── Find image ──
        image_path = find_image_for_dicom(patient_path, dicom_id)
        if image_path is None:
            counters['skipped_no_image'] += 1
            continue

        # ── Load fixations ──
        fixations_csv = os.path.join(session_path, 'fixations.csv')
        if not os.path.exists(fixations_csv):
            counters['skipped_no_fixations'] += 1
            continue

        try:
            fix_df = pd.read_csv(fixations_csv)
            from PIL import Image as PILImage
            img_w, img_h = PILImage.open(image_path).size
            fixations_list = process_reflacx_fixations(fix_df, img_w, img_h)
        except Exception:
            counters['skipped_no_fixations'] += 1
            continue

        if len(fixations_list) == 0:
            counters['skipped_no_fixations'] += 1
            continue

        # ── Load transcription ──
        trans_path = os.path.join(session_path, 'transcription.txt')
        if not os.path.exists(trans_path):
            counters['skipped_no_text'] += 1
            continue

        text = open(trans_path, encoding='utf-8', errors='ignore').read().strip()
        if not text:
            counters['skipped_no_text'] += 1
            continue

        # ── Label from anomaly ellipses ──
        label = get_reflacx_label(session_path)

        # ── Write outputs ──
        sample_id = f"{patient_id}_reflacx_{reflacx_id}"

        with open(os.path.join(output_dir, 'gaze', f"{sample_id}.json"), 'w') as f:
            json.dump({'fixations': fixations_list}, f)

        with open(os.path.join(output_dir, 'text', f"{sample_id}.txt"), 'w',
                  encoding='utf-8') as f:
            f.write(' '.join(text.split()))  # normalize whitespace

        samples[sample_id] = {
            'split':         split,
            'label':         int(label),
            'patient_id':    patient_id,
            'source':        'reflacx',
            'image_path':    image_path,
            'num_fixations': len(fixations_list),
            'text_length':   len(text),
        }
        counters['reflacx_processed'] += 1

    return samples


def get_reflacx_label(session_path):
    """
    Binary label from anomaly_location_ellipses.csv.
    Any ellipse rows present → Abnormal (1), else Normal (0).
    """
    ellipses_csv = os.path.join(session_path, 'anomaly_location_ellipses.csv')
    if not os.path.exists(ellipses_csv):
        return 0
    try:
        df = pd.read_csv(ellipses_csv)
        return 1 if len(df) > 0 else 0
    except Exception:
        return 0


def process_reflacx_fixations(fixations_df, img_w, img_h):
    """
    Normalize REFLACX fixations from pixel coordinates to [0, 1].
    Expected columns: x_position, y_position (pixels),
                      timestamp_start_fixation, timestamp_end_fixation (seconds)
    Duration is computed as (end - start) * 1000 → milliseconds.
    """
    fixations = []
    cols      = {c.lower(): c for c in fixations_df.columns}

    x_col     = cols.get('x_position')
    y_col     = cols.get('y_position')
    start_col = cols.get('timestamp_start_fixation')
    end_col   = cols.get('timestamp_end_fixation')

    if not all([x_col, y_col, start_col, end_col]):
        return fixations

    for _, row in fixations_df.iterrows():
        x, y     = row[x_col], row[y_col]
        t_start  = row[start_col]
        t_end    = row[end_col]

        if any(pd.isna(v) for v in [x, y, t_start, t_end]):
            continue

        fixations.append({
            'x':        float(np.clip(x / img_w, 0, 1)),
            'y':        float(np.clip(y / img_h, 0, 1)),
            'duration': float(max((t_end - t_start) * 1000, 0)),  # ms
        })

    return fixations


# ── EyeGaze processing ────────────────────────────────────────────────────────

def process_eyegaze_patient(patient_path, patient_id, split,
                             label_map, output_dir, counters):
    """
    Process all EyeGaze sessions for a patient.
    Each unique DICOM_ID in fixations.csv is one sample.
    """
    samples       = {}
    eyegaze_path  = os.path.join(patient_path, 'EyeGaze')
    fixations_csv = os.path.join(eyegaze_path, 'fixations.csv')

    if not os.path.exists(fixations_csv):
        return samples

    try:
        fix_df = pd.read_csv(fixations_csv)
    except Exception:
        return samples

    for dicom_id in fix_df['DICOM_ID'].unique():
        dicom_id = str(dicom_id)

        # ── Label ──
        if dicom_id not in label_map:
            counters['skipped_no_label'] += 1
            continue
        label = label_map[dicom_id]

        # ── Image ──
        image_path = find_image_for_dicom(patient_path, dicom_id)
        if image_path is None:
            counters['skipped_no_image'] += 1
            continue

        # ── Fixations ──
        dicom_fix = fix_df[fix_df['DICOM_ID'].astype(str) == dicom_id]
        fixations_list = process_eyegaze_fixations(dicom_fix)
        if len(fixations_list) == 0:
            counters['skipped_no_fixations'] += 1
            continue

        # ── Text from transcript.json ──
        transcript_path = os.path.join(
            eyegaze_path, 'audio_segmentation_transcripts',
            dicom_id, 'transcript.json'
        )
        text = None
        if os.path.exists(transcript_path):
            try:
                data = json.load(open(transcript_path))
                text = data.get('full_text', '').strip()
            except Exception:
                pass

        if not text:
            counters['skipped_no_text'] += 1
            continue

        # ── Write outputs ──
        sample_id = f"{patient_id}_eyegaze_{dicom_id}"

        with open(os.path.join(output_dir, 'gaze', f"{sample_id}.json"), 'w') as f:
            json.dump({'fixations': fixations_list}, f)

        with open(os.path.join(output_dir, 'text', f"{sample_id}.txt"), 'w',
                  encoding='utf-8') as f:
            f.write(' '.join(text.split()))

        samples[sample_id] = {
            'split':         split,
            'label':         int(label),
            'patient_id':    patient_id,
            'source':        'eyegaze',
            'image_path':    image_path,
            'num_fixations': len(fixations_list),
            'text_length':   len(text),
        }
        counters['eyegaze_processed'] += 1

    return samples


def process_eyegaze_fixations(fixations_df):
    """
    EyeGaze fixations are already normalized [0,1].
    FPOGX, FPOGY: normalized coordinates. FPOGD: duration in seconds → ms.
    """
    fixations = []
    for _, row in fixations_df.iterrows():
        x, y, dur = row.get('FPOGX'), row.get('FPOGY'), row.get('FPOGD')
        if any(pd.isna(v) for v in [x, y, dur]):
            continue
        fixations.append({
            'x':        float(np.clip(x, 0, 1)),
            'y':        float(np.clip(y, 0, 1)),
            'duration': float(max(dur * 1000, 0)),  # seconds → ms
        })
    return fixations


# ── Shared utilities ──────────────────────────────────────────────────────────

def find_image_for_dicom(patient_path, dicom_id):
    """
    Locate JPG for a given DICOM ID under patient_path/CXR-JPG/s{study_id}/.
    """
    cxr_path = os.path.join(patient_path, 'CXR-JPG')
    if not os.path.exists(cxr_path):
        return None

    for item in os.listdir(cxr_path):
        study_path = os.path.join(cxr_path, item)
        if not os.path.isdir(study_path):
            continue
        image_file = os.path.join(study_path, f"{dicom_id}.jpg")
        if os.path.exists(image_file):
            return image_file

    return None


def create_eyegaze_label_mapping(master_df):
    """
    Build dicom_id → binary label from EyeGaze master_sheet.
    Abnormal (1) if any pathology column is set, else Normal (0).
    """
    abnormality_cols = [
        'CHF', 'pneumonia', 'consolidation',
        'enlarged_cardiac_silhouette', 'linear__patchy_atelectasis',
        'lobar__segmental_collapse',
        'not_otherwise_specified_opacity___pleural__parenchymal_opacity__',
        'pleural_effusion_or_thickening', 'pulmonary_edema__hazy_opacity'
    ]

    label_map = {}
    for _, row in master_df.iterrows():
        dicom_id = str(row['dicom_id'])

        has_abnormality = any(
            col in row.index and row[col] == 1
            for col in abnormality_cols
        )
        is_normal = (
            ('normal_reports' in row.index and row['normal_reports'] == 1) or
            ('Normal' in row.index and row['Normal'] == 1)
        )

        label_map[dicom_id] = 1 if has_abnormality else 0

    return label_map


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_ROOT  = '/media/16TB_Storage/kavin/dataset/mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye'
    OUTPUT_DIR = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye'

    prepare_mimic_eye_metadata(DATA_ROOT, OUTPUT_DIR)