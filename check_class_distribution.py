import json

# Load metadata
metadata_path = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye/metadata.json'

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Count labels per split
splits = {'train': {0: 0, 1: 0}, 'val': {0: 0, 1: 0}, 'test': {0: 0, 1: 0}}

for sample_id, info in metadata.items():
    split = info['split']
    label = info['label']
    splits[split][label] += 1

# Print results
print("="*60)
print("Class Distribution Analysis")
print("="*60)

for split in ['train', 'val', 'test']:
    total = splits[split][0] + splits[split][1]
    normal_pct = (splits[split][0] / total) * 100
    abnormal_pct = (splits[split][1] / total) * 100
    
    print(f"\n{split.upper()} ({total} samples):")
    print(f"  Normal (0):   {splits[split][0]:3d} ({normal_pct:.1f}%)")
    print(f"  Abnormal (1): {splits[split][1]:3d} ({abnormal_pct:.1f}%)")

# Overall
total_normal = sum(splits[s][0] for s in splits)
total_abnormal = sum(splits[s][1] for s in splits)
total = total_normal + total_abnormal

print(f"\nOVERALL ({total} samples):")
print(f"  Normal (0):   {total_normal} ({100*total_normal/total:.1f}%)")
print(f"  Abnormal (1): {total_abnormal} ({100*total_abnormal/total:.1f}%)")
print("="*60)




# ============================================================
# Class Distribution Analysis
# ============================================================

# TRAIN (873 samples):
#   Normal (0):   294 (33.7%)
#   Abnormal (1): 579 (66.3%)

# VAL (108 samples):
#   Normal (0):    36 (33.3%)
#   Abnormal (1):  72 (66.7%)

# TEST (102 samples):
#   Normal (0):    30 (29.4%)
#   Abnormal (1):  72 (70.6%)

# OVERALL (1083 samples):
#   Normal (0):   360 (33.2%)
#   Abnormal (1): 723 (66.8%)
# ============================================================