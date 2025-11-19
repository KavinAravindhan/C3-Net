import os
import json
import torch
from torch.utils.data import Dataset
from .preprocessing import ImagePreprocessor, GazePreprocessor


class MIMICEyeDataset(Dataset):
    """
    Dataset class for MIMIC-Eye data.
    
    Expected directory structure:
    mimic_eye/
    ├── images/
    │   ├── patient1_study1.jpg
    │   ├── patient2_study1.jpg
    │   └── ...
    ├── gaze/
    │   ├── patient1_study1.json
    │   ├── patient2_study1.json
    │   └── ...
    └── labels.json  # Contains image_id -> label mapping
    
    Each gaze JSON file format:
    {
        "fixations": [
            {"x": 0.45, "y": 0.32, "duration": 250},
            {"x": 0.52, "y": 0.38, "duration": 180},
            ...
        ]
    }
    """
    
    def __init__(self, root_dir, split='train', config=None):
        """
        Args:
            root_dir: Path to MIMIC-Eye dataset directory
            split: 'train', 'val', or 'test'
            config: Configuration dictionary with preprocessing params
        """
        self.root_dir = root_dir
        self.split = split
        
        # Initialize preprocessors
        image_size = config['data']['image_size'] if config else 224
        heatmap_size = config['data']['gaze_heatmap_size'] if config else 224
        max_fixations = config['data']['max_fixations'] if config else 50
        fixation_sigma = config['data']['fixation_sigma'] if config else 10
        
        self.image_preprocessor = ImagePreprocessor(image_size=image_size)
        self.gaze_preprocessor = GazePreprocessor(
            image_size=image_size,
            heatmap_size=heatmap_size,
            fixation_sigma=fixation_sigma,
            max_fixations=max_fixations
        )
        
        # Load data splits
        self.samples = self._load_split()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_split(self):
        """
        Load the data split from dataset.
        
        Returns:
            List of sample dictionaries with keys:
                - image_path: path to image file
                - gaze_path: path to gaze JSON file  
                - label: 0 or 1 (for binary classification)
        """
        # Load metadata
        metadata_path = os.path.join(self.root_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get samples for this split
        samples = []
        for sample_id, info in metadata.items():
            if info['split'] != self.split:
                continue
            
            image_path = os.path.join(self.root_dir, 'images', f"{sample_id}.jpg")
            gaze_path = os.path.join(self.root_dir, 'gaze', f"{sample_id}.json")
            
            # Verify files exist
            if os.path.exists(image_path) and os.path.exists(gaze_path):
                samples.append({
                    'image_path': image_path,
                    'gaze_path': gaze_path,
                    'label': info['label'],
                    'sample_id': sample_id
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - image: Tensor [3, 224, 224]
                - gaze_heatmap: Tensor [224, 224]
                - gaze_sequence: Tensor [50, 3]
                - gaze_seq_length: int
                - label: int (0 or 1)
                - sample_id: str
        """
        sample_info = self.samples[idx]
        
        # Load and preprocess image
        # Input: image file → Output: [3, 224, 224]
        image = self.image_preprocessor(sample_info['image_path'])
        
        # Load gaze data
        with open(sample_info['gaze_path'], 'r') as f:
            gaze_data = json.load(f)
        
        fixations = gaze_data['fixations']
        
        # Preprocess gaze
        # Input: List of fixations → Output: heatmap [224, 224], sequence [50, 3]
        gaze_heatmap, gaze_sequence, seq_length = self.gaze_preprocessor(fixations)
        
        return {
            'image': image,                    # [3, 224, 224]
            'gaze_heatmap': gaze_heatmap,      # [224, 224]
            'gaze_sequence': gaze_sequence,    # [50, 3]
            'gaze_seq_length': seq_length,     # scalar
            'label': torch.tensor(sample_info['label'], dtype=torch.long),  # scalar
            'sample_id': sample_info['sample_id']
        }


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles variable-length sequences and creates proper batches.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Dictionary with batched tensors:
            - images: [B, 3, 224, 224]
            - gaze_heatmaps: [B, 224, 224]
            - gaze_sequences: [B, 50, 3]
            - gaze_seq_lengths: [B]
            - labels: [B]
    """
    images = torch.stack([item['image'] for item in batch])  # [B, 3, 224, 224]
    gaze_heatmaps = torch.stack([item['gaze_heatmap'] for item in batch])  # [B, 224, 224]
    gaze_sequences = torch.stack([item['gaze_sequence'] for item in batch])  # [B, 50, 3]
    gaze_seq_lengths = torch.tensor([item['gaze_seq_length'] for item in batch])  # [B]
    labels = torch.stack([item['label'] for item in batch])  # [B]
    sample_ids = [item['sample_id'] for item in batch]
    
    return {
        'images': images,
        'gaze_heatmaps': gaze_heatmaps,
        'gaze_sequences': gaze_sequences,
        'gaze_seq_lengths': gaze_seq_lengths,
        'labels': labels,
        'sample_ids': sample_ids
    }