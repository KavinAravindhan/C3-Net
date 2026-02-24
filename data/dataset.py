import os
import json
import torch
from torch.utils.data import Dataset
from .preprocessing import ImagePreprocessor, GazePreprocessor, TextPreprocessor
from .transforms import MedicalImageAugmentation, NoAugmentation


class MIMICEyeDataset(Dataset):
    """
    Dataset class for MIMIC-Eye data with full multimodal support.
    
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
    ├── text/
    │   ├── patient1_study1.txt
    │   ├── patient2_study1.txt
    │   └── ...
    └── metadata.json  # Contains image_id -> label mapping
    
    Each gaze JSON file format:
    {
        "fixations": [
            {"x": 0.45, "y": 0.32, "duration": 250},
            {"x": 0.52, "y": 0.38, "duration": 180},
            ...
        ]
    }
    
    Each text file contains cleaned radiology report.
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
        
        # Initialize augmentation (only for training)
        if split == 'train':
            self.augmentation = MedicalImageAugmentation(image_size=image_size)
            print("  Using data augmentation for training")
        else:
            self.augmentation = NoAugmentation()
            print("  No augmentation for validation/test")

        self.gaze_preprocessor = GazePreprocessor(
            image_size=image_size,
            heatmap_size=heatmap_size,
            fixation_sigma=fixation_sigma,
            max_fixations=max_fixations
        )
        
        # Initialize text preprocessor
        self.text_preprocessor = TextPreprocessor(
            model_name='emilyalsentzer/Bio_ClinicalBERT',
            max_length=128
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
                - text_path: path to text file
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
            text_path = os.path.join(self.root_dir, 'text', f"{sample_id}.txt")
            
            # Verify all files exist
            if os.path.exists(image_path) and os.path.exists(gaze_path) and os.path.exists(text_path):
                samples.append({
                    'image_path': image_path,
                    'gaze_path': gaze_path,
                    'text_path': text_path,
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
                - text_token_ids: Tensor [128]
                - text_attention_mask: Tensor [128]
                - label: int (0 or 1)
                - sample_id: str
        """
        sample_info = self.samples[idx]

        # Load image
        from PIL import Image
        pil_image = Image.open(sample_info['image_path']).convert('RGB')

        # Apply augmentation (training only)
        pil_image = self.augmentation(pil_image)

        # Preprocess image
        # Input: PIL image → Output: [3, 224, 224]
        image = self.image_preprocessor.preprocess_pil(pil_image)
        
        # Load gaze data
        with open(sample_info['gaze_path'], 'r') as f:
            gaze_data = json.load(f)
        
        fixations = gaze_data['fixations']
        
        # Preprocess gaze
        # Input: List of fixations → Output: heatmap [224, 224], sequence [50, 3]
        gaze_heatmap, gaze_sequence, seq_length = self.gaze_preprocessor(fixations)
        
        # Load text
        with open(sample_info['text_path'], 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Preprocess text
        # Input: Raw text → Output: token_ids [128], attention_mask [128]
        text_token_ids, text_attention_mask = self.text_preprocessor(text)
        
        return {
            'image': image,                                # [3, 224, 224]
            'gaze_heatmap': gaze_heatmap,                  # [224, 224]
            'gaze_sequence': gaze_sequence,                # [50, 3]
            'gaze_seq_length': seq_length,                 # scalar
            'text_token_ids': text_token_ids,              # [128]
            'text_attention_mask': text_attention_mask,    # [128]
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
            - text_token_ids: [B, 128]
            - text_attention_masks: [B, 128]
            - labels: [B]
            - sample_ids: List[str]
    """
    images = torch.stack([item['image'] for item in batch])  # [B, 3, 224, 224]
    gaze_heatmaps = torch.stack([item['gaze_heatmap'] for item in batch])  # [B, 224, 224]
    gaze_sequences = torch.stack([item['gaze_sequence'] for item in batch])  # [B, 50, 3]
    gaze_seq_lengths = torch.tensor([item['gaze_seq_length'] for item in batch])  # [B]
    text_token_ids = torch.stack([item['text_token_ids'] for item in batch])  # [B, 128]
    text_attention_masks = torch.stack([item['text_attention_mask'] for item in batch])  # [B, 128]
    labels = torch.stack([item['label'] for item in batch])  # [B]
    sample_ids = [item['sample_id'] for item in batch]
    
    return {
        'images': images,
        'gaze_heatmaps': gaze_heatmaps,
        'gaze_sequences': gaze_sequences,
        'gaze_seq_lengths': gaze_seq_lengths,
        'text_token_ids': text_token_ids,
        'text_attention_masks': text_attention_masks,
        'labels': labels,
        'sample_ids': sample_ids
    }