import os
import json
import torch
from torch.utils.data import Dataset
from .preprocessing import ImagePreprocessor, GazePreprocessor, TextPreprocessor, MedGemmaTextPreprocessor
from .transforms import MedicalImageAugmentation, NoAugmentation


class MIMICEyeDataset(Dataset):
    """
    Dataset class for MIMIC-Eye data with full multimodal support.
    Supports both REFLACX and EyeGaze sources via unified metadata.json.

    Expected processed directory structure:
    processed_mimic_eye/
    ├── gaze/
    │   ├── {sample_id}.json
    │   └── ...
    ├── text/
    │   ├── {sample_id}.txt
    │   └── ...
    └── metadata.json

    metadata.json schema per sample:
    {
        "split":         "train" | "val" | "test",
        "label":         0 | 1,
        "patient_id":    str,
        "source":        "reflacx" | "eyegaze",
        "image_path":    str,       # absolute path to original CXR-JPG
        "num_fixations": int,
        "text_length":   int
    }

    Each gaze JSON file format:
    {
        "fixations": [
            {"x": 0.45, "y": 0.32, "duration": 250},
            ...
        ]
    }

    Each text file contains cleaned radiologist transcription
    (REFLACX transcription.txt or EyeGaze transcript.json full_text).
    """

    def __init__(self, root_dir, split='train', config=None):
        """
        Args:
            root_dir: Path to processed MIMIC-Eye directory
            split: 'train', 'val', or 'test'
            config: Configuration dictionary with preprocessing params
        """
        self.root_dir = root_dir
        self.split    = split
        self.config   = config

        # Initialize preprocessors
        image_size     = config['data']['image_size']         if config else 224
        heatmap_size   = config['data']['gaze_heatmap_size']  if config else 224
        max_fixations  = config['data']['max_fixations']      if config else 50
        fixation_sigma = config['data']['fixation_sigma']     if config else 10
        use_medgemma   = config['model']['use_medgemma']      if config else False

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

        # Conditionally select text preprocessor based on use_medgemma flag
        if use_medgemma:
            medgemma_model  = config['model']['medgemma']['model_name'] if config else 'google/medgemma-4b-it'
            medgemma_maxlen = config['model']['medgemma']['max_length']  if config else 256
            self.text_preprocessor = MedGemmaTextPreprocessor(
                model_name=medgemma_model,
                max_length=medgemma_maxlen
            )
            print("  Text preprocessor: MedGemmaTextPreprocessor")
        else:
            self.text_preprocessor = TextPreprocessor(
                model_name='emilyalsentzer/Bio_ClinicalBERT',
                max_length=128
            )
            print("  Text preprocessor: TextPreprocessor (BioClinicalBERT)")

        # Load data splits
        self.samples = self._load_split()

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_split(self):
        """
        Load the data split from metadata.json.

        Returns:
            List of sample dictionaries with keys:
                - image_path:  absolute path to CXR-JPG image
                - gaze_path:   path to gaze JSON file
                - text_path:   path to text file
                - label:       0 or 1
                - sample_id:   str
                - source:      'reflacx' or 'eyegaze'
        """
        metadata_path = os.path.join(self.root_dir, 'metadata.json')

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        samples = []
        for sample_id, info in metadata.items():
            if info['split'] != self.split:
                continue

            # Image path is stored directly in metadata (no copying)
            image_path = info['image_path']
            gaze_path  = os.path.join(self.root_dir, 'gaze', f"{sample_id}.json")
            text_path  = os.path.join(self.root_dir, 'text', f"{sample_id}.txt")

            if os.path.exists(image_path) and os.path.exists(gaze_path) and os.path.exists(text_path):
                samples.append({
                    'image_path': image_path,
                    'gaze_path':  gaze_path,
                    'text_path':  text_path,
                    'label':      info['label'],
                    'sample_id':  sample_id,
                    'source':     info.get('source', 'unknown'),
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - image:               Tensor [3, 224, 224]
                - gaze_heatmap:        Tensor [196]  — patch-aligned gaze scores
                - gaze_sequence:       Tensor [50, 3]
                - gaze_seq_length:     int
                - text_token_ids:      Tensor [max_length]
                - text_attention_mask: Tensor [max_length]
                - report_text:         str  — raw transcription for decoder
                - label:               int (0 or 1)
                - sample_id:           str
                - source:              str ('reflacx' or 'eyegaze')
        """
        sample_info = self.samples[idx]

        # Load and augment image
        from PIL import Image
        pil_image = Image.open(sample_info['image_path']).convert('RGB')
        pil_image = self.augmentation(pil_image)

        # Preprocess image
        # Input: PIL image → Output: [3, 224, 224]
        image = self.image_preprocessor.preprocess_pil(pil_image)

        # Load gaze data
        with open(sample_info['gaze_path'], 'r') as f:
            gaze_data = json.load(f)

        fixations = gaze_data['fixations']

        # Preprocess gaze
        # Input: List of fixations → Output: heatmap [196], sequence [50, 3]
        gaze_heatmap, gaze_sequence, seq_length = self.gaze_preprocessor(fixations)

        # Load text
        with open(sample_info['text_path'], 'r', encoding='utf-8') as f:
            text = f.read()

        # Preprocess text — BioClinicalBERT or MedGemma tokenizer depending on config
        # Input: Raw text → Output: token_ids [max_length], attention_mask [max_length]
        text_token_ids, text_attention_mask = self.text_preprocessor(text)

        return {
            'image':               image,                 # [3, 224, 224]
            'gaze_heatmap':        gaze_heatmap,          # [196]
            'gaze_sequence':       gaze_sequence,         # [50, 3]
            'gaze_seq_length':     seq_length,            # scalar
            'text_token_ids':      text_token_ids,        # [max_length]
            'text_attention_mask': text_attention_mask,   # [max_length]
            'report_text':         text,                  # str — raw text for decoder
            'label':               torch.tensor(sample_info['label'], dtype=torch.long),
            'sample_id':           sample_info['sample_id'],
            'source':              sample_info['source'],  # 'reflacx' or 'eyegaze'
        }


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Returns:
        Dictionary with batched tensors:
            - images:               [B, 3, 224, 224]
            - gaze_heatmaps:        [B, 196]
            - gaze_sequences:       [B, 50, 3]
            - gaze_seq_lengths:     [B]
            - text_token_ids:       [B, max_length]
            - text_attention_masks: [B, max_length]
            - report_texts:         List[str]
            - labels:               [B]
            - sample_ids:           List[str]
            - sources:              List[str]
    """
    images               = torch.stack([item['image']               for item in batch])  # [B, 3, 224, 224]
    gaze_heatmaps        = torch.stack([item['gaze_heatmap']        for item in batch])  # [B, 196]
    gaze_sequences       = torch.stack([item['gaze_sequence']       for item in batch])  # [B, 50, 3]
    gaze_seq_lengths     = torch.tensor([item['gaze_seq_length']    for item in batch])  # [B]
    text_token_ids       = torch.stack([item['text_token_ids']      for item in batch])  # [B, max_length]
    text_attention_masks = torch.stack([item['text_attention_mask'] for item in batch])  # [B, max_length]
    report_texts         = [item['report_text']  for item in batch]                      # List[str]
    labels               = torch.stack([item['label']               for item in batch])  # [B]
    sample_ids           = [item['sample_id']    for item in batch]
    sources              = [item['source']       for item in batch]

    return {
        'images':               images,
        'gaze_heatmaps':        gaze_heatmaps,
        'gaze_sequences':       gaze_sequences,
        'gaze_seq_lengths':     gaze_seq_lengths,
        'text_token_ids':       text_token_ids,
        'text_attention_masks': text_attention_masks,
        'report_texts':         report_texts,
        'labels':               labels,
        'sample_ids':           sample_ids,
        'sources':              sources,
    }