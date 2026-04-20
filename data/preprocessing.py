import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image
import cv2
from transformers import AutoTokenizer
import os

from dotenv import load_dotenv
load_dotenv()

class ImagePreprocessor:
    """
    Preprocesses medical images (X-ray or OCT) for ViT input.
    
    Mathematical operations:
    1. Resize: I_resized = Resize(I_raw, size=(H, W))
    2. Normalize: I_norm = (I_resized - μ) / σ
       where μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225] (ImageNet stats)
    """
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
        # ImageNet normalization (standard for ViT)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess_pil(self, pil_image):
        """
        Preprocess a PIL image (used with augmentation pipeline).
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            image_tensor: Preprocessed image [C=3, H=224, W=224]
        """
        # Resize
        image = pil_image.resize((self.image_size, self.image_size), 
                                Image.BILINEAR)  # [224, 224, 3]
        
        # Convert to numpy array and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0  # [224, 224, 3]
        
        # Normalize using ImageNet statistics
        # I_norm = (I - μ) / σ
        image = (image - self.mean) / self.std  # [224, 224, 3]
        
        # Convert to PyTorch tensor [C, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # [3, 224, 224]
        
        return image_tensor
    
    def __call__(self, image_path):
        """
        Args:
            image_path: Path to image file
            
        Returns:
            image_tensor: Preprocessed image [C=3, H=224, W=224]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')  # [H, W, C=3]
        
        # Resize
        image = image.resize((self.image_size, self.image_size), 
                           Image.BILINEAR)  # [224, 224, 3]
        
        # Convert to numpy array and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0  # [224, 224, 3]
        
        # Normalize using ImageNet statistics
        # I_norm = (I - μ) / σ
        image = (image - self.mean) / self.std  # [224, 224, 3]
        
        # Convert to PyTorch tensor [C, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # [3, 224, 224]
        
        return image_tensor


class GazePreprocessor:
    """
    Processes gaze tracking data into two representations:
    1. Spatial heatmap: H(x,y) showing fixation density, downsampled to ViT patch grid
    2. Temporal sequence: [(x₁,y₁,t₁), (x₂,y₂,t₂), ..., (xₙ,yₙ,tₙ)]
    
    Mathematical formulation:
    
    Heatmap generation:
        H(x,y) = Σᵢ G(x,y; xᵢ,yᵢ,σ)
        where G is a 2D Gaussian: G(x,y; μₓ,μᵧ,σ) = exp(-(x-μₓ)²+(y-μᵧ)²)/(2σ²))
    
        The full-resolution heatmap [224, 224] is then downsampled to the ViT patch
        grid [14, 14] by averaging each 16×16 patch region, then flattened to [196].
        This aligns one heatmap score per ViT patch token.
    
    Sequence encoding:
        S = [(x₁,y₁,d₁), ..., (xₙ,yₙ,dₙ)]
        where dᵢ = fixation duration in milliseconds
    """
    
    def __init__(self, image_size=224, heatmap_size=224, 
                 fixation_sigma=10, max_fixations=50):
        self.image_size     = image_size
        self.heatmap_size   = heatmap_size
        self.fixation_sigma = fixation_sigma
        self.max_fixations  = max_fixations
        
        # ViT-Base patch grid: 224 / 16 = 14 → 14×14 = 196 patch tokens
        self.patch_size      = 16
        self.num_patches_1d  = image_size // self.patch_size  # 14
        self.num_patches     = self.num_patches_1d ** 2       # 196
    
    def create_heatmap(self, fixations):
        """
        Generate spatial heatmap from fixation points, aligned to ViT patch grid.
        
        Args:
            fixations: List of dicts with keys ['x', 'y', 'duration']
                      x, y are normalized coordinates in [0, 1]
                      
        Returns:
            heatmap_patches: Tensor [196] — one attention score per ViT patch token
        
        Process:
            1. Build full-resolution heatmap [224, 224]
            2. Apply Gaussian smoothing
            3. Normalize to [0, 1]
            4. Downsample to patch grid [14, 14] by averaging 16×16 blocks
            5. Flatten to [196]
        """
        # Step 1: Build full-resolution heatmap [224, 224]
        heatmap = np.zeros((self.heatmap_size, self.heatmap_size), 
                           dtype=np.float32)
        
        if len(fixations) == 0:
            # Return uniform attention if no fixations
            return torch.zeros(self.num_patches, dtype=torch.float32)  # [196]
        
        for fix in fixations:
            # Normalized [0,1] → pixel coordinates [0, 224]
            px = int(fix['x'] * self.heatmap_size)
            py = int(fix['y'] * self.heatmap_size)
            
            px = np.clip(px, 0, self.heatmap_size - 1)
            py = np.clip(py, 0, self.heatmap_size - 1)
            
            # Weight by fixation duration (longer = more salient)
            weight = fix['duration'] / 1000.0  # ms → seconds
            heatmap[py, px] += weight
        
        # Step 2: Gaussian smoothing: H_smooth = G_σ * H
        heatmap = gaussian_filter(heatmap, sigma=self.fixation_sigma)
        
        # Step 3: Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Step 4: Downsample [224, 224] → [14, 14] by averaging 16×16 patch blocks
        # Reshape to (14, 16, 14, 16) then mean over the patch dimensions
        heatmap_patches = heatmap.reshape(
            self.num_patches_1d, self.patch_size,
            self.num_patches_1d, self.patch_size
        ).mean(axis=(1, 3))  # [14, 14]
        
        # Step 5: Flatten to [196]
        heatmap_patches = heatmap_patches.flatten()  # [196]
        
        return torch.from_numpy(heatmap_patches).float()  # [196]
    
    def create_sequence(self, fixations):
        """
        Create temporal sequence representation.
        
        Args:
            fixations: List of dicts with keys ['x', 'y', 'duration']
            
        Returns:
            sequence: Tensor [max_fixations=50, 3] 
                     Each row: [x_normalized, y_normalized, duration_normalized]
            seq_length: Actual number of valid fixations
        """
        num_fixations = len(fixations)
        
        # Initialize sequence tensor with zeros (padding)
        sequence = np.zeros((self.max_fixations, 3), dtype=np.float32)  # [50, 3]
        
        # Fill in actual fixations
        actual_length = min(num_fixations, self.max_fixations)
        
        for i in range(actual_length):
            fix = fixations[i]
            sequence[i, 0] = fix['x']  # Normalized x ∈ [0, 1]
            sequence[i, 1] = fix['y']  # Normalized y ∈ [0, 1]
            
            # Normalize duration: d_norm = log(1 + d) / log(1 + d_max)
            # (log transform handles large variations)
            duration_normalized = np.log(1 + fix['duration']) / np.log(1 + 5000)
            sequence[i, 2] = duration_normalized  # ∈ [0, ~1]
        
        return torch.from_numpy(sequence), actual_length  # [50, 3], scalar
    
    def __call__(self, fixations):
        """
        Complete gaze preprocessing.
        
        Args:
            fixations: List of fixation dictionaries
            
        Returns:
            gaze_heatmap: Tensor [196]  — patch-aligned gaze attention scores
            gaze_sequence: Tensor [50, 3]
            seq_length: int
        """
        heatmap           = self.create_heatmap(fixations)           # [196]
        sequence, seq_length = self.create_sequence(fixations)       # [50, 3], scalar
        
        return heatmap, sequence, seq_length


class TextPreprocessor:
    """
    Preprocesses clinical radiology reports for BERT-based text encoding.
    
    Uses BioClinicalBERT tokenizer optimized for clinical text.
    
    Process:
        1. Load cleaned report text
        2. Tokenize using BioClinicalBERT tokenizer
        3. Pad/truncate to max_length
        4. Create attention masks
        
    Output:
        - token_ids: [max_length] - BERT input token IDs
        - attention_mask: [max_length] - 1 for real tokens, 0 for padding
    """
    
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', max_length=128):
        """
        Args:
            model_name: HuggingFace model identifier for tokenizer
            max_length: Maximum sequence length (default 128 tokens)
        """
        self.max_length = max_length
        
        # Load BioClinicalBERT tokenizer
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Initialized TextPreprocessor")
        print(f"  Max length: {max_length} tokens")
        print(f"  Tokenizer vocab size: {len(self.tokenizer)}")
    
    def __call__(self, text):
        """
        Tokenize clinical text.
        
        Args:
            text: Raw or cleaned report text (string)
            
        Returns:
            token_ids: Tensor [max_length] - tokenized input IDs
            attention_mask: Tensor [max_length] - attention mask
        """
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # Add [CLS] and [SEP]
            max_length=self.max_length,
            padding='max_length',          # Pad to max_length
            truncation=True,               # Truncate if longer
            return_tensors='pt'            # Return PyTorch tensors
        )
        
        # Extract token IDs and attention mask
        # Shape: [1, max_length] → squeeze to [max_length]
        token_ids      = encoding['input_ids'].squeeze(0)       # [max_length]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [max_length]
        
        return token_ids, attention_mask
    
    def decode(self, token_ids):
        """
        Decode token IDs back to text (useful for debugging).
        
        Args:
            token_ids: Tensor [max_length]
            
        Returns:
            text: Decoded string
        """
        # Remove padding tokens (token_id = 0)
        # Convert to list if tensor
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text


class MedGemmaTextPreprocessor:
    """
    Preprocesses clinical text using MedGemma's tokenizer (Gemma-3 based).

    Used when config['model']['use_medgemma'] is True. Replaces BioClinicalBERT
    as the text encoder input preprocessor. Interface is identical to
    TextPreprocessor so dataset.py requires no structural changes.

    Key differences vs TextPreprocessor:
    - Tokenizer: Gemma-3 SentencePiece (vocab ~256k) vs BERT WordPiece (~30k)
    - No [CLS]/[SEP] tokens — Gemma uses BOS/EOS instead
    - Default max_length increased to 256 to match Gemma's longer context handling
      of radiologist transcriptions (avg ~150 chars, some up to 500+)
    - padding_side: left (Gemma convention for decoder-style models)

    Output shapes are identical to TextPreprocessor:
        - token_ids:      Tensor [max_length]
        - attention_mask: Tensor [max_length]
    """

    def __init__(self, model_name='google/medgemma-4b-it', max_length=256):
        """
        Args:
            model_name: HuggingFace MedGemma model identifier
            max_length: Maximum token sequence length (default 256)
        """
        token = os.environ.get('HF_TOKEN')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        self.max_length = max_length

        print(f"Loading MedGemma tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Gemma tokenizer has no default pad token — set to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Left-padding: for decoder models, padding on the left keeps
        # the actual tokens contiguous at the right end of the sequence
        self.tokenizer.padding_side = 'left'

        print(f"Initialized MedGemmaTextPreprocessor")
        print(f"  Max length: {max_length} tokens")
        print(f"  Tokenizer vocab size: {len(self.tokenizer)}")
        print(f"  Pad token: {self.tokenizer.pad_token}")

    def __call__(self, text):
        """
        Tokenize clinical text using Gemma tokenizer.

        Args:
            text: Raw or cleaned transcription string

        Returns:
            token_ids:      Tensor [max_length]
            attention_mask: Tensor [max_length]
        """
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,   # Adds BOS token (Gemma convention)
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        token_ids      = encoding['input_ids'].squeeze(0)       # [max_length]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [max_length]

        return token_ids, attention_mask

    def decode(self, token_ids):
        """
        Decode token IDs back to text (useful for debugging generation output).

        Args:
            token_ids: Tensor [max_length]

        Returns:
            text: Decoded string
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text