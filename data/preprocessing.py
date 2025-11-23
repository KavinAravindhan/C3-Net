import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image
import cv2

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
    1. Spatial heatmap: H(x,y) showing fixation density
    2. Temporal sequence: [(x₁,y₁,t₁), (x₂,y₂,t₂), ..., (xₙ,yₙ,tₙ)]
    
    Mathematical formulation:
    
    Heatmap generation:
        H(x,y) = Σᵢ G(x,y; xᵢ,yᵢ,σ)
        where G is a 2D Gaussian: G(x,y; μₓ,μᵧ,σ) = exp(-(x-μₓ)²+(y-μᵧ)²)/(2σ²))
    
    Sequence encoding:
        S = [(x₁,y₁,d₁), ..., (xₙ,yₙ,dₙ)]
        where dᵢ = fixation duration in milliseconds
    """
    
    def __init__(self, image_size=224, heatmap_size=224, 
                 fixation_sigma=10, max_fixations=50):
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.fixation_sigma = fixation_sigma
        self.max_fixations = max_fixations
    
    def create_heatmap(self, fixations):
        """
        Generate spatial heatmap from fixation points.
        
        Args:
            fixations: List of dicts with keys ['x', 'y', 'duration']
                      x, y are normalized coordinates in [0, 1]
                      
        Returns:
            heatmap: Tensor [H=224, W=224] representing fixation density
        """
        # Initialize empty heatmap
        heatmap = np.zeros((self.heatmap_size, self.heatmap_size), 
                          dtype=np.float32)  # [224, 224]
        
        if len(fixations) == 0:
            return torch.from_numpy(heatmap)  # Return zeros if no fixations
        
        # Convert normalized coordinates to pixel coordinates
        for fix in fixations:
            # x, y in [0, 1] → pixel coordinates in [0, 224]
            px = int(fix['x'] * self.heatmap_size)
            py = int(fix['y'] * self.heatmap_size)
            
            # Clip to valid range
            px = np.clip(px, 0, self.heatmap_size - 1)
            py = np.clip(py, 0, self.heatmap_size - 1)
            
            # Weight by duration (longer fixations = more salient)
            weight = fix['duration'] / 1000.0  # Convert ms to seconds
            
            # Add weighted point
            heatmap[py, px] += weight
        
        # Apply Gaussian smoothing: H_smooth = G_σ * H
        heatmap = gaussian_filter(heatmap, sigma=self.fixation_sigma)  # [224, 224]
        
        # Normalize to [0, 1]: H_norm = (H - min(H)) / (max(H) - min(H))
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return torch.from_numpy(heatmap)  # [224, 224]
    
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
            gaze_heatmap: Tensor [224, 224]
            gaze_sequence: Tensor [50, 3]
            seq_length: int
        """
        heatmap = self.create_heatmap(fixations)  # [224, 224]
        sequence, seq_length = self.create_sequence(fixations)  # [50, 3], scalar
        
        return heatmap, sequence, seq_length