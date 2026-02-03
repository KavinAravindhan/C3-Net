import torch
import torchvision.transforms as T
from PIL import Image
import random


class MedicalImageAugmentation:
    """
    Data augmentation for medical images (X-ray/OCT).
    
    Applies random transformations to improve generalization.
    Only applied during training, not validation/test.
    """
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        
        # Define augmentation pipeline
        self.transform = T.Compose([
            # Random horizontal flip (50% chance)
            T.RandomHorizontalFlip(p=0.5),
            
            # Random rotation (±10 degrees)
            T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BILINEAR),
            
            # Random affine (slight translation and scaling)
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),  # ±5% translation
                scale=(0.95, 1.05),      # ±5% zoom
                interpolation=T.InterpolationMode.BILINEAR
            ),
            
            # Color jitter (brightness, contrast)
            T.ColorJitter(
                brightness=0.2,  # ±20%
                contrast=0.2,    # ±20%
                saturation=0.1,  # ±10%
                hue=0.05         # ±5%
            ),
        ])
        
        print("Initialized MedicalImageAugmentation")
        print("  - Random horizontal flip (p=0.5)")
        print("  - Random rotation (±10°)")
        print("  - Random affine (translate ±5%, scale ±5%)")
        print("  - Color jitter (brightness/contrast ±20%)")
    
    def __call__(self, image):
        """
        Apply augmentation to PIL image.
        
        Args:
            image: PIL Image
            
        Returns:
            augmented_image: PIL Image (same size)
        """
        return self.transform(image)


class NoAugmentation:
    """
    No augmentation - just return the image as-is.
    Used for validation and test sets.
    """
    
    def __call__(self, image):
        return image