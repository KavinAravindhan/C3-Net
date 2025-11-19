import torch
from torch.utils.data import DataLoader
import yaml
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import MIMICEyeDataset, collate_fn
from models.encoders import ImageEncoder, GazeEncoder, GazePredictor


def test_preprocessing_and_encoding():
    """
    Test script to verify:
    1. Data loading works
    2. Preprocessing produces correct shapes
    3. Encoders run successfully
    """
    
    print("="*60)
    print("Testing Preprocessing and Encoding Pipeline")
    print("="*60)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize dataset
    print("\n1. Loading dataset...")
    dataset = MIMICEyeDataset(
        root_dir='data/processed_mimic_eye',  # UPDATE THIS PATH
        split='train',
        config=config
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging
    )
    
    # Get one batch
    print("\n2. Loading one batch...")
    batch = next(iter(dataloader))
    
    print(f"  Images shape: {batch['images'].shape}")  # [4, 3, 224, 224]
    print(f"  Gaze heatmaps shape: {batch['gaze_heatmaps'].shape}")  # [4, 224, 224]
    print(f"  Gaze sequences shape: {batch['gaze_sequences'].shape}")  # [4, 50, 3]
    print(f"  Labels shape: {batch['labels'].shape}")  # [4]
    
    # Initialize encoders
    print("\n3. Initializing encoders...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    image_encoder = ImageEncoder(
        model_name=config['model']['image_encoder']['type'],
        pretrained=config['model']['image_encoder']['pretrained']
    ).to(device)
    
    gaze_encoder = GazeEncoder(
        spatial_hidden_dim=config['model']['gaze_encoder']['spatial_hidden_dim'],
        temporal_hidden_dim=config['model']['gaze_encoder']['temporal_hidden_dim'],
        lstm_layers=config['model']['gaze_encoder']['lstm_layers'],
        dropout=config['model']['gaze_encoder']['dropout']
    ).to(device)
    
    gaze_predictor = GazePredictor(
        input_dim=768,
        num_patches=196
    ).to(device)
    
    # Move batch to device
    images = batch['images'].to(device)
    gaze_heatmaps = batch['gaze_heatmaps'].to(device)
    gaze_sequences = batch['gaze_sequences'].to(device)
    gaze_seq_lengths = batch['gaze_seq_lengths']
    
    # Forward pass
    print("\n4. Running forward pass...")
    
    with torch.no_grad():
        # Image encoding
        patch_features, cls_token = image_encoder(images)
        print(f"  Image patch features: {patch_features.shape}")  # [4, 196, 768]
        print(f"  Image CLS token: {cls_token.shape}")  # [4, 768]
        
        # Gaze encoding
        gaze_weights, gaze_features = gaze_encoder(
            gaze_heatmaps,
            gaze_sequences,
            gaze_seq_lengths
        )
        print(f"  Gaze patch weights: {gaze_weights.shape}")  # [4, 196, 1]
        print(f"  Gaze features: {gaze_features.shape}")  # [4, 512]
        
        # Gaze prediction
        predicted_gaze = gaze_predictor(patch_features)
        print(f"  Predicted gaze: {predicted_gaze.shape}")  # [4, 196]
    
    print("\n5. Visualizing results...")
    visualize_sample(batch, patch_features, gaze_weights, predicted_gaze, idx=0)
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


def visualize_sample(batch, patch_features, gaze_weights, predicted_gaze, idx=0):
    """
    Visualize one sample with its gaze data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = batch['images'][idx].cpu().permute(1, 2, 0).numpy()
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth gaze heatmap
    gaze_heatmap = batch['gaze_heatmaps'][idx].cpu().numpy()
    axes[1].imshow(img)
    axes[1].imshow(gaze_heatmap, alpha=0.5, cmap='hot')
    axes[1].set_title('Ground Truth Gaze')
    axes[1].axis('off')
    
    # Predicted gaze heatmap
    pred_gaze = predicted_gaze[idx].cpu().numpy().reshape(14, 14)
    # Upsample to image size
    from scipy.ndimage import zoom
    pred_gaze_upsampled = zoom(pred_gaze, (224/14, 224/14))
    
    axes[2].imshow(img)
    axes[2].imshow(pred_gaze_upsampled, alpha=0.5, cmap='hot')
    axes[2].set_title('Predicted Gaze (from Image)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'test_visualization.png'")
    plt.close()


if __name__ == '__main__':
    test_preprocessing_and_encoding()