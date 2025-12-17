import torch
from torch.utils.data import DataLoader
import yaml
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import MIMICEyeDataset, collate_fn
from models.encoders import ImageEncoder, GazeEncoder
from models.attention import GazeGuidedFusion


def test_level1_fusion():
    """
    Test Level 1: Gaze-Guided Visual Attention
    """
    print("="*80)
    print("Testing Level 1: Gaze-Guided Visual Attention")
    print("="*80)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize dataset
    print("\n1. Loading dataset...")
    dataset = MIMICEyeDataset(
        root_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye',
        split='train',
        config=config
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"  Loaded batch with {len(batch['images'])} samples")
    
    # Initialize models
    print("\n2. Initializing models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image_encoder = ImageEncoder(
        model_name='vit_base_patch16_224',
        pretrained=True
    ).to(device)
    
    gaze_encoder = GazeEncoder(
        spatial_hidden_dim=256,
        temporal_hidden_dim=256
    ).to(device)
    
    # NEW: Level 1 Fusion Module
    gaze_fusion = GazeGuidedFusion(
        image_dim=768,
        gaze_dim=512,
        num_heads=8
    ).to(device)
    
    # Move batch to device
    images = batch['images'].to(device)
    gaze_heatmaps = batch['gaze_heatmaps'].to(device)
    gaze_sequences = batch['gaze_sequences'].to(device)
    gaze_seq_lengths = batch['gaze_seq_lengths']
    
    print("\n3. Running forward pass...")
    
    with torch.no_grad():
        # Encode image
        patch_features, cls_token = image_encoder(images)
        print(f"  ✓ Image encoded: patches {patch_features.shape}, CLS {cls_token.shape}")
        
        # Encode gaze
        gaze_weights, gaze_features = gaze_encoder(
            gaze_heatmaps, 
            gaze_sequences, 
            gaze_seq_lengths
        )
        print(f"  ✓ Gaze encoded: weights {gaze_weights.shape}, features {gaze_features.shape}")
        
        # Level 1 Fusion: Gaze-Guided Attention
        fused_features, attention_map = gaze_fusion(
            patch_features,
            cls_token,
            gaze_weights,
            gaze_features
        )
        print(f"  ✓ Fusion complete: fused {fused_features.shape}, attention {attention_map.shape}")
    
    print("\n4. Analyzing attention patterns...")
    
    # Compare gaze weights vs learned attention
    gaze_weights_flat = gaze_weights.squeeze(-1).cpu().numpy()  # [B, 196]
    learned_attention = attention_map.cpu().numpy()  # [B, 196]
    
    for i in range(min(2, len(images))):
        correlation = np.corrcoef(gaze_weights_flat[i], learned_attention[i])[0, 1]
        print(f"  Sample {i}: Correlation between gaze and learned attention = {correlation:.3f}")
    
    print("\n5. Visualizing attention...")
    visualize_attention(batch, gaze_weights, attention_map, fused_features, idx=0)
    
    print("\n" + "="*80)
    print("✓ Level 1 Fusion Test Complete!")
    print("="*80)
    print("\nKey Insights:")
    print("  • Gaze weights guide the attention mechanism")
    print("  • Learned attention can deviate to learn optimal patterns")
    print("  • Fused features combine raw image + gaze-guided attention")
    print("  • Ready for downstream classification!")


def visualize_attention(batch, gaze_weights, attention_map, fused_features, idx=0):
    """
    Visualize the attention patterns from Level 1 fusion.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get sample
    img = batch['images'][idx].cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    gaze_heatmap = batch['gaze_heatmaps'][idx].cpu().numpy()
    gaze_weights_map = gaze_weights[idx].squeeze(-1).cpu().numpy().reshape(14, 14)
    learned_attention_map = attention_map[idx].cpu().numpy().reshape(14, 14)
    
    # Upsample to image size for visualization
    from scipy.ndimage import zoom
    gaze_weights_upsampled = zoom(gaze_weights_map, (224/14, 224/14))
    learned_upsampled = zoom(learned_attention_map, (224/14, 224/14))
    
    # Row 1: Input visualizations
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img)
    axes[0, 1].imshow(gaze_heatmap, alpha=0.5, cmap='hot')
    axes[0, 1].set_title('Ground Truth Gaze Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img)
    axes[0, 2].imshow(gaze_weights_upsampled, alpha=0.5, cmap='hot')
    axes[0, 2].set_title('Gaze Weights (from Encoder)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Fusion outputs
    axes[1, 0].imshow(img)
    axes[1, 0].imshow(learned_upsampled, alpha=0.5, cmap='viridis')
    axes[1, 0].set_title('Learned Attention (Level 1)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Difference map
    diff = learned_upsampled - gaze_weights_upsampled
    im = axes[1, 1].imshow(diff, cmap='RdBu_r', vmin=-diff.max(), vmax=diff.max())
    axes[1, 1].set_title('Difference (Learned - Gaze)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Feature magnitude visualization
    fused_magnitude = torch.norm(fused_features[idx], p=2).item()
    axes[1, 2].bar(['Fused Features'], [fused_magnitude], color='green', alpha=0.7)
    axes[1, 2].set_ylabel('L2 Norm', fontsize=10)
    axes[1, 2].set_title(f'Fused Feature Magnitude\n(L2 = {fused_magnitude:.2f})', 
                        fontsize=12, fontweight='bold')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('level1_fusion_visualization.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'level1_fusion_visualization.png'")
    plt.close()


if __name__ == '__main__':
    test_level1_fusion()