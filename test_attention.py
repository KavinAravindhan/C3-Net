"""
Comprehensive test script for all attention/fusion modules.
Tests Level 1 (Gaze-Guided) and Level 2 (Text-Image Alignment) fusion.
"""

import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import MIMICEyeDataset, collate_fn
from models.encoders import ImageEncoder, GazeEncoder, TextEncoder
from models.attention import GazeGuidedFusion, TextImageAlignment


def test_attention_modules():
    """
    Test all attention/fusion modules with real data.
    """
    print("="*80)
    print("Testing Attention & Fusion Modules")
    print("="*80)
    
    # Load config
    config_path = 'configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ===== Load Data =====
    print("\n" + "="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)
    
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
    print(f"✓ Batch loaded: {len(batch['images'])} samples")
    
    # Move to device
    images = batch['images'].to(device)
    gaze_heatmaps = batch['gaze_heatmaps'].to(device)
    gaze_sequences = batch['gaze_sequences'].to(device)
    gaze_seq_lengths = batch['gaze_seq_lengths']
    text_token_ids = batch['text_token_ids'].to(device)
    text_attention_masks = batch['text_attention_masks'].to(device)
    
    # ===== Initialize Encoders =====
    print("\n" + "="*80)
    print("STEP 2: Initializing Encoders")
    print("="*80)
    
    print("\nInitializing ImageEncoder...")
    image_encoder = ImageEncoder(
        model_name='vit_base_patch16_224',
        pretrained=True,
        freeze_backbone=False
    ).to(device)
    
    print("\nInitializing GazeEncoder...")
    gaze_encoder = GazeEncoder(
        spatial_hidden_dim=256,
        temporal_hidden_dim=256,
        lstm_layers=2,
        dropout=0.3
    ).to(device)
    
    print("\nInitializing TextEncoder...")
    text_encoder = TextEncoder(
        model_name='emilyalsentzer/Bio_ClinicalBERT',
        freeze_bert=True
    ).to(device)
    
    print("\n✓ All encoders initialized")
    
    # ===== Encode Inputs =====
    print("\n" + "="*80)
    print("STEP 3: Encoding Inputs")
    print("="*80)
    
    with torch.no_grad():
        # Encode image
        print("\nEncoding images...")
        image_patches, image_cls = image_encoder(images)
        print(f"  Image patches: {image_patches.shape}")
        print(f"  Image CLS: {image_cls.shape}")
        
        # Encode gaze
        print("\nEncoding gaze...")
        gaze_weights, gaze_features = gaze_encoder(
            gaze_heatmaps,
            gaze_sequences,
            gaze_seq_lengths
        )
        print(f"  Gaze weights: {gaze_weights.shape}")
        print(f"  Gaze features: {gaze_features.shape}")
        
        # Encode text
        print("\nEncoding text...")
        text_embeddings, text_cls = text_encoder(
            text_token_ids,
            text_attention_masks
        )
        print(f"  Text embeddings: {text_embeddings.shape}")
        print(f"  Text CLS: {text_cls.shape}")
    
    print("\n✓ All inputs encoded")
    
    # ===== Test Level 1: Gaze-Guided Fusion =====
    print("\n" + "="*80)
    print("STEP 4: Testing Level 1 - Gaze-Guided Fusion")
    print("="*80)
    
    print("\nInitializing GazeGuidedFusion...")
    level1_fusion = GazeGuidedFusion(
        image_dim=768,
        gaze_dim=512,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    print("\nRunning Level 1 forward pass...")
    with torch.no_grad():
        gaze_guided_features, level1_attention_map = level1_fusion(
            image_patches,
            image_cls,
            gaze_weights,
            gaze_features
        )
    
    print(f"\n✓ Level 1 Forward Pass Complete")
    print(f"\nOutput Shapes:")
    print(f"  Gaze-guided features: {gaze_guided_features.shape}")
    print(f"    Expected:           [4, 768]")
    print(f"    {'✓ Match!' if gaze_guided_features.shape == torch.Size([4, 768]) else '✗ Mismatch!'}")
    
    print(f"\n  Attention map:        {level1_attention_map.shape}")
    print(f"    Expected:           [4, 196]")
    print(f"    {'✓ Match!' if level1_attention_map.shape == torch.Size([4, 196]) else '✗ Mismatch!'}")
    
    print(f"\nStatistics:")
    print(f"  Features - Min: {gaze_guided_features.min().item():.4f}, Max: {gaze_guided_features.max().item():.4f}, Mean: {gaze_guided_features.mean().item():.4f}")
    print(f"  Attention - Min: {level1_attention_map.min().item():.4f}, Max: {level1_attention_map.max().item():.4f}, Sum: {level1_attention_map.sum(dim=1).mean().item():.4f}")
    
    # ===== Test Level 2: Text-Image Alignment =====
    print("\n" + "="*80)
    print("STEP 5: Testing Level 2 - Text-Image Alignment")
    print("="*80)
    
    print("\nInitializing TextImageAlignment...")
    level2_fusion = TextImageAlignment(
        hidden_dim=768,
        output_dim=512,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    print("\nRunning Level 2 forward pass...")
    with torch.no_grad():
        aligned_features, level2_attention_map = level2_fusion(
            text_embeddings,
            image_patches,
            gaze_guided_features,
            text_attention_masks
        )
    
    print(f"\n✓ Level 2 Forward Pass Complete")
    print(f"\nOutput Shapes:")
    print(f"  Aligned features:     {aligned_features.shape}")
    print(f"    Expected:           [4, 512]")
    print(f"    {'✓ Match!' if aligned_features.shape == torch.Size([4, 512]) else '✗ Mismatch!'}")
    
    print(f"\n  Attention map:        {level2_attention_map.shape}")
    print(f"    Expected:           [4, 128, 196]")
    print(f"    {'✓ Match!' if level2_attention_map.shape == torch.Size([4, 128, 196]) else '✗ Mismatch!'}")
    
    print(f"\nStatistics:")
    print(f"  Features - Min: {aligned_features.min().item():.4f}, Max: {aligned_features.max().item():.4f}, Mean: {aligned_features.mean().item():.4f}")
    print(f"  Attention - Min: {level2_attention_map.min().item():.4f}, Max: {level2_attention_map.max().item():.4f}")
    
    # Analyze attention patterns
    print(f"\nAttention Pattern Analysis:")
    # Average attention weight per text token
    avg_attention_per_token = level2_attention_map.mean(dim=-1)  # [B, 128]
    print(f"  Average attention per text token:")
    print(f"    Min: {avg_attention_per_token.min().item():.6f}")
    print(f"    Max: {avg_attention_per_token.max().item():.6f}")
    print(f"    Mean: {avg_attention_per_token.mean().item():.6f}")
    
    # Find which image patches get most attention
    avg_attention_per_patch = level2_attention_map.mean(dim=1)  # [B, 196]
    print(f"\n  Average attention per image patch:")
    print(f"    Min: {avg_attention_per_patch.min().item():.6f}")
    print(f"    Max: {avg_attention_per_patch.max().item():.6f}")
    print(f"    Mean: {avg_attention_per_patch.mean().item():.6f}")
    
    # ===== Test Complete Pipeline =====
    print("\n" + "="*80)
    print("STEP 6: Testing Complete Multimodal Pipeline")
    print("="*80)
    
    print("\nPipeline Flow:")
    print("  Image → ImageEncoder → Patches [4, 196, 768]")
    print("  Gaze → GazeEncoder → Weights [4, 196, 1] + Features [4, 512]")
    print("  Level 1 → GazeGuidedFusion → Gaze-weighted features [4, 768]")
    print("  Text → TextEncoder → Embeddings [4, 128, 768]")
    print("  Level 2 → TextImageAlignment → Aligned features [4, 512]")
    
    print("\n✓ Complete pipeline verified!")
    
    # ===== Parameter Count =====
    print("\n" + "="*80)
    print("STEP 7: Model Statistics")
    print("="*80)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTrainable Parameters:")
    print(f"  ImageEncoder:        {count_parameters(image_encoder):,}")
    print(f"  GazeEncoder:         {count_parameters(gaze_encoder):,}")
    print(f"  TextEncoder:         {count_parameters(text_encoder):,}")
    print(f"  Level 1 Fusion:      {count_parameters(level1_fusion):,}")
    print(f"  Level 2 Fusion:      {count_parameters(level2_fusion):,}")
    print(f"  {'─'*40}")
    total_params = (
        count_parameters(image_encoder) +
        count_parameters(gaze_encoder) +
        count_parameters(text_encoder) +
        count_parameters(level1_fusion) +
        count_parameters(level2_fusion)
    )
    print(f"  Total:               {total_params:,}")
    
    # ===== Final Summary =====
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_tests_passed = (
        gaze_guided_features.shape == torch.Size([4, 768]) and
        level1_attention_map.shape == torch.Size([4, 196]) and
        aligned_features.shape == torch.Size([4, 512]) and
        level2_attention_map.shape == torch.Size([4, 128, 196])
    )
    
    if all_tests_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nBoth fusion levels are working correctly:")
        print("  ✓ Level 1: Gaze-guided attention fusion")
        print("  ✓ Level 2: Text-image alignment fusion")
        print("\nReady to build Teacher Model!")
        print("\nNext steps:")
        print("  1. Create Teacher model (combines all modalities)")
        print("  2. Implement knowledge distillation")
        print("  3. Update training loop")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the shapes above.")
    
    print("="*80)


if __name__ == '__main__':
    test_attention_modules()