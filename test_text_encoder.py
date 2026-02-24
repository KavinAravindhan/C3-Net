"""
Test script to verify TextEncoder (BioClinicalBERT).
"""

import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import MIMICEyeDataset, collate_fn
from models.encoders import TextEncoder


def test_text_encoder():
    """
    Test the TextEncoder with real data from the dataset.
    """
    print("="*80)
    print("Testing TextEncoder (BioClinicalBERT)")
    print("="*80)
    
    # Load config
    config_path = 'configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Loading dataset...")
    
    # Create dataset
    dataset = MIMICEyeDataset(
        root_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye',
        split='train',
        config=config
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Create dataloader
    print("\n2. Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Get one batch
    print("\n3. Loading one batch...")
    batch = next(iter(dataloader))
    
    text_token_ids = batch['text_token_ids']
    text_attention_masks = batch['text_attention_masks']
    
    print(f"✓ Batch loaded")
    print(f"  Text token IDs shape: {text_token_ids.shape}")
    print(f"  Text attention masks shape: {text_attention_masks.shape}")
    
    # Initialize TextEncoder
    print("\n4. Initializing TextEncoder...")
    text_encoder = TextEncoder(
        model_name='emilyalsentzer/Bio_ClinicalBERT',
        freeze_bert=True  # Freeze initially
    )
    
    print(f"\n✓ TextEncoder initialized")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n5. Moving to device: {device}")
    
    text_encoder = text_encoder.to(device)
    text_token_ids = text_token_ids.to(device)
    text_attention_masks = text_attention_masks.to(device)
    
    # Forward pass
    print("\n6. Running forward pass...")
    
    with torch.no_grad():  # No gradients needed for testing
        token_embeddings, cls_token = text_encoder(
            text_token_ids,
            text_attention_masks
        )
    
    print(f"✓ Forward pass complete")
    
    # Print shapes
    print("\n" + "="*80)
    print("OUTPUT TENSOR SHAPES")
    print("="*80)
    
    print(f"\nToken Embeddings:      {token_embeddings.shape}")
    print(f"  Expected:            [4, 128, 768]")
    print(f"  ✓ Match!" if token_embeddings.shape == torch.Size([4, 128, 768]) else "  ✗ Mismatch!")
    
    print(f"\nCLS Token:             {cls_token.shape}")
    print(f"  Expected:            [4, 768]")
    print(f"  ✓ Match!" if cls_token.shape == torch.Size([4, 768]) else "  ✗ Mismatch!")
    
    # Print statistics
    print("\n" + "="*80)
    print("OUTPUT STATISTICS")
    print("="*80)
    
    print(f"\nToken Embeddings:")
    print(f"  Min:  {token_embeddings.min().item():.4f}")
    print(f"  Max:  {token_embeddings.max().item():.4f}")
    print(f"  Mean: {token_embeddings.mean().item():.4f}")
    print(f"  Std:  {token_embeddings.std().item():.4f}")
    
    print(f"\nCLS Token:")
    print(f"  Min:  {cls_token.min().item():.4f}")
    print(f"  Max:  {cls_token.max().item():.4f}")
    print(f"  Mean: {cls_token.mean().item():.4f}")
    print(f"  Std:  {cls_token.std().item():.4f}")
    
    # Test freeze/unfreeze
    print("\n" + "="*80)
    print("TESTING FREEZE/UNFREEZE")
    print("="*80)
    
    print("\nInitial state: Frozen =", text_encoder.frozen)
    
    # Count trainable parameters
    trainable_before = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in text_encoder.parameters())
    
    print(f"Trainable parameters: {trainable_before:,} / {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_before / total_params:.2f}%")
    
    # Unfreeze
    print("\nUnfreezing BERT...")
    text_encoder.unfreeze()
    
    trainable_after = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    
    print(f"Trainable parameters: {trainable_after:,} / {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_after / total_params:.2f}%")
    
    # Freeze again
    print("\nFreezing BERT again...")
    text_encoder.freeze()
    
    trainable_refrozen = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    
    print(f"Trainable parameters: {trainable_refrozen:,} / {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_refrozen / total_params:.2f}%")
    
    # Decode one sample to show what BERT sees
    print("\n" + "="*80)
    print("SAMPLE TEXT (first sample, decoded)")
    print("="*80)
    
    first_tokens = batch['text_token_ids'][0].cpu()
    decoded_text = dataset.text_preprocessor.decode(first_tokens)
    
    print(f"\nDecoded text preview (first 300 chars):")
    print("-"*80)
    print(decoded_text[:300])
    if len(decoded_text) > 300:
        print("...")
    print("-"*80)
    
    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    shapes_correct = (
        token_embeddings.shape == torch.Size([4, 128, 768]) and
        cls_token.shape == torch.Size([4, 768])
    )
    
    freeze_works = (
        trainable_before == 0 and
        trainable_after == total_params and
        trainable_refrozen == 0
    )
    
    if shapes_correct and freeze_works:
        print("\n✅ ALL TESTS PASSED!")
        print("\nTextEncoder is working correctly!")
        print(f"  - Token embeddings: [B, 128, 768] ✓")
        print(f"  - CLS token: [B, 768] ✓")
        print(f"  - Freeze/unfreeze mechanism: ✓")
        print(f"  - Total parameters: {total_params:,}")
        print("\nNext steps:")
        print("  1. Create Level 2 fusion (text-image alignment)")
        print("  2. Build teacher model")
        print("  3. Implement knowledge distillation")
    else:
        print("\n❌ SOME TESTS FAILED!")
        if not shapes_correct:
            print("  - Output shapes are incorrect")
        if not freeze_works:
            print("  - Freeze/unfreeze mechanism is not working")
    
    print("="*80)


if __name__ == '__main__':
    test_text_encoder()