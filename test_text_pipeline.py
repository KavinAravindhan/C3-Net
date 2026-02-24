"""
Test script to verify full multimodal data pipeline (Image + Gaze + Text).
"""

import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import MIMICEyeDataset, collate_fn


def test_text_pipeline():
    """
    Test loading data with all three modalities.
    """
    print("="*80)
    print("Testing Full Multimodal Data Pipeline: Image + Gaze + Text")
    print("="*80)
    
    # Load config
    config_path = 'configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n1. Loading dataset...")
    
    # Create dataset (train split for testing)
    dataset = MIMICEyeDataset(
        root_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye',
        split='train',
        config=config
    )
    
    print(f"\n✓ Dataset loaded: {len(dataset)} samples")
    
    # Create dataloader
    print("\n2. Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    print(f"✓ DataLoader created with batch_size=4")
    
    # Get one batch
    print("\n3. Loading one batch...")
    batch = next(iter(dataloader))
    
    print(f"\n✓ Batch loaded successfully!")
    
    # Print shapes
    print("\n" + "="*80)
    print("BATCH TENSOR SHAPES")
    print("="*80)
    
    print(f"\nImages:                {batch['images'].shape}")
    print(f"  Expected:            [4, 3, 224, 224]")
    print(f"  ✓ Match!" if batch['images'].shape == torch.Size([4, 3, 224, 224]) else "  ✗ Mismatch!")
    
    print(f"\nGaze Heatmaps:         {batch['gaze_heatmaps'].shape}")
    print(f"  Expected:            [4, 224, 224]")
    print(f"  ✓ Match!" if batch['gaze_heatmaps'].shape == torch.Size([4, 224, 224]) else "  ✗ Mismatch!")
    
    print(f"\nGaze Sequences:        {batch['gaze_sequences'].shape}")
    print(f"  Expected:            [4, 50, 3]")
    print(f"  ✓ Match!" if batch['gaze_sequences'].shape == torch.Size([4, 50, 3]) else "  ✗ Mismatch!")
    
    print(f"\nGaze Seq Lengths:      {batch['gaze_seq_lengths'].shape}")
    print(f"  Expected:            [4]")
    print(f"  ✓ Match!" if batch['gaze_seq_lengths'].shape == torch.Size([4]) else "  ✗ Mismatch!")
    
    print(f"\nText Token IDs:        {batch['text_token_ids'].shape}")
    print(f"  Expected:            [4, 128]")
    print(f"  ✓ Match!" if batch['text_token_ids'].shape == torch.Size([4, 128]) else "  ✗ Mismatch!")
    
    print(f"\nText Attention Masks:  {batch['text_attention_masks'].shape}")
    print(f"  Expected:            [4, 128]")
    print(f"  ✓ Match!" if batch['text_attention_masks'].shape == torch.Size([4, 128]) else "  ✗ Mismatch!")
    
    print(f"\nLabels:                {batch['labels'].shape}")
    print(f"  Expected:            [4]")
    print(f"  ✓ Match!" if batch['labels'].shape == torch.Size([4]) else "  ✗ Mismatch!")
    
    print(f"\nSample IDs:            {len(batch['sample_ids'])} items")
    print(f"  Expected:            4")
    print(f"  ✓ Match!" if len(batch['sample_ids']) == 4 else "  ✗ Mismatch!")
    
    # Print sample statistics
    print("\n" + "="*80)
    print("SAMPLE STATISTICS")
    print("="*80)
    
    print(f"\nImage pixel values:")
    print(f"  Min:  {batch['images'].min().item():.3f}")
    print(f"  Max:  {batch['images'].max().item():.3f}")
    print(f"  Mean: {batch['images'].mean().item():.3f}")
    
    print(f"\nGaze sequence lengths: {batch['gaze_seq_lengths'].tolist()}")
    
    print(f"\nLabels: {batch['labels'].tolist()}")
    print(f"  Normal (0):   {(batch['labels'] == 0).sum().item()}")
    print(f"  Abnormal (1): {(batch['labels'] == 1).sum().item()}")
    
    print(f"\nText token statistics:")
    print(f"  Non-padding tokens per sample (first 4):")
    for i in range(min(4, len(batch['text_attention_masks']))):
        num_tokens = batch['text_attention_masks'][i].sum().item()
        print(f"    Sample {i}: {num_tokens} tokens")
    
    # Decode one text sample to verify
    print("\n" + "="*80)
    print("SAMPLE TEXT (first sample, decoded)")
    print("="*80)
    
    # Get first sample's tokens
    first_tokens = batch['text_token_ids'][0]
    
    # Decode using the tokenizer from dataset
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
    
    all_correct = (
        batch['images'].shape == torch.Size([4, 3, 224, 224]) and
        batch['gaze_heatmaps'].shape == torch.Size([4, 224, 224]) and
        batch['gaze_sequences'].shape == torch.Size([4, 50, 3]) and
        batch['gaze_seq_lengths'].shape == torch.Size([4]) and
        batch['text_token_ids'].shape == torch.Size([4, 128]) and
        batch['text_attention_masks'].shape == torch.Size([4, 128]) and
        batch['labels'].shape == torch.Size([4]) and
        len(batch['sample_ids']) == 4
    )
    
    if all_correct:
        print("\n✅ ALL TESTS PASSED!")
        print("Full multimodal pipeline (Image + Gaze + Text) is working correctly!")
        print("\nNext steps:")
        print("  1. Create text encoder (BioClinicalBERT)")
        print("  2. Implement Level 2 fusion (text-image alignment)")
        print("  3. Build teacher model")
        print("  4. Implement knowledge distillation")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the shapes above.")
    
    print("="*80)


if __name__ == '__main__':
    test_text_pipeline()