"""
Test script to verify MultimodalTeacher forward pass.
"""

import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import MIMICEyeDataset, collate_fn
from models.teacher import MultimodalTeacher


def test_teacher():
    print("="*80)
    print("Testing MultimodalTeacher")
    print("="*80)

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

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
    images               = batch['images'].to(device)
    gaze_heatmaps        = batch['gaze_heatmaps'].to(device)
    gaze_sequences       = batch['gaze_sequences'].to(device)
    gaze_seq_lengths     = batch['gaze_seq_lengths']
    text_token_ids       = batch['text_token_ids'].to(device)
    text_attention_masks = batch['text_attention_masks'].to(device)
    labels               = batch['labels'].to(device)

    # ===== Initialize Teacher =====
    print("\n" + "="*80)
    print("STEP 2: Initializing MultimodalTeacher")
    print("="*80)

    teacher = MultimodalTeacher(config=config).to(device)
    print("\n✓ Teacher initialized")

    # ===== Forward Pass =====
    print("\n" + "="*80)
    print("STEP 3: Forward Pass")
    print("="*80)

    with torch.no_grad():
        logits, attention_maps = teacher(
            images,
            gaze_heatmaps,
            gaze_sequences,
            gaze_seq_lengths,
            text_token_ids,
            text_attention_masks
        )

    print(f"\n✓ Forward pass complete")

    # ===== Check Output Shapes =====
    print("\n" + "="*80)
    print("STEP 4: Output Shape Verification")
    print("="*80)

    print(f"\nLogits:              {logits.shape}")
    print(f"  Expected:          [4, 2]")
    print(f"  {'✓ Match!' if logits.shape == torch.Size([4, 2]) else '✗ Mismatch!'}")

    print(f"\nLevel 1 Attention:   {attention_maps['level1'].shape}")
    print(f"  Expected:          [4, 196]")
    print(f"  {'✓ Match!' if attention_maps['level1'].shape == torch.Size([4, 196]) else '✗ Mismatch!'}")

    print(f"\nLevel 2 Attention:   {attention_maps['level2'].shape}")
    print(f"  Expected:          [4, 128, 196]")
    print(f"  {'✓ Match!' if attention_maps['level2'].shape == torch.Size([4, 128, 196]) else '✗ Mismatch!'}")

    # ===== Output Statistics =====
    print("\n" + "="*80)
    print("STEP 5: Output Statistics")
    print("="*80)

    print(f"\nLogits:")
    print(f"  Min:  {logits.min().item():.4f}")
    print(f"  Max:  {logits.max().item():.4f}")
    print(f"  Mean: {logits.mean().item():.4f}")

    probs = torch.softmax(logits, dim=1)
    print(f"\nSoftmax Probabilities:")
    for i in range(len(labels)):
        pred = probs[i].argmax().item()
        label = labels[i].item()
        print(f"  Sample {i}: P(Normal)={probs[i][0]:.3f}, P(Abnormal)={probs[i][1]:.3f} | "
              f"Pred={pred}, GT={label}")

    # ===== Test Loss Computation =====
    print("\n" + "="*80)
    print("STEP 6: Loss Computation")
    print("="*80)

    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(f"\n✓ Cross-entropy loss: {loss.item():.4f}")
    print(f"  (Expected ~0.69 for random predictions on binary task)")

    # ===== Test Backward Pass =====
    print("\n" + "="*80)
    print("STEP 7: Backward Pass (Gradient Check)")
    print("="*80)

    teacher.train()
    logits_train, _ = teacher(
        images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
        text_token_ids, text_attention_masks
    )
    loss_train = criterion(logits_train, labels)
    loss_train.backward()

    # Check gradients exist on classifier
    has_grads = all(
        p.grad is not None
        for p in teacher.classifier.parameters()
    )
    print(f"\n✓ Backward pass complete")
    print(f"  Classifier has gradients: {'✓ Yes' if has_grads else '✗ No'}")

    # ===== Test Freeze/Unfreeze BERT =====
    print("\n" + "="*80)
    print("STEP 8: BERT Freeze/Unfreeze")
    print("="*80)

    teacher.freeze_bert()
    frozen_params = sum(p.numel() for p in teacher.text_encoder.parameters() if p.requires_grad)
    print(f"  Trainable BERT params after freeze:   {frozen_params:,}")

    teacher.unfreeze_bert()
    unfrozen_params = sum(p.numel() for p in teacher.text_encoder.parameters() if p.requires_grad)
    print(f"  Trainable BERT params after unfreeze: {unfrozen_params:,}")

    # Refreeze for clean state
    teacher.freeze_bert()

    # ===== Parameter Count =====
    print("\n" + "="*80)
    print("STEP 9: Parameter Count")
    print("="*80)

    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def count_total(module):
        return sum(p.numel() for p in module.parameters())

    print(f"\nTrainable Parameters (BERT frozen):")
    print(f"  ImageEncoder:    {count_params(teacher.image_encoder):>12,}")
    print(f"  GazeEncoder:     {count_params(teacher.gaze_encoder):>12,}")
    print(f"  TextEncoder:     {count_params(teacher.text_encoder):>12,}  (frozen)")
    print(f"  Level1 Fusion:   {count_params(teacher.level1_fusion):>12,}")
    print(f"  Level2 Fusion:   {count_params(teacher.level2_fusion):>12,}")
    print(f"  Classifier:      {count_params(teacher.classifier):>12,}")
    print(f"  {'─'*40}")
    print(f"  Total trainable: {count_params(teacher):>12,}")
    print(f"  Total params:    {count_total(teacher):>12,}")

    # ===== Final Summary =====
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = (
        logits.shape == torch.Size([4, 2]) and
        attention_maps['level1'].shape == torch.Size([4, 196]) and
        attention_maps['level2'].shape == torch.Size([4, 128, 196]) and
        has_grads and
        frozen_params == 0 and
        unfrozen_params > 0
    )

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nMultimodalTeacher is working correctly:")
        print("  ✓ Forward pass: logits [B, 2]")
        print("  ✓ Attention maps: level1 [B, 196], level2 [B, 128, 196]")
        print("  ✓ Loss computation")
        print("  ✓ Backward pass with gradients")
        print("  ✓ BERT freeze/unfreeze")
        print("\nReady to update train.py for teacher training!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        if logits.shape != torch.Size([4, 2]):
            print("  ✗ Logits shape incorrect")
        if attention_maps['level1'].shape != torch.Size([4, 196]):
            print("  ✗ Level 1 attention shape incorrect")
        if attention_maps['level2'].shape != torch.Size([4, 128, 196]):
            print("  ✗ Level 2 attention shape incorrect")
        if not has_grads:
            print("  ✗ Gradients not flowing to classifier")
        if frozen_params != 0:
            print("  ✗ BERT freeze not working")
        if unfrozen_params == 0:
            print("  ✗ BERT unfreeze not working")

    print("="*80)


if __name__ == '__main__':
    test_teacher()