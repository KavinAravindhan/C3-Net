"""
visualize.py — Standalone visualization script for C3-Net.

Loads a trained checkpoint and generates publication-quality figures:
    1. Gaze attention overlays (real vs predicted)
    2. Gaze comparison grid across Normal/Abnormal samples
    3. Per-sample attention map export

Usage:
    python visualize.py --config configs/config.yaml \
                        --checkpoint /media/16TB_Storage/kavin/models/c3-net/best_model.pth \
                        --split test \
                        --output_dir outputs/visualizations \
                        --num_samples 16
"""

import os
import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from datetime import datetime

from data.dataset import MIMICEyeDataset, collate_fn
from models.medgemma_model import MedGemmaModel
from models.teacher import MultimodalTeacher
from utils.visualization import (
    save_attention_grid,
    save_gaze_comparison,
    overlay_attention_on_image,
    attention_map_to_heatmap
)


# ==============================================================================
# Loaders
# ==============================================================================

def load_model(config, checkpoint_path, device):
    """
    Load model from checkpoint based on use_medgemma flag.

    Args:
        config:          dict — loaded config.yaml
        checkpoint_path: str — path to .pth checkpoint
        device:          torch.device

    Returns:
        model: MedGemmaModel or MultimodalTeacher, in eval mode
    """
    use_medgemma = config['model'].get('use_medgemma', False)

    if use_medgemma:
        print("Loading MedGemmaModel...")
        model = MedGemmaModel(config=config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded from: {checkpoint_path}")
        print(f"  ✓ Checkpoint epoch: {checkpoint['epoch'] + 1}")
    else:
        print("Loading MultimodalTeacher...")
        model = MultimodalTeacher(config=config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['teacher_state_dict'])
        print(f"  ✓ Loaded from: {checkpoint_path}")
        print(f"  ✓ Checkpoint epoch: {checkpoint['epoch'] + 1}")

    if 'val_metrics' in checkpoint:
        m = checkpoint['val_metrics']
        print(f"  ✓ Val AUC: {m['auc']:.4f}  |  Accuracy: {m['accuracy']*100:.2f}%")

    model.eval()
    return model


def load_dataset(config, split, num_samples):
    """
    Load dataset split and return a DataLoader capped at num_samples.
    """
    dataset = MIMICEyeDataset(
        root_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye',
        split=split,
        config=config
    )
    # Subsample deterministically
    indices = list(range(min(num_samples, len(dataset))))
    subset  = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=min(num_samples, 8),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    print(f"  ✓ Loaded {len(subset)} samples from '{split}' split")
    return loader


# ==============================================================================
# Visualization Routines
# ==============================================================================

def visualize_medgemma(model, loader, device, output_dir, max_samples):
    """
    Generate visualizations for MedGemma pipeline.
    Saves:
        - real_gaze_attention.png     — real gaze overlays
        - predicted_gaze_attention.png — predicted gaze overlays
        - gaze_comparison.png          — real vs predicted side-by-side
        - normal_vs_abnormal.png       — balanced Normal/Abnormal samples
    """
    all_images        = []
    all_real_attn     = []
    all_pred_attn     = []
    all_labels        = []
    all_sample_ids    = []

    with torch.no_grad():
        for batch in loader:
            images           = batch['images'].to(device)
            gaze_heatmaps    = batch['gaze_heatmaps'].to(device)
            gaze_sequences   = batch['gaze_sequences'].to(device)
            gaze_seq_lengths = batch['gaze_seq_lengths']

            # Real gaze attention
            _, _, real_attn = model._encode_image_with_gaze(
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                use_predicted_gaze=False
            )

            # Predicted gaze attention (inference scenario)
            _, _, pred_attn = model._encode_image_with_gaze(
                images, use_predicted_gaze=True
            )

            all_images.extend(images.cpu())
            all_real_attn.extend(real_attn.cpu())
            all_pred_attn.extend(pred_attn.cpu())
            all_labels.extend(batch['labels'].tolist())
            all_sample_ids.extend(batch['sample_ids'])

            if len(all_images) >= max_samples:
                break

    # Cap to max_samples
    all_images     = all_images[:max_samples]
    all_real_attn  = all_real_attn[:max_samples]
    all_pred_attn  = all_pred_attn[:max_samples]
    all_labels     = all_labels[:max_samples]
    all_sample_ids = all_sample_ids[:max_samples]

    import torch as _torch
    images_tensor    = _torch.stack(all_images)
    real_attn_tensor = _torch.stack(all_real_attn)
    pred_attn_tensor = _torch.stack(all_pred_attn)

    # 1. Real gaze attention grid
    save_attention_grid(
        images_tensor, real_attn_tensor, all_labels, all_sample_ids,
        save_path=os.path.join(output_dir, 'real_gaze_attention.png'),
        title='C3-Net — Real Gaze Attention Maps',
        max_samples=max_samples
    )

    # 2. Predicted gaze attention grid
    save_attention_grid(
        images_tensor, pred_attn_tensor, all_labels, all_sample_ids,
        save_path=os.path.join(output_dir, 'predicted_gaze_attention.png'),
        title='C3-Net — Predicted Gaze Attention Maps (Inference)',
        max_samples=max_samples
    )

    # 3. Real vs predicted comparison
    save_gaze_comparison(
        images_tensor, real_attn_tensor, pred_attn_tensor,
        all_labels, all_sample_ids,
        save_path=os.path.join(output_dir, 'gaze_comparison.png'),
        max_samples=max_samples
    )

    # 4. Balanced Normal vs Abnormal grid
    normal_idx   = [i for i, l in enumerate(all_labels) if l == 0]
    abnormal_idx = [i for i, l in enumerate(all_labels) if l == 1]
    n_each       = min(4, len(normal_idx), len(abnormal_idx))
    selected     = normal_idx[:n_each] + abnormal_idx[:n_each]

    if selected:
        save_attention_grid(
            images_tensor[selected],
            real_attn_tensor[selected],
            [all_labels[i]     for i in selected],
            [all_sample_ids[i] for i in selected],
            save_path=os.path.join(output_dir, 'normal_vs_abnormal.png'),
            title='C3-Net — Normal vs Abnormal Gaze Attention',
            max_samples=len(selected)
        )


def visualize_teacher(model, loader, device, output_dir, max_samples):
    """
    Generate visualizations for BioClinicalBERT + BioGPT pipeline.
    Saves level1 (gaze-guided) and level2 (text-image) attention maps.
    """
    all_images     = []
    all_level1     = []
    all_labels     = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in loader:
            images               = batch['images'].to(device)
            gaze_heatmaps        = batch['gaze_heatmaps'].to(device)
            gaze_sequences       = batch['gaze_sequences'].to(device)
            gaze_seq_lengths     = batch['gaze_seq_lengths']
            text_token_ids       = batch['text_token_ids'].to(device)
            text_attention_masks = batch['text_attention_masks'].to(device)

            _, attention_maps = model(
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                text_token_ids, text_attention_masks
            )

            all_images.extend(images.cpu())
            all_labels.extend(batch['labels'].tolist())
            all_sample_ids.extend(batch['sample_ids'])

            if 'level1' in attention_maps:
                all_level1.extend(attention_maps['level1'].cpu())

            if len(all_images) >= max_samples:
                break

    import torch as _torch
    all_images = all_images[:max_samples]
    all_labels = all_labels[:max_samples]
    all_sample_ids = all_sample_ids[:max_samples]
    images_tensor  = _torch.stack(all_images)

    if all_level1:
        all_level1 = all_level1[:max_samples]
        save_attention_grid(
            images_tensor,
            _torch.stack(all_level1),
            all_labels,
            all_sample_ids,
            save_path=os.path.join(output_dir, 'level1_gaze_attention.png'),
            title='C3-Net Teacher — Level 1 Gaze-Guided Attention',
            max_samples=max_samples
        )


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='C3-Net Visualization')
    parser.add_argument('--config',      type=str, default='configs/config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--split',       type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--output_dir',  type=str, default='outputs/visualizations',
                        help='Directory to save output figures')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to visualize')
    parser.add_argument('--gpu_id',      type=int, default=None,
                        help='GPU ID override (default: uses config)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    gpu_id = args.gpu_id if args.gpu_id is not None else config['training']['gpu_id']
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output directory
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print("\nLoading model...")
    model = load_model(config, args.checkpoint, device)

    # Load data
    print("\nLoading dataset...")
    loader = load_dataset(config, args.split, args.num_samples)

    # Visualize
    print("\nGenerating visualizations...")
    use_medgemma = config['model'].get('use_medgemma', False)

    if use_medgemma:
        visualize_medgemma(model, loader, device, output_dir, args.num_samples)
    else:
        visualize_teacher(model, loader, device, output_dir, args.num_samples)

    print(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

# python visualize.py --checkpoint /media/16TB_Storage/kavin/models/c3-net/stage2_medgemma/best_model.pth --split test --num_samples 16