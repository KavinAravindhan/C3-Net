"""
t-SNE / UMAP feature visualization for C3-Net ablation study.

Extracts teacher features from all four modality variants on the test set
and generates:
  1. One t-SNE plot per modality variant (colored by Normal/Abnormal)
  2. Subspace plots for image_gaze_text (image / gaze / text dims separately)

Usage:
    cd /home/kavin/C3-Net
    conda activate c3net
    python interpretability/tsne_features.py

Outputs saved to: interpretability/outputs/tsne/
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from data.dataset import MIMICEyeDataset, collate_fn
from models.teacher import MultimodalTeacher


# ==============================================================================
CHECKPOINT_DIR = '/media/16TB_Storage/kavin/models/c3-net/hparam_search'
DATASET_ROOT   = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye'
OUTPUT_DIR     = 'interpretability/outputs/tsne'
CONFIG_PATH    = 'configs/config.yaml'

MODALITIES = ['image_only', 'image_gaze', 'image_text', 'image_gaze_text']

# Feature subspace dims for image_gaze_text (3328 total)
# image: 768, gaze: 512+768=1280 (spatial+temporal), text: 768+512=1280
# Simplified split based on your architecture:
SUBSPACES = {
    'image': (0, 768),
    'gaze':  (768, 768 + 1280),
    'text':  (768 + 1280, 3328),
}

COLORS = {0: '#2196F3', 1: '#F44336'}  # blue=Normal, red=Abnormal
LABELS = {0: 'Normal', 1: 'Abnormal'}


# ==============================================================================

def load_teacher(config, modality, checkpoint_path, device):
    """Load frozen teacher for a given modality."""
    config = dict(config)
    config['model'] = dict(config['model'])
    config['model']['modality'] = modality

    model = MultimodalTeacher(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_key = 'teacher_state_dict' if 'teacher_state_dict' in checkpoint else 'model_state_dict'
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def extract_features(model, loader, config, device):
    """Run inference on test set and return (features, labels)."""
    modality   = config['model']['modality']
    uses_gaze  = modality in ('image_gaze', 'image_gaze_text')
    uses_text  = modality in ('image_text', 'image_gaze_text')

    all_features = []
    all_labels   = []

    with torch.no_grad():
        for batch in loader:
            images               = batch['images'].to(device)
            labels               = batch['labels']
            gaze_heatmaps        = batch['gaze_heatmaps'].to(device)       if uses_gaze else None
            gaze_sequences       = batch['gaze_sequences'].to(device)      if uses_gaze else None
            gaze_seq_lengths     = batch['gaze_seq_lengths']               if uses_gaze else None
            text_token_ids       = batch['text_token_ids'].to(device)      if uses_text else None
            text_attention_masks = batch['text_attention_masks'].to(device) if uses_text else None

            # Extract combined features (before classification head)
            features, _ = model.extract_features(
                images=images,
                gaze_heatmaps=gaze_heatmaps,
                gaze_sequences=gaze_sequences,
                gaze_seq_lengths=gaze_seq_lengths,
                text_token_ids=text_token_ids,
                text_attention_masks=text_attention_masks
            )

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels   = np.concatenate(all_labels,   axis=0)
    return features, labels


def run_tsne(features, perplexity=20, random_state=42):
    """Run t-SNE and return 2D embedding."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        init='pca'
    )
    return tsne.fit_transform(features)


def plot_tsne(embedding, labels, title, save_path):
    """Plot 2D t-SNE embedding colored by class label."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for cls in [0, 1]:
        mask = labels == cls
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=COLORS[cls],
            label=LABELS[cls],
            alpha=0.7,
            s=25,
            edgecolors='none'
        )

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_id = config['training'].get('gpu_id', 0)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test dataset once (modality-agnostic loading)
    test_dataset = MIMICEyeDataset(
        root_dir=DATASET_ROOT,
        split='test',
        config=config
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    all_embeddings = {}

    # ===== Per-modality t-SNE =====
    print("\n" + "="*60)
    print("Extracting features and running t-SNE per modality")
    print("="*60)

    for modality in MODALITIES:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{modality}.pth')
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {modality} — checkpoint not found")
            continue

        print(f"\n[{modality}]")
        mod_config = dict(config)
        mod_config['model'] = dict(config['model'])
        mod_config['model']['modality'] = modality

        model    = load_teacher(mod_config, modality, ckpt_path, device)
        features, labels = extract_features(model, test_loader, mod_config, device)

        print(f"  Features shape: {features.shape}")
        print(f"  Running t-SNE...")

        embedding = run_tsne(features)
        all_embeddings[modality] = (embedding, labels)

        save_path = os.path.join(OUTPUT_DIR, f'tsne_{modality}.png')
        plot_tsne(embedding, labels, f't-SNE: {modality}', save_path)

        del model
        torch.cuda.empty_cache()

    # ===== Subspace plots for image_gaze_text =====
    print("\n" + "="*60)
    print("Running subspace t-SNE for image_gaze_text")
    print("="*60)

    igt_ckpt = os.path.join(CHECKPOINT_DIR, 'best_image_gaze_text.pth')
    if os.path.exists(igt_ckpt):
        igt_config = dict(config)
        igt_config['model'] = dict(config['model'])
        igt_config['model']['modality'] = 'image_gaze_text'

        model    = load_teacher(igt_config, 'image_gaze_text', igt_ckpt, device)
        features, labels = extract_features(model, test_loader, igt_config, device)

        for subspace_name, (start, end) in SUBSPACES.items():
            print(f"\n  Subspace: {subspace_name} (dims {start}:{end})")
            sub_features = features[:, start:end]
            embedding    = run_tsne(sub_features)

            save_path = os.path.join(OUTPUT_DIR, f'tsne_subspace_{subspace_name}.png')
            plot_tsne(
                embedding, labels,
                f't-SNE: image_gaze_text — {subspace_name} subspace',
                save_path
            )

        del model
        torch.cuda.empty_cache()

    # ===== Combined comparison figure =====
    print("\n" + "="*60)
    print("Generating combined comparison figure")
    print("="*60)

    available = [(m, all_embeddings[m]) for m in MODALITIES if m in all_embeddings]
    if len(available) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (modality, (embedding, labels)) in enumerate(available):
            ax = axes[idx]
            for cls in [0, 1]:
                mask = labels == cls
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=COLORS[cls],
                    label=LABELS[cls],
                    alpha=0.7,
                    s=20,
                    edgecolors='none'
                )
            ax.set_title(modality, fontsize=11, fontweight='bold')
            ax.set_xlabel('t-SNE dim 1', fontsize=9)
            ax.set_ylabel('t-SNE dim 2', fontsize=9)
            ax.grid(True, alpha=0.2)

        handles = [
            mpatches.Patch(color=COLORS[0], label='Normal'),
            mpatches.Patch(color=COLORS[1], label='Abnormal')
        ]
        fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=11, 
                   bbox_to_anchor=(0.5, 0.01))
        fig.suptitle('t-SNE Feature Space: C3-Net Modality Ablation (Test Set)',
                     fontsize=14, fontweight='bold', y=1.01)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'tsne_comparison_all_modalities.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

    print("\n" + "="*60)
    print(f"All t-SNE plots saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()