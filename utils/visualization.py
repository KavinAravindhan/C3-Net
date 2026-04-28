import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


# ==============================================================================
# Core Visualization Functions
# ==============================================================================

def attention_map_to_heatmap(attention_map, image_size=224, patch_size=16):
    """
    Convert a flat patch attention map [196] to a full-resolution heatmap [224, 224].

    Args:
        attention_map: Tensor or ndarray [196] — patch-level attention scores
        image_size:    int — output resolution (default 224)
        patch_size:    int — ViT patch size (default 16)

    Returns:
        heatmap: ndarray [224, 224] — upsampled attention map, values in [0, 1]
    """
    num_patches_1d = image_size // patch_size  # 14

    if torch.is_tensor(attention_map):
        attention_map = attention_map.detach().cpu().numpy()

    attention_map = attention_map.reshape(num_patches_1d, num_patches_1d)  # [14, 14]

    # Normalize to [0, 1]
    if attention_map.max() > attention_map.min():
        attention_map = (attention_map - attention_map.min()) / \
                        (attention_map.max() - attention_map.min())

    # Upsample [14, 14] → [224, 224] using bilinear interpolation
    heatmap_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float()
    heatmap_tensor = F.interpolate(
        heatmap_tensor,
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False
    )
    heatmap = heatmap_tensor.squeeze().numpy()  # [224, 224]

    return heatmap


def overlay_attention_on_image(image, attention_map, alpha=0.5, colormap='jet',
                                image_size=224, patch_size=16):
    """
    Overlay a patch attention map on the original image as a heatmap.

    Args:
        image:        Tensor [3, 224, 224] (normalized ImageNet) or ndarray [224, 224, 3]
        attention_map: Tensor or ndarray [196] — patch attention scores
        alpha:        float — heatmap opacity (0=image only, 1=heatmap only)
        colormap:     str — matplotlib colormap name
        image_size:   int
        patch_size:   int

    Returns:
        overlay: ndarray [224, 224, 3] uint8 — blended image+heatmap
        heatmap_rgb: ndarray [224, 224, 3] uint8 — heatmap only (for side-by-side)
    """
    # ── Denormalize image ──
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    if image.shape[0] == 3:
        # [3, H, W] → [H, W, 3]
        image = np.transpose(image, (1, 2, 0))

    # Reverse ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    image = (image * std + mean)
    image = np.clip(image, 0, 1)

    # ── Build heatmap ──
    heatmap = attention_map_to_heatmap(attention_map, image_size, patch_size)

    # Apply colormap
    cmap        = cm.get_cmap(colormap)
    heatmap_rgb = cmap(heatmap)[:, :, :3]  # [224, 224, 3], drop alpha channel

    # ── Blend ──
    overlay = (1 - alpha) * image + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    # Convert to uint8
    overlay     = (overlay     * 255).astype(np.uint8)
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)

    return overlay, heatmap_rgb


def save_attention_grid(images, attention_maps, labels, sample_ids,
                        save_path, title=None, alpha=0.5, max_samples=8):
    """
    Save a grid of [original | heatmap | overlay] for a batch of samples.

    Args:
        images:        Tensor [B, 3, 224, 224]
        attention_maps: Tensor [B, 196] — gaze attention maps
        labels:        list[int] — ground truth labels (0=Normal, 1=Abnormal)
        sample_ids:    list[str]
        save_path:     str — full path to save the figure (e.g. .../epoch5_attention.png)
        title:         str — optional figure title
        alpha:         float — heatmap opacity
        max_samples:   int — max number of samples to show (caps at B)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    B = min(len(images), max_samples)

    fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
    if B == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing works for single sample

    col_titles = ['Original', 'Gaze Heatmap', 'Overlay']
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=13, fontweight='bold')

    for i in range(B):
        image         = images[i]           # [3, 224, 224]
        attention_map = attention_maps[i]   # [196]
        label_str     = 'Abnormal' if labels[i] == 1 else 'Normal'
        sid           = sample_ids[i]

        overlay, heatmap_rgb = overlay_attention_on_image(image, attention_map, alpha=alpha)

        # Denormalize original for display
        img_np = image.detach().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 1].imshow(heatmap_rgb)
        axes[i, 2].imshow(overlay)

        for col in range(3):
            axes[i, col].axis('off')

        axes[i, 0].set_ylabel(f"{sid}\n({label_str})", fontsize=9, rotation=0,
                               labelpad=80, va='center')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved attention grid → {save_path}")


def save_gaze_comparison(images, real_attention_maps, pred_attention_maps,
                          labels, sample_ids, save_path, alpha=0.5, max_samples=6):
    """
    Side-by-side comparison of real gaze vs predicted gaze overlays.
    Useful for evaluating GazePredictor quality.

    Grid columns: [Original | Real Gaze Overlay | Predicted Gaze Overlay]

    Args:
        images:               Tensor [B, 3, 224, 224]
        real_attention_maps:  Tensor [B, 196] — from GazeEncoder (ground truth)
        pred_attention_maps:  Tensor [B, 196] — from GazePredictor
        labels:               list[int]
        sample_ids:           list[str]
        save_path:            str
        alpha:                float
        max_samples:          int
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    B = min(len(images), max_samples)

    fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
    if B == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original', 'Real Gaze', 'Predicted Gaze']
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=13, fontweight='bold')

    for i in range(B):
        image     = images[i]
        label_str = 'Abnormal' if labels[i] == 1 else 'Normal'
        sid       = sample_ids[i]

        real_overlay, _ = overlay_attention_on_image(image, real_attention_maps[i], alpha=alpha)
        pred_overlay, _ = overlay_attention_on_image(image, pred_attention_maps[i], alpha=alpha)

        img_np = image.detach().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 1].imshow(real_overlay)
        axes[i, 2].imshow(pred_overlay)

        for col in range(3):
            axes[i, col].axis('off')

        axes[i, 0].set_ylabel(f"{sid}\n({label_str})", fontsize=9, rotation=0,
                               labelpad=80, va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved gaze comparison → {save_path}")


# ==============================================================================
# Training Callback Helper
# ==============================================================================

def save_epoch_attention(model, val_loader, device, save_dir, epoch,
                          use_medgemma=True, max_samples=8):
    """
    Called during training every N epochs to save attention overlays.
    Pulls one batch from val_loader, runs a forward pass, saves the grid.

    Args:
        model:        MedGemmaModel or MultimodalTeacher
        val_loader:   DataLoader
        device:       torch.device
        save_dir:     str — root directory for visualization outputs
        epoch:        int — current epoch number (1-indexed)
        use_medgemma: bool — determines which forward path to use
        max_samples:  int
    """
    model.eval()

    vis_dir = os.path.join(save_dir, 'attention_maps')
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        batch = next(iter(val_loader))

        images           = batch['images'].to(device)
        gaze_heatmaps    = batch['gaze_heatmaps'].to(device)
        gaze_sequences   = batch['gaze_sequences'].to(device)
        gaze_seq_lengths = batch['gaze_seq_lengths']
        labels           = batch['labels'].tolist()
        sample_ids       = batch['sample_ids']

        if use_medgemma:
            # MedGemma path — use _encode_image_with_gaze directly
            _, _, attn_map = model._encode_image_with_gaze(
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                use_predicted_gaze=False
            )
            # Also get predicted gaze for comparison
            _, _, pred_attn_map = model._encode_image_with_gaze(
                images, use_predicted_gaze=True
            )

            # Save real gaze overlay
            save_attention_grid(
                images[:max_samples],
                attn_map[:max_samples],
                labels[:max_samples],
                sample_ids[:max_samples],
                save_path=os.path.join(vis_dir, f'epoch_{epoch:03d}_real_gaze.png'),
                title=f'Epoch {epoch} — Real Gaze Attention',
                max_samples=max_samples
            )

            # Save real vs predicted comparison
            save_gaze_comparison(
                images[:max_samples],
                attn_map[:max_samples],
                pred_attn_map[:max_samples],
                labels[:max_samples],
                sample_ids[:max_samples],
                save_path=os.path.join(vis_dir, f'epoch_{epoch:03d}_gaze_comparison.png'),
                max_samples=max_samples
            )

        else:
            # BioClinicalBERT path — extract from attention_maps dict
            text_token_ids       = batch['text_token_ids'].to(device)
            text_attention_masks = batch['text_attention_masks'].to(device)

            _, attention_maps = model(
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                text_token_ids, text_attention_masks
            )

            if 'level1' in attention_maps:
                save_attention_grid(
                    images[:max_samples],
                    attention_maps['level1'][:max_samples],
                    labels[:max_samples],
                    sample_ids[:max_samples],
                    save_path=os.path.join(vis_dir, f'epoch_{epoch:03d}_level1_attention.png'),
                    title=f'Epoch {epoch} — Level 1 Gaze-Guided Attention',
                    max_samples=max_samples
                )

    model.train()