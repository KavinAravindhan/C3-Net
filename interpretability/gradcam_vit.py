"""
GradCAM visualization for C3-Net ViT image encoder.

Computes GradCAM on the last ViT transformer block and overlays on chest X-rays.
Also overlays radiologist gaze heatmaps for direct comparison.

Generates:
  1. GradCAM overlays for correct vs incorrect predictions
  2. Side-by-side: image_only vs image_gaze_text GradCAM on same samples
  3. Normal vs Abnormal GradCAM pattern comparison

Usage:
    cd /home/kavin/C3-Net
    conda activate c3net
    python interpretability/gradcam_vit.py

Outputs saved to: interpretability/outputs/gradcam/
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.utils.data import DataLoader
import cv2

from data.dataset import MIMICEyeDataset, collate_fn
from models.teacher import MultimodalTeacher


# ==============================================================================
CHECKPOINT_DIR = '/media/16TB_Storage/kavin/models/c3-net/hparam_search'
DATASET_ROOT   = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye'
OUTPUT_DIR     = 'interpretability/outputs/gradcam'
CONFIG_PATH    = 'configs/config.yaml'

NUM_SAMPLES    = 10   # number of test samples to visualize
IMG_SIZE       = 224
PATCH_SIZE     = 16
NUM_PATCHES_1D = IMG_SIZE // PATCH_SIZE  # 14


# ==============================================================================

class GradCAMViT:
    """
    GradCAM for ViT last transformer block.

    Hooks into the last block's output to capture:
      - activations: [B, 197, 768]  (197 = 1 cls token + 196 patch tokens)
      - gradients:   [B, 197, 768]

    GradCAM map = ReLU(sum over channels of gradient-weighted activations)
    projected onto the 14x14 patch grid then upsampled to 224x224.
    """

    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the last transformer block of the ViT encoder."""
        # timm ViT: model.image_encoder.model.blocks[-1]
        last_block = self.model.image_encoder.vit.blocks[-1]

        def forward_hook(module, input, output):
            # output: [B, 197, 768]
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: [B, 197, 768]
            self.gradients = grad_output[0].detach()

        self._hooks.append(last_block.register_forward_hook(forward_hook))
        self._hooks.append(last_block.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def compute(self, images, target_class, gaze_heatmaps=None, gaze_sequences=None,
                gaze_seq_lengths=None, text_token_ids=None, text_attention_masks=None):
        """
        Compute GradCAM map for a batch.

        Args:
            images:        [B, 3, 224, 224]
            target_class:  int — class index to compute gradient for (0=Normal, 1=Abnormal)
            ...rest:       modality-specific inputs

        Returns:
            cam_maps: [B, 224, 224] — upsampled GradCAM maps, values in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        # Enable gradients for this pass only
        images = images.requires_grad_(True)

        logits, _ = self.model(
            images=images,
            gaze_heatmaps=gaze_heatmaps,
            gaze_sequences=gaze_sequences,
            gaze_seq_lengths=gaze_seq_lengths,
            text_token_ids=text_token_ids,
            text_attention_masks=text_attention_masks
        )

        # Backprop w.r.t. target class score
        score = logits[:, target_class].sum()
        score.backward()

        # activations / gradients: [B, 197, 768]
        # Exclude CLS token (index 0), keep patch tokens (1:197)
        patch_activations = self.activations[:, 1:, :]  # [B, 196, 768]
        patch_gradients   = self.gradients[:, 1:, :]    # [B, 196, 768]

        # Global average pool gradients over channel dim → weights [B, 196]
        weights = patch_gradients.mean(dim=-1)           # [B, 196]

        # Weight activations and sum over channels → raw cam [B, 196]
        cam = (weights.unsqueeze(-1) * patch_activations).sum(dim=-1)  # [B, 196]

        # ReLU — keep only positive activations
        cam = torch.clamp(cam, min=0)

        # Reshape to [B, 14, 14] and upsample to [B, 224, 224]
        B = cam.shape[0]
        cam = cam.reshape(B, NUM_PATCHES_1D, NUM_PATCHES_1D)  # [B, 14, 14]

        cam_maps = []
        for i in range(B):
            c = cam[i].cpu().numpy()
            # Normalize to [0, 1]
            if c.max() > c.min():
                c = (c - c.min()) / (c.max() - c.min())
            # Upsample to 224x224
            c = cv2.resize(c, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            cam_maps.append(c)

        return np.stack(cam_maps, axis=0)  # [B, 224, 224]


# ==============================================================================

def load_teacher(config, modality, checkpoint_path, device):
    """Load teacher model for a given modality."""
    cfg = dict(config)
    cfg['model'] = dict(config['model'])
    cfg['model']['modality'] = modality

    model = MultimodalTeacher(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = 'teacher_state_dict' if 'teacher_state_dict' in checkpoint else 'model_state_dict'
    model.load_state_dict(checkpoint[state_key])
    return model


def denormalize_image(tensor):
    """
    Reverse ImageNet normalization and return [H, W, 3] numpy array in [0, 1].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    img  = std * img + mean
    return np.clip(img, 0, 1)


def overlay_heatmap(image, cam, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlay a GradCAM heatmap onto a grayscale or RGB image.

    Args:
        image: [H, W, 3] float in [0, 1]
        cam:   [H, W]    float in [0, 1]
        alpha: blend factor for heatmap

    Returns:
        blended: [H, W, 3] float in [0, 1]
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    blended = (1 - alpha) * image + alpha * heatmap
    return np.clip(blended, 0, 1)


def plot_gradcam_row(image, gaze_heatmap, cam_image_only, cam_image_gaze_text,
                     label, pred_io, pred_igt, save_path):
    """
    Plot one row: original | gaze heatmap | GradCAM image_only | GradCAM image_gaze_text.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    class_names = {0: 'Normal', 1: 'Abnormal'}
    label_str   = class_names[label]

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'Original\nGT: {label_str}', fontsize=10)
    axes[0].axis('off')

    # Gaze heatmap overlay
    gaze_norm = gaze_heatmap.cpu().numpy()
    if gaze_norm.max() > gaze_norm.min():
        gaze_norm = (gaze_norm - gaze_norm.min()) / (gaze_norm.max() - gaze_norm.min())
    gaze_overlay = overlay_heatmap(image, gaze_norm, alpha=0.5, colormap=cv2.COLORMAP_HOT)
    axes[1].imshow(gaze_overlay)
    axes[1].set_title('Radiologist Gaze', fontsize=10)
    axes[1].axis('off')

    # GradCAM image_only
    io_overlay = overlay_heatmap(image, cam_image_only)
    pred_io_str = class_names[pred_io]
    correct_io  = '✓' if pred_io == label else '✗'
    axes[2].imshow(io_overlay)
    axes[2].set_title(f'GradCAM: image_only\nPred: {pred_io_str} {correct_io}', fontsize=10)
    axes[2].axis('off')

    # GradCAM image_gaze_text
    igt_overlay = overlay_heatmap(image, cam_image_gaze_text)
    pred_igt_str = class_names[pred_igt]
    correct_igt  = '✓' if pred_igt == label else '✗'
    axes[3].imshow(igt_overlay)
    axes[3].set_title(f'GradCAM: image_gaze_text\nPred: {pred_igt_str} {correct_igt}', fontsize=10)
    axes[3].axis('off')

    fig.suptitle(f'GradCAM Comparison — Sample GT: {label_str}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_id = config['training'].get('gpu_id', 0)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test dataset
    test_dataset = MIMICEyeDataset(
        root_dir=DATASET_ROOT,
        split='test',
        config=config
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,       # process one sample at a time for GradCAM
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # Load both models
    io_ckpt  = os.path.join(CHECKPOINT_DIR, 'best_image_only.pth')
    igt_ckpt = os.path.join(CHECKPOINT_DIR, 'best_image_gaze_text.pth')

    print("\nLoading image_only model...")
    io_config = dict(config)
    io_config['model'] = dict(config['model'])
    io_config['model']['modality'] = 'image_only'
    model_io = load_teacher(io_config, 'image_only', io_ckpt, device)

    print("\nLoading image_gaze_text model...")
    igt_config = dict(config)
    igt_config['model'] = dict(config['model'])
    igt_config['model']['modality'] = 'image_gaze_text'
    model_igt = load_teacher(igt_config, 'image_gaze_text', igt_ckpt, device)

    gradcam_io  = GradCAMViT(model_io)
    gradcam_igt = GradCAMViT(model_igt)

    print(f"\nGenerating GradCAM for {NUM_SAMPLES} test samples...")
    print("="*60)

    sample_count    = 0
    correct_io_list = []
    wrong_io_list   = []

    for batch_idx, batch in enumerate(test_loader):
        if sample_count >= NUM_SAMPLES:
            break

        images               = batch['images'].to(device)
        labels               = batch['labels']
        gaze_heatmaps        = batch['gaze_heatmaps'].to(device)
        gaze_sequences       = batch['gaze_sequences'].to(device)
        gaze_seq_lengths     = batch['gaze_seq_lengths']
        text_token_ids       = batch['text_token_ids'].to(device)
        text_attention_masks = batch['text_attention_masks'].to(device)

        label = labels[0].item()

        # Predictions (no grad)
        with torch.no_grad():
            logits_io,  _ = model_io(images=images)
            logits_igt, _ = model_igt(
                images=images,
                gaze_heatmaps=gaze_heatmaps,
                gaze_sequences=gaze_sequences,
                gaze_seq_lengths=gaze_seq_lengths,
                text_token_ids=text_token_ids,
                text_attention_masks=text_attention_masks
            )

        pred_io  = logits_io.argmax(dim=1)[0].item()
        pred_igt = logits_igt.argmax(dim=1)[0].item()

        # GradCAM w.r.t. ground truth class
        cam_io  = gradcam_io.compute(images, target_class=label)
        cam_igt = gradcam_igt.compute(
            images, target_class=label,
            gaze_heatmaps=gaze_heatmaps,
            gaze_sequences=gaze_sequences,
            gaze_seq_lengths=gaze_seq_lengths,
            text_token_ids=text_token_ids,
            text_attention_masks=text_attention_masks
        )

        # Denormalize image for visualization
        image_np      = denormalize_image(images[0])
        gaze_heatmap  = gaze_heatmaps[0]  # [224, 224]

        # Determine save subdirectory based on prediction outcome
        # Case: image_only wrong, image_gaze_text correct — most interesting
        if pred_io != label and pred_igt == label:
            subdir = 'io_wrong_igt_correct'
        elif pred_io == label and pred_igt == label:
            subdir = 'both_correct'
        elif pred_io != label and pred_igt != label:
            subdir = 'both_wrong'
        else:
            subdir = 'io_correct_igt_wrong'

        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, subdir, f'sample_{batch_idx:04d}.png')

        plot_gradcam_row(
            image=image_np,
            gaze_heatmap=gaze_heatmap,
            cam_image_only=cam_io[0],
            cam_image_gaze_text=cam_igt[0],
            label=label,
            pred_io=pred_io,
            pred_igt=pred_igt,
            save_path=save_path
        )

        print(f"  Sample {sample_count+1:02d} | GT: {'Normal' if label==0 else 'Abnormal'} "
              f"| IO: {'✓' if pred_io==label else '✗'} "
              f"| IGT: {'✓' if pred_igt==label else '✗'} "
              f"| Saved: {subdir}/")

        sample_count += 1

    gradcam_io.remove_hooks()
    gradcam_igt.remove_hooks()

    print("\n" + "="*60)
    print(f"GradCAM outputs saved to: {OUTPUT_DIR}")
    print(f"Subdirectories: io_wrong_igt_correct / both_correct / both_wrong / io_correct_igt_wrong")
    print("="*60)


if __name__ == '__main__':
    main()