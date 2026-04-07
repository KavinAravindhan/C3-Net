"""
BioGPT Attention Rollout for C3-Net decoder diagnosis.

Uses teacher-forcing forward pass with output_attentions=True to measure
how much BioGPT attends to the multimodal conditioning prefix (position 0)
at each token position and each layer.

Key question: Does BioGPT attend to the projected teacher features (position 0)
or does it rely purely on language model priors (attending only to prior tokens)?

Low attention to position 0 → decoder ignoring conditioning → garbled output.

Architecture reminder:
    inputs_embeds = [prefix_token, BOS, token_1, ..., token_T]
    Position 0 = projected multimodal features (the conditioning)
    Position 1 = BOS token
    Positions 2..T+1 = report tokens

Usage:
    cd /home/kavin/C3-Net
    conda activate c3net
    python interpretability/biogpt_attention_rollout.py

Outputs saved to: interpretability/outputs/attention_rollout/
"""

import os
import re
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from transformers import BioGptTokenizer

from data.dataset import MIMICEyeDataset, collate_fn
from models.teacher_decoder import C3NetTeacherDecoder


# ==============================================================================
TEACHER_CKPT   = '/media/16TB_Storage/kavin/models/c3-net/hparam_search/best_image_gaze_text.pth'
DECODER_CKPT   = '/media/16TB_Storage/kavin/models/c3-net/decoder_image_gaze_text_20260323_143007/best_decoder.pth'
DATASET_ROOT   = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye'
OUTPUT_DIR     = 'interpretability/outputs/attention_rollout'
CONFIG_PATH    = 'configs/config.yaml'

NUM_SAMPLES    = 8      # test samples to analyse
MAX_REPORT_LEN = 100    # max tokens for teacher forcing (keep manageable)
NUM_LAYERS     = 24     # BioGPT has 24 transformer layers


# ==============================================================================

def load_model(config, device):
    """Load C3NetTeacherDecoder with frozen teacher and trained decoder."""
    model = C3NetTeacherDecoder(
        config=config,
        teacher_checkpoint_path=TEACHER_CKPT
    ).to(device)

    decoder_ckpt = torch.load(DECODER_CKPT, map_location=device, weights_only=False)
    model.decoder.load_state_dict(decoder_ckpt['decoder_state_dict'])
    model.set_eval_mode()
    print("Model loaded and set to eval mode.")
    return model


def clean_report(text):
    """Remove MIMIC anonymization artifacts for cleaner tokenization."""
    text = re.sub(r'_{2,}', '[ANON]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_report(text, tokenizer, max_length):
    """
    Tokenize a report string with BioGPT tokenizer.
    Returns input_ids: [1, max_length]
    """
    encoding = tokenizer(
        text,
        max_length=max_length - 1,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return encoding['input_ids'], encoding['attention_mask']


def compute_attention_rollout(attentions):
    """
    Compute attention rollout across all BioGPT layers.

    Args:
        attentions: tuple of length num_layers, each [1, num_heads, seq_len, seq_len]

    Returns:
        rollout: [seq_len, seq_len] — total attention flow matrix
                 rollout[i, j] = how much position j influences position i
    """
    num_layers = len(attentions)
    seq_len    = attentions[0].shape[-1]

    # Start with identity
    rollout = torch.eye(seq_len)

    for layer_attn in attentions:
        # layer_attn: [1, num_heads, seq_len, seq_len]
        # Average across heads: [seq_len, seq_len]
        attn = layer_attn[0].mean(dim=0).cpu()

        # Add residual connection: A_hat = 0.5 * A + 0.5 * I
        attn_hat = 0.5 * attn + 0.5 * torch.eye(seq_len)

        # Normalize rows to sum to 1
        attn_hat = attn_hat / (attn_hat.sum(dim=-1, keepdim=True) + 1e-8)

        # Multiply rollout
        rollout = attn_hat @ rollout

    return rollout.numpy()  # [seq_len, seq_len]


def get_per_layer_conditioning_attention(attentions):
    """
    For each layer, compute mean attention to position 0 (conditioning prefix)
    across all non-prefix token positions.

    Args:
        attentions: tuple of [1, num_heads, seq_len, seq_len] per layer

    Returns:
        per_layer_scores: [num_layers] — mean conditioning attention per layer
    """
    scores = []
    for layer_attn in attentions:
        # layer_attn: [1, num_heads, seq_len, seq_len]
        # Mean across heads, then mean across positions 1..seq_len attention to position 0
        attn = layer_attn[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
        # attn[i, 0] = attention from position i to position 0 (conditioning)
        # Average over positions 1 onward (exclude self-attention of position 0)
        cond_attn = attn[1:, 0].mean()
        scores.append(cond_attn)
    return np.array(scores)


# ==============================================================================

def plot_rollout_heatmap(rollout, cond_per_layer, reference_text,
                         generated_text, label, save_path, max_tokens=60):
    """
    Three-panel figure:
      Left:   Full rollout attention matrix (seq_len x seq_len)
      Middle: Attention to position 0 per generation position (from rollout)
      Right:  Attention to position 0 per layer (direct, not rolled out)
    """
    seq_len  = min(rollout.shape[0], max_tokens)
    mat      = rollout[:seq_len, :seq_len]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    class_names = {0: 'Normal', 1: 'Abnormal'}

    # ===== Panel 1: Rollout heatmap =====
    ax = axes[0]
    im = ax.imshow(mat, aspect='auto', cmap='Blues', interpolation='nearest',
                   vmin=0, vmax=mat.max())
    ax.set_xlabel('Source position', fontsize=10)
    ax.set_ylabel('Target position', fontsize=10)
    ax.set_title('Attention Rollout Matrix\n(all 24 layers combined)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04)

    # Highlight conditioning column (position 0)
    ax.axvline(x=0.5, color='red', linewidth=2, linestyle='--')
    ax.text(1.5, seq_len * 0.05, 'Pos 0\n(conditioning)',
            color='red', fontsize=7, ha='left')

    # ===== Panel 2: Conditioning attention per position =====
    ax2 = axes[1]
    cond_per_pos = mat[:, 0]  # attention to position 0 at each target position
    ax2.bar(range(seq_len), cond_per_pos, color='steelblue', alpha=0.8, width=0.8)
    ax2.axhline(y=cond_per_pos.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {cond_per_pos.mean():.4f}')
    ax2.set_xlabel('Token position', fontsize=10)
    ax2.set_ylabel('Rollout attention to pos 0', fontsize=10)
    ax2.set_title('Conditioning Attention\nper Token Position', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')

    # ===== Panel 3: Per-layer conditioning attention =====
    ax3 = axes[2]
    ax3.plot(range(len(cond_per_layer)), cond_per_layer,
             marker='o', color='darkorange', linewidth=2, markersize=5)
    ax3.fill_between(range(len(cond_per_layer)), cond_per_layer, alpha=0.2, color='darkorange')
    ax3.axhline(y=cond_per_layer.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {cond_per_layer.mean():.4f}')
    ax3.set_xlabel('Layer index', fontsize=10)
    ax3.set_ylabel('Attn to conditioning (pos 0)', fontsize=10)
    ax3.set_title('Conditioning Attention\nper BioGPT Layer', fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # Text references at bottom
    label_str  = class_names[label]
    ref_trunc  = reference_text[:100] + '...' if len(reference_text) > 100 else reference_text
    gen_trunc  = generated_text[:100] + '...' if len(generated_text) > 100 else generated_text

    fig.suptitle(f'BioGPT Attention Rollout — GT: {label_str}',
                 fontsize=13, fontweight='bold')
    fig.text(0.01, -0.02, f'Reference: {ref_trunc}', fontsize=7, color='green')
    fig.text(0.01, -0.06, f'Generated: {gen_trunc}', fontsize=7, color='darkorange')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary(all_rollout_scores, all_layer_scores, all_labels, save_path):
    """
    Summary across all samples:
      Left:  Mean conditioning attention score per sample (bar chart)
      Right: Mean per-layer conditioning attention averaged across samples
    """
    if not all_rollout_scores:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Panel 1: Per-sample conditioning score =====
    ax = axes[0]
    colors = ['#2196F3' if l == 0 else '#F44336' for l in all_labels]
    ax.bar(range(len(all_rollout_scores)), all_rollout_scores,
           color=colors, alpha=0.8)
    ax.axhline(y=np.mean(all_rollout_scores), color='black', linestyle='--',
               linewidth=1.5, label=f'Mean: {np.mean(all_rollout_scores):.4f}')
    ax.set_xlabel('Sample index', fontsize=11)
    ax.set_ylabel('Mean rollout attention to pos 0', fontsize=11)
    ax.set_title('Per-Sample Conditioning Attention\n'
                 '(low = decoder ignoring teacher features)', fontsize=11)
    legend_elements = [
        mpatches.Patch(facecolor='#2196F3', label='Normal'),
        mpatches.Patch(facecolor='#F44336', label='Abnormal'),
        plt.Line2D([0], [0], color='black', linestyle='--',
                   label=f'Mean: {np.mean(all_rollout_scores):.4f}')
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # ===== Panel 2: Average per-layer attention across all samples =====
    ax2 = axes[1]
    mean_layer = np.mean(all_layer_scores, axis=0)
    std_layer  = np.std(all_layer_scores, axis=0)

    ax2.plot(range(len(mean_layer)), mean_layer,
             marker='o', color='darkorange', linewidth=2, markersize=5,
             label='Mean across samples')
    ax2.fill_between(range(len(mean_layer)),
                     mean_layer - std_layer,
                     mean_layer + std_layer,
                     alpha=0.2, color='darkorange', label='±1 std')
    ax2.set_xlabel('BioGPT layer index', fontsize=11)
    ax2.set_ylabel('Attention to conditioning (pos 0)', fontsize=11)
    ax2.set_title('Per-Layer Conditioning Attention\n(averaged across all samples)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    config['model']['modality'] = 'image_gaze_text'

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_id = config['training'].get('gpu_id', 0)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model     = load_model(config, device)
    tokenizer = model.tokenizer

    test_dataset = MIMICEyeDataset(
        root_dir=DATASET_ROOT,
        split='test',
        config=config
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    all_rollout_scores = []
    all_layer_scores   = []
    all_labels         = []

    print(f"\nRunning attention rollout on {NUM_SAMPLES} test samples...")
    print("="*60)

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= NUM_SAMPLES:
            break

        images               = batch['images'].to(device)
        labels               = batch['labels']
        report_texts         = batch['report_texts']
        gaze_heatmaps        = batch['gaze_heatmaps'].to(device)
        gaze_sequences       = batch['gaze_sequences'].to(device)
        gaze_seq_lengths     = batch['gaze_seq_lengths']
        text_token_ids       = batch['text_token_ids'].to(device)
        text_attention_masks = batch['text_attention_masks'].to(device)

        label          = labels[0].item()
        reference_text = report_texts[0]
        cleaned_report = clean_report(reference_text)

        print(f"\n[Sample {batch_idx+1}] GT: {'Normal' if label==0 else 'Abnormal'}")

        # ===== Step 1: Extract teacher features (frozen, no_grad) =====
        with torch.no_grad():
            combined_features = model.get_combined_features(
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                text_token_ids, text_attention_masks
            )
            # combined_features: [1, 3328]

            # ===== Step 2: Project features → BioGPT prefix =====
            prefix        = model.decoder.projection(combined_features)  # [1, 1024]
            prefix_embed  = prefix.unsqueeze(1)                           # [1, 1, 1024]

            # ===== Step 3: Tokenize reference report =====
            report_ids, report_mask = tokenize_report(
                cleaned_report, tokenizer, MAX_REPORT_LEN
            )
            report_ids  = report_ids.to(device)
            report_mask = report_mask.to(device)

            # ===== Step 4: Get BioGPT token embeddings =====
            token_embeds  = model.decoder.biogpt.biogpt.embed_tokens(report_ids)
            # token_embeds: [1, MAX_REPORT_LEN, 1024]

            # ===== Step 5: Prepend prefix — position 0 is the conditioning =====
            inputs_embeds = torch.cat([prefix_embed, token_embeds], dim=1)
            # inputs_embeds: [1, MAX_REPORT_LEN+1, 1024]

            prefix_mask   = torch.ones(1, 1, device=device, dtype=report_mask.dtype)
            full_mask     = torch.cat([prefix_mask, report_mask], dim=1)
            # full_mask: [1, MAX_REPORT_LEN+1]

            # ===== Step 6: Forward pass with output_attentions=True =====
            outputs = model.decoder.biogpt(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                output_attentions=True
            )
            # outputs.attentions: tuple of 24 x [1, num_heads, seq_len, seq_len]

        attentions = outputs.attentions
        if attentions is None:
            print("  Warning: no attention weights returned — skipping sample")
            continue

        # ===== Step 7: Generate text for qualitative comparison =====
        with torch.no_grad():
            generated_texts = model.generate_report(
                images, gaze_heatmaps, gaze_sequences, gaze_seq_lengths,
                text_token_ids, text_attention_masks
            )
        generated_text = generated_texts[0]

        # ===== Step 8: Compute rollout and per-layer scores =====
        rollout        = compute_attention_rollout(attentions)
        layer_scores   = get_per_layer_conditioning_attention(attentions)
        rollout_score  = rollout[1:, 0].mean()  # mean attention to pos 0, excl. pos 0 itself

        all_rollout_scores.append(rollout_score)
        all_layer_scores.append(layer_scores)
        all_labels.append(label)

        print(f"  Rollout conditioning score: {rollout_score:.4f}")
        print(f"  Reference (80): {reference_text[:80]}")
        print(f"  Generated (80): {generated_text[:80]}")

        # ===== Step 9: Save per-sample plot =====
        save_path = os.path.join(
            OUTPUT_DIR,
            f'rollout_sample_{batch_idx:02d}_{"normal" if label==0 else "abnormal"}.png'
        )
        plot_rollout_heatmap(
            rollout, layer_scores,
            reference_text, generated_text,
            label, save_path
        )

    # ===== Summary plot =====
    if all_rollout_scores:
        summary_path = os.path.join(OUTPUT_DIR, 'conditioning_attention_summary.png')
        plot_summary(all_rollout_scores, all_layer_scores, all_labels, summary_path)

        print(f"\n{'='*60}")
        print(f"Mean conditioning attention (rollout): {np.mean(all_rollout_scores):.4f}")
        print(f"  → If close to 0: decoder largely ignoring teacher features")
        print(f"  → If close to 1: decoder strongly conditioned on teacher features")
        print(f"\nAll outputs saved to: {OUTPUT_DIR}")
        print("="*60)


if __name__ == '__main__':
    main()