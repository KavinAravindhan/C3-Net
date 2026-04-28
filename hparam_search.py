"""
Hyperparameter search for C3-Net teacher model.

Runs 20 random configs per modality (80 total).
Saves results to CSV and best checkpoint per modality.

Usage:
    cd /home/kavin/C3-Net
    conda activate c3net
    python hparam_search.py

    # To search a single modality only:
    python hparam_search.py --modality image_only
"""

import os
import csv
import copy
import random
import argparse
import shutil
from datetime import datetime

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import MIMICEyeDataset, collate_fn
from train import C3NetTrainer


# ==============================================================================
# Search spaces
# ==============================================================================

UNIVERSAL_SPACE = {
    'learning_rate': [1e-4, 5e-5, 1e-5],
    'dropout':       [0.2, 0.3, 0.4],
    'weight_decay':  [0.001, 0.01, 0.1],
    'batch_size':    [16, 32],
}

GAZE_SPACE = {
    'lambda_gaze_pred': [0.3, 0.5, 0.7],
}

TEXT_SPACE = {
    'bert_freeze_epochs': [3, 5, 10],
}

MODALITY_SPACES = {
    'image_only':      {**UNIVERSAL_SPACE},
    'image_gaze':      {**UNIVERSAL_SPACE, **GAZE_SPACE},
    'image_text':      {**UNIVERSAL_SPACE, **TEXT_SPACE},
    'image_gaze_text': {**UNIVERSAL_SPACE, **GAZE_SPACE, **TEXT_SPACE},
}

NUM_CONFIGS    = 20
DATASET_ROOT   = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye'
CHECKPOINT_DIR = '/media/16TB_Storage/kavin/models/c3-net/hparam_search'
RESULTS_CSV    = os.path.join(CHECKPOINT_DIR, 'results.csv')


# ==============================================================================
# Helpers
# ==============================================================================

def sample_config(space):
    """Sample one random config from the search space."""
    return {k: random.choice(v) for k, v in space.items()}


def apply_overrides(base_config, sampled):
    """
    Deep copy base config and apply sampled hyperparameters.
    Returns the modified config.
    """
    cfg = copy.deepcopy(base_config)

    # Universal
    if 'learning_rate' in sampled:
        cfg['training']['learning_rate'] = sampled['learning_rate']
    if 'dropout' in sampled:
        cfg['model']['dropout'] = sampled['dropout']
        cfg['model']['gaze_encoder']['dropout'] = sampled['dropout']
    if 'weight_decay' in sampled:
        cfg['training']['weight_decay'] = sampled['weight_decay']
    if 'batch_size' in sampled:
        cfg['training']['batch_size'] = sampled['batch_size']

    # Gaze-specific
    if 'lambda_gaze_pred' in sampled:
        cfg['training']['lambda_gaze_pred'] = sampled['lambda_gaze_pred']

    # Text-specific
    if 'bert_freeze_epochs' in sampled:
        cfg['model']['bert_freeze_epochs'] = sampled['bert_freeze_epochs']

    return cfg


def init_csv(path, modalities):
    """Write CSV header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        'modality', 'run_id',
        'learning_rate', 'dropout', 'weight_decay', 'batch_size',
        'lambda_gaze_pred', 'bert_freeze_epochs',
        'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_auc',
        'best_epoch'
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return fieldnames


def append_csv(path, fieldnames, row):
    """Append one result row to CSV."""
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def build_loaders(config):
    """Build train/val DataLoaders for a given config (batch_size may vary)."""
    train_dataset = MIMICEyeDataset(
        root_dir=DATASET_ROOT, split='train', config=config
    )
    val_dataset = MIMICEyeDataset(
        root_dir=DATASET_ROOT, split='val', config=config
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader


# ==============================================================================
# Main search loop
# ==============================================================================

def run_search(modalities, base_config, num_configs=NUM_CONFIGS):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    fieldnames = init_csv(RESULTS_CSV, modalities)

    # Track global best per modality
    best_per_modality = {m: {'auc': 0.0, 'checkpoint': None, 'config': None} for m in modalities}

    for modality in modalities:
        print("\n" + "="*80)
        print(f"HYPERPARAMETER SEARCH — {modality.upper()}")
        print("="*80)

        space = MODALITY_SPACES[modality]

        # Sample num_configs random configs (deduplicated)
        sampled_configs = []
        seen = set()
        attempts = 0
        while len(sampled_configs) < num_configs and attempts < num_configs * 10:
            cfg = sample_config(space)
            key = str(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                sampled_configs.append(cfg)
            attempts += 1

        print(f"Sampled {len(sampled_configs)} unique configs to evaluate\n")

        for run_idx, sampled in enumerate(sampled_configs):
            print(f"\n--- {modality} | Run {run_idx+1}/{len(sampled_configs)} ---")
            print(f"    Config: {sampled}")

            # Build run config
            cfg = apply_overrides(base_config, sampled)
            cfg['model']['modality'] = modality

            # Temp checkpoint dir for this run
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name  = f"hparam_{modality}_r{run_idx+1}_{timestamp}"
            tmp_dir   = os.path.join(CHECKPOINT_DIR, 'tmp', run_name)
            os.makedirs(tmp_dir, exist_ok=True)

            # Build loaders (respects batch_size override)
            train_loader, val_loader = build_loaders(cfg)

            # Train
            try:
                trainer = C3NetTrainer(cfg, train_dataset=train_loader.dataset, run_name=run_name)
                best_metrics = trainer.train(
                    train_loader,
                    val_loader,
                    num_epochs=cfg['training']['num_epochs'],
                    save_dir=tmp_dir,
                    save_periodic_checkpoints=False   # saves disk space during search
                )
            except Exception as e:
                print(f"  ✗ Run failed: {e}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            # Record result
            row = {
                'modality':          modality,
                'run_id':            run_name,
                'learning_rate':     sampled.get('learning_rate', 'N/A'),
                'dropout':           sampled.get('dropout', 'N/A'),
                'weight_decay':      sampled.get('weight_decay', 'N/A'),
                'batch_size':        sampled.get('batch_size', 'N/A'),
                'lambda_gaze_pred':  sampled.get('lambda_gaze_pred', 'N/A'),
                'bert_freeze_epochs':sampled.get('bert_freeze_epochs', 'N/A'),
                'val_accuracy':      round(best_metrics['accuracy'] * 100, 4),
                'val_precision':     round(best_metrics['precision'] * 100, 4),
                'val_recall':        round(best_metrics['recall'] * 100, 4),
                'val_f1':            round(best_metrics['f1'] * 100, 4),
                'val_auc':           round(best_metrics['auc'], 6),
                'best_epoch':        trainer.best_epoch,
            }
            append_csv(RESULTS_CSV, fieldnames, row)

            # Update global best for this modality
            if best_metrics['auc'] > best_per_modality[modality]['auc']:
                best_per_modality[modality]['auc']        = best_metrics['auc']
                best_per_modality[modality]['config']     = sampled
                best_per_modality[modality]['best_metrics'] = best_metrics

                # Copy best checkpoint
                best_ckpt_dst = os.path.join(CHECKPOINT_DIR, f'best_{modality}.pth')
                best_ckpt_src = os.path.join(tmp_dir, 'best_model.pth')
                if os.path.exists(best_ckpt_src):
                    shutil.copy(best_ckpt_src, best_ckpt_dst)
                    print(f"\n  ★ New best for {modality}! AUC={best_metrics['auc']:.4f} → saved to best_{modality}.pth")

            # Remove temp dir to save disk
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ==== Final summary ====
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*80)
    for modality in modalities:
        best = best_per_modality[modality]
        if best['config']:
            m = best['best_metrics']
            print(f"\n{modality}:")
            print(f"  Best config: {best['config']}")
            print(f"  Val AUC:      {m['auc']:.4f}")
            print(f"  Val Accuracy: {m['accuracy']*100:.2f}%")
            print(f"  Val F1:       {m['f1']*100:.2f}%")
    print(f"\nAll results saved to: {RESULTS_CSV}")
    print(f"Best checkpoints in:  {CHECKPOINT_DIR}/best_{{modality}}.pth")
    print("="*80)

    return best_per_modality


# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modality', type=str, default=None,
        choices=['image_only', 'image_gaze', 'image_text', 'image_gaze_text'],
        help='Run search for a single modality only. Omit to run all 4.'
    )
    args = parser.parse_args()

    with open('configs/config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    modalities = [args.modality] if args.modality else [
        'image_only', 'image_gaze', 'image_text', 'image_gaze_text'
    ]

    run_search(modalities, base_config, num_configs=NUM_CONFIGS)


if __name__ == '__main__':
    main()
