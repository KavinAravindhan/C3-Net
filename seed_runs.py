"""
5-seed final runs for DeLong's test.

Uses the best hyperparameter config found per modality during hparam_search.
For each seed run:
  - Trains from scratch with that config + fixed seed
  - Evaluates best checkpoint on test set
  - Saves predicted probabilities and metrics to disk

Usage:
    cd /home/kavin/C3-Net
    conda activate c3net
    python seed_runs.py

    # Single modality:
    python seed_runs.py --modality image_only
"""

import os
import copy
import json
import argparse
import numpy as np
import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader

from data.dataset import MIMICEyeDataset, collate_fn
from train import C3NetTrainer


# ==============================================================================
# Best configs from hparam search (taken directly from results.csv analysis)
# ==============================================================================

BEST_CONFIGS = {
    'image_only': {
        'learning_rate': 1e-5,
        'dropout':       0.4,
        'weight_decay':  0.1,
        'batch_size':    16,
    },
    'image_gaze': {
        'learning_rate':    5e-5,
        'dropout':          0.2,
        'weight_decay':     0.1,
        'batch_size':       16,
        'lambda_gaze_pred': 0.3,
    },
    'image_text': {
        'learning_rate':    5e-5,
        'dropout':          0.2,
        'weight_decay':     0.01,
        'batch_size':       32,
        'bert_freeze_epochs': 5,
    },
    'image_gaze_text': {
        'learning_rate':    1e-4,
        'dropout':          0.2,
        'weight_decay':     0.001,
        'batch_size':       16,
        'lambda_gaze_pred': 0.7,
        'bert_freeze_epochs': 3,
    },
}

SEEDS          = [42, 123, 456, 789, 1024]
DATASET_ROOT   = '/media/16TB_Storage/kavin/dataset/processed_mimic_eye'
OUTPUT_DIR     = '/media/16TB_Storage/kavin/models/c3-net/seed_runs'


# ==============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def apply_best_config(base_config, modality, best):
    cfg = copy.deepcopy(base_config)
    cfg['model']['modality'] = modality

    cfg['training']['learning_rate'] = best['learning_rate']
    cfg['training']['weight_decay']  = best['weight_decay']
    cfg['training']['batch_size']    = best['batch_size']
    cfg['model']['dropout']          = best['dropout']
    cfg['model']['gaze_encoder']['dropout'] = best['dropout']

    if 'lambda_gaze_pred' in best:
        cfg['training']['lambda_gaze_pred'] = best['lambda_gaze_pred']
    if 'bert_freeze_epochs' in best:
        cfg['model']['bert_freeze_epochs'] = best['bert_freeze_epochs']

    return cfg


def build_loaders(config):
    train_dataset = MIMICEyeDataset(root_dir=DATASET_ROOT, split='train', config=config)
    val_dataset   = MIMICEyeDataset(root_dir=DATASET_ROOT, split='val',   config=config)
    test_dataset  = MIMICEyeDataset(root_dir=DATASET_ROOT, split='test',  config=config)

    bs = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=bs, shuffle=False,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# ==============================================================================

def run_seed_runs(modalities, base_config):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Summary container: modality → list of per-seed results
    all_results = {m: [] for m in modalities}

    for modality in modalities:
        print("\n" + "="*80)
        print(f"SEED RUNS — {modality.upper()}")
        print("="*80)

        best = BEST_CONFIGS[modality]
        modality_dir = os.path.join(OUTPUT_DIR, modality)
        os.makedirs(modality_dir, exist_ok=True)

        for seed in SEEDS:
            print(f"\n--- {modality} | Seed {seed} ---")
            set_seed(seed)

            cfg       = apply_best_config(base_config, modality, best)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name  = f"seed_{modality}_s{seed}_{timestamp}"
            save_dir  = os.path.join(modality_dir, f'seed_{seed}')
            os.makedirs(save_dir, exist_ok=True)

            train_loader, val_loader, test_loader = build_loaders(cfg)

            # Train
            trainer      = C3NetTrainer(cfg, run_name=run_name)
            best_metrics = trainer.train(
                train_loader, val_loader,
                num_epochs=cfg['training']['num_epochs'],
                save_dir=save_dir,
                save_periodic_checkpoints=False,
                finish_wandb=False        # keep run open
            )

            # Load best checkpoint then evaluate on test
            ckpt_path = os.path.join(save_dir, 'best_model.pth')
            ckpt      = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            trainer.teacher.load_state_dict(ckpt['teacher_state_dict'])
            trainer.teacher.to(trainer.device)

            test_metrics, test_probs, test_labels = trainer.evaluate_test(test_loader)

            # Save probabilities and labels for DeLong's test
            probs_path = os.path.join(save_dir, 'test_probs.json')
            with open(probs_path, 'w') as f:
                json.dump({
                    'modality':    modality,
                    'seed':        seed,
                    'test_probs':  test_probs,
                    'test_labels': test_labels,
                    'test_metrics': test_metrics,
                    'val_metrics':  best_metrics,
                }, f, indent=2)

            # Log to wandb
            import wandb
            wandb.log({
                "test/accuracy":  test_metrics['accuracy'] * 100,
                "test/precision": test_metrics['precision'] * 100,
                "test/recall":    test_metrics['recall'] * 100,
                "test/f1":        test_metrics['f1'] * 100,
                "test/auc":       test_metrics['auc'],
            })
            wandb.finish()

            all_results[modality].append({
                'seed':         seed,
                'val_auc':      best_metrics['auc'],
                'test_auc':     test_metrics['auc'],
                'test_acc':     test_metrics['accuracy'],
                'test_f1':      test_metrics['f1'],
                'test_precision': test_metrics['precision'],
                'test_recall':  test_metrics['recall'],
            })

            print(f"  Seed {seed} done — Test AUC: {test_metrics['auc']:.4f}")

        # Print modality summary
        aucs = [r['test_auc'] for r in all_results[modality]]
        print(f"\n{modality} — Test AUC across 5 seeds:")
        print(f"  Mean ± Std: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Min: {np.min(aucs):.4f}  Max: {np.max(aucs):.4f}")

        # Save modality-level summary
        summary_path = os.path.join(modality_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results[modality], f, indent=2)

    # Final summary across all modalities
    print("\n" + "="*80)
    print("SEED RUNS COMPLETE — FINAL SUMMARY")
    print("="*80)
    print(f"\n{'Modality':<20} {'Mean AUC':>10} {'Std AUC':>10} {'Mean Acc':>10} {'Mean F1':>10}")
    print("-"*60)
    for modality in modalities:
        results = all_results[modality]
        aucs  = [r['test_auc'] for r in results]
        accs  = [r['test_acc'] for r in results]
        f1s   = [r['test_f1']  for r in results]
        print(f"{modality:<20} {np.mean(aucs):>10.4f} {np.std(aucs):>10.4f} "
              f"{np.mean(accs)*100:>9.2f}% {np.mean(f1s)*100:>9.2f}%")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Run delong_test.py next to compute p-values.")
    print("="*80)


# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--modality', type=str, default=None,
        choices=['image_only', 'image_gaze', 'image_text', 'image_gaze_text'],
        help='Run seeds for a single modality. Omit to run all 4.'
    )
    args = parser.parse_args()

    with open('configs/config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    modalities = [args.modality] if args.modality else [
        'image_only', 'image_gaze', 'image_text', 'image_gaze_text'
    ]

    run_seed_runs(modalities, base_config)


if __name__ == '__main__':
    main()