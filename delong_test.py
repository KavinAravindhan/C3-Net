"""
DeLong's test for comparing AUC between modality pairs.

Loads predicted probabilities saved by seed_runs.py and computes
pairwise statistical significance tests.

Usage:
    cd /home/kavin/C3-Net
    conda activate c3net
    python delong_test.py
"""

import os
import json
import numpy as np
from itertools import combinations

from sklearn.metrics import roc_auc_score
from utils.metrics import delong_test


SEED_RUNS_DIR = '/media/16TB_Storage/kavin/models/c3-net/seed_runs'
MODALITIES    = ['image_only', 'image_gaze', 'image_text', 'image_gaze_text']
SEEDS         = [42, 123, 456, 789, 1024]


def load_probs(modality, seed):
    """Load predicted probabilities for a given modality and seed."""
    path = os.path.join(SEED_RUNS_DIR, modality, f'seed_{seed}', 'test_probs.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data['test_labels'], data['test_probs']


def average_probs_across_seeds(modality):
    """
    Average predicted probabilities across all 5 seeds for a modality.
    Averaging probability outputs gives a more stable estimate than picking one seed.
    """
    all_probs  = []
    all_labels = None

    for seed in SEEDS:
        labels, probs = load_probs(modality, seed)
        all_probs.append(probs)
        if all_labels is None:
            all_labels = labels

    avg_probs = np.mean(all_probs, axis=0).tolist()
    return all_labels, avg_probs


def main():
    print("="*80)
    print("DeLong's Test — C3-Net Modality Comparison")
    print("="*80)
    print("Reference: DeLong et al. (1988), Biometrics")
    print()

    # Load averaged probs per modality
    modality_data = {}
    print("Loading predicted probabilities...")
    for modality in MODALITIES:
        try:
            labels, probs = average_probs_across_seeds(modality)
            modality_data[modality] = (labels, probs)
            auc = roc_auc_score(labels, probs)
            print(f"  {modality:<20} — AUC (avg across seeds): {auc:.4f}")
        except FileNotFoundError as e:
            print(f"  {modality:<20} — NOT FOUND: {e}")

    # Pairwise DeLong's test
    print("\n" + "="*80)
    print("Pairwise DeLong's Test Results")
    print("="*80)
    print(f"\n{'Modality A':<22} {'Modality B':<22} {'AUC A':>8} {'AUC B':>8} {'Z':>8} {'p-value':>10} {'Sig':>6}")
    print("-"*90)

    pairs   = list(combinations(MODALITIES, 2))
    results = []

    for mod_a, mod_b in pairs:
        if mod_a not in modality_data or mod_b not in modality_data:
            continue

        labels_a, probs_a = modality_data[mod_a]
        labels_b, probs_b = modality_data[mod_b]

        auc_a, auc_b, z, p = delong_test(labels_a, probs_a, labels_b, probs_b)

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        print(f"{mod_a:<22} {mod_b:<22} {auc_a:>8.4f} {auc_b:>8.4f} {z:>8.3f} {p:>10.4f} {sig:>6}")

        results.append({
            'modality_a':  mod_a,
            'modality_b':  mod_b,
            'auc_a':       round(auc_a, 4),
            'auc_b':       round(auc_b, 4),
            'z_stat':      round(z, 4),
            'p_value':     round(p, 6),
            'significant': bool(p < 0.05),
        })

    # Key comparison for paper
    print("\n" + "="*80)
    print("Key Comparison for Paper: image_only vs image_gaze_text")
    print("="*80)
    key = next((r for r in results
                if set([r['modality_a'], r['modality_b']]) == {'image_only', 'image_gaze_text'}), None)
    if key:
        print(f"  AUC improvement: {key['auc_a']:.4f} → {key['auc_b']:.4f}")
        print(f"  Z-statistic:     {key['z_stat']:.3f}")
        print(f"  p-value:         {key['p_value']:.4f}")
        print(f"  Significant (p<0.05): {key['significant']}")

    # Save results
    out_path = os.path.join(SEED_RUNS_DIR, 'delong_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFull results saved to: {out_path}")
    print("\nSignificance legend: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
    print("="*80)


if __name__ == '__main__':
    main()