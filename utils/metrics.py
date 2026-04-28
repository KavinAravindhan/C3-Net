"""
utils/metrics.py — Centralized evaluation metrics for C3-Net.

Classification:
    - DeLong's test for AUC comparison between modality pairs
      Ref: DeLong et al. (1988), Biometrics

Generation:
    - ROUGE-L (via rouge_score)
    - BERTScore (via bert_score)
"""

import numpy as np
from scipy import stats


# ==============================================================================
# DeLong's Test
# ==============================================================================

def compute_auc_variance(labels, probs):
    """
    Compute AUC and its variance using DeLong's structural components method.

    Args:
        labels: array-like of ground truth binary labels
        probs:  array-like of predicted probabilities for class 1

    Returns:
        auc:      float
        variance: float
    """
    labels = np.array(labels)
    probs  = np.array(probs)

    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    n_pos = len(pos_probs)
    n_neg = len(neg_probs)

    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0

    # Structural components
    # V10[i] = P(score of pos_i > neg) — placement of positive i
    # V01[j] = P(score of pos > neg_j) — placement of negative j
    V10 = np.array([
        np.mean((pos_probs[i] > neg_probs) + 0.5 * (pos_probs[i] == neg_probs))
        for i in range(n_pos)
    ])
    V01 = np.array([
        np.mean((pos_probs > neg_probs[j]) + 0.5 * (pos_probs == neg_probs[j]))
        for j in range(n_neg)
    ])

    auc = np.mean(V10)

    S10  = np.var(V10, ddof=1) / n_pos
    S01  = np.var(V01, ddof=1) / n_neg
    var  = S10 + S01

    return auc, var


def delong_test(labels_a, probs_a, labels_b, probs_b):
    """
    DeLong's test comparing AUC of two models on the same test set.

    Args:
        labels_a, probs_a: ground truth + predicted probs for model A
        labels_b, probs_b: ground truth + predicted probs for model B

    Returns:
        auc_a:   float
        auc_b:   float
        z_stat:  float
        p_value: float (two-tailed)
    """
    labels_a = np.array(labels_a)
    labels_b = np.array(labels_b)
    probs_a  = np.array(probs_a)
    probs_b  = np.array(probs_b)

    auc_a, var_a = compute_auc_variance(labels_a, probs_a)
    auc_b, var_b = compute_auc_variance(labels_b, probs_b)

    pos_a = probs_a[labels_a == 1]
    neg_a = probs_a[labels_a == 0]
    pos_b = probs_b[labels_b == 1]
    neg_b = probs_b[labels_b == 0]

    n_pos = len(pos_a)
    n_neg = len(neg_a)

    V10_a = np.array([
        np.mean((pos_a[i] > neg_a) + 0.5 * (pos_a[i] == neg_a))
        for i in range(n_pos)
    ])
    V10_b = np.array([
        np.mean((pos_b[i] > neg_b) + 0.5 * (pos_b[i] == neg_b))
        for i in range(n_pos)
    ])
    V01_a = np.array([
        np.mean((pos_a > neg_a[j]) + 0.5 * (pos_a == neg_a[j]))
        for j in range(n_neg)
    ])
    V01_b = np.array([
        np.mean((pos_b > neg_b[j]) + 0.5 * (pos_b == neg_b[j]))
        for j in range(n_neg)
    ])

    cov10 = np.cov(V10_a, V10_b)[0, 1] / n_pos if n_pos > 1 else 0.0
    cov01 = np.cov(V01_a, V01_b)[0, 1] / n_neg if n_neg > 1 else 0.0
    cov   = cov10 + cov01

    var_diff = var_a + var_b - 2 * cov

    if var_diff <= 0:
        return auc_a, auc_b, 0.0, 1.0

    z_stat  = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

    return auc_a, auc_b, z_stat, p_value


# ==============================================================================
# Generation Metrics
# ==============================================================================

def compute_rouge_l(predictions, references):
    """
    Compute ROUGE-L F1 scores for a list of generated vs reference texts.
    Requires: pip install rouge_score

    Args:
        predictions: list[str] — generated transcriptions
        references:  list[str] — ground truth transcriptions

    Returns:
        scores: dict with keys 'rougeL_mean', 'rougeL_scores' (per-sample list)
    """
    from rouge_score import rouge_scorer

    scorer  = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores  = []

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score['rougeL'].fmeasure)

    return {
        'rougeL_mean':   float(np.mean(scores)),
        'rougeL_scores': scores,
    }


def compute_bertscore(predictions, references, model_type='distilbert-base-uncased'):
    """
    Compute BERTScore F1 for a list of generated vs reference texts.
    Requires: pip install bert_score

    Args:
        predictions: list[str] — generated transcriptions
        references:  list[str] — ground truth transcriptions
        model_type:  str — underlying BERT model for scoring
                     Use 'distilbert-base-uncased' for speed,
                     'microsoft/deberta-xlarge-mnli' for highest quality

    Returns:
        scores: dict with keys 'bertscore_mean', 'bertscore_scores' (per-sample list)
    """
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions,
        references,
        model_type=model_type,
        verbose=False
    )

    f1_scores = F1.tolist()

    return {
        'bertscore_mean':   float(np.mean(f1_scores)),
        'bertscore_scores': f1_scores,
    }


def compute_generation_metrics(predictions, references):
    """
    Compute all generation metrics at once.
    Convenience wrapper for ROUGE-L + BERTScore.

    Args:
        predictions: list[str]
        references:  list[str]

    Returns:
        metrics: dict with rougeL_mean, bertscore_mean, and per-sample scores
    """
    rouge  = compute_rouge_l(predictions, references)
    bscore = compute_bertscore(predictions, references)

    return {**rouge, **bscore}