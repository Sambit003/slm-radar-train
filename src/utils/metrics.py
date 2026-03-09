import numpy as np
import torch
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef
)
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction):
    """
    Compute metrics for multi-task classification:
    - Head 1: Threat (Binary)
    - Head 2: Category (Multi-class)
    - Head 3: Subcategory (Multi-class)
    """
    logits, labels = eval_pred

    # Unpack logits (tuple of 3) and labels (tuple of 3)
    # Note: Trainer passes labels as a tuple if the dataset returns
    # multiple label columns
    # However, depending on Trainer version,
    # 'labels' might be stacked or a tuple.
    # Our MultiHeadTrainer passes 3 label columns,
    # but standard HF Trainer might stack them if they have same shape.
    # For safety, we'll assume standard Trainer behavior where labels
    # are passed as a single tensor if easy,
    # but since we have multiple names, let's rely on prediction structure.

    # logits is a tuple: (logits_threat, logits_category, logits_subcategory)
    logits_threat, logits_category, logits_subcategory = logits

    # We need to ensure we align with how the dataset provides labels.
    # IMPORTANT: The standard compute_metrics receives `label_ids`
    # which matches the structure of `predictions`.

    # Standard behavior: labels is a tuple matching predictions tuple
    # Or labels might be a unified array. Let's handle the tuple case
    # which our model returns.

    labels_threat = labels[0]
    labels_category = labels[1]
    labels_subcategory = labels[2]

    # --- THREAT HEAD (Binary) ---
    pred_threat = np.argmax(logits_threat, axis=1)
    acc_t = accuracy_score(labels_threat, pred_threat)
    p_t, r_t, f1_t, _ = precision_recall_fscore_support(labels_threat,
                                                        pred_threat,
                                                        average='binary')
    mcc_t = matthews_corrcoef(labels_threat, pred_threat)

    # AUC needs probabilities for positive class
    probs_t = torch.softmax(torch.tensor(logits_threat), dim=1)[:, 1].numpy()
    try:
        auc_t = roc_auc_score(labels_threat, probs_t)
    except ValueError:
        auc_t = 0.0
    if not np.isfinite(auc_t):
        auc_t = 0.0

    # --- THREAT THRESHOLD SWEEP ---
    # Find threshold that maximizes precision while keeping recall >= 0.90
    best_thresh = 0.5
    best_precision = 0.0
    best_recall = 0.0

    # Sweep from 0.3 to 0.95
    # Use numpy for the range
    thresholds = np.arange(0.3, 0.96, 0.05)

    # Track metrics for each threshold
    candidates = []

    for th in thresholds:
        preds = (probs_t >= th).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            labels_threat, preds, average='binary', zero_division=0
        )
        candidates.append((th, p, r))

        # Priority: Recall >= 0.90, then maximize Precision
        if r >= 0.90:
            if p > best_precision:
                best_precision = p
                best_recall = r
                best_thresh = th

    # If no threshold met recall >= 0.90, pick the one with highest recall
    if best_precision == 0.0 and len(candidates) > 0:
        # Find candidate with highest recall
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_thresh, best_precision, best_recall = candidates[0]
        # Or alternatively: maximize precision among those with
        # recall closest to 0.9

    # Use the best threshold logic, but only if we found something reasonable
    if best_precision == 0.0:
        best_thresh = 0.5
        best_precision = p_t
        best_recall = r_t

    logger.info(
        f"Threshold Sweep:: Optimal Thresh={best_thresh:.2f}, "
        f"Precision={best_precision:.4f}, Recall={best_recall:.4f}"
    )

    # --- CATEGORY HEAD (Multi-class) ---
    pred_cat = np.argmax(logits_category, axis=1)
    acc_c = accuracy_score(labels_category, pred_cat)
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(labels_category,
                                                        pred_cat,
                                                        average='weighted',
                                                        zero_division=0)

    # --- SUBCATEGORY HEAD (Multi-class) ---
    pred_sub = np.argmax(logits_subcategory, axis=1)
    acc_s = accuracy_score(labels_subcategory, pred_sub)
    p_s, r_s, f1_s, _ = precision_recall_fscore_support(labels_subcategory,
                                                        pred_sub,
                                                        average='weighted',
                                                        zero_division=0)

    return {
        # Threat Standard Metrics (0.5)
        "threat_accuracy": acc_t,
        "threat_f1": f1_t,
        "threat_precision": p_t,
        "threat_recall": r_t,
        "threat_auc": auc_t,
        "threat_mcc": mcc_t,

        # Threat Optimal Metrics
        "threat_opt_thresh": best_thresh,
        "threat_opt_precision": best_precision,
        "threat_opt_recall": best_recall,

        # Category Metrics
        "category_accuracy": acc_c,
        "category_f1": f1_c,
        "category_precision": p_c,
        "category_recall": r_c,
        # Subcategory Metrics
        "subcategory_accuracy": acc_s,
        "subcategory_f1": f1_s,
        "subcategory_precision": p_s,
        "subcategory_recall": r_s,
        # Combined
        "eval_combined_accuracy": (acc_t + acc_c + acc_s) / 3
    }
