import numpy as np
from transformers import EvalPrediction

from src.utils.metrics import compute_metrics


def test_compute_metrics_returns_expected_keys(sample_eval_prediction):
    metrics = compute_metrics(sample_eval_prediction)

    assert set(metrics.keys()) == {
        "threat_accuracy",
        "threat_f1",
        "threat_precision",
        "threat_recall",
        "threat_auc",
        "threat_mcc",
        "threat_opt_thresh",
        "threat_opt_precision",
        "threat_opt_recall",
        "category_accuracy",
        "category_f1",
        "category_precision",
        "category_recall",
        "subcategory_accuracy",
        "subcategory_f1",
        "subcategory_precision",
        "subcategory_recall",
        "eval_combined_accuracy",
    }


def test_compute_metrics_calculates_expected_standard_and_optimal_metrics(
    sample_eval_prediction,
):
    metrics = compute_metrics(sample_eval_prediction)

    assert metrics["threat_accuracy"] == 1.0
    assert metrics["threat_precision"] == 1.0
    assert metrics["threat_recall"] == 1.0
    assert metrics["threat_f1"] == 1.0
    assert metrics["threat_opt_thresh"] == 0.35
    assert metrics["threat_opt_precision"] == 1.0
    assert metrics["threat_opt_recall"] == 1.0
    assert metrics["category_accuracy"] == 1.0
    assert metrics["subcategory_accuracy"] == 1.0
    assert metrics["eval_combined_accuracy"] == 1.0


def test_compute_metrics_falls_back_to_best_available_recall():
    probs = np.array([0.95, 0.80, 0.20, 0.10, 0.90], dtype=np.float32)
    logits_positive = np.log(probs / (1.0 - probs))
    logits_threat = np.stack(
        [np.zeros_like(logits_positive), logits_positive], axis=1
    )
    labels = np.array([1, 1, 1, 1, 0], dtype=np.int64)

    prediction = EvalPrediction(
        predictions=(
            logits_threat,
            np.tile([[3.0, 1.0, 0.0]], (5, 1)).astype(np.float32),
            np.tile([[3.0, 1.0]], (5, 1)).astype(np.float32),
        ),
        label_ids=(
            labels,
            np.zeros(5, dtype=np.int64),
            np.zeros(5, dtype=np.int64),
        ),
    )

    metrics = compute_metrics(prediction)

    assert metrics["threat_opt_thresh"] == 0.3
    assert metrics["threat_opt_recall"] == 0.5


def test_compute_metrics_handles_single_class_auc_case():
    probs = np.array([0.90, 0.80, 0.70], dtype=np.float32)
    logits_positive = np.log(probs / (1.0 - probs))
    logits_threat = np.stack(
        [np.zeros_like(logits_positive), logits_positive], axis=1
    )
    labels = np.array([1, 1, 1], dtype=np.int64)

    prediction = EvalPrediction(
        predictions=(
            logits_threat,
            np.tile([[2.0, 1.0]], (3, 1)).astype(np.float32),
            np.tile([[2.0, 1.0]], (3, 1)).astype(np.float32),
        ),
        label_ids=(
            labels,
            np.zeros(3, dtype=np.int64),
            np.zeros(3, dtype=np.int64),
        ),
    )

    metrics = compute_metrics(prediction)

    assert metrics["threat_auc"] == 0.0
