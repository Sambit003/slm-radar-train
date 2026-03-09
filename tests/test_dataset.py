import json
import pickle

import numpy as np
import pytest
import torch

from src.data.dataset import ThreatDataset


def test_threat_dataset_loads_splits_and_fits_encoders(sample_jsonl_files):
    dataset = ThreatDataset(
        sample_jsonl_files["train"],
        sample_jsonl_files["validation"],
        sample_jsonl_files["test"],
    )

    assert set(dataset.dataset.keys()) == {"train", "validation", "test"}
    assert {"true", "false"}.issubset(set(dataset.encoders["threat"].classes_))
    assert "unknown" in set(dataset.encoders["category"].classes_)
    assert "unknown" in set(dataset.encoders["subcategory"].classes_)


def test_save_encoders_writes_pickle_file(tmp_path, sample_jsonl_files):
    dataset = ThreatDataset(sample_jsonl_files["train"])

    dataset.save_encoders(str(tmp_path))

    with (tmp_path / "encoders.pkl").open("rb") as handle:
        encoders = pickle.load(handle)

    assert set(encoders.keys()) == {"threat", "category", "subcategory"}


def test_get_hf_dataset_returns_torch_formatted_splits(
    sample_jsonl_files, mock_tokenizer
):
    dataset = ThreatDataset(
        sample_jsonl_files["train"],
        sample_jsonl_files["validation"],
        sample_jsonl_files["test"],
    )

    processed = dataset.get_hf_dataset(
        mock_tokenizer, max_length=6, batch_size=2
    )
    example = processed["train"][0]

    assert set(example.keys()) == {
        "input_ids",
        "attention_mask",
        "labels_threat",
        "labels_category",
        "labels_subcategory",
    }
    assert isinstance(example["input_ids"], torch.Tensor)
    assert example["input_ids"].shape == (6,)


def test_get_hf_dataset_maps_none_values_to_unknown(
    sample_jsonl_files, mock_tokenizer
):
    dataset = ThreatDataset(sample_jsonl_files["train"])

    processed = dataset.get_hf_dataset(mock_tokenizer, max_length=5)
    unknown_category = dataset.encoders["category"].transform(["unknown"])[0]
    unknown_subcategory = dataset.encoders["subcategory"].transform(
        ["unknown"]
    )[0]

    assert processed["train"][1]["labels_category"].item() == unknown_category
    assert (
        processed["train"][1]["labels_subcategory"].item()
        == unknown_subcategory
    )


def test_fit_encoders_handles_missing_category_columns(tmp_path):
    train_file = tmp_path / "train.jsonl"
    rows = [
        {"prompt": "message one", "is_threat": True},
        {"prompt": "message two", "is_threat": False},
    ]
    with train_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    dataset = ThreatDataset(str(train_file))

    assert list(dataset.encoders["category"].classes_) == ["unknown"]
    assert list(dataset.encoders["subcategory"].classes_) == ["unknown"]


def test_get_class_weights_are_normalized_around_one(sample_jsonl_files):
    dataset = ThreatDataset(sample_jsonl_files["train"])

    threat_weights, category_weights, subcategory_weights = (
        dataset.get_class_weights()
    )

    assert np.isclose(threat_weights.mean(), 1.0)
    assert np.isclose(category_weights.mean(), 1.0)
    assert np.isclose(subcategory_weights.mean(), 1.0)


def test_get_class_weights_handle_single_unknown_class(tmp_path):
    train_file = tmp_path / "train.jsonl"
    rows = [
        {"prompt": "message one", "is_threat": True},
        {"prompt": "message two", "is_threat": False},
    ]
    with train_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    dataset = ThreatDataset(str(train_file))
    _, category_weights, subcategory_weights = dataset.get_class_weights()

    assert np.array_equal(category_weights, np.array([1.0], dtype=np.float32))
    assert np.array_equal(
        subcategory_weights, np.array([1.0], dtype=np.float32)
    )


def test_get_hf_dataset_raises_for_unseen_labels(tmp_path, mock_tokenizer):
    train_file = tmp_path / "train.jsonl"
    test_file = tmp_path / "test.jsonl"

    train_rows = [
        {
            "prompt": "urgent malware report",
            "is_threat": "true",
            "category": "malware",
            "sub-category": "trojan",
        },
        {
            "prompt": "normal user request",
            "is_threat": "false",
            "category": "phishing",
            "sub-category": "credential-theft",
        },
    ]
    test_rows = [
        {
            "prompt": "general product question",
            "is_threat": "false",
            "category": "benign",
            "sub-category": "general",
        }
    ]

    with train_file.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row) + "\n")

    with test_file.open("w", encoding="utf-8") as handle:
        for row in test_rows:
            handle.write(json.dumps(row) + "\n")

    dataset = ThreatDataset(str(train_file), test_file=str(test_file))

    with pytest.raises(ValueError, match="previously unseen labels"):
        dataset.get_hf_dataset(mock_tokenizer, max_length=6)
