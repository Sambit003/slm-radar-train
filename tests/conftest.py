import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import EvalPrediction


class FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 8, vocab_size: int = 64):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.gc_enable_kwargs = None
        self.gc_disable_called = False

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding.weight.dtype

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = torch.zeros((1, 1), dtype=torch.long)
        last_hidden_state = self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=last_hidden_state)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs=None
    ):
        self.gc_enable_kwargs = gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self):
        self.gc_disable_called = True

    def save_pretrained(self, save_directory: str):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_backbone.bin").write_text("fake", encoding="utf-8")


class MockTokenizer:
    def __call__(
        self,
        prompts,
        truncation=True,
        padding="max_length",
        max_length=8,
    ):
        input_ids = []
        attention_mask = []
        for index, prompt in enumerate(prompts):
            length = min(max(1, len(str(prompt).split())), max_length)
            token_ids = list(range(index + 1, index + 1 + length))
            input_ids.append(token_ids + [0] * (max_length - length))
            attention_mask.append([1] * length + [0] * (max_length - length))
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def dummy_backbone():
    return FakeBackbone(hidden_size=8)


@pytest.fixture
def sample_batch():
    return {
        "input_ids": torch.tensor(
            [[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long
        ),
        "labels_threat": torch.tensor([1, 0], dtype=torch.long),
        "labels_category": torch.tensor([2, 1], dtype=torch.long),
        "labels_subcategory": torch.tensor([3, 0], dtype=torch.long),
    }


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def sample_jsonl_files(tmp_path):
    train_rows = [
        {
            "prompt": "urgent malware report",
            "is_threat": True,
            "category": "malware",
            "sub-category": "trojan",
        },
        {
            "prompt": "normal user request",
            "is_threat": False,
            "category": None,
            "sub-category": None,
        },
        {
            "prompt": "credential phishing alert",
            "is_threat": True,
            "category": "phishing",
            "sub-category": "credential-theft",
        },
    ]
    validation_rows = [
        {
            "prompt": "suspicious login",
            "is_threat": True,
            "category": "phishing",
            "sub-category": "credential-theft",
        }
    ]
    test_rows = [
        {
            "prompt": "general product question",
            "is_threat": False,
            "category": "phishing",
            "sub-category": "credential-theft",
        }
    ]

    def write_jsonl(path: Path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    test_file = tmp_path / "test.jsonl"
    write_jsonl(train_file, train_rows)
    write_jsonl(validation_file, validation_rows)
    write_jsonl(test_file, test_rows)

    return {
        "train": str(train_file),
        "validation": str(validation_file),
        "test": str(test_file),
    }


@pytest.fixture
def sample_eval_prediction():
    probs = np.array([0.10, 0.95, 0.85, 0.34], dtype=np.float32)
    logits_positive = np.log(probs / (1.0 - probs))
    logits_threat = np.stack(
        [np.zeros_like(logits_positive), logits_positive], axis=1
    )

    logits_category = np.array(
        [
            [5.0, 1.0, 0.0],
            [0.0, 5.0, 1.0],
            [0.0, 1.0, 5.0],
            [4.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    logits_subcategory = np.array(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )

    labels = (
        np.array([0, 1, 1, 0], dtype=np.int64),
        np.array([0, 1, 2, 0], dtype=np.int64),
        np.array([0, 1, 2, 3], dtype=np.int64),
    )
    predictions = (logits_threat, logits_category, logits_subcategory)
    return EvalPrediction(predictions=predictions, label_ids=labels)
