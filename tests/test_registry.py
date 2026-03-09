from types import SimpleNamespace

import pytest

import src.model  # noqa: F401
from src.model.base import BaseMultiHeadClassifier
from src.model.bert import BertMultiHeadClassifier
from src.model.gemma import GemmaMultiHeadClassifier
from src.model.llama import LlamaMultiHeadClassifier
from src.model.registry import (
    MODEL_REGISTRY,
    _candidate_keys,
    get_model_class,
    register_model,
)


@register_model("UnitTestAlias", "UnitTest-Extra")
class RegisteredTempClassifier(BaseMultiHeadClassifier):
    @classmethod
    def get_lora_target_modules(cls):
        return ("proj",)

    def _get_hidden_size(self) -> int:
        return int(self.config.hidden_size)

    def _pool(self, last_hidden_state, input_ids, attention_mask):
        return last_hidden_state[:, 0]


def test_builtin_model_aliases_are_registered():
    assert MODEL_REGISTRY["bert"] is BertMultiHeadClassifier
    assert MODEL_REGISTRY["gemma"] is GemmaMultiHeadClassifier
    assert MODEL_REGISTRY["llama"] is LlamaMultiHeadClassifier
    assert MODEL_REGISTRY["mistral"] is LlamaMultiHeadClassifier


def test_register_model_lowercases_all_aliases():
    assert MODEL_REGISTRY["unittestalias"] is RegisteredTempClassifier
    assert MODEL_REGISTRY["unittest-extra"] is RegisteredTempClassifier


def test_candidate_keys_split_and_deduplicate_parts():
    assert _candidate_keys("google/gemma-3-270m", "gemma3") == (
        "gemma3",
        "google/gemma-3-270m",
        "google",
        "gemma-3-270m",
        "google/gemma",
        "3",
        "270m",
    )


def test_get_model_class_uses_detected_model_type(monkeypatch):
    monkeypatch.setattr(
        "src.model.registry.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="deberta-v3"),
    )

    model_class = get_model_class("some/private-model")

    assert model_class is BertMultiHeadClassifier


def test_get_model_class_falls_back_to_name_when_config_lookup_fails(
    monkeypatch,
):
    def raise_lookup_error(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(
        "src.model.registry.AutoConfig.from_pretrained",
        raise_lookup_error,
    )

    model_class = get_model_class("mistral-7b")

    assert model_class is LlamaMultiHeadClassifier


def test_get_model_class_raises_for_unknown_architecture(monkeypatch):
    monkeypatch.setattr(
        "src.model.registry.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="unsupported"),
    )

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        get_model_class("unsupported-model")
