from types import SimpleNamespace

import torch
import transformers

from src.model.gemma import GemmaMultiHeadClassifier


def test_gemma_default_lora_target_modules_match_decoder_layers():
    assert GemmaMultiHeadClassifier.get_lora_target_modules() == (
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_gemma_pool_uses_last_non_padding_token(dummy_backbone):
    model = GemmaMultiHeadClassifier(
        dummy_backbone, num_categories=3, num_subcategories=4
    )
    hidden = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    input_ids = torch.tensor([[1, 2, 3, 0], [5, 6, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long
    )

    pooled = model._pool(
        hidden, input_ids=input_ids, attention_mask=attention_mask
    )

    assert torch.equal(pooled[0], hidden[0, 2])
    assert torch.equal(pooled[1], hidden[1, 1])


def test_gemma_pool_falls_back_to_last_position_without_mask(dummy_backbone):
    model = GemmaMultiHeadClassifier(
        dummy_backbone, num_categories=3, num_subcategories=4
    )
    hidden = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)

    pooled = model._pool(hidden, input_ids=None, attention_mask=None)

    assert torch.equal(pooled, hidden[:, -1])


def test_gemma_load_train_backbone_extracts_underlying_decoder(
    monkeypatch, dummy_backbone
):
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model=dummy_backbone),
    )

    backbone = GemmaMultiHeadClassifier.load_train_backbone(
        "google/gemma-test"
    )

    assert backbone is dummy_backbone
