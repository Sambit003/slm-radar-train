import torch

from src.model.llama import LlamaMultiHeadClassifier
from src.model.registry import MODEL_REGISTRY


def test_llama_and_mistral_aliases_point_to_same_classifier():
    assert MODEL_REGISTRY["llama"] is LlamaMultiHeadClassifier
    assert MODEL_REGISTRY["mistral"] is LlamaMultiHeadClassifier


def test_llama_pool_matches_last_non_padding_token_behavior(dummy_backbone):
    model = LlamaMultiHeadClassifier(
        dummy_backbone, num_categories=3, num_subcategories=4
    )
    hidden = torch.arange(2 * 5 * 8, dtype=torch.float32).reshape(2, 5, 8)
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 0], [9, 8, 0, 0, 0]], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.long
    )

    pooled = model._pool(
        hidden, input_ids=input_ids, attention_mask=attention_mask
    )

    assert torch.equal(pooled[0], hidden[0, 3])
    assert torch.equal(pooled[1], hidden[1, 1])


def test_llama_pool_uses_last_position_without_attention_mask(dummy_backbone):
    model = LlamaMultiHeadClassifier(
        dummy_backbone, num_categories=3, num_subcategories=4
    )
    hidden = torch.arange(2 * 5 * 8, dtype=torch.float32).reshape(2, 5, 8)

    pooled = model._pool(hidden, input_ids=None, attention_mask=None)

    assert torch.equal(pooled, hidden[:, -1])
