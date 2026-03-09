import torch

from src.model.bert import BertMultiHeadClassifier


def test_bert_classifier_uses_encoder_mode():
    assert BertMultiHeadClassifier.AUTO_MODEL_KIND == "encoder"


def test_bert_default_lora_targets_cover_bert_and_deberta_names():
    assert BertMultiHeadClassifier.get_lora_target_modules() == (
        "query",
        "key",
        "value",
        "query_proj",
        "key_proj",
        "value_proj",
    )


def test_bert_pool_uses_cls_token(dummy_backbone):
    model = BertMultiHeadClassifier(
        dummy_backbone, num_categories=3, num_subcategories=4
    )
    hidden = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)

    pooled = model._pool(hidden, input_ids=None, attention_mask=None)

    assert torch.equal(pooled, hidden[:, 0])


def test_bert_hidden_size_comes_from_backbone_config(dummy_backbone):
    model = BertMultiHeadClassifier(
        dummy_backbone, num_categories=3, num_subcategories=4
    )

    assert model._get_hidden_size() == 8
