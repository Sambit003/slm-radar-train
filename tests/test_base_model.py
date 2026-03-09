import json

import torch
import torch.nn.functional as F

from src.model.base import BaseMultiHeadClassifier, focal_loss


class StubClassifier(BaseMultiHeadClassifier):
    @classmethod
    def get_lora_target_modules(cls):
        return ("stub_proj",)

    def _get_hidden_size(self) -> int:
        return int(self.config.hidden_size)

    def _pool(self, last_hidden_state, input_ids, attention_mask):
        return last_hidden_state[:, 0]


def test_focal_loss_matches_cross_entropy_when_gamma_is_zero():
    logits = torch.tensor([[3.0, 0.5], [0.5, 2.5]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    actual = focal_loss(logits, targets, gamma=0.0)
    expected = F.cross_entropy(logits, targets)

    assert torch.allclose(actual, expected)


def test_focal_loss_downweights_easy_examples():
    logits = torch.tensor([[3.0, 0.5], [0.5, 2.5]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    base_loss = focal_loss(logits, targets, gamma=0.0)
    focused_loss = focal_loss(logits, targets, gamma=2.0)

    assert focused_loss < base_loss


def test_focal_loss_supports_label_smoothing_and_class_weights():
    logits = torch.tensor([[1.5, 0.5], [0.3, 1.2]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    unsmoothed = focal_loss(logits, targets, gamma=0.0)
    smoothed = focal_loss(logits, targets, gamma=0.0, label_smoothing=0.1)
    weighted = focal_loss(
        logits,
        targets,
        gamma=0.0,
        weight=torch.tensor([1.0, 3.0], dtype=torch.float32),
    )

    assert not torch.allclose(smoothed, unsmoothed)
    assert not torch.allclose(weighted, unsmoothed)


def test_forward_without_labels_returns_three_logit_tensors(
    dummy_backbone, sample_batch
):
    model = StubClassifier(
        dummy_backbone, num_categories=3, num_subcategories=4
    )

    outputs = model(
        input_ids=sample_batch["input_ids"],
        attention_mask=sample_batch["attention_mask"],
    )

    assert len(outputs) == 3
    assert outputs[0].shape == (2, 2)
    assert outputs[1].shape == (2, 3)
    assert outputs[2].shape == (2, 4)


def test_forward_with_labels_returns_loss_and_logits(
    dummy_backbone, sample_batch
):
    model = StubClassifier(
        dummy_backbone, num_categories=3, num_subcategories=4
    )

    outputs = model(**sample_batch)

    assert len(outputs) == 4
    assert outputs[0].ndim == 0
    assert torch.isfinite(outputs[0])
    assert outputs[1].shape == (2, 2)


def test_scalar_focal_gamma_broadcasts_to_all_heads(dummy_backbone):
    model = StubClassifier(
        dummy_backbone,
        num_categories=3,
        num_subcategories=4,
        focal_gamma=1.5,
    )

    assert model.focal_gamma == (1.5, 1.5, 1.5)


def test_class_weights_are_registered_as_buffers(dummy_backbone):
    model = StubClassifier(
        dummy_backbone,
        num_categories=3,
        num_subcategories=4,
        class_weights={
            "threat": torch.tensor([1.0, 2.0]),
            "category": torch.tensor([1.0, 1.5, 2.0]),
            "subcategory": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        },
    )

    buffers = dict(model.named_buffers())
    assert "weight_threat" in buffers
    assert "weight_category" in buffers
    assert "weight_subcategory" in buffers


def test_loss_weights_scale_each_head_loss(
    monkeypatch, dummy_backbone, sample_batch
):
    values = iter([torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)])

    monkeypatch.setattr(
        "src.model.base.focal_loss",
        lambda *args, **kwargs: next(values),
    )
    model = StubClassifier(
        dummy_backbone,
        num_categories=3,
        num_subcategories=4,
        loss_weights=(2.0, 3.0, 4.0),
    )

    outputs = model(**sample_batch)

    assert torch.equal(outputs[0], torch.tensor(20.0))


def test_save_pretrained_and_from_pretrained_restore_heads(
    tmp_path, monkeypatch, dummy_backbone
):
    import transformers

    model = StubClassifier(
        dummy_backbone,
        num_categories=3,
        num_subcategories=4,
        loss_weights=(1.2, 1.3, 1.4),
        focal_gamma=(0.5, 1.0, 1.5),
        head_dropout=0.25,
    )
    save_dir = tmp_path / "checkpoint"

    model.save_pretrained(str(save_dir))

    meta = json.loads(
        (save_dir / "classifier_config.json").read_text(encoding="utf-8")
    )
    assert (save_dir / "heads.pt").exists()
    assert meta["loss_weights"] == [1.2, 1.3, 1.4]
    assert meta["focal_gamma"] == [0.5, 1.0, 1.5]
    assert meta["head_dropout"] == 0.25

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: dummy_backbone.__class__(hidden_size=8),
    )

    loaded = StubClassifier.from_pretrained(str(save_dir))

    assert torch.allclose(loaded.threat_head.weight, model.threat_head.weight)
    assert torch.allclose(
        loaded.category_head.weight, model.category_head.weight
    )
    assert torch.allclose(
        loaded.subcategory_head.weight, model.subcategory_head.weight
    )
    assert loaded.loss_weights == (1.2, 1.3, 1.4)
    assert loaded.focal_gamma == (0.5, 1.0, 1.5)


def test_gradient_checkpointing_methods_delegate_to_backbone(dummy_backbone):
    model = StubClassifier(
        dummy_backbone, num_categories=3, num_subcategories=4
    )

    model.gradient_checkpointing_enable({"checkpoint_ratio": 0.5})
    model.gradient_checkpointing_disable()

    assert model.backbone.gc_enable_kwargs == {
        "checkpoint_ratio": 0.5,
        "use_reentrant": False,
    }
    assert model.backbone.gc_disable_called is True
