import torch
from torch import nn
from transformers import TrainingArguments

from src.utils.trainer import MultiHeadTrainer


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.last_kwargs = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels_threat=None,
        labels_category=None,
        labels_subcategory=None,
        label_smoothing=0.0,
    ):
        self.last_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_threat": labels_threat,
            "labels_category": labels_category,
            "labels_subcategory": labels_subcategory,
            "label_smoothing": label_smoothing,
        }
        batch_size = input_ids.shape[0]
        pooled = self.proj(
            torch.ones(
                batch_size,
                4,
                dtype=self.proj.weight.dtype,
                device=self.proj.weight.device,
            )
        )
        loss = pooled.mean()
        logits_threat = pooled[:, :2]
        logits_category = pooled[:, :3]
        logits_subcategory = pooled
        return loss, logits_threat, logits_category, logits_subcategory


def build_trainer(tmp_path, model):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        report_to=[],
        disable_tqdm=True,
    )
    return MultiHeadTrainer(model=model, args=args, label_smoothing=0.2)


def build_inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
        "attention_mask": torch.tensor(
            [[1, 1, 1], [1, 1, 0]], dtype=torch.long
        ),
        "labels_threat": torch.tensor([1, 0], dtype=torch.long),
        "labels_category": torch.tensor([2, 1], dtype=torch.long),
        "labels_subcategory": torch.tensor([3, 0], dtype=torch.long),
    }


def test_compute_loss_passes_all_multi_head_inputs(tmp_path):
    model = RecordingModel()
    trainer = build_trainer(tmp_path, model)
    inputs = build_inputs()

    loss = trainer.compute_loss(model, inputs)

    assert loss.ndim == 0
    assert model.last_kwargs["label_smoothing"] == 0.2
    assert torch.equal(
        model.last_kwargs["labels_threat"], inputs["labels_threat"]
    )


def test_compute_loss_can_return_outputs(tmp_path):
    model = RecordingModel()
    trainer = build_trainer(tmp_path, model)
    inputs = build_inputs()

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    assert loss.ndim == 0
    assert len(outputs) == 4


def test_prediction_step_returns_loss_logits_and_labels(tmp_path):
    model = RecordingModel()
    trainer = build_trainer(tmp_path, model)
    inputs = build_inputs()

    loss, logits, labels = trainer.prediction_step(
        model, inputs, prediction_loss_only=False
    )

    assert loss.ndim == 0
    assert len(logits) == 3
    assert len(labels) == 3
    assert torch.equal(labels[0].cpu(), inputs["labels_threat"].cpu())


def test_prediction_step_can_return_loss_only(tmp_path):
    model = RecordingModel()
    trainer = build_trainer(tmp_path, model)
    inputs = build_inputs()

    loss, logits, labels = trainer.prediction_step(
        model, inputs, prediction_loss_only=True
    )

    assert loss.ndim == 0
    assert logits is None
    assert labels is None


def test_training_step_records_grad_norm_every_hundred_steps(
    tmp_path, monkeypatch
):
    from transformers import Trainer

    model = RecordingModel()
    trainer = build_trainer(tmp_path, model)
    trainer.state.global_step = 100

    def fake_training_step(self, model, inputs, num_items_in_batch=None):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        return torch.tensor(0.5)

    monkeypatch.setattr(Trainer, "training_step", fake_training_step)

    loss = trainer.training_step(model, build_inputs())

    assert loss.item() == 0.5
    assert len(trainer._grad_norm_history) == 1
    assert trainer._grad_norm_history[0] > 0
