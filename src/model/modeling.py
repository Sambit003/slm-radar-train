import os
from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing for training stability."""

    def __init__(self, smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing == 0.0:
            return F.cross_entropy(logits, targets, reduction=self.reduction)

        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GemmaMultiHeadClassifier(nn.Module):
    """
    Multi-head classifier for Gemma-3 with three outputs:
    - Threat Detection (Binary)
    - Category Classification (Multi-class)
    - Sub-category Classification (Multi-class)

    Includes training stabilization features:
    - Label smoothing support
    - Weighted loss for imbalanced heads
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_categories: int,
        num_subcategories: int,
        loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        super().__init__()
        self.backbone = base_model
        self.config = base_model.config
        hidden_size = self.config.hidden_size
        self.loss_weights = loss_weights

        # Classification Heads with optional dropout for regularization
        self.head_dropout = nn.Dropout(p=0.1)
        self.threat_head = nn.Linear(hidden_size, 2)
        self.category_head = nn.Linear(hidden_size, num_categories)
        self.subcategory_head = nn.Linear(hidden_size, num_subcategories)

        # Cast heads to the same dtype as the backbone
        self.threat_head.to(self.backbone.dtype)
        self.category_head.to(self.backbone.dtype)
        self.subcategory_head.to(self.backbone.dtype)

        # Default loss function (can be overridden in forward with label_smoothing)
        self.loss_fct = nn.CrossEntropyLoss()

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Forward gradient checkpointing enable to the backbone."""
        # Use non-reentrant checkpointing to avoid XLA/device detection issues
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        elif "use_reentrant" not in gradient_checkpointing_kwargs:
            gradient_checkpointing_kwargs["use_reentrant"] = False
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Forward gradient checkpointing disable to the backbone."""
        self.backbone.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels_threat: Optional[torch.LongTensor] = None,
        labels_category: Optional[torch.LongTensor] = None,
        labels_subcategory: Optional[torch.LongTensor] = None,
        label_smoothing: float = 0.0,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Forward pass for the multi-head model with label smoothing support."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        last_hidden_state = outputs.last_hidden_state

        # EOS token pooling or Last Token Pooling
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            indices = torch.arange(batch_size, device=input_ids.device)
            pooled_output = last_hidden_state[indices, sequence_lengths]
        else:
            pooled_output = last_hidden_state[:, -1]

        pooled_output = self.head_dropout(pooled_output)

        logits_threat = self.threat_head(pooled_output)
        logits_category = self.category_head(pooled_output)
        logits_subcategory = self.subcategory_head(pooled_output)

        loss = None
        all_labels_present = (
            labels_threat is not None
            and labels_category is not None
            and labels_subcategory is not None
        )
        if all_labels_present:
            # Use label smoothing loss if specified
            if label_smoothing > 0.0:
                loss_fn = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            else:
                loss_fn = self.loss_fct

            loss_t = loss_fn(logits_threat, labels_threat) * self.loss_weights[0]
            loss_c = loss_fn(logits_category, labels_category) * self.loss_weights[1]
            loss_s = loss_fn(logits_subcategory, labels_subcategory) * self.loss_weights[2]
            loss = loss_t + loss_c + loss_s

        output = (logits_threat, logits_category, logits_subcategory)
        if loss is not None:
            return (loss,) + output
        return output

    def save_pretrained(self, save_directory: str):
        """Saves the model backbone and head weights."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.backbone.save_pretrained(save_directory)

        heads_state = {
            'threat_head': self.threat_head.state_dict(),
            'category_head': self.category_head.state_dict(),
            'subcategory_head': self.subcategory_head.state_dict()
        }
        torch.save(heads_state, os.path.join(save_directory, "heads.pt"))
