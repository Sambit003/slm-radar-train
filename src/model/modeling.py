import os
from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn


class GemmaMultiHeadClassifier(nn.Module):
    """
    Multi-head classifier for Gemma-3 with three outputs:
    - Threat Detection (Binary)
    - Category Classification (Multi-class)
    - Sub-category Classification (Multi-class)
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_categories: int,
        num_subcategories: int
    ):
        super().__init__()
        self.backbone = base_model
        self.config = base_model.config
        hidden_size = self.config.hidden_size

        # Classification Heads
        self.threat_head = nn.Linear(hidden_size, 2)
        self.category_head = nn.Linear(hidden_size, num_categories)
        self.subcategory_head = nn.Linear(hidden_size, num_subcategories)

        # Loss function
        self.loss_fct = nn.CrossEntropyLoss()

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Forward gradient checkpointing enable to the backbone."""
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
        **kwargs
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Forward pass for the multi-head model."""
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
            loss_t = self.loss_fct(logits_threat, labels_threat)
            loss_c = self.loss_fct(logits_category, labels_category)
            loss_s = self.loss_fct(logits_subcategory, labels_subcategory)
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
