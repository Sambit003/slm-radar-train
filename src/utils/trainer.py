from typing import Dict, Union, Any, Tuple, Optional, List

import torch
from torch import nn
from transformers import Trainer


class MultiHeadTrainer(Trainer):
    """Custom Trainer to handle multi-task loss calculation."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        """Computes the combined loss from all classification heads."""
        labels_threat = inputs.get("labels_threat")
        labels_category = inputs.get("labels_category")
        labels_subcategory = inputs.get("labels_subcategory")

        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            labels_threat=labels_threat,
            labels_category=labels_category,
            labels_subcategory=labels_subcategory
        )

        loss = outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]], Optional[Tuple[torch.Tensor, ...]]]:
        """Perform an evaluation step with multi-head outputs."""
        inputs = self._prepare_inputs(inputs)

        labels_threat = inputs.get("labels_threat")
        labels_category = inputs.get("labels_category")
        labels_subcategory = inputs.get("labels_subcategory")

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels_threat=labels_threat,
                labels_category=labels_category,
                labels_subcategory=labels_subcategory
            )
            # outputs = (loss, logits_threat, logits_category, logits_subcategory)
            loss = outputs[0]
            logits = outputs[1:]  # tuple of 3 logits tensors

        if prediction_loss_only:
            return (loss, None, None)

        # Stack labels as a tuple
        labels = (labels_threat, labels_category, labels_subcategory)

        return (loss, logits, labels)
