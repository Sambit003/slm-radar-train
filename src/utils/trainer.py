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
