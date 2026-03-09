from typing import Optional, Tuple

import torch

from src.model.base import BaseMultiHeadClassifier
from src.model.registry import register_model


@register_model(
    "bert",
    "roberta",
    "distilbert",
    "albert",
    "electra",
    "deberta",
    "deberta-v2",
    "deberta-v3",
)
class BertMultiHeadClassifier(BaseMultiHeadClassifier):
    """BERT/DeBERTa-family encoder backbone with CLS-token pooling."""

    AUTO_MODEL_KIND = "encoder"

    @classmethod
    def get_lora_target_modules(cls) -> Tuple[str, ...]:
        # BERT-style models use query/key/value while DeBERTa commonly uses
        # query_proj/key_proj/value_proj naming.
        return (
            "query",
            "key",
            "value",
            "query_proj",
            "key_proj",
            "value_proj",
        )

    def _get_hidden_size(self) -> int:
        return int(self.config.hidden_size)

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # BERT-family models conventionally use the first token (CLS) as
        # sequence representation.
        return last_hidden_state[:, 0]
