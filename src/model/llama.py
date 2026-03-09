from typing import Optional, Tuple

import torch

from src.model.base import BaseMultiHeadClassifier
from src.model.registry import register_model


@register_model("llama", "mistral")
class LlamaMultiHeadClassifier(BaseMultiHeadClassifier):
    """LLaMA/Mistral-style decoder backbone with EOS/last-token pooling."""

    AUTO_MODEL_KIND = "causal_lm"

    @classmethod
    def get_lora_target_modules(cls) -> Tuple[str, ...]:
        return (
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )

    def _get_hidden_size(self) -> int:
        return int(self.config.hidden_size)

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is not None and input_ids is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            indices = torch.arange(batch_size, device=input_ids.device)
            return last_hidden_state[indices, sequence_lengths]
        return last_hidden_state[:, -1]
