import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.LongTensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal loss: down-weights easy examples."""
    if weight is not None:
        weight = weight.to(logits.dtype)

    ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        label_smoothing=label_smoothing,
        weight=weight,
    )
    p = F.softmax(logits.detach(), dim=-1)
    p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce).mean()


class BaseMultiHeadClassifier(nn.Module, ABC):
    """Backbone-agnostic classifier for threat/category/subcategory."""

    AUTO_MODEL_KIND = "causal_lm"

    def __init__(
        self,
        base_model: nn.Module,
        num_categories: int,
        num_subcategories: int,
        loss_weights: Tuple[float, ...] = (1.0, 1.0, 1.0),
        focal_gamma: Union[float, Tuple[float, ...]] = 2.0,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = base_model
        self.config = base_model.config
        hidden_size = self._get_hidden_size()
        self.loss_weights = tuple(float(x) for x in loss_weights)
        self.head_dropout_prob = float(head_dropout)

        if isinstance(focal_gamma, (float, int)):
            self.focal_gamma = (float(focal_gamma),) * 3
        else:
            self.focal_gamma = tuple(float(g) for g in focal_gamma)

        if class_weights is not None:
            if "threat" in class_weights:
                self.register_buffer("weight_threat", class_weights["threat"])
            if "category" in class_weights:
                self.register_buffer(
                    "weight_category", class_weights["category"]
                )
            if "subcategory" in class_weights:
                self.register_buffer(
                    "weight_subcategory", class_weights["subcategory"]
                )
        else:
            self.weight_threat = None
            self.weight_category = None
            self.weight_subcategory = None

        self.head_dropout = nn.Dropout(p=self.head_dropout_prob)
        self.threat_head = nn.Linear(hidden_size, 2)
        self.category_head = nn.Linear(hidden_size, num_categories)
        self.subcategory_head = nn.Linear(hidden_size, num_subcategories)

        self.threat_head.to(self.backbone.dtype)
        self.category_head.to(self.backbone.dtype)
        self.subcategory_head.to(self.backbone.dtype)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def load_train_backbone(
        cls,
        model_name_or_path: str,
        *,
        token: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ) -> nn.Module:
        """Load the trainable backbone for this architecture."""
        if cls.AUTO_MODEL_KIND == "causal_lm":
            from transformers import AutoModelForCausalLM

            causal_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                token=token,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            return cls.extract_backbone(causal_model)

        from transformers import AutoModel

        return AutoModel.from_pretrained(
            model_name_or_path,
            token=token,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    @classmethod
    def extract_backbone(cls, loaded_model: nn.Module) -> nn.Module:
        """Extract a backbone from a loaded model."""
        return getattr(loaded_model, "model", loaded_model)

    @classmethod
    @abstractmethod
    def get_lora_target_modules(cls) -> Tuple[str, ...]:
        """Return default LoRA module names for this architecture."""

    @abstractmethod
    def _get_hidden_size(self) -> int:
        """Return hidden size from the underlying backbone config."""

    @abstractmethod
    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return pooled hidden representation for head inputs."""

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs=None
    ):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        elif "use_reentrant" not in gradient_checkpointing_kwargs:
            gradient_checkpointing_kwargs["use_reentrant"] = False
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels_threat: Optional[torch.LongTensor] = None,
        labels_category: Optional[torch.LongTensor] = None,
        labels_subcategory: Optional[torch.LongTensor] = None,
        label_smoothing: float = 0.0,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self._pool(
            last_hidden_state, input_ids, attention_mask
        )
        pooled_output = self.head_dropout(pooled_output)

        logits_threat = self.threat_head(pooled_output)
        logits_category = self.category_head(pooled_output)
        logits_subcategory = self.subcategory_head(pooled_output)

        loss = None
        if (
            labels_threat is not None
            and labels_category is not None
            and labels_subcategory is not None
        ):
            loss_t = focal_loss(
                logits_threat.float(),
                labels_threat.long(),
                gamma=self.focal_gamma[0],
                label_smoothing=label_smoothing,
                weight=getattr(self, "weight_threat", None),
            ) * self.loss_weights[0]

            loss_c = focal_loss(
                logits_category.float(),
                labels_category.long(),
                gamma=self.focal_gamma[1],
                label_smoothing=label_smoothing,
                weight=getattr(self, "weight_category", None),
            ) * self.loss_weights[1]

            loss_s = focal_loss(
                logits_subcategory.float(),
                labels_subcategory.long(),
                gamma=self.focal_gamma[2],
                label_smoothing=label_smoothing,
                weight=getattr(self, "weight_subcategory", None),
            ) * self.loss_weights[2]

            loss = loss_t + loss_c + loss_s

        output = (logits_threat, logits_category, logits_subcategory)
        if loss is not None:
            return (loss,) + output
        return output

    def save_pretrained(self, save_directory: str):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        self.backbone.save_pretrained(save_directory)
        heads_state = {
            "threat_head": self.threat_head.state_dict(),
            "category_head": self.category_head.state_dict(),
            "subcategory_head": self.subcategory_head.state_dict(),
        }
        torch.save(heads_state, os.path.join(save_directory, "heads.pt"))

        classifier_meta = {
            "num_categories": int(self.category_head.weight.shape[0]),
            "num_subcategories": int(self.subcategory_head.weight.shape[0]),
            "loss_weights": [float(x) for x in self.loss_weights],
            "focal_gamma": [float(x) for x in self.focal_gamma],
            "head_dropout": float(self.head_dropout_prob),
            "model_class": self.__class__.__name__,
        }
        meta_path = os.path.join(save_directory, "classifier_config.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(classifier_meta, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        *,
        model_name_or_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        loss_weights: Optional[Tuple[float, ...]] = None,
        focal_gamma: Union[float, Tuple[float, ...]] = 2.0,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        head_dropout: Optional[float] = None,
    ):
        heads_path = os.path.join(checkpoint_dir, "heads.pt")
        if not os.path.exists(heads_path):
            raise FileNotFoundError(
                f"Missing heads.pt in checkpoint: {checkpoint_dir}"
            )

        heads_state: Dict[str, Any] = torch.load(
            heads_path, map_location="cpu"
        )
        num_categories = int(heads_state["category_head"]["weight"].shape[0])
        num_subcategories = int(
            heads_state["subcategory_head"]["weight"].shape[0]
        )

        classifier_config_path = os.path.join(
            checkpoint_dir, "classifier_config.json"
        )
        if os.path.exists(classifier_config_path):
            try:
                with open(classifier_config_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if loss_weights is None and "loss_weights" in meta:
                    loss_weights = tuple(
                        float(x) for x in meta["loss_weights"]
                    )
                if "focal_gamma" in meta:
                    fg = meta["focal_gamma"]
                    if isinstance(fg, (float, int)):
                        focal_gamma = float(fg)
                    elif isinstance(fg, list):
                        focal_gamma = tuple(float(x) for x in fg)
                if head_dropout is None and "head_dropout" in meta:
                    head_dropout = float(meta["head_dropout"])
            except Exception:
                pass

        is_lora_adapter = os.path.exists(
            os.path.join(checkpoint_dir, "adapter_config.json")
        )
        if is_lora_adapter:
            from peft import PeftConfig, PeftModel

            peft_cfg = PeftConfig.from_pretrained(checkpoint_dir)
            base_name = model_name_or_path or peft_cfg.base_model_name_or_path
            if base_name is None:
                raise ValueError(
                    "Unable to determine base model for LoRA checkpoint. "
                    "Pass model_name_or_path explicitly."
                )
            base_backbone = cls.load_train_backbone(
                base_name,
                token=hf_token,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            backbone = PeftModel.from_pretrained(
                base_backbone,
                checkpoint_dir,
                is_trainable=False,
            )
        else:
            from transformers import AutoModel

            backbone = AutoModel.from_pretrained(
                checkpoint_dir,
                token=hf_token,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
            )

        if loss_weights is None:
            loss_weights = (1.0, 1.0, 1.0)
        if head_dropout is None:
            head_dropout = 0.1

        model = cls(
            backbone,
            num_categories=num_categories,
            num_subcategories=num_subcategories,
            loss_weights=loss_weights,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            head_dropout=head_dropout,
        )
        model.threat_head.load_state_dict(heads_state["threat_head"])
        model.category_head.load_state_dict(heads_state["category_head"])
        model.subcategory_head.load_state_dict(heads_state["subcategory_head"])
        return model
