from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """Arguments for model/config/tokenizer fine-tuning."""

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Model identifier from huggingface.co/models"}
    )
    hf_token: str = field(
        default=None,
        metadata={"help": "HuggingFace token for gated models"}
    )
    finetune_mode: str = field(
        default="lora",
        metadata={
            "help": (
                "Fine-tuning mode: 'lora' (LoRA adapters) or 'full' "
                "(train all backbone weights)."
            )
        },
    )
    lora_r: int = field(
        default=16,
        metadata={
            "help": (
                "LoRA attention dimension "
                "(used only when finetune_mode='lora')"
            )
        }
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha (used only when finetune_mode='lora')"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout (used only when finetune_mode='lora')"}
    )
    gpu_type: str = field(
        default=None,
        metadata={
            "help": (
                "GPU type for hardware specific optimizations "
                "(e.g., 'nvidia-t4')"
            )
        }
    )
    fp32: bool = field(
        default=False,
        metadata={"help": "Whether to use fp32 mode."}
    )


@dataclass
class DataArguments:
    """Arguments for input data configuration."""

    train_file: str = field(
        default=None,
        metadata={"help": "Path to training dataset file (jsonl)."}
    )
    validation_file: str = field(
        default=None,
        metadata={"help": "Path to validation dataset file (jsonl)."}
    )
    test_file: str = field(
        default=None,
        metadata={"help": "Path to test dataset file (jsonl)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Max input sequence length after tokenization."}
    )
    use_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing to save memory."}
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": (
                "Disable gradient checkpointing and force use_cache=True."
            )
        }
    )
    mlflow_experiment: str = field(
        default="slm-radar-finetune",
        metadata={"help": "MLflow experiment name."}
    )
