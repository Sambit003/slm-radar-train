from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """Arguments for model/config/tokenizer fine-tuning."""

    model_name_or_path: str = field(
        default="google/gemma-3-270m",
        metadata={"help": "Model identifier from huggingface.co/models"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )


@dataclass
class DataArguments:
    """Arguments for input data configuration."""

    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to dataset file (jsonl)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Max input sequence length after tokenization."}
    )
    test_size: float = field(
        default=0.1,
        metadata={"help": "Proportion of dataset for validation."}
    )
