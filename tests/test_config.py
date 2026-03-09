from transformers import HfArgumentParser

from src.utils.config import DataArguments, ModelArguments


def test_model_arguments_defaults_are_stable():
    args = ModelArguments()

    assert args.finetune_mode == "lora"
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_dropout == 0.05
    assert args.loss_weights == "1.5,1.0,1.0"
    assert args.focal_gamma == "0.0,2.0,2.0"
    assert args.head_dropout == 0.1
    assert args.early_stopping_patience == 3
    assert args.fp32 is False


def test_data_arguments_defaults_are_stable():
    args = DataArguments()

    assert args.max_seq_length == 512
    assert args.use_gradient_checkpointing is True
    assert args.disable_gradient_checkpointing is False
    assert args.mlflow_experiment == "slm-radar-finetune"


def test_hf_argument_parser_parses_model_and_data_dataclasses():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses(
        args=[
            "--model_name_or_path",
            "google/gemma-3-270m",
            "--train_file",
            "train.jsonl",
            "--validation_file",
            "validation.jsonl",
            "--max_seq_length",
            "256",
        ]
    )

    assert model_args.model_name_or_path == "google/gemma-3-270m"
    assert data_args.train_file == "train.jsonl"
    assert data_args.validation_file == "validation.jsonl"
    assert data_args.max_seq_length == 256
