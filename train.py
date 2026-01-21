import os
import sys
import logging

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    HfArgumentParser,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

from src.utils.config import ModelArguments, DataArguments
from src.utils.device import get_device
from src.utils.trainer import MultiHeadTrainer
from src.model.modeling import GemmaMultiHeadClassifier
from src.data.dataset import ThreatDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    """Main training entry point."""
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        json_path = os.path.abspath(sys.argv[1])
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=json_path
        )
    else:
        model_args, data_args, training_args = (
            parser.parse_args_into_dataclasses()
        )

    # Detect Device and Set Seed
    device = get_device()
    set_seed(training_args.seed)
    logger.info(f"Training on device: {device}")

    # 1. Load Data
    logger.info(f"Loading data from {data_args.dataset_path}")
    threat_dataset = ThreatDataset(data_args.dataset_path)

    # Save encoders for inference usage later
    threat_dataset.save_encoders(training_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Process Dataset
    logger.info("Tokenizing and processing dataset...")
    hf_dataset = threat_dataset.get_hf_dataset(
        tokenizer,
        data_args.max_seq_length
    )

    # Split
    dataset_splits = hf_dataset.train_test_split(test_size=data_args.test_size)
    train_ds = dataset_splits["train"]
    eval_ds = dataset_splits["test"]

    logger.info(f"Loading base model: {model_args.model_name_or_path}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    base_model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=target_modules
    )

    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    num_cats = len(threat_dataset.encoders['category'].classes_)
    num_subcats = len(threat_dataset.encoders['subcategory'].classes_)
    model = GemmaMultiHeadClassifier(
        base_model,
        num_categories=num_cats,
        num_subcategories=num_subcats
    )

    # 5. Trainer
    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # 6. Train
    logger.info("Starting training...")
    trainer.train()

    # 7. Save
    logger.info(f"Saving model to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
