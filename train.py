import os
import sys
import logging

import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

from src.utils.config import ModelArguments, DataArguments
from src.utils.device import get_device
from src.utils.trainer import MultiHeadTrainer
from src.utils.metrics import compute_metrics
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

    # Ensure Trainer gathers our custom labels
    training_args.label_names = [
        "labels_threat", "labels_category", "labels_subcategory"
    ]

    # Set MLFlow Experiment
    if "mlflow" in training_args.report_to or training_args.report_to == "all":
        mlflow.set_experiment(data_args.mlflow_experiment)
        logger.info(f"MLflow Experiment set to: {data_args.mlflow_experiment}")

    # 1. Load Data
    logger.info(f"Loading data from {data_args.dataset_path}")
    threat_dataset = ThreatDataset(data_args.dataset_path)

    # Save encoders for inference usage later
    threat_dataset.save_encoders(training_args.output_dir)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        token=model_args.hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Process Dataset
    logger.info("Tokenizing and processing dataset...")
    hf_dataset = threat_dataset.get_hf_dataset(
        tokenizer,
        data_args.max_seq_length
    )

    # Split: train / val / test (3-way)
    test_val_size = data_args.val_size + data_args.test_size
    temp_split = hf_dataset.train_test_split(
        test_size=test_val_size,
        seed=training_args.seed
    )
    train_ds = temp_split["train"]

    # Split remaining into val and test
    val_test_ratio = data_args.test_size / test_val_size
    val_test_split = temp_split["test"].train_test_split(
        test_size=val_test_ratio,
        seed=training_args.seed
    )
    eval_ds = val_test_split["train"]  # validation
    test_ds = val_test_split["test"]   # held-out test

    logger.info(
        f"Dataset splits - Train: {len(train_ds)}, "
        f"Val: {len(eval_ds)}, Test: {len(test_ds)}"
    )

    logger.info(f"Loading base model: {model_args.model_name_or_path}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        token=model_args.hf_token
    ).model

    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
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

    if str(device).startswith("xla"):
        training_args.optim = "adamw_torch"
        training_args.dataloader_pin_memory = False
        logger.info("TPU detected: using adamw_torch optimizer")

    if data_args.use_gradient_checkpointing:
        training_args.gradient_checkpointing = True
        logger.info("Gradient Checkpointing ENABLED")

    # 5. Trainer
    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 6. Train
    logger.info("Starting training...")
    trainer.train()

    # 7. Save
    logger.info(f"Saving model to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Create & Save Model Card
    model_card = f"""---
language: en
tags:
- gemma
- threat-detection
- classification
- slm-radar
metrics:
- accuracy
- f1
---

# SLM Radar: Gemma-3 Threat Classifier

Fine-tuned Gemma-3-270M for multi-head classification:
1. **Threat Detection** (Safe/Unsafe)
2. **Category** (Harm Category)
3. **Subcategory** (Specific Harm Type)

## Performance
- **Threat Accuracy**: {trainer.state.log_history[-1].get('eval_threat_accuracy', 'N/A')}
- **Threat F1**: {trainer.state.log_history[-1].get('eval_threat_f1', 'N/A')}
- **Combined Accuracy**: {trainer.state.log_history[-1].get('eval_combined_accuracy', 'N/A')}

## Training Config
- **Epochs**: {training_args.num_train_epochs}
- **Batch Size**: {training_args.per_device_train_batch_size}
- **Gradient Checkpointing**: {training_args.gradient_checkpointing}
"""
    with open(os.path.join(training_args.output_dir, "README.md"), "w") as f:
        f.write(model_card)

    logger.info("Model Card generated.")


if __name__ == "__main__":
    main()
