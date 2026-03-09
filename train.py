import os
import sys
import logging
import subprocess
import torch
import time
from typing import Optional, Tuple

import mlflow
from pyngrok import ngrok
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

from src.utils.config import ModelArguments, DataArguments
from src.utils.device import get_device, get_device_type
from src.utils.trainer import MultiHeadTrainer
from src.utils.metrics import compute_metrics
from src.model import get_model_class
from src.data.dataset import ThreatDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("pyngrok").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def _parse_triplet(raw: str, arg_name: str) -> Tuple[float, float, float]:
    parts = [x.strip() for x in (raw or "").split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"--{arg_name} must have exactly 3 comma-separated values. "
            f"Got: {raw!r}"
        )
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_lora_target_modules(
    override: Optional[str],
    default_targets: Tuple[str, ...],
) -> Tuple[str, ...]:
    if override is None or not override.strip():
        return default_targets
    parts = tuple(x.strip() for x in override.split(",") if x.strip())
    if not parts:
        raise ValueError(
            "--lora_target_modules was provided but no valid "
            "module names were found."
        )
    return parts


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

    if model_args.hf_token:
        login(token=model_args.hf_token)

    if model_args.gpu_type == "nvidia-t4":
        logger.info(
            "Detected 'nvidia-t4' GPU type. "
            "Forcing FP16 and disabling BF16."
        )
        training_args.fp16 = True
        training_args.bf16 = False

    # Detect Device and Set Seed
    device = get_device()
    set_seed(training_args.seed)
    logger.info(f"Training on device: {device}")

    os.environ["WANDB_DISABLED"] = "true"
    if "wandb" in training_args.report_to:
        if isinstance(training_args.report_to, list):
            training_args.report_to.remove("wandb")
        elif training_args.report_to == "wandb":
            training_args.report_to = "none"

    # Ensure Trainer gathers our custom labels
    training_args.label_names = [
        "labels_threat", "labels_category", "labels_subcategory"
    ]

    # Metric for Best Model (Early Stopping)
    training_args.metric_for_best_model = "eval_loss"
    training_args.load_best_model_at_end = True
    training_args.greater_is_better = False
    training_args.eval_strategy = "epoch"
    training_args.save_strategy = "epoch"

    logger.info(
        "Training Stabilization: max_grad_norm=%s, warmup_ratio=%s, "
        "lr_scheduler=%s, weight_decay=%s, label_smoothing=%s",
        training_args.max_grad_norm,
        training_args.warmup_ratio,
        training_args.lr_scheduler_type,
        training_args.weight_decay,
        training_args.label_smoothing_factor,
    )

    # Set MLFlow Experiment
    if "mlflow" in training_args.report_to or training_args.report_to == "all":
        # Enable System Metrics Logging
        try:
            mlflow.enable_system_metrics_logging()
            logger.info("MLflow System Metrics Logging Enabled")
        except AttributeError:
            logger.warning(
                "mlflow.enable_system_metrics_logging() not found "
                "(update mlflow?)"
            )

        mlflow.set_experiment(data_args.mlflow_experiment)
        logger.info(f"MLflow Experiment set to: {data_args.mlflow_experiment}")

        # Start MLflow UI and Ngrok Tunnel
        try:
            # Check if MLflow UI is already running
            # This is a basic background start.
            logger.info("Starting MLflow UI in the background...")
            subprocess.Popen(
                ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Give it a moment to start
            time.sleep(3)

            # Open Ngrok Tunnel
            port = 5000
            public_url = ngrok.connect(port, host_header="rewrite").public_url
            logger.info(f"Ngrok Tunnel created for port {port}")
            print(
                f"\n{'='*60}\n🚀 MLflow Dashboard available at: "
                f"{public_url}\n{'='*60}\n"
            )

        except Exception as e:
            logger.warning(
                "Failed to set up Ngrok/MLflow UI automatically: %s",
                e
            )

    # 1. Load Data
    logger.info("Loading data...")
    threat_dataset = ThreatDataset(
        data_args.train_file,
        validation_file=data_args.validation_file,
        test_file=data_args.test_file
    )

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

    # Get splits from the processed DatasetDict
    train_ds = hf_dataset['train']
    eval_ds = hf_dataset.get('validation')
    test_ds = hf_dataset.get('test')

    logger.info(f"Dataset splits - Train: {len(train_ds)}")
    if eval_ds:
        logger.info(f"Val: {len(eval_ds)}")
    if test_ds:
        logger.info(f"Test: {len(test_ds)}")

    logger.info(f"Loading base model: {model_args.model_name_or_path}")

    model_class = get_model_class(
        model_args.model_name_or_path,
        hf_token=model_args.hf_token,
        trust_remote_code=True,
    )
    logger.info("Resolved classifier class: %s", model_class.__name__)

    if model_args.pooling_strategy != "auto":
        logger.warning(
            "pooling_strategy=%s requested, but current built-in models use "
            "their "
            "architecture defaults.",
            model_args.pooling_strategy,
        )

    if model_args.fp32:
        dtype = torch.float32
    else:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32

    base_model = model_class.load_train_backbone(
        model_args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        token=model_args.hf_token,
    )

    finetune_mode = (model_args.finetune_mode or "lora").lower().strip()
    if finetune_mode not in {"lora", "full"}:
        raise ValueError(
            "Invalid --finetune_mode. Expected one of: 'lora', 'full'. "
            f"Got: {model_args.finetune_mode!r}"
        )

    if finetune_mode == "lora":
        if model_args.lora_r is None or model_args.lora_r <= 0:
            raise ValueError(
                "When finetune_mode='lora', --lora_r must be > 0. "
                f"Got: {model_args.lora_r}"
            )

        target_modules = _parse_lora_target_modules(
            model_args.lora_target_modules,
            model_class.get_lora_target_modules(),
        )
        logger.info("LoRA target modules: %s", list(target_modules))
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
    else:
        # Full fine-tuning: ensure the entire backbone is trainable.
        base_model.requires_grad_(True)
        total = sum(p.numel() for p in base_model.parameters())
        trainable = sum(
            p.numel() for p in base_model.parameters() if p.requires_grad
        )
        logger.info(
            "Full fine-tuning enabled: trainable backbone params=%s/%s "
            "(%.2f%%)",
            trainable,
            total,
            100.0 * (trainable / max(total, 1)),
        )

    num_cats = len(threat_dataset.encoders['category'].classes_)
    num_subcats = len(threat_dataset.encoders['subcategory'].classes_)

    w_threat, w_cat, w_subcat = _parse_triplet(
        model_args.loss_weights, "loss_weights"
    )
    logger.info(
        "Loss head weights: threat=%.4f, category=%.4f, "
        "subcategory=%.4f",
        w_threat, w_cat, w_subcat
    )

    # 1. Compute Class Weights
    cw_threat, cw_cat, cw_subcat = threat_dataset.get_class_weights()
    logger.info(f"Class Weights - Threat: {cw_threat}")

    class_weights_dict = {
        "threat": torch.tensor(cw_threat, dtype=dtype),
        "category": torch.tensor(cw_cat, dtype=dtype),
        "subcategory": torch.tensor(cw_subcat, dtype=dtype)
    }

    focal_gamma = _parse_triplet(model_args.focal_gamma, "focal_gamma")

    model = model_class(
        base_model,
        num_categories=num_cats,
        num_subcategories=num_subcats,
        loss_weights=(w_threat, w_cat, w_subcat),
        focal_gamma=focal_gamma,
        class_weights=class_weights_dict,
        head_dropout=model_args.head_dropout,
    )

    # Device-specific optimizations
    device_type = get_device_type()
    if device_type == "cuda":
        training_args.dataloader_pin_memory = True
        logger.info("GPU detected: enabling pin_memory for dataloaders")

    # Gradient checkpointing with proper kwargs
    if data_args.disable_gradient_checkpointing:
        training_args.gradient_checkpointing = False
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = True
            logger.info(
                "Gradient Checkpointing DISABLED. use_cache=True FORCED."
            )
        else:
            logger.info("Gradient Checkpointing DISABLED.")
    elif data_args.use_gradient_checkpointing:
        training_args.gradient_checkpointing = True
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        logger.info("Gradient Checkpointing ENABLED (use_reentrant=False)")

    # 5. Trainer
    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=model_args.early_stopping_patience,
            early_stopping_threshold=0.0
            )],
        label_smoothing=training_args.label_smoothing_factor
    )

    # 6. Train
    logger.info("Starting training...")
    trainer.train()

    if test_ds is not None:
        logger.info("Running final evaluation on test set")
        test_metrics = trainer.evaluate(eval_dataset=test_ds)
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    # 7. Save
    logger.info(f"Saving model to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Create & Save Model Card
    # Try to find the last evaluation log
    eval_logs = [entry for entry in trainer.state.log_history
                 if "eval_threat_accuracy" in entry]
    if eval_logs:
        last_eval = eval_logs[-1]
    else:
        last_eval = {}

    threat_acc = last_eval.get("eval_threat_accuracy", "N/A")
    combined_acc = last_eval.get("eval_combined_accuracy", "N/A")

    # Get optimal threshold stats
    opt_thresh = last_eval.get("eval_threat_opt_thresh", "N/A")
    opt_prec = last_eval.get("eval_threat_opt_precision", "N/A")
    opt_rec = last_eval.get("eval_threat_opt_recall", "N/A")

    model_name = model_args.model_name_or_path
    model_family = model_class.__name__.replace("MultiHeadClassifier", "")
    model_card = f"""---
language: en
tags:
- {model_name}
- threat-detection
- classification
- slm-radar
metrics:
- accuracy
- f1
---

# SLM Radar: {model_family} Threat Classifier

Fine-tuned {model_name} for multi-head classification:
1. **Threat Detection** (Safe/Unsafe)
2. **Category** (Harm Category)
3. **Subcategory** (Specific Harm Type)

## Performance
- **Threat Accuracy**: {threat_acc}
- **Threat F1**: {last_eval.get('eval_threat_f1', 'N/A')}
- **Combined Accuracy**: {combined_acc}

### Optimal Threshold (Recall >= 0.90)
- **Threshold**: {opt_thresh}
- **Precision**: {opt_prec}
- **Recall**: {opt_rec}

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
