# SLM Radar Train

This project provides a training pipeline for fine-tuning Small Language Models (SLMs) with a modular multi-head classifier architecture, designed for radar/threat detection tasks. It utilizes Hugging Face `transformers`, optional `peft` for LoRA fine-tuning, and `mlflow` for tracking.

## Features

- **Modular Model Layer**: Uses a model registry/factory so multiple backbones can plug into the same training pipeline.
- **Built-in Architectures**: Includes Gemma, LLaMA/Mistral, and BERT/DeBERTa-family wrappers.
- **LoRA Support**: Implements Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.
- **Full Fine-tuning**: Supports training all backbone weights (no adapters).
- **Tracking**: Integrated MLflow tracking with ngrok support for remote monitoring.
- **Precision**: Handles BF16/FP32 precision settings automatically based on device support (e.g., specific handling for NVIDIA T4).

## Project Structure

```text
slm-radar-train/
├── train.py                # Main training script
├── requirements.txt        # Python dependencies
├── configs/                # JSON configuration files
│   └── stable_training.json
└── src/
    ├── data/               # Dataset loading and processing
    ├── model/              # Modular model registry + classifier definitions
    └── utils/              # Configuration, metrics, and trainer utilities
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Sambit003/slm-radar-train.git
   cd slm-radar-train
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ngrok authtoken** (for MLflow tracking):
   - Sign up at [ngrok.com](https://ngrok.com/) and get your authtoken.
   - Run the following command to set it up: **(Most recommended)**

        ```bash
        ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
        ```

     or
   - Set the authtoken as an environment variable:

        ```bash
        export NGROK_AUTHTOKEN=<YOUR_NGROK_AUTHTOKEN>
        ```

## Usage

You can run the training script either by passing command-line arguments or by using a JSON configuration file.

### Option 1: Command Line Arguments

Run `train.py` with the desired parameters. Below is a comprehensive example:

```bash
python train.py \
    --model_name_or_path google/gemma-3-270m \
   --finetune_mode lora \
    --gpu_type nvidia-t4 \ #This argument is only needed when you're on T4 gpu
   --train_file <TRAIN_JSONL_PATH> \
   --validation_file <VAL_JSONL_PATH> \
   --test_file <TEST_JSONL_PATH> \
    --output_dir ./output_gemma_radar \
    --hf_token "<YOUR_HF_TOKEN>" \ # HF token should be within "" (quotes)
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --lora_r 16 \
   --loss_weights 1.5,1.0,1.0 \
   --focal_gamma 0.0,2.0,2.0 \
   --head_dropout 0.1 \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --disable_gradient_checkpointing

# For full fine-tuning (no LoRA adapters), use:
#   --finetune_mode full

# To override LoRA targets explicitly, use:
#   --lora_target_modules q_proj,v_proj,k_proj,o_proj

# Example BERT-family run:
#   --model_name_or_path bert-base-uncased
```

## Add a New Model Architecture

1. Create a new class in `src/model/` that extends `BaseMultiHeadClassifier`.
2. Register it with `@register_model("your_model_type")`.
3. Implement:
   - `get_lora_target_modules()`
   - `_get_hidden_size()`
   - `_pool(...)`
4. Run training with `--model_name_or_path ...`; the class is auto-selected from HF `config.model_type`.

### Option 2: JSON Configuration

You can store your arguments in a JSON file (e.g., `configs/stable_training.json`) and pass it to the script:

```bash
python train.py configs/stable_training.json
```

## Notebooks

- `slm_radar_train_notebook.ipynb`: A Jupyter/Colab notebook containing setup steps and execution cells for running the training in an interactive environment (like Google Colab).
