import argparse
import json
import logging
import os
import time
import warnings
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from utils.data_utils import format_medical_sample, format_ultrachat

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Ignore FutureWarnings from Transformers
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*attention mask.*deprecated.*", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 on Medical Data")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-0.5B-Instruct", help="Base model name")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument(
        "--output_dir", type=str, default="./medical_qwen_lora", help="Output directory for saved model"
    )
    return parser.parse_args()


def prepare_data(tokenizer: Any, max_seq_len: int) -> Dataset:
    logger.info("📥 Loading Medical Meadow Flashcards dataset...")
    raw_dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    df = raw_dataset.to_pandas()
    df = df.rename(columns={"input": "question", "output": "answer"})
    df["q_len"] = df["question"].str.split().str.len()
    df["a_len"] = df["answer"].str.split().str.len()
    df["total_len"] = df["q_len"] + df["a_len"]

    # Filter and sample
    df_filtered = df[df["total_len"] <= 200].reset_index(drop=True)
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = df_filtered.iloc[:1000]  # Use 1000 samples for training

    medical_samples = [format_medical_sample(row) for _, row in df_train.iterrows()]
    medical_hf = Dataset.from_list(medical_samples)

    logger.info("📥 Loading general replay buffer (UltraChat-200k)...")
    try:
        ultrachat_raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=False)
        replay_size = int(len(df_train) * 0.10)
        replay_raw = ultrachat_raw.shuffle(seed=42).select(range(min(replay_size, len(ultrachat_raw))))

        replay_formatted = replay_raw.map(format_ultrachat, remove_columns=replay_raw.column_names)
        replay_formatted = replay_formatted.filter(lambda x: len(x["text"]) > 50)

        combined_dataset = concatenate_datasets([medical_hf, replay_formatted]).shuffle(seed=42)
    except Exception as e:
        logger.warning(f"⚠️ Could not load replay buffer: {e}")
        combined_dataset = medical_hf.shuffle(seed=42)

    def tokenize_and_check(example):
        tokens = tokenizer(example["text"], truncation=False)
        example["token_count"] = len(tokens["input_ids"])
        return example

    train_dataset = combined_dataset.map(tokenize_and_check, num_proc=2, desc="Tokenizing")
    train_dataset = train_dataset.filter(lambda x: x["token_count"] <= max_seq_len - 32)
    logger.info(f"✅ Final training set size: {len(train_dataset)}")

    return train_dataset


def main() -> None:
    args = parse_args()

    # 1. Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for training, but none was found.")

    # 2. Load Model & Tokenizer
    logger.info(f"🔄 Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.max_length = None

    # 3. Configure QLoRA
    logger.info("⚙️ Configuring QLoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable_params / total_params) * 100
    logger.info(f"📊 Params: Total {total_params:,}, Trainable {trainable_params:,} ({trainable_pct:.2f}%)")

    # 4. Prepare Dataset
    train_dataset = prepare_data(tokenizer, args.max_seq_length)

    # 5. Setup Training Config
    training_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        seed=42,
        eval_strategy="no",
        max_steps=-1,
        dataloader_num_workers=0,
        dataset_num_proc=2,
        report_to="none",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_config,
    )

    # 6. Train
    logger.info("🚀 Starting training...")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info(f"🔋 Memory BEFORE training: Used {(total - free) / 1e9:.2f}GB / Total {total / 1e9:.2f}GB")

    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info(f"🔋 Memory AFTER training: Used {(total - free) / 1e9:.2f}GB / Total {total / 1e9:.2f}GB")

    logger.info(f"🎉 TRAINING COMPLETE! Time: {elapsed / 60:.1f} min. Final Loss: {train_result.training_loss:.4f}")

    # 7. Save
    logger.info(f"💾 Saving LoRA adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training logs for visualization
    log_history = trainer.state.log_history
    train_logs = [x for x in log_history if "loss" in x and "eval_loss" not in x]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_logs.json"), "w") as f:
        json.dump(train_logs, f, indent=2)


if __name__ == "__main__":
    main()
