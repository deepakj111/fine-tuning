import os
import argparse
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# We set wandb project name before importing transformers/unsloth
os.environ["WANDB_PROJECT"] = "medical-qwen2.5-finetuning"

import torch
import wandb
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# Ignore FutureWarnings from Transformers
import warnings
warnings.filterwarnings("ignore", message=".*AttentionMaskConverter.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*attention mask.*deprecated.*", category=FutureWarning)

from unsloth import FastLanguageModel

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 on Medical Data")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-0.5B-Instruct", help="Base model name")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--output_dir", type=str, default="./medical_qwen_lora", help="Output directory for saved model")
    parser.add_argument("--push_to_hub", action="store_true", help="Push final model to Hugging Face Hub")
    parser.add_argument("--hub_repo_id", type=str, default="your_username/medical-qwen2.5-0.5B-lora", help="HF Hub Repo ID to push to")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B tracking")
    return parser.parse_args()

def prepare_data(tokenizer, max_seq_len):
    print("📥 Loading Medical Meadow Flashcards dataset...")
    raw_dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    df = raw_dataset.to_pandas()
    df = df.rename(columns={"input": "question", "output": "answer"})
    df["q_len"] = df["question"].str.split().str.len()
    df["a_len"] = df["answer"].str.split().str.len()
    df["total_len"] = df["q_len"] + df["a_len"]
    
    # Filter and sample
    df_filtered = df[df["total_len"] <= 200].reset_index(drop=True)
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = df_filtered.iloc[:1000] # Use 1000 samples for training
    
    def format_medical_sample(row):
        return {
            "text": (
                f"<|im_start|>system\n"
                f"You are a medical education assistant. "
                f"Answer the following medical question clearly and accurately.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n{row['question']}\n<|im_end|>\n"
                f"<|im_start|>assistant\n{row['answer']}<|im_end|>"
            )
        }
    
    medical_samples = [format_medical_sample(row) for _, row in df_train.iterrows()]
    medical_hf = Dataset.from_list(medical_samples)
    
    print("📥 Loading general replay buffer (UltraChat-200k)...")
    try:
        ultrachat_raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=False)
        replay_size = int(len(df_train) * 0.10)
        replay_raw = ultrachat_raw.shuffle(seed=42).select(range(min(replay_size, len(ultrachat_raw))))
        
        def format_ultrachat(example):
            messages = example.get("messages", [])
            text = ""
            for msg in messages[:4]:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:300]
                text += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
            return {"text": text.strip()}
            
        replay_formatted = replay_raw.map(format_ultrachat, remove_columns=replay_raw.column_names)
        replay_formatted = replay_formatted.filter(lambda x: len(x["text"]) > 50)
        
        combined_dataset = concatenate_datasets([medical_hf, replay_formatted]).shuffle(seed=42)
    except Exception as e:
        print(f"⚠️ Could not load replay buffer: {e}")
        combined_dataset = medical_hf.shuffle(seed=42)
        
    def tokenize_and_check(example):
        tokens = tokenizer(example["text"], truncation=False)
        example["token_count"] = len(tokens["input_ids"])
        return example
        
    train_dataset = combined_dataset.map(tokenize_and_check)
    train_dataset = train_dataset.filter(lambda x: x["token_count"] <= max_seq_len - 32)
    print(f"✅ Final training set size: {len(train_dataset)}")
    
    return train_dataset

def main():
    args = parse_args()
    
    # 1. Check GPU & Setup Tracking
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for training, but none was found.")
    
    report_to = "none" if args.disable_wandb else "wandb"
    if report_to == "wandb":
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project="medical-qwen2.5-finetuning", config=vars(args))
        print("✅ Weights & Biases tracking enabled")
        
    # 2. Load Model & Tokenizer
    print(f"🔄 Loading base model: {args.model_name}")
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
    print("⚙️ Configuring QLoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
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
        report_to=report_to,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_config,
    )
    
    # 6. Train
    print("🚀 Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time
    
    print(f"\n🎉 TRAINING COMPLETE! Time: {elapsed/60:.1f} min. Final Loss: {train_result.training_loss:.4f}")
    
    # 7. Save & Push
    print(f"💾 Saving LoRA adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if args.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
            print(f"☁️ Pushing to Hugging Face Hub: {args.hub_repo_id}")
            model.push_to_hub(args.hub_repo_id)
            tokenizer.push_to_hub(args.hub_repo_id)
        else:
            print("⚠️ HF_TOKEN not found in environment. Skipping push to hub.")
            
    if report_to == "wandb":
        wandb.finish()

if __name__ == "__main__":
    main()
