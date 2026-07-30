import argparse
import json
import logging
import os

import evaluate
import nltk
import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from unsloth import FastLanguageModel

# Ensure required NLTK data is downloaded for BLEU score
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Base and Fine-Tuned Qwen2.5 Models")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen2.5-0.5B-Instruct", help="Base model name")
    parser.add_argument("--adapter_dir", type=str, default="./medical_qwen_lora", help="Path to saved LoRA adapters")
    parser.add_argument("--num_samples", type=int, default=150, help="Number of samples to evaluate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    return parser.parse_args()


GENERAL_TESTS = [
    {
        "task": "General Knowledge",
        "question": "What is the capital of France and what is it famous for?",
        "ref_keywords": ["paris", "eiffel", "louvre", "france", "city"],
    },
    {
        "task": "Simple Math Reasoning",
        "question": "If a train travels at 60 km/h and needs to cover 180 km, how long will it take?",
        "ref_keywords": ["3", "three", "hours", "hour"],
    },
    {
        "task": "Common Sense",
        "question": "Why do we need to sleep every night? Give three reasons.",
        "ref_keywords": ["rest", "brain", "memory", "health", "body", "energy"],
    },
    {
        "task": "Language Understanding",
        "question": "Explain the difference between a metaphor and a simile with examples.",
        "ref_keywords": ["like", "as", "comparison", "directly", "example"],
    },
]


def keyword_score(text: str, keywords: list) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / len(keywords)


def check_catastrophic_forgetting(model, tokenizer):
    logger.info("🛡️  Running Catastrophic Forgetting Check on Fine-Tuned Model...")
    results = []
    for test in GENERAL_TESTS:
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
            f"<|im_start|>user\n{test['question']}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        score = keyword_score(response, test["ref_keywords"])
        results.append({"task": test["task"], "score": round(score, 4)})
        logger.info(f"Task: {test['task']} | Score: {score:.2f}")

    avg_retention = sum(r["score"] for r in results) / len(results)
    logger.info(f"Average General Task Retention Score: {avg_retention:.2f}")
    if avg_retention >= 0.3:
        logger.info("✅ PASS — Replay buffer strategy prevented catastrophic forgetting.")
    else:
        logger.warning("⚠️  WARNING — Model shows signs of catastrophic forgetting.")

    return results


def build_prompt(question: str, include_answer: bool = False, answer: str = "") -> str:
    prompt = (
        f"<|im_start|>system\n"
        f"You are a medical education assistant. "
        f"Answer the following medical question clearly and accurately.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if include_answer:
        prompt += answer + "<|im_end|>"
    return prompt


def evaluate_model(model, tokenizer, eval_df, max_seq_length, num_samples, label):
    FastLanguageModel.for_inference(model)
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")

    predictions, references, perplexities = [], [], []
    eval_sample = eval_df.sample(min(num_samples, len(eval_df)), random_state=42)

    for _, row in tqdm(eval_sample.iterrows(), total=len(eval_sample), desc=f"Evaluating [{label}]"):
        question, answer = row["question"], row["answer"]
        prompt = build_prompt(question)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_length=None,
                max_new_tokens=120,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )
        pred = tokenizer.decode(gen_out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        predictions.append(pred)
        references.append(answer)

        full_text = build_prompt(question, include_answer=True, answer=answer)
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_seq_length).to("cuda")
        with torch.no_grad():
            loss = model(**enc, labels=enc["input_ids"]).loss
        perplexities.append(torch.exp(loss).item())

    rouge_scores = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
    bleu_result = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])

    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu_result["bleu"],
        "perplexity": np.mean(perplexities),
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for evaluation.")

    logger.info("📥 Loading Medical Meadow Flashcards dataset (Validation split)...")
    raw_dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    df = raw_dataset.to_pandas()
    df = df.rename(columns={"input": "question", "output": "answer"})
    df["total_len"] = df["question"].str.split().str.len() + df["answer"].str.split().str.len()

    df_filtered = df[df["total_len"] <= 200].reset_index(drop=True)
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    # Using 150 samples after the first 1000 used for training
    eval_df = df_filtered.iloc[1000 : 1000 + args.num_samples]

    logger.info("🔄 Loading Base Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.max_length = None

    logger.info(f"📊 Evaluating Base Model on {len(eval_df)} samples...")
    base_scores = evaluate_model(model, tokenizer, eval_df, args.max_seq_length, args.num_samples, "Base Model")

    logger.info("🔄 Freeing GPU memory...")
    import gc

    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("🔄 Loading Fine-Tuned Model (LoRA Adapters)...")
    if not os.path.exists(args.adapter_dir):
        logger.error(f"Adapter directory {args.adapter_dir} not found. Please train the model first.")
        return

    model_ft, tokenizer_ft = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer_ft.pad_token is None:
        tokenizer_ft.pad_token = tokenizer_ft.eos_token
        tokenizer_ft.pad_token_id = tokenizer_ft.eos_token_id
        model_ft.config.pad_token_id = tokenizer_ft.eos_token_id
    model_ft.generation_config.max_length = None

    logger.info(f"📊 Evaluating Fine-Tuned Model on {len(eval_df)} samples...")
    ft_scores = evaluate_model(model_ft, tokenizer_ft, eval_df, args.max_seq_length, args.num_samples, "Fine-Tuned")

    # Catastrophic forgetting check
    forgetting_results = check_catastrophic_forgetting(model_ft, tokenizer_ft)

    results = {
        "dataset": "medalpaca/medical_meadow_medical_flashcards",
        "eval_samples": len(eval_df),
        "baseline_scores": {k: round(float(v), 4) for k, v in base_scores.items()},
        "finetuned_scores": {k: round(float(v), 4) for k, v in ft_scores.items()},
        "forgetting_checks": forgetting_results,
    }

    os.makedirs(args.adapter_dir, exist_ok=True)
    out_path = os.path.join(args.adapter_dir, "training_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Results saved to {out_path}")
    for k in results["baseline_scores"].keys():
        logger.info(f"{k.upper()}: Base={results['baseline_scores'][k]:.4f} -> FT={results['finetuned_scores'][k]:.4f}")


if __name__ == "__main__":
    main()
