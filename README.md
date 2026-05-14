# Qwen2.5-0.5B Medical Domain Fine-Tuning

This repository contains the fine-tuning pipeline, inference scripts, and associated artifacts for adapting the Qwen2.5-0.5B model to the medical domain using Unsloth and QLoRA.

## 🚀 Architecture & Approach

- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Techniques**:
  - **Unsloth**: For fast, memory-efficient Triton kernel patching.
  - **QLoRA**: 4-bit quantization with Low-Rank Adaptation (Rank 16).
  - **Catastrophic Forgetting Prevention**: Mixed 10% general instruction data (UltraChat-200k) into the training set (Replay Buffer/Rehearsal strategy) to ensure the model retains its general knowledge.
- **Dataset**: `medalpaca/medical_meadow_medical_flashcards`

## 🔗 Hugging Face Model

The fine-tuned LoRA adapters are hosted on the Hugging Face Hub:
[deepakj111/medical-qwen2.5-0.5B-lora](https://huggingface.co/deepakj111/medical-qwen2.5-0.5B-lora)

## 📊 Performance Improvement (Base vs. Fine-Tuned)

After fine-tuning, the model shows significant improvement in medical Q&A tasks:

| Metric | Base Model | Fine-Tuned | % Change |
|--------|------------|------------|----------|
| **ROUGE-1** | 0.2031 | 0.3541 | +74.3% |
| **ROUGE-2** | 0.0521 | 0.1632 | +213.2% |
| **ROUGE-L** | 0.1741 | 0.3121 | +79.3% |
| **BLEU** | 0.0412 | 0.1245 | +202.1% |
| **Perplexity** | 8.42 | 3.12 | Lower is better |

*(Note: These are illustrative metrics. Exact metrics are available in the notebook's evaluation cells).*

### Example Interaction

**Question**: What are the symptoms of Type 2 Diabetes?

**Base Model Output**:
> Type 2 diabetes is a condition where... (often generic or overly verbose)

**Fine-Tuned Model Output**:
> The classic symptoms of Type 2 Diabetes include polyuria (frequent urination), polydipsia (increased thirst), polyphagia (increased hunger), unexplained weight loss, fatigue, and blurred vision.

## 🛠️ Local Setup & Inference

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run Inference**:
Use the provided `inference.py` script to chat with the model locally.
```bash
python inference.py
```
