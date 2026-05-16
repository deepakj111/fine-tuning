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

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic, production-grade dependency management.

1. **Install `uv`**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Sync the Environment**:
```bash
uv sync
```

3. **Environment Variables**:
Copy the example environment file and add your tokens:
```bash
cp .env.example .env
```
- `HF_TOKEN`: Required to push models to Hugging Face.
- `WANDB_API_KEY`: Required for Weights & Biases experiment tracking.

4. **Production Training Script (Recommended)**:
Run the production-grade training pipeline via terminal:
```bash
uv run python train.py --epochs 3 --batch_size 2
```

5. **Run the Jupyter Notebook Locally**:
```bash
uv run jupyter notebook
```

6. **Run the Streamlit UI**:
```bash
uv run streamlit run app.py
```

7. **Run Terminal Inference**:
```bash
uv run python inference.py
```
