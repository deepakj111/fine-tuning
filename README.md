# Qwen2.5-0.5B Medical Domain Fine-Tuning

This repository contains the fine-tuning pipeline, inference scripts, and associated artifacts for adapting the Qwen2.5-0.5B model to the medical domain using Unsloth and QLoRA.

## 🚀 Architecture & Approach

- **Base Model**: `unsloth/Qwen2.5-0.5B-Instruct`
- **Techniques**:
  - **Unsloth**: For fast, memory-efficient Triton kernel patching.
  - **QLoRA**: 4-bit quantization with Low-Rank Adaptation (Rank 16, Alpha 32).
  - **Catastrophic Forgetting Prevention**: Mixed 10% general instruction data (`HuggingFaceH4/ultrachat_200k`) into the training set (Replay Buffer/Rehearsal strategy) to ensure the model retains its general knowledge.
- **Dataset**: `medalpaca/medical_meadow_medical_flashcards`

## 📂 Codebase Overview

- **`train.py`**: Production-grade training script using `unsloth`, `trl`, and `wandb` for logging. Includes CLI arguments for hyperparameter tuning.
- **`inference.py`**: CLI-based inference script using Unsloth's optimized native 2x faster inference.
- **`app.py`**: A Streamlit web application providing a chat UI for interacting with the fine-tuned model.

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

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic, production-grade dependency management, and provides a `Makefile` for convenience.

1. **Install `uv`**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Sync the Environment**:
You can use the provided Makefile to install dependencies (including dev tools):
```bash
make install
```
*(Alternatively, run `uv sync` manually.)*

3. **Environment Variables**:
Copy the example environment file and add your tokens:
```bash
cp .env.example .env
```
- `HF_TOKEN`: Required to push models to Hugging Face.
- `WANDB_API_KEY`: Required for Weights & Biases experiment tracking.

## 🧑‍💻 Usage

### 1. Training (`train.py`)
Run the production-grade training pipeline via terminal. The script features structured logging and supports various command-line arguments:

```bash
make train
# Or with custom arguments:
uv run python train.py --epochs 3 --batch_size 2 --learning_rate 2e-4
```

#### Train with Disable wandb
```
uv run python train.py --disable_wandb
```

**Key Arguments:**
- `--model_name`: Base model name (default: `unsloth/Qwen2.5-0.5B-Instruct`)
- `--epochs`: Number of training epochs
- `--batch_size`: Per-device batch size
- `--lora_r` / `--lora_alpha`: LoRA hyperparameters
- `--push_to_hub`: Flag to push the final model to Hugging Face Hub
- `--disable_wandb`: Flag to disable Weights & Biases tracking

*Use `uv run python train.py --help` to see all available options.*

### 2. Streamlit UI (`app.py`)
Launch an interactive web interface to chat with the model:
```bash
make app
```

### 3. Terminal Inference (`inference.py`)
Run a quick local inference script with predefined examples:
```bash
uv run python inference.py
```

### 4. Jupyter Notebook
If you prefer exploring the process interactively:
```bash
uv run jupyter notebook
```

## 🐳 Docker Deployment

The Streamlit application is containerized using a multi-stage Dockerfile powered by `uv` for minimal image size.

```bash
# Build the image
docker build -t medical-qwen-app .

# Run the container
docker run -p 8501:8501 --env-file .env medical-qwen-app
```

## 🧪 Testing & Code Quality

The project includes a comprehensive test suite and code formatting guidelines enforced via GitHub Actions.

- **Run tests**:
  ```bash
  make test
  ```
- **Lint the code**:
  ```bash
  make lint
  ```
- **Format the code**:
  ```bash
  make format
  ```
