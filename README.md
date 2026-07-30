# Qwen2.5-0.5B Medical Domain Fine-Tuning

This repository contains a complete pipeline for adapting the **Qwen2.5-0.5B-Instruct** model to the medical domain using **Unsloth** and **QLoRA**. It features a robust defense against catastrophic forgetting and includes interactive web UI and visualization tools.

## 📓 Notebook vs 📜 Scripts: The Dual Workflow

This project offers two parallel ways to run the pipeline, catering to both research and production:

1. **Jupyter Notebook (`Fine_Tuning_...ipynb`)**:
   Designed for **rapid prototyping**, visual step-by-step learning, and interactive Exploratory Data Analysis (EDA). It is highly optimized to run smoothly on constrained cloud environments like the **Google Colab Free Tier (T4 GPU)**.
2. **Production Scripts (`scripts/`)**:
   A modular, production-grade pipeline built for **automation, scale, and local execution** (e.g., local Linux/WSL environments on GPU-enabled laptops). The scripts perfectly mirror the notebook's performance and logic but decouple the processes (training, evaluation, visualization, upload) into discrete, maintainable steps managed by a `Makefile`.

---

## 🚀 Architecture & Technical Deep Dive

### 1. Base Model & Dataset
- **Base Model**: `unsloth/Qwen2.5-0.5B-Instruct`
- **Dataset**: `medalpaca/medical_meadow_medical_flashcards`

### 2. Fine-Tuning via QLoRA & Unsloth
To train efficiently on consumer hardware, we utilize **QLoRA** (Quantized Low-Rank Adaptation):
- The massive base model weights are frozen and loaded in **4-bit precision** to drastically cut VRAM usage.
- We attach and train tiny "adapter" matrices (Rank `r=16`, `alpha=32`) which represent only ~6.3% of the total parameters.
- **Unsloth Integration**: By setting `lora_dropout=0` and utilizing Unsloth's custom Triton kernels, we achieve up to **2x faster training and inference** with significantly lower memory overhead compared to standard Hugging Face Transformers.

### 3. Catastrophic Forgetting Prevention (Rehearsal Strategy)
A common issue in domain-specific fine-tuning is **catastrophic forgetting**, where the model "forgets" how to have normal conversations or do basic logic.
To prevent this, our pipeline implements a **Replay Buffer (Rehearsal)** strategy:
- We dynamically inject a 10% slice of general instruction data (`HuggingFaceH4/ultrachat_200k`) into the medical training mix.
- This forces the model to constantly "rehearse" general capabilities (math, common sense, structure) while learning deep medical knowledge.

### 4. Fully Local & Private Tracking
The pipeline relies entirely on local tracking (`training_logs.json`). There are no forced external dependencies or API keys required for telemetry (like W&B). Detailed VRAM profiling and parameter counts are natively logged to your console.

---

## 🛠️ Step-by-Step Execution Guide

We use [`uv`](https://github.com/astral-sh/uv) for lightning-fast, deterministic dependency management. All core workflows are orchestrated via the `Makefile`.

### Step 1: Environment Setup
Ensure you have `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then run:
```bash
make install
```
*This installs all project dependencies locally in a virtual environment (`.venv`).*

### Step 2: Model Training
```bash
make train
```
**What this does:**
1. Loads the base model in 4-bit precision and configures QLoRA.
2. Prepares the medical dataset and merges it with the UltraChat general replay buffer.
3. Fine-tunes the model (logs VRAM usage before/after).
4. Saves the lightweight LoRA adapters to `./medical_qwen_lora` and outputs `training_logs.json`.

*(Optional Customization: `PYTHONPATH=. uv run python scripts/train.py --epochs 3 --batch_size 2 --learning_rate 2e-4`)*

### Step 3: Evaluation & Validation
```bash
make evaluate
```
**What this does:**
1. Evaluates the **Base Model** against a validation set of medical flashcards.
2. Evaluates your **Fine-Tuned Model** against the exact same set.
3. Computes **ROUGE-1/2/L, BLEU, and Perplexity** (lower is better).
4. Runs a dedicated **Catastrophic Forgetting Check** (asking math and general knowledge questions) to ensure the model retained its foundation.
5. Saves all comparison metrics to `training_results.json`.

### Step 4: Visualizations
```bash
make visualize_data
make visualize_results
```
**What this does:**
- `visualize_data`: Generates interactive HTML Plotly charts in `./plots/` showing Q/A length distributions and LoRA parameter pie charts.
- `visualize_results`: Reads your local JSON logs and generates beautiful Radar charts, Grouped Bar charts, and Loss Curves comparing the Before & After metrics.

### Step 5: Test the Model (Inference)
There are two ways to talk to your model:
```bash
# Option A: Run a fast CLI inference script with predefined examples
make test-inference

# Option B: Launch an interactive Streamlit Chat Web App
make app
```

### Step 6: Publish to Hugging Face
Once you are happy with the results, share your model with the world:
```bash
# Note: Requires HF_TOKEN in your .env file
make upload
```
**What this does:**
Pushes your LoRA adapters to the Hugging Face Hub and dynamically generates a rich Model Card containing your training parameters and exact evaluation metrics.

---

## 📊 Performance Improvement (Base vs. Fine-Tuned)

After fine-tuning, the model shows significant improvement in medical Q&A tasks:

| Metric | Base Model | Fine-Tuned | % Change |
|--------|------------|------------|----------|
| **ROUGE-1** | 0.2031 | 0.3541 | +74.3% |
| **ROUGE-2** | 0.0521 | 0.1632 | +213.2% |
| **ROUGE-L** | 0.1741 | 0.3121 | +79.3% |
| **BLEU** | 0.0412 | 0.1245 | +202.1% |
| **Perplexity** | 8.42 | 3.12 | Lower is better |


### Example Interaction

**Question**: What are the symptoms of Type 2 Diabetes?

**Base Model Output**:
> Type 2 diabetes is a condition where... *(often generic, short, or overly verbose)*

**Fine-Tuned Model Output**:
> The classic symptoms of Type 2 Diabetes include polyuria (frequent urination), polydipsia (increased thirst), polyphagia (increased hunger), unexplained weight loss, fatigue, and blurred vision.

---

## 🔗 Hugging Face Model

The fine-tuned LoRA adapters are hosted on the Hugging Face Hub:
[deepakj111/medical-qwen2.5-0.5B-lora](https://huggingface.co/deepakj111/medical-qwen2.5-0.5B-lora)

---

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

- **Run tests**: `make test`
- **Lint the code**: `make lint`
- **Format the code**: `make format`
