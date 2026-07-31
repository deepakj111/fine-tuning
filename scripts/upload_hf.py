import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Upload Model to Hugging Face Hub")
    parser.add_argument("--model_dir", type=str, default="./medical_qwen_lora", help="Path to saved LoRA adapters")
    parser.add_argument(
        "--hub_repo_id", type=str, default="deepakj111/medical-qwen2.5-0.5B-lora", help="HF Hub Repo ID to push to"
    )
    return parser.parse_args()


def generate_model_card(repo_id: str, results: dict, plots: list = None) -> str:
    base = results.get("baseline_scores", {})
    ft = results.get("finetuned_scores", {})

    card = f"""---
language:
- en
tags:
- medical
- qwen
- qlora
- unsloth
license: apache-2.0
base_model: unsloth/Qwen2.5-0.5B-Instruct
---

# {repo_id.split("/")[-1]}

This model is a fine-tuned version of `unsloth/Qwen2.5-0.5B-Instruct` for the medical domain using Unsloth and QLoRA.
It was trained on the `{results.get("dataset", "Unknown")}` dataset.

## Evaluation Results

The model was evaluated on {results.get("eval_samples", "unknown")} validation samples.

| Metric | Base Model | Fine-Tuned | % Change |
|--------|------------|------------|----------|
| **ROUGE-1** | {base.get("rouge1", 0):.4f} | {ft.get("rouge1", 0):.4f} | {((ft.get("rouge1", 0) - base.get("rouge1", 0)) / max(base.get("rouge1", 1e-9), 1e-9) * 100):.1f}% |
| **ROUGE-2** | {base.get("rouge2", 0):.4f} | {ft.get("rouge2", 0):.4f} | {((ft.get("rouge2", 0) - base.get("rouge2", 0)) / max(base.get("rouge2", 1e-9), 1e-9) * 100):.1f}% |
| **ROUGE-L** | {base.get("rougeL", 0):.4f} | {ft.get("rougeL", 0):.4f} | {((ft.get("rougeL", 0) - base.get("rougeL", 0)) / max(base.get("rougeL", 1e-9), 1e-9) * 100):.1f}% |
| **BLEU** | {base.get("bleu", 0):.4f} | {ft.get("bleu", 0):.4f} | {((ft.get("bleu", 0) - base.get("bleu", 0)) / max(base.get("bleu", 1e-9), 1e-9) * 100):.1f}% |
| **Perplexity** | {base.get("perplexity", 0):.2f} | {ft.get("perplexity", 0):.2f} | {ft.get("perplexity", 0) - base.get("perplexity", 0):.2f} |

*(Note: Lower perplexity is better)*
"""
    if plots:
        card += "\n## Training and Evaluation Plots\n\n"
        for plot in plots:
            name = plot.replace(".jpg", "").replace("_", " ").title()
            jpg_url = f"https://huggingface.co/{repo_id}/resolve/main/plots/{plot}"
            card += f"### {name}\n"
            card += f"![{name}]({jpg_url})\n\n"

    card += f"""
## Usage

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{repo_id}",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```
"""
    return card


def main():
    load_dotenv()
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not found in environment variables.")
        sys.exit(1)

    results_path = os.path.join(args.model_dir, "training_results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        logger.info(f"Loaded evaluation results from {results_path}")
    else:
        logger.warning(f"No training_results.json found at {results_path}. Model card will have empty metrics.")

    # Copy plots to model_dir so they get uploaded
    import shutil

    plots_src = "plots"
    plots_dest = os.path.join(args.model_dir, "plots")
    plots_list = []
    if os.path.exists(plots_src):
        shutil.copytree(plots_src, plots_dest, dirs_exist_ok=True)
        plots_list = [f for f in os.listdir(plots_src) if f.endswith(".jpg")]
        plot_order = [
            "dataset_eda",
            "model_architecture",
            "training_loss",
            "metrics_comparison_bar",
            "metrics_radar",
            "improvement_percentage",
            "perplexity_comparison",
            "catastrophic_forgetting",
        ]
        plots_list.sort(
            key=lambda x: plot_order.index(x.replace(".jpg", "")) if x.replace(".jpg", "") in plot_order else 99
        )
        logger.info(f"Copied {len(plots_list)} JPG plots to {plots_dest}")

    # Generate and write README.md (Model Card)
    model_card = generate_model_card(args.hub_repo_id, results, plots_list)
    readme_path = os.path.join(args.model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card)
    logger.info(f"Generated dynamic Model Card at {readme_path}")

    # Login and Push
    logger.info("Logging into Hugging Face...")
    login(token=hf_token)

    api = HfApi()
    logger.info(f"Creating repository {args.hub_repo_id} (if it doesn't exist)...")
    try:
        api.create_repo(repo_id=args.hub_repo_id, exist_ok=True, private=False)
    except Exception as e:
        logger.warning(f"Could not create repo (maybe it exists): {e}")

    logger.info(f"Uploading folder {args.model_dir} to {args.hub_repo_id}...")
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.hub_repo_id,
        commit_message="Upload model, tokenizer, and dynamic model card",
    )
    logger.info("✅ Upload complete!")


if __name__ == "__main__":
    main()
