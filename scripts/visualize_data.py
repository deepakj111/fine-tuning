import argparse
import os
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
from plotly.subplots import make_subplots
from unsloth import FastLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Dataset and Model Architecture")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./plots")
    return parser.parse_args()


def plot_dataset_eda(output_dir: str):
    print("📥 Loading Medical Meadow Flashcards dataset...")
    raw_dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    df = raw_dataset.to_pandas()
    df = df.rename(columns={"input": "question", "output": "answer"})
    df["q_len"] = df["question"].str.split().str.len()
    df["a_len"] = df["answer"].str.split().str.len()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Question Length Distribution",
            "Answer Length Distribution",
            "Q vs A Length Scatter",
            "Top 20 Medical Terms in Questions",
        ],
        specs=[[{"type": "histogram"}, {"type": "histogram"}], [{"type": "scatter"}, {"type": "bar"}]],
    )

    fig.add_trace(
        go.Histogram(x=df["q_len"].clip(upper=100), nbinsx=50, marker_color="#4ECDC4", name="Q Length"), row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df["a_len"].clip(upper=200), nbinsx=50, marker_color="#FF6B6B", name="A Length"), row=1, col=2
    )

    sample = df.sample(min(1000, len(df)), random_state=42)
    fig.add_trace(
        go.Scatter(
            x=sample["q_len"],
            y=sample["a_len"],
            mode="markers",
            opacity=0.4,
            marker=dict(size=4, color="#45B7D1"),
            name="Q vs A",
        ),
        row=2,
        col=1,
    )

    all_words = " ".join(df["question"].tolist()).lower()
    stop_words = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "is",
        "to",
        "and",
        "or",
        "for",
        "with",
        "by",
        "are",
        "be",
        "was",
        "what",
        "which",
        "that",
        "this",
        "it",
        "its",
        "as",
        "can",
        "how",
        "does",
        "do",
        "when",
        "where",
        "why",
        "not",
        "from",
        "at",
    }
    words = [w for w in re.findall(r"\b[a-z]{4,}\b", all_words) if w not in stop_words]
    top_words = Counter(words).most_common(20)
    tw_df = pd.DataFrame(top_words, columns=["term", "count"])

    fig.add_trace(
        go.Bar(
            x=tw_df["count"],
            y=tw_df["term"],
            orientation="h",
            marker_color=px.colors.sequential.Viridis_r[:20],
            name="Term Freq",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title_text="<b>Medical Flashcards Dataset EDA</b>",
        title_x=0.5,
        height=750,
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
    )

    out_path = os.path.join(output_dir, "dataset_eda.jpg")
    fig.write_image(out_path)
    print(f"✅ Dataset EDA plot saved to {out_path}")


def plot_architecture(model_name: str, output_dir: str):
    print(f"🔄 Loading base model: {model_name} for architecture analysis...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    module_sizes = {}
    for name, param in model.named_parameters():
        top_module = name.split(".")[0]
        module_sizes[top_module] = module_sizes.get(top_module, 0) + param.numel()

    labels = list(module_sizes.keys())
    values = list(module_sizes.values())
    total_params = sum(values)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                textinfo="label+percent",
                marker=dict(line=dict(color="white", width=2)),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=f"<b>{model_name}</b> — Parameter Distribution", x=0.5),
        annotations=[
            dict(text=f"<b>{total_params / 1e6:.0f}M</b><br>params", x=0.5, y=0.5, font_size=16, showarrow=False)
        ],
        height=500,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    out_path = os.path.join(output_dir, "model_architecture.jpg")
    fig.write_image(out_path)
    print(f"✅ Architecture plot saved to {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dataset_eda(args.output_dir)
    plot_architecture(args.model_name, args.output_dir)


if __name__ == "__main__":
    main()
