import argparse
import json
import os
import re
import pandas as pd
import plotly.graph_objects as go


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Evaluation Results")
    parser.add_argument("--model_dir", type=str, default="./medical_qwen_lora", help="Path to saved LoRA adapters")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Path to save plots")
    return parser.parse_args()


def plot_metrics_comparison(results: dict, output_dir: str):
    base = results.get("baseline_scores", {})
    ft = results.get("finetuned_scores", {})

    metrics_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
    metrics_keys = ["rouge1", "rouge2", "rougeL", "bleu"]

    baseline_vals = [base.get(k, 0) for k in metrics_keys]
    finetuned_vals = [ft.get(k, 0) for k in metrics_keys]
    deltas = [f - b for f, b in zip(finetuned_vals, baseline_vals)]
    pct_changes = [(d / max(b, 1e-9)) * 100 for d, b in zip(deltas, baseline_vals)]

    # 1. Grouped Bar Chart
    fig1 = go.Figure()
    fig1.add_trace(
        go.Bar(
            name="Base Model",
            x=metrics_labels,
            y=baseline_vals,
            marker_color="#D3D3D3",
            text=[f"{v:.4f}" for v in baseline_vals],
            textposition="outside",
        )
    )
    fig1.add_trace(
        go.Bar(
            name="Fine-Tuned",
            x=metrics_labels,
            y=finetuned_vals,
            marker_color="#4ECDC4",
            text=[f"{v:.4f}" for v in finetuned_vals],
            textposition="outside",
        )
    )
    fig1.update_layout(
        title_text="<b>Text Generation Metrics: Base vs Fine-Tuned</b>",
        title_x=0.5,
        barmode="group",
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        height=480,
    )
    fig1.write_image(os.path.join(output_dir, "metrics_comparison_bar.jpg"))

    # 2. Radar Chart
    theta = metrics_labels + [metrics_labels[0]]
    r_base = baseline_vals + [baseline_vals[0]]
    r_ft = finetuned_vals + [finetuned_vals[0]]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(r=r_base, theta=theta, fill="toself", name="Base Model", line_color="#D3D3D3"))
    fig2.add_trace(go.Scatterpolar(r=r_ft, theta=theta, fill="toself", name="Fine-Tuned Model", line_color="#4ECDC4"))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(finetuned_vals) * 1.2])),
        title_text="<b>Metric Coverage — Radar View</b>",
        title_x=0.5,
        height=500,
        paper_bgcolor="white",
    )
    fig2.write_image(os.path.join(output_dir, "metrics_radar.jpg"))

    # 3. Improvement % Bar Chart
    colors = ["#FF6B6B" if d < 0 else "#4ECDC4" for d in deltas]
    fig3 = go.Figure(
        go.Bar(
            x=metrics_labels,
            y=pct_changes,
            marker_color=colors,
            text=[f"{p:+.1f}%" for p in pct_changes],
            textposition="outside",
        )
    )
    fig3.update_layout(
        title_text="<b>Percentage Improvement After Fine-Tuning</b>",
        title_x=0.5,
        yaxis_title="% Change",
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        height=420,
    )
    fig3.write_image(os.path.join(output_dir, "improvement_percentage.jpg"))

    # 4. Perplexity Comparison
    b_ppl = base.get("perplexity", 0)
    f_ppl = ft.get("perplexity", 0)
    fig4 = go.Figure(
        go.Bar(
            x=["Base Model", "Fine-Tuned Model"],
            y=[b_ppl, f_ppl],
            marker_color=["#D3D3D3", "#FF6B6B"],
            text=[f"{b_ppl:.2f}", f"{f_ppl:.2f}"],
            textposition="outside",
            width=0.4,
        )
    )
    fig4.update_layout(
        title_text="<b>Perplexity (Lower = Better)</b>",
        title_x=0.5,
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        height=420,
    )
    fig4.write_image(os.path.join(output_dir, "perplexity_comparison.jpg"))

    print("✅ Evaluation metric plots generated.")


def plot_catastrophic_forgetting(results: dict, output_dir: str):
    forgetting = results.get("forgetting_checks", [])
    if not forgetting:
        return

    tasks = [x["task"] for x in forgetting]
    scores = [x["score"] for x in forgetting]
    colors = ["#4ECDC4" if s >= 0.3 else "#FF6B6B" for s in scores]

    fig = go.Figure(
        go.Bar(
            x=tasks, y=scores, marker_color=colors, text=[f"{s:.2f}" for s in scores], textposition="outside", width=0.5
        )
    )
    fig.add_hline(y=0.3, line_dash="dash", line_color="#888", annotation_text="0.3 threshold")
    fig.update_layout(
        title_text="<b>Catastrophic Forgetting Check — Retention</b>",
        title_x=0.5,
        yaxis=dict(range=[0, 1.2]),
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        height=450,
    )
    fig.write_image(os.path.join(output_dir, "catastrophic_forgetting.jpg"))
    print("✅ Catastrophic forgetting plot generated.")


def plot_training_loss(model_dir: str, output_dir: str):
    log_path = os.path.join(model_dir, "training_logs.json")
    if not os.path.exists(log_path):
        return

    with open(log_path) as f:
        logs = json.load(f)

    if not logs:
        return

    steps = [x["step"] for x in logs]
    losses = [x["loss"] for x in logs]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=losses,
            mode="lines+markers",
            name="Training Loss",
            line=dict(color="#FF6B6B"),
            marker=dict(size=5),
        )
    )

    rolling_avg = pd.Series(losses).rolling(window=5, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=rolling_avg.tolist(),
            mode="lines",
            name="5-Step Rolling Avg",
            line=dict(color="#4ECDC4", dash="dot"),
        )
    )

    fig.update_layout(
        title_text="<b>Training Loss Curve</b>",
        title_x=0.5,
        xaxis_title="Step",
        yaxis_title="Loss",
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        height=450,
    )
    fig.write_image(os.path.join(output_dir, "training_loss.jpg"))
    print("✅ Training loss curve generated.")


def update_readme_table(results: dict):
    base = results.get("baseline_scores", {})
    ft = results.get("finetuned_scores", {})
    if not base or not ft:
        return

    table = f"""| Metric | Base Model | Fine-Tuned | % Change |
|--------|------------|------------|----------|
| **ROUGE-1** | {base.get("rouge1", 0):.4f} | {ft.get("rouge1", 0):.4f} | {((ft.get("rouge1", 0) - base.get("rouge1", 0)) / max(base.get("rouge1", 1e-9), 1e-9) * 100):.1f}% |
| **ROUGE-2** | {base.get("rouge2", 0):.4f} | {ft.get("rouge2", 0):.4f} | {((ft.get("rouge2", 0) - base.get("rouge2", 0)) / max(base.get("rouge2", 1e-9), 1e-9) * 100):.1f}% |
| **ROUGE-L** | {base.get("rougeL", 0):.4f} | {ft.get("rougeL", 0):.4f} | {((ft.get("rougeL", 0) - base.get("rougeL", 0)) / max(base.get("rougeL", 1e-9), 1e-9) * 100):.1f}% |
| **BLEU** | {base.get("bleu", 0):.4f} | {ft.get("bleu", 0):.4f} | {((ft.get("bleu", 0) - base.get("bleu", 0)) / max(base.get("bleu", 1e-9), 1e-9) * 100):.1f}% |
| **Perplexity** | {base.get("perplexity", 0):.2f} | {ft.get("perplexity", 0):.2f} | {ft.get("perplexity", 0) - base.get("perplexity", 0):.2f} |"""

    readme_path = "README.md"
    if not os.path.exists(readme_path):
        return

    with open(readme_path) as f:
        content = f.read()

    pattern = r"(<!-- RESULTS_TABLE_START -->\n)(.*?)(\n<!-- RESULTS_TABLE_END -->)"
    new_content = re.sub(pattern, lambda m: m.group(1) + table + m.group(3), content, flags=re.DOTALL)

    with open(readme_path, "w") as f:
        f.write(new_content)
    print("✅ README.md table updated dynamically.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    res_path = os.path.join(args.model_dir, "training_results.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            results = json.load(f)
        plot_metrics_comparison(results, args.output_dir)
        plot_catastrophic_forgetting(results, args.output_dir)
        update_readme_table(results)
    else:
        print(f"⚠️  No training_results.json found at {res_path}. Run eval_model.py first.")

    plot_training_loss(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
