#!/usr/bin/env python3
"""Generate publication-quality plots from results.json for the Lean 4 autoformalization paper/blogpost.

Outputs:
  - scripts/plots/fig7_compile_scaling.png
  - scripts/plots/fig8_throughput_vs_accuracy.png
  - scripts/plots/fig9_error_breakdown.png
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Ensure target directory exists
plots_dir = Path("scripts/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# Load results.json
with open("results.json", "r") as f:
    data = json.load(f)

models_data = data["models"]

# Set publication style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Human-readable labels and color palette
COLOR_MAP = {
    "qwen-dora-postminif2f-vF": ("#08519c", "o", "-", "QDoRA + RLCF (miniF2F)"),
    "qwen-lora-prominif2f-vF": ("#3182bd", "s", "--", "QLoRA + RLCF (miniF2F)"),
    "qwen-dora-postleanworkbook-vF": ("#006d2c", "D", "-", "QDoRA + RLCF (Workbook)"),
    "qwen-lora-proleanworkbook-vF": ("#41ab5d", "^", "--", "QLoRA + RLCF (Workbook)"),
    "qwen-dora-postsyntax-vF": ("#a50f15", "x", "-", "QDoRA + SFT (Syntax)"),
    "qwen-lora-prosyntax-vF": ("#fb6a4a", "+", "--", "QLoRA + SFT (Syntax)"),
}

# =============================================================================
# Plot 1: Figure 7 - Compile@k Scaling Curves
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

k_vals = [1, 2, 3, 4, 5]

for m_key, (color, marker, ls, label) in COLOR_MAP.items():
    if m_key in models_data:
        m_info = models_data[m_key]
        comp_at_k = m_info["compile_at_k"]
        accuracies = [comp_at_k[f"compile@{k}"]["percentage"] for k in k_vals]
        ax.plot(k_vals, accuracies, label=label, color=color, marker=marker,
                linestyle=ls, linewidth=2.0, markersize=6)

ax.set_xlabel("Compiler Feedback Attempt ($k$)")
ax.set_ylabel("Statement Type-Check Pass Rate (%)")
ax.set_title("Iterative Compile Rate (compile@k) Scaling Across Stages")
ax.set_xticks(k_vals)
ax.set_ylim(25, 70)
ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")

plt.tight_layout()
fig7_path = plots_dir / "fig7_compile_scaling.png"
fig.savefig(fig7_path)
plt.close(fig)
print(f"Saved Figure 7 to {fig7_path}")

# =============================================================================
# Plot 2: Figure 8 - Throughput vs. Compile@5 Accuracy Scatter
# =============================================================================
fig, ax = plt.subplots(figsize=(6.5, 4.5))

for m_key, (color, marker, ls, label) in COLOR_MAP.items():
    if m_key in models_data:
        m_info = models_data[m_key]
        acc = m_info["pass_at_5_final"]["percentage"]
        tps = m_info["throughput"]["tokens_per_sec"]
        
        ax.scatter(tps, acc, color=color, s=90, zorder=5, edgecolors="black", linewidths=0.8)
        
        # Label offset for clarity
        xytext = (5, 5)
        if "dora" in m_key:
            xytext = (-10, 8)
        ax.annotate(label.split(" (")[0], (tps, acc), textcoords="offset points",
                    xytext=xytext, ha="left" if tps > 5 else "right", fontsize=9,
                    fontweight="medium",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

ax.set_xlabel("Token Generation Throughput (tokens/sec)")
ax.set_ylabel("Final Compile@5 / Pass@5 Accuracy (%)")
ax.set_title("Generation Throughput vs. Compile@5 Accuracy")
ax.set_xlim(2.0, 8.0)
ax.set_ylim(40, 68)

# Highlight LoRA vs DoRA clusters
ax.axvspan(2.5, 3.8, color="#fee0d2", alpha=0.3, label="QDoRA Cluster (~3 tok/s)")
ax.axvspan(6.0, 7.5, color="#deebf7", alpha=0.3, label="QLoRA Cluster (~7 tok/s)")
ax.legend(loc="upper left", frameon=True, framealpha=0.9)

plt.tight_layout()
fig8_path = plots_dir / "fig8_throughput_vs_accuracy.png"
fig.savefig(fig8_path)
plt.close(fig)
print(f"Saved Figure 8 to {fig8_path}")

# =============================================================================
# Plot 3: Figure 9 - Compiler Error Mode Breakdown Across Unsolved Problems
# =============================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))

# Focus on QLoRA progression across stages for clean comparison
eval_models = [
    ("qwen-lora-prosyntax-vF", "QLoRA SFT"),
    ("qwen-lora-proleanworkbook-vF", "QLoRA RLCF (WB)"),
    ("qwen-lora-prominif2f-vF", "QLoRA RLCF (F2F)"),
    ("qwen-dora-postsyntax-vF", "QDoRA SFT"),
    ("qwen-dora-postleanworkbook-vF", "QDoRA RLCF (WB)"),
    ("qwen-dora-postminif2f-vF", "QDoRA RLCF (F2F)"),
]

x_labels = [label for _, label in eval_models]
error_categories = ["syntax", "unknown_ident", "typeclass", "other"]
cat_labels = ["Syntax Error", "Unknown Identifier", "Typeclass Failure", "Other"]
cat_colors = ["#d95f02", "#7570b3", "#e7298a", "#66a61e"]

# Collect percentage data per model
data_by_cat = {cat: [] for cat in error_categories}

for m_key, _ in eval_models:
    err_dict = models_data[m_key]["error_breakdown_unsolved"]
    total_errs = sum(err_dict.values())
    for cat in error_categories:
        count = err_dict.get(cat, 0)
        pct = (count / total_errs * 100) if total_errs > 0 else 0
        data_by_cat[cat].append(pct)

bottoms = np.zeros(len(eval_models))
x_indices = np.arange(len(eval_models))

for cat, cat_label, color in zip(error_categories, cat_labels, cat_colors):
    values = np.array(data_by_cat[cat])
    bars = ax.bar(x_indices, values, bottom=bottoms, label=cat_label, color=color, width=0.55, edgecolor="white")
    
    # Add percentage labels inside bars if large enough
    for i, (val, bot) in enumerate(zip(values, bottoms)):
        if val > 8:
            ax.text(x_indices[i], bot + val / 2, f"{val:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=8.5)
    bottoms += values

ax.set_xticks(x_indices)
ax.set_xticklabels(x_labels, rotation=15, ha="right")
ax.set_ylabel("Percentage of Unsolved Failures (%)")
ax.set_title("Shift in Compiler Error Modes Across Training Stages")
ax.set_ylim(0, 105)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), frameon=True)

plt.tight_layout()
fig9_path = plots_dir / "fig9_error_breakdown.png"
fig.savefig(fig9_path)
plt.close(fig)
print(f"Saved Figure 9 to {fig9_path}")

print("All plots generated successfully.")
