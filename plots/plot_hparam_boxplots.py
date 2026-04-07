"""
C3-Net Hyperparameter Search — Box Plot
Usage: python plot_hparam_boxplots.py
Output: boxplot_hparam.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

CSV_PATH   = "/media/16TB_Storage/kavin/models/c3-net/hparam_search/results.csv"
OUTPUT     = "/home/kavin/C3-Net/plots/boxplot_hparam.pdf"

MODALITIES  = ["image_only", "image_gaze", "image_text", "image_gaze_text"]
MOD_LABELS  = ["Image only", "Image + Gaze", "Image + Text", "All modalities"]
COLORS      = ["#D3D1C7", "#9FE1CB", "#CECBF6", "#B5D4F4"]
EDGE        = ["#5F5E5A", "#0F6E56", "#534AB7", "#185FA5"]

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

data = {
    mod: {
        "f1":  df[df["modality"] == mod]["val_f1"].values,
        "auc": df[df["modality"] == mod]["val_auc"].values,
    }
    for mod in MODALITIES
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.35)

METRICS = [("f1", "Val F1 (%)"), ("auc", "Val AUC")]

for ax, (met, ylabel) in zip(axes, METRICS):
    positions = np.arange(len(MODALITIES))

    for i, mod in enumerate(MODALITIES):
        vals = data[mod][met]
        bp = ax.boxplot(
            vals,
            positions=[i],
            widths=0.45,
            patch_artist=True,
            notch=False,
            showfliers=True,
            flierprops=dict(marker="o", markersize=4, linestyle="none",
                            markerfacecolor="none", markeredgecolor=EDGE[i],
                            markeredgewidth=0.8),
            medianprops=dict(color="#1a1a1a", linewidth=2),
            whiskerprops=dict(color=EDGE[i], linewidth=1),
            capprops=dict(color=EDGE[i], linewidth=1),
            boxprops=dict(facecolor=COLORS[i], edgecolor=EDGE[i], linewidth=1),
        )

        # Mean dot
        mean_val = vals.mean()
        ax.plot(i, mean_val, marker="o", markersize=5,
                color=EDGE[i], zorder=5, label="_nolegend_")

        # Best-run diamond
        best_val = vals.max()
        ax.plot(i, best_val, marker="D", markersize=5,
                color=EDGE[i], zorder=5, label="_nolegend_")

    ax.set_xticks(positions)
    ax.set_xticklabels(MOD_LABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

# Shared legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS[i], edgecolor=EDGE[i], label=MOD_LABELS[i])
    for i in range(len(MODALITIES))
] + [
    plt.Line2D([0], [0], marker="o", color="gray", markersize=5,
               linestyle="none", label="Mean"),
    plt.Line2D([0], [0], marker="D", color="gray", markersize=5,
               linestyle="none", label="Best run"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=6,
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

fig.suptitle("Hyperparameter Search — Val F1 & AUC by Modality",
             fontsize=11, fontweight="normal", y=1.01)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=300)
print(f"Saved → {OUTPUT}")