"""Fig 6: Contact recapitulation heatmap.

Shows which gold-standard BMPR1A interface residues (from 1REW) are
contacted by the top 4 candidates from each design tool. Includes a
frequency bar showing how often each residue is contacted across all
16 candidates.

Input:
  - data/results/master_scores.csv
  - data/results/contact_scores.csv
  - data/structures/bmpr1a_interface_residues.json

Output:
  - figures/fig6_contact_heatmap.png (300 dpi)
  - figures/fig6_contact_heatmap.pdf
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"

TOOL_ORDER = ["PepMLM", "RFdiffusion", "BindCraft", "RFpeptides"]
TOOL_COLORS = {
    "BindCraft": "#E63946",
    "RFdiffusion": "#457B9D",
    "PepMLM": "#2A9D8F",
    "RFpeptides": "#E9C46A",
}
TOP_N_PER_TOOL = 4


def main():
    # Load data
    cs = pd.read_csv(DATA_DIR / "results" / "contact_scores.csv")
    ms = pd.read_csv(DATA_DIR / "results" / "master_scores.csv")

    with open(DATA_DIR / "structures" / "bmpr1a_interface_residues.json") as f:
        gs = json.load(f)
    gold_sorted = sorted(gs["bmpr1a_interface_residues"], key=lambda x: x[1])
    gold_residues = [r[1] for r in gold_sorted]
    gold_names = [f"{r[0][:3]}{r[1]}" for r in gold_sorted]

    # Select top N from each tool by composite rank
    designed = ms[~ms["source_tool"].str.contains("control|random", case=False)]
    top_per_tool = []
    for tool in TOOL_ORDER:
        tool_df = designed[designed["source_tool"] == tool].nsmallest(
            TOP_N_PER_TOOL, "composite_rank"
        )
        top_per_tool.append(tool_df)
    selected = pd.concat(top_per_tool)
    merged = selected.merge(
        cs[["peptide_id", "recapitulated_residues"]], on="peptide_id", how="left"
    )

    # Build binary contact matrix
    n_candidates = len(merged)
    n_residues = len(gold_residues)
    matrix = np.zeros((n_candidates, n_residues))
    labels = []
    tool_list = []

    for i, (_, row) in enumerate(merged.iterrows()):
        res = (
            ast.literal_eval(row["recapitulated_residues"])
            if pd.notna(row["recapitulated_residues"])
            else []
        )
        for r in res:
            if r in gold_residues:
                j = gold_residues.index(r)
                matrix[i, j] = 1
        pid = (
            row["peptide_id"]
            .replace("bindcraft_bmpr1a_", "bc_")
            .replace("pepmlm_", "pm_")
            .replace("rfdiff_", "rf_")
            .replace("rfpep_", "rp_")
        )
        frac = row["contact_recap"]
        labels.append(f"{pid} ({frac:.2f})")
        tool_list.append(row["source_tool"])

    # ── Figure ──
    fig, (ax_main, ax_freq) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0.05},
        sharex=True,
    )

    # Main heatmap
    cmap = plt.cm.colors.ListedColormap(["#F0F0F0", "#2A9D8F"])
    ax_main.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    # Y-axis labels colored by tool
    ax_main.set_yticks(range(n_candidates))
    ax_main.set_yticklabels(labels, fontsize=11, fontfamily="monospace")
    for i, tool in enumerate(tool_list):
        color = TOOL_COLORS.get(tool, "#333333")
        ax_main.get_yticklabels()[i].set_color(color)
        ax_main.get_yticklabels()[i].set_fontweight("bold")

    # Horizontal separators between tool groups
    current_tool = tool_list[0]
    for i, tool in enumerate(tool_list):
        if tool != current_tool:
            ax_main.axhline(i - 0.5, color="#888888", linewidth=1.5)
            current_tool = tool

    # Tool group labels on right side
    tool_starts = {}
    current_tool = tool_list[0]
    start = 0
    for i, tool in enumerate(tool_list):
        if tool != current_tool:
            tool_starts[current_tool] = (start, i - 1)
            current_tool = tool
            start = i
    tool_starts[current_tool] = (start, n_candidates - 1)

    for tool, (s, e) in tool_starts.items():
        mid = (s + e) / 2
        ax_main.text(
            n_residues + 0.5, mid, tool,
            fontsize=12, fontweight="bold", va="center", ha="left",
            color=TOOL_COLORS.get(tool, "#333333"),
        )

    # Grid lines
    ax_main.set_xticks(np.arange(-0.5, n_residues, 1), minor=True)
    ax_main.set_yticks(np.arange(-0.5, n_candidates, 1), minor=True)
    ax_main.grid(which="minor", color="white", linewidth=2)
    ax_main.tick_params(which="minor", size=0)
    ax_main.tick_params(axis="x", bottom=False, labelbottom=False)

    ax_main.set_title(
        f"Contact Recapitulation: Top {TOP_N_PER_TOOL} Candidates Per Tool vs. 1REW Interface",
        fontsize=15, fontweight="bold", pad=12,
    )

    ax_main.set_xlim(-0.5, n_residues + 4)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=TOOL_COLORS[t], label=t) for t in TOOL_ORDER
    ]
    ax_main.legend(
        handles=legend_elements, loc="upper right", fontsize=11,
        framealpha=0.95, edgecolor="#CCCCCC", bbox_to_anchor=(0.85, 1.0),
    )

    # ── Frequency bar ──
    freq = matrix.sum(axis=0) / n_candidates
    ax_freq.bar(range(n_residues), freq, color="#2A9D8F", alpha=0.7, width=0.8)
    ax_freq.set_xlim(-0.5, n_residues + 4)
    ax_freq.set_ylim(0, 1.05)
    ax_freq.set_ylabel("Frequency", fontsize=11)
    ax_freq.set_xticks(range(n_residues))
    ax_freq.set_xticklabels(gold_names, fontsize=10, rotation=55, ha="right")
    ax_freq.set_xlabel(
        "BMPR1A Gold-Standard Interface Residues (from 1REW)", fontsize=12
    )

    for i, f in enumerate(freq):
        if f >= 0.5:
            ax_freq.text(i, f + 0.02, f"{f:.0%}", ha="center", fontsize=8, color="#333333")

    # Save
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig6_contact_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig6_contact_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved fig6_contact_heatmap.png/pdf to {FIG_DIR}")


if __name__ == "__main__":
    main()
