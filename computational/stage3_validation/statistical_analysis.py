"""Task 12: Statistical analysis and publication-ready figures.

Performs:
  1. Kruskal-Wallis tests (tool comparisons, candidates vs controls)
  2. Pairwise Dunn post-hoc tests with Bonferroni correction
  3. Cross-method correlation analysis
  4. Publication figures (violin plots, heatmaps, composite ranking)

Output: figures/ directory + stats summary to stdout
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path
from itertools import combinations

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"

# Color palette for tools
TOOL_COLORS = {
    "BindCraft": "#E63946",
    "RFdiffusion": "#457B9D",
    "PepMLM": "#2A9D8F",
    "RFpeptides": "#E9C46A",
    "scrambled_control": "#AAAAAA",
    "random_control": "#CCCCCC",
}
TOOL_ORDER = ["BindCraft", "RFdiffusion", "PepMLM", "RFpeptides",
              "scrambled_control", "random_control"]
TOOL_LABELS = {
    "BindCraft": "BindCraft",
    "RFdiffusion": "RFdiffusion",
    "PepMLM": "PepMLM",
    "RFpeptides": "RFpeptides",
    "scrambled_control": "Scrambled",
    "random_control": "Random",
}


def load_data():
    """Load master scores table."""
    df = pd.read_csv(DATA_DIR / "results" / "master_scores.csv")
    # Ensure tool order for plotting
    df["source_tool"] = pd.Categorical(
        df["source_tool"], categories=TOOL_ORDER, ordered=True
    )
    return df


# ── Statistical tests ────────────────────────────────────────────────

def kruskal_wallis_tests(df):
    """Run Kruskal-Wallis + Dunn post-hoc for key metrics."""
    metrics = {
        "iptm": ("AF3 ipTM", False),
        "rosetta_dG": ("Rosetta dG (REU)", True),
        "foldx_dG": ("FoldX ΔG (kcal/mol)", True),
    }
    if "contact_recap" in df.columns:
        metrics["contact_recap"] = ("Contact Recapitulation", False)

    results = []

    print("=" * 70)
    print("KRUSKAL-WALLIS TESTS (tool comparison)")
    print("=" * 70)

    for metric, (label, lower_better) in metrics.items():
        col = df[metric].dropna()
        groups = [df.loc[df["source_tool"] == t, metric].dropna().values
                  for t in TOOL_ORDER if len(df.loc[df["source_tool"] == t, metric].dropna()) > 0]
        group_names = [t for t in TOOL_ORDER
                       if len(df.loc[df["source_tool"] == t, metric].dropna()) > 0]

        if len(groups) < 2:
            continue

        stat, p_val = stats.kruskal(*groups)
        print(f"\n{label}:")
        print(f"  H-statistic = {stat:.3f}, p = {p_val:.2e}")
        print(f"  {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

        results.append({
            "metric": metric, "label": label, "H_stat": stat,
            "p_value": p_val, "significant": p_val < 0.05
        })

        # Dunn post-hoc (pairwise Mann-Whitney with Bonferroni)
        if p_val < 0.05:
            n_comparisons = len(groups) * (len(groups) - 1) // 2
            print(f"  Pairwise comparisons (Bonferroni α = {0.05/n_comparisons:.4f}):")
            for i, j in combinations(range(len(groups)), 2):
                if len(groups[i]) > 0 and len(groups[j]) > 0:
                    u_stat, p_pair = stats.mannwhitneyu(
                        groups[i], groups[j], alternative="two-sided"
                    )
                    p_adj = min(p_pair * n_comparisons, 1.0)
                    sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
                    print(f"    {TOOL_LABELS.get(group_names[i], group_names[i]):12s} vs "
                          f"{TOOL_LABELS.get(group_names[j], group_names[j]):12s}: "
                          f"p_adj = {p_adj:.4f} {sig}")

    return results


def candidates_vs_controls(df):
    """Mann-Whitney U tests: designed candidates vs controls."""
    print(f"\n{'=' * 70}")
    print("CANDIDATES vs CONTROLS (Mann-Whitney U)")
    print("=" * 70)

    candidates = df[~df["source_tool"].str.contains("control")]
    controls = df[df["source_tool"].str.contains("control")]

    metrics = {
        "iptm": "AF3 ipTM",
        "rosetta_dG": "Rosetta dG",
        "foldx_dG": "FoldX ΔG",
    }
    if "contact_recap" in df.columns:
        metrics["contact_recap"] = "Contact Recap"

    for metric, label in metrics.items():
        c_vals = candidates[metric].dropna()
        ctrl_vals = controls[metric].dropna()
        u_stat, p_val = stats.mannwhitneyu(c_vals, ctrl_vals, alternative="two-sided")
        effect_r = 1 - (2 * u_stat) / (len(c_vals) * len(ctrl_vals))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {label:20s}: candidates={c_vals.mean():.3f} vs controls={ctrl_vals.mean():.3f}  "
              f"U={u_stat:.0f}, p={p_val:.4f} {sig}, r={effect_r:.3f}")


def correlation_analysis(df):
    """Cross-method correlation matrix."""
    print(f"\n{'=' * 70}")
    print("CROSS-METHOD CORRELATIONS (Spearman)")
    print("=" * 70)

    cols = ["iptm", "rosetta_dG", "foldx_dG", "plddt_mean_B", "pae_interface_mean"]
    if "contact_recap" in df.columns:
        cols.append("contact_recap")

    labels = {
        "iptm": "ipTM", "rosetta_dG": "Rosetta dG", "foldx_dG": "FoldX ΔG",
        "plddt_mean_B": "pLDDT", "pae_interface_mean": "PAE",
        "contact_recap": "Contact Recap"
    }

    existing = [c for c in cols if c in df.columns]
    sub = df[existing].dropna()

    corr_matrix = sub.corr(method="spearman")
    corr_matrix.index = [labels.get(c, c) for c in corr_matrix.index]
    corr_matrix.columns = [labels.get(c, c) for c in corr_matrix.columns]
    print(corr_matrix.round(3).to_string())

    return corr_matrix, existing, labels


# ── Figures ──────────────────────────────────────────────────────────

def fig1_violin_plots(df):
    """Figure 1: Violin plots of key metrics by source tool."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    metrics = [
        ("iptm", "AF3 ipTM", False),
        ("rosetta_dG", "Rosetta dG (REU)", True),
        ("foldx_dG", "FoldX ΔG (kcal/mol)", True),
    ]

    for ax, (metric, label, invert) in zip(axes, metrics):
        plot_df = df[["source_tool", metric]].dropna()
        plot_df["tool_label"] = plot_df["source_tool"].map(TOOL_LABELS)

        order = [TOOL_LABELS[t] for t in TOOL_ORDER if t in plot_df["source_tool"].values]
        palette = {TOOL_LABELS[k]: v for k, v in TOOL_COLORS.items()}

        sns.violinplot(
            data=plot_df, x="tool_label", y=metric, order=order,
            palette=palette, inner="box", cut=0, ax=ax, linewidth=0.8
        )
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=11)
        ax.tick_params(axis="x", rotation=40, labelsize=9)
        ax.set_title(label, fontsize=12, fontweight="bold")

        # Add median line
        for i, tool in enumerate(order):
            vals = plot_df.loc[plot_df["tool_label"] == tool, metric]
            if len(vals) > 0:
                ax.hlines(vals.median(), i - 0.3, i + 0.3, color="white",
                          linewidth=2, zorder=5)

    plt.subplots_adjust(bottom=0.18, wspace=0.35)
    fig.savefig(FIG_DIR / "fig2_violin_metrics.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig2_violin_metrics.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig2_violin_metrics.png/pdf")


def fig3_correlation_heatmap(corr_matrix):
    """Figure 2: Cross-method correlation heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, ax=ax,
        linewidths=0.5, cbar_kws={"label": "Spearman ρ"}
    )
    ax.set_title("Cross-Method Correlation (Spearman)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_correlation_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig3_correlation_heatmap.png/pdf")


def fig4_composite_ranking(df):
    """Figure 3: Top 30 candidates by composite rank — horizontal bar chart."""
    candidates = df[~df["source_tool"].str.contains("control")].copy()
    top30 = candidates.nsmallest(30, "composite_rank")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [TOOL_COLORS.get(t, "#888888") for t in top30["source_tool"]]
    y_pos = range(len(top30))

    bars = ax.barh(y_pos, top30["composite_rank"].max() - top30["composite_rank"] + 1,
                   color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top30["peptide_id"].str.replace("bindcraft_bmpr1a_", "bc_"),
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Composite Score (higher = better)", fontsize=11)
    ax.set_title("Top 30 Candidates by Composite Rank", fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=TOOL_COLORS[t], label=TOOL_LABELS[t])
                       for t in TOOL_ORDER if t in candidates["source_tool"].values]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_composite_ranking.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig4_composite_ranking.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig4_composite_ranking.png/pdf")


def fig5_candidates_vs_controls(df):
    """Figure 4: Candidates vs controls comparison."""
    df = df.copy()
    df["group"] = df["source_tool"].apply(
        lambda x: "Controls" if "control" in x else "Designed"
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = [
        ("iptm", "AF3 ipTM"),
        ("rosetta_dG", "Rosetta dG (REU)"),
        ("foldx_dG", "FoldX ΔG (kcal/mol)"),
        ("contact_recap", "Contact Recapitulation"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        sns.boxplot(
            data=df, x="group", y=metric, ax=ax,
            palette={"Designed": "#2A9D8F", "Controls": "#AAAAAA"},
            width=0.5, linewidth=1.2, showfliers=False
        )
        sns.stripplot(
            data=df, x="group", y=metric, ax=ax,
            color="black", alpha=0.3, size=3, jitter=True
        )
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=15)

        # Add p-value annotation inside plot area
        designed = df.loc[df["group"] == "Designed", metric].dropna()
        controls = df.loc[df["group"] == "Controls", metric].dropna()
        _, p = stats.mannwhitneyu(designed, controls, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(0.5, 0.97, f"p={p:.3f} {sig}",
                ha="center", fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.subplots_adjust(wspace=0.35)
    fig.savefig(FIG_DIR / "fig5_candidates_vs_controls.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig5_candidates_vs_controls.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig5_candidates_vs_controls.png/pdf")


def fig7_physicochemical(df):
    """Figure 5: Physicochemical properties of filtered vs removed candidates."""
    candidates = df[~df["source_tool"].str.contains("control")].copy()

    # Filtered = passed all filters
    top = pd.read_csv(DATA_DIR / "results" / "top_candidates.csv")
    top_ids = set(top["peptide_id"])
    candidates["passed_filters"] = candidates["peptide_id"].isin(top_ids)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    props = [
        ("gravy", "GRAVY Score", 0.5),
        ("instability_index", "Instability Index", 40),
        ("net_charge_7_4", "Net Charge (pH 7.4)", None),
    ]

    for ax, (prop, label, threshold) in zip(axes, props):
        passed = candidates[candidates["passed_filters"]][prop].dropna()
        failed = candidates[~candidates["passed_filters"]][prop].dropna()
        ax.hist([passed, failed], bins=20, color=["#2A9D8F", "#E63946"],
                alpha=0.7, label=["Passed", "Filtered"], stacked=False)
        if threshold is not None:
            ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.text(threshold + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                    ax.get_ylim()[1] * 0.85, f"cutoff={threshold}",
                    fontsize=9, va="top")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")

    plt.suptitle("Physicochemical Filtering", fontsize=13, fontweight="bold")
    plt.subplots_adjust(wspace=0.3, top=0.90)
    fig.savefig(FIG_DIR / "fig7_physicochemical.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig7_physicochemical.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig7_physicochemical.png/pdf")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} peptides\n")

    # Statistical tests
    kw_results = kruskal_wallis_tests(df)
    candidates_vs_controls(df)
    corr_matrix, corr_cols, corr_labels = correlation_analysis(df)

    # Figures
    print(f"\n{'=' * 70}")
    print("GENERATING FIGURES")
    print("=" * 70)
    fig1_violin_plots(df)
    fig3_correlation_heatmap(corr_matrix)
    fig4_composite_ranking(df)
    fig5_candidates_vs_controls(df)
    fig7_physicochemical(df)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
