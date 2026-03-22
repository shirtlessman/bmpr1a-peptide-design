"""Render static PNG images of top peptide-BMPR1A complexes for manuscript.

Uses Playwright to screenshot the py3Dmol HTML viewers in different modes.
Produces a composite panel figure (Fig 6) for the manuscript.

Output:
  - figures/structures/static/{peptide_id}_{mode}.png (individual)
  - figures/fig6_structure_panel.png (composite 2x3 panel)
"""
import subprocess
import time
import signal
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
STRUCT_DIR = FIG_DIR / "structures"
STATIC_DIR = STRUCT_DIR / "static"

# Top candidate per tool for the panel figure
PANEL_CANDIDATES = [
    "pepmlm_L15_0026",
    "rfdiff_d2_n5",
    "bindcraft_bmpr1a_l26_s635301",
]

MODES = ["cartoon", "surface"]


def render_screenshots():
    """Start local server, render each candidate in each mode."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Start HTTP server
    server = subprocess.Popen(
        ["python3", "-m", "http.server", "8766"],
        cwd=str(STRUCT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000})

            for pid in PANEL_CANDIDATES:
                url = f"http://localhost:8766/{pid}.html"
                page.goto(url)
                page.wait_for_timeout(4000)  # Wait for 3Dmol to render

                for mode in MODES:
                    # Click the mode button
                    if mode == "cartoon":
                        page.click("button:has-text('Cartoon')")
                    elif mode == "surface":
                        page.click("button:has-text('Surface')")

                    page.wait_for_timeout(4000)  # Wait for render

                    # Ensure consistent zoom to full complex
                    page.evaluate("viewer.zoomTo(); viewer.render();")
                    page.wait_for_timeout(1000)

                    # Screenshot just the viewer div
                    viewer = page.locator("#viewer")
                    out_path = STATIC_DIR / f"{pid}_{mode}.png"
                    viewer.screenshot(path=str(out_path))
                    print(f"  {pid} [{mode}] -> {out_path.name}")

            browser.close()
    finally:
        server.send_signal(signal.SIGTERM)
        server.wait()


def create_panel_figure():
    """Create composite 2x3 panel figure for manuscript."""
    # Load master scores for labels
    master = pd.read_csv(DATA_DIR / "results" / "master_scores.csv")
    scores = {r["peptide_id"]: r for _, r in master.iterrows()}

    tool_labels = {
        "pepmlm_L15_0026": "PepMLM",
        "rfdiff_d2_n5": "RFdiffusion",
        "bindcraft_bmpr1a_l26_s635301": "BindCraft",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for col, pid in enumerate(PANEL_CANDIDATES):
        s = scores.get(pid, {})
        tool = tool_labels.get(pid, "")
        iptm = s.get("iptm", 0)
        foldx = s.get("foldx_dG", 0)

        for row, mode in enumerate(MODES):
            ax = axes[row, col]
            img_path = STATIC_DIR / f"{pid}_{mode}.png"

            if img_path.exists():
                img = mpimg.imread(str(img_path))
                ax.imshow(img)

            ax.axis("off")

            if row == 0:
                # Title on top row
                short_id = pid.replace("bindcraft_bmpr1a_", "bc_")
                ax.set_title(
                    f"{tool}\n{short_id}\nipTM={iptm:.2f}  FoldX={foldx:.1f} kcal/mol",
                    fontsize=11, fontweight="bold", pad=8
                )

        # Row labels
        if col == 0:
            axes[0, 0].set_ylabel("Cartoon", fontsize=13, fontweight="bold",
                                   rotation=0, labelpad=60, va="center")
            axes[1, 0].set_ylabel("Surface", fontsize=13, fontweight="bold",
                                   rotation=0, labelpad=60, va="center")

    # Panel labels
    for i, label in enumerate(["A", "B", "C", "D", "E", "F"]):
        row, col = divmod(i, 3)
        axes[row, col].text(-0.05, 1.05, label, transform=axes[row, col].transAxes,
                            fontsize=16, fontweight="bold", va="top")

    plt.suptitle("Top Peptide–BMPR1A Complexes by Design Tool",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.subplots_adjust(wspace=0.05, hspace=0.15, top=0.90)

    out_path = FIG_DIR / "fig6_structure_panel.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / "fig6_structure_panel.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Panel figure saved: {out_path}")


def main():
    print("Rendering static structure images...")
    render_screenshots()
    print("\nCreating panel figure...")
    create_panel_figure()
    print("\nDone!")


if __name__ == "__main__":
    main()
