"""Generate interactive 3D structure visualizations of top peptide-BMPR1A complexes.

Creates HTML files with py3Dmol for each top candidate, showing:
  - BMPR1A receptor (chain A) in light blue cartoon
  - Designed peptide (chain B) in red cartoon
  - Interface residues highlighted as sticks
  - Rotatable, zoomable in any browser

Also creates a gallery HTML page with all top candidates side by side.

Output: figures/structures/*.html
"""
import json
import zipfile
import tempfile
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures" / "structures"
RESULTS_DIR = PROJECT_DIR.parent / "alphafold results"


def extract_best_cif(peptide_id: str, best_model_idx: int) -> str:
    """Find and extract the best CIF file for a peptide from AF3 result zips."""
    # Normalize ID for matching zip contents (AF3 lowercases)
    search_id = peptide_id.lower()

    for zf_path in sorted(RESULTS_DIR.glob("*.zip")):
        try:
            with zipfile.ZipFile(zf_path, "r") as zf:
                for name in zf.namelist():
                    if (search_id in name.lower()
                            and f"model_{best_model_idx}.cif" in name
                            and "template" not in name):
                        return zf.read(name).decode("utf-8", errors="replace")
        except zipfile.BadZipFile:
            continue
    return ""


def cif_to_pdb_string(cif_content: str) -> str:
    """Convert CIF to PDB format string using BioPython."""
    from Bio.PDB import MMCIFParser, PDBIO
    import io

    with tempfile.NamedTemporaryFile(suffix=".cif", mode="w", delete=False) as tmp:
        tmp.write(cif_content)
        tmp_path = tmp.name

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("complex", tmp_path)
        io_pdb = PDBIO()
        io_pdb.set_structure(structure)
        string_io = io.StringIO()
        io_pdb.save(string_io)
        return string_io.getvalue()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def create_3dmol_html(pdb_string: str, peptide_id: str, metrics: dict) -> str:
    """Create an HTML page with py3Dmol viewer for a peptide-BMPR1A complex."""
    # Escape PDB string for JavaScript
    pdb_escaped = pdb_string.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    iptm = metrics.get("iptm", "N/A")
    rosetta = metrics.get("rosetta_dG", "N/A")
    foldx = metrics.get("foldx_dG", "N/A")
    tool = metrics.get("source_tool", "N/A")

    if isinstance(iptm, float):
        iptm = f"{iptm:.3f}"
    if isinstance(rosetta, float):
        rosetta = f"{rosetta:.1f}"
    if isinstance(foldx, float):
        foldx = f"{foldx:.1f}"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{peptide_id} — BMPR1A Complex</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ font-size: 1.4em; color: #333; margin-bottom: 5px; }}
        .metrics {{ display: flex; gap: 15px; margin: 10px 0 12px 0; flex-wrap: wrap; }}
        .metric {{ background: white; padding: 8px 16px; border-radius: 8px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric .label {{ font-size: 0.8em; color: #666; }}
        .metric .value {{ font-size: 1.2em; font-weight: bold; color: #333; }}
        .controls {{ display: flex; gap: 8px; margin: 0 0 10px 0; flex-wrap: wrap; }}
        .btn {{ padding: 7px 16px; border: 2px solid #ddd; border-radius: 6px;
                background: white; cursor: pointer; font-size: 0.9em; color: #444;
                transition: all 0.15s; }}
        .btn:hover {{ border-color: #999; }}
        .btn.active {{ border-color: #E63946; background: #FFF0F0; color: #E63946; font-weight: 600; }}
        .viewer {{ width: 100%; height: 600px; position: relative;
                   border-radius: 8px; overflow: hidden;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .legend {{ margin-top: 10px; font-size: 0.9em; color: #555; }}
        .legend span {{ display: inline-block; width: 14px; height: 14px;
                       border-radius: 3px; vertical-align: middle; margin-right: 5px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>{peptide_id}</h1>
    <p style="color:#666; margin-top:0;">Source: {tool} | AF3-predicted complex with BMPR1A ECD</p>

    <div class="metrics">
        <div class="metric">
            <div class="label">AF3 ipTM</div>
            <div class="value">{iptm}</div>
        </div>
        <div class="metric">
            <div class="label">Rosetta dG</div>
            <div class="value">{rosetta} REU</div>
        </div>
        <div class="metric">
            <div class="label">FoldX ΔG</div>
            <div class="value">{foldx} kcal/mol</div>
        </div>
    </div>

    <div class="controls">
        <button class="btn active" onclick="setView('cartoon', this)">Cartoon</button>
        <button class="btn" onclick="setView('surface', this)">Surface</button>
        <button class="btn" onclick="setView('sticks', this)">Cartoon + Sticks</button>
        <button class="btn" onclick="setView('interface', this)">Interface Focus</button>
        <button class="btn" onclick="toggleSpin()" id="spinBtn">Spin</button>
    </div>

    <div id="viewer" class="viewer"></div>

    <div class="legend" id="legend-cartoon">
        <span style="background:#7CB9E8;"></span> BMPR1A receptor (chain A)
        &nbsp;&nbsp;
        <span style="background:#E63946;"></span> Designed peptide (chain B)
    </div>
</div>

<script>
var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
var pdbData = `{pdb_escaped}`;
viewer.addModel(pdbData, "pdb");
var spinning = false;

function clearBtnActive() {{
    document.querySelectorAll('.btn').forEach(function(b) {{
        if (b.id !== 'spinBtn') b.classList.remove('active');
    }});
}}

function setView(mode, btn) {{
    clearBtnActive();
    if (btn) btn.classList.add('active');
    else {{
        // Find button by text content for programmatic calls
        document.querySelectorAll('.btn').forEach(function(b) {{
            if (b.textContent.trim().toLowerCase().replace(/ \\+ /g,'') === mode.replace(/ /g,''))
                b.classList.add('active');
        }});
    }}

    // Reset all styles and surfaces
    viewer.removeAllSurfaces();
    viewer.setStyle({{}}, {{}});

    if (mode === 'cartoon') {{
        // Clean cartoon — no sticks
        viewer.setStyle({{chain: "A"}}, {{cartoon: {{color: "#7CB9E8", opacity: 0.9}}}});
        viewer.setStyle({{chain: "B"}}, {{cartoon: {{color: "#E63946"}}}});
    }}
    else if (mode === 'surface') {{
        // Surface view — quaternary structure style
        viewer.setStyle({{chain: "A"}}, {{cartoon: {{color: "#7CB9E8", opacity: 0.4}}}});
        viewer.setStyle({{chain: "B"}}, {{cartoon: {{color: "#E63946", opacity: 0.4}}}});
        viewer.addSurface($3Dmol.SurfaceType.VDW, {{
            opacity: 0.75, color: "#7CB9E8"
        }}, {{chain: "A"}});
        viewer.addSurface($3Dmol.SurfaceType.VDW, {{
            opacity: 0.85, color: "#E63946"
        }}, {{chain: "B"}});
    }}
    else if (mode === 'sticks') {{
        // Cartoon + side chain sticks (interface only)
        viewer.setStyle({{chain: "A"}}, {{cartoon: {{color: "#7CB9E8", opacity: 0.85}}}});
        viewer.setStyle({{chain: "B"}}, {{
            cartoon: {{color: "#E63946"}},
            stick: {{color: "#E63946", radius: 0.12}}
        }});
        // Show sticks only for interface residues on chain A
        viewer.setStyle(
            {{chain: "A", byres: true, within: {{distance: 4, sel: {{chain: "B"}}}}}},
            {{cartoon: {{color: "#2A9D8F", opacity: 1.0}},
              stick: {{color: "#2A9D8F", radius: 0.12}}}}
        );
    }}
    else if (mode === 'interface') {{
        // Focus on binding interface — hide distant parts
        viewer.setStyle({{chain: "A"}}, {{cartoon: {{color: "#7CB9E8", opacity: 0.3}}}});
        viewer.setStyle({{chain: "B"}}, {{
            cartoon: {{color: "#E63946"}},
            stick: {{color: "#E63946", radius: 0.14}}
        }});
        // Interface residues on A with sticks
        viewer.setStyle(
            {{chain: "A", byres: true, within: {{distance: 4.5, sel: {{chain: "B"}}}}}},
            {{cartoon: {{color: "#2A9D8F", opacity: 1.0}},
              stick: {{color: "#2A9D8F", radius: 0.14}}}}
        );
        // Zoom to interface
        viewer.zoomTo({{chain: "B"}});
    }}

    viewer.render();
}}

function toggleSpin() {{
    spinning = !spinning;
    viewer.spin(spinning);
    var btn = document.getElementById('spinBtn');
    btn.classList.toggle('active', spinning);
}}

// Default: clean cartoon
setView('cartoon');
viewer.zoomTo();
viewer.render();
</script>
</body>
</html>"""
    return html


def create_gallery_html(candidates: list) -> str:
    """Create a gallery page linking all structure views."""
    cards = ""
    for c in candidates:
        iptm = f"{c['iptm']:.3f}" if isinstance(c['iptm'], float) else "N/A"
        foldx = f"{c['foldx_dG']:.1f}" if isinstance(c['foldx_dG'], float) else "N/A"
        rank = c.get('rank', '?')
        cards += f"""
        <a href="{c['peptide_id']}.html" class="card">
            <div class="rank">#{rank}</div>
            <div class="name">{c['peptide_id'].replace('bindcraft_bmpr1a_', 'bc_')}</div>
            <div class="tool">{c['source_tool']}</div>
            <div class="stats">ipTM: {iptm} | FoldX: {foldx}</div>
        </a>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Top Peptide-BMPR1A Complexes</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               margin: 0; padding: 30px; background: #f5f5f5; }}
        h1 {{ color: #333; text-align: center; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                 gap: 15px; max-width: 1200px; margin: 20px auto; }}
        .card {{ background: white; padding: 20px; border-radius: 10px;
                 box-shadow: 0 2px 6px rgba(0,0,0,0.1); text-decoration: none;
                 color: #333; transition: transform 0.15s; }}
        .card:hover {{ transform: translateY(-3px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
        .rank {{ font-size: 1.5em; font-weight: bold; color: #E63946; }}
        .name {{ font-size: 1.1em; font-weight: 600; margin: 5px 0; }}
        .tool {{ font-size: 0.9em; color: #666; }}
        .stats {{ font-size: 0.85em; color: #888; margin-top: 8px; }}
    </style>
</head>
<body>
    <h1>Top Peptide–BMPR1A Complexes</h1>
    <p style="text-align:center; color:#666;">Click any card to view the interactive 3D structure</p>
    <div class="grid">{cards}</div>
</body>
</html>"""


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load master scores
    master = pd.read_csv(DATA_DIR / "results" / "master_scores.csv")

    # Get top 10 candidates by composite rank (no controls)
    candidates = master[~master["source_tool"].str.contains("control")].copy()
    top10 = candidates.nsmallest(10, "composite_rank")

    # Load best model indices
    af3 = pd.read_csv(DATA_DIR / "af3_results" / "round1_scores.csv")
    best_models = dict(zip(af3["peptide_id"], af3["best_model_idx"].astype(int)))

    print(f"Generating 3D structure views for top {len(top10)} candidates...")

    gallery_data = []
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        pid = row["peptide_id"]
        best_idx = best_models.get(pid, 0)

        print(f"  {rank}. {pid} (model {best_idx})...", end=" ")

        cif_content = extract_best_cif(pid, best_idx)
        if not cif_content:
            print("CIF not found, skipping")
            continue

        try:
            pdb_string = cif_to_pdb_string(cif_content)
        except Exception as e:
            print(f"conversion error: {e}")
            continue

        metrics = {
            "iptm": row.get("iptm"),
            "rosetta_dG": row.get("rosetta_dG"),
            "foldx_dG": row.get("foldx_dG"),
            "source_tool": row.get("source_tool"),
        }

        html = create_3dmol_html(pdb_string, pid, metrics)
        out_path = FIG_DIR / f"{pid}.html"
        out_path.write_text(html)
        print(f"saved")

        gallery_data.append({
            "peptide_id": pid,
            "rank": rank,
            **metrics,
        })

    # Create gallery page
    gallery_html = create_gallery_html(gallery_data)
    (FIG_DIR / "index.html").write_text(gallery_html)
    print(f"\nGallery page: {FIG_DIR / 'index.html'}")
    print(f"Open in browser: file://{FIG_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
