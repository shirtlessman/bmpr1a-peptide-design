"""Compare predicted peptide-BMPR1A contacts against the 1REW gold standard.

Reads AF3 CIF structures directly from batch zip files, extracts
peptide-BMPR1A contacts at 5Å cutoff, and scores against gold standard
interface residues from 1REW.

Output: data/results/contact_scores.csv
"""
import json
import zipfile
import tempfile
import re
import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import MMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Polypeptide import is_aa

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR.parent / "alphafold results"

DISTANCE_CUTOFF = 5.0  # Angstroms


def normalize_peptide_id(raw_id: str) -> str:
    """Normalize AF3 subdirectory name back to canonical peptide ID."""
    pid = re.sub(r'_(\d)$', '', raw_id) if re.search(r'(?<=\d)_\d$', raw_id) else raw_id
    pid = re.sub(r'pepmlm_l(\d+)', lambda m: f'pepmlm_L{m.group(1)}', pid)
    pid = pid.replace('scrambled_pepmlm', 'scrambled_PepMLM')
    pid = pid.replace('scrambled_rfdiffusion', 'scrambled_RFdiffusion')
    pid = pid.replace('scrambled_rfpeptides', 'scrambled_RFpeptides')
    return pid


def load_gold_standard() -> set:
    """Load BMPR1A interface residue numbers from Stage 1."""
    interface_path = DATA_DIR / "structures" / "bmpr1a_interface_residues.json"
    data = json.loads(interface_path.read_text())
    return set(r[1] for r in data["bmpr1a_interface_residues"])


def get_predicted_contacts_from_cif(cif_content: bytes,
                                     distance_cutoff: float = DISTANCE_CUTOFF) -> set:
    """Extract BMPR1A residues contacted by the peptide in an AF3 CIF structure."""
    parser = MMCIFParser(QUIET=True)

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(cif_content)
        tmp_path = tmp.name

    try:
        structure = parser.get_structure("pred", tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    model = structure[0]
    chains = list(model.get_chains())
    if len(chains) < 2:
        return set()

    # AF3 convention: first chain = BMPR1A (A), second chain = peptide (B)
    chain_a = chains[0]
    chain_b = chains[1]

    # Build neighbor search on BMPR1A atoms
    bmpr1a_atoms = list(chain_a.get_atoms())
    if not bmpr1a_atoms:
        return set()
    ns = NeighborSearch(bmpr1a_atoms)

    # Find BMPR1A residues within cutoff of any peptide atom
    contacted = set()
    for atom in chain_b.get_atoms():
        nearby = ns.search(atom.coord, distance_cutoff)
        for nb in nearby:
            res = nb.get_parent()
            if is_aa(res, standard=True):
                contacted.add(res.id[1])

    return contacted


def contact_recapitulation_score(predicted: set, gold_standard: set) -> dict:
    """Calculate fraction of gold-standard contacts recapitulated."""
    overlap = predicted & gold_standard
    novel = predicted - gold_standard
    return {
        "contacts_recapitulated": len(overlap),
        "total_gold_standard": len(gold_standard),
        "recapitulation_fraction": len(overlap) / len(gold_standard) if gold_standard else 0,
        "novel_contacts": len(novel),
        "total_predicted_contacts": len(predicted),
        "recapitulated_residues": sorted(overlap) if overlap else [],
    }


def process_all_zips(results_dir: Path, gold_standard: set,
                     scores_df: pd.DataFrame) -> pd.DataFrame:
    """Extract best CIF from each job in all zips and compute contacts."""
    contact_results = []
    seen = set()

    # Map peptide_id -> best_model_idx from AF3 scores
    best_models = {}
    for _, row in scores_df.iterrows():
        pid = row["peptide_id"]
        idx = int(row.get("best_model_idx", 0))
        best_models[pid] = idx

    zip_files = sorted(results_dir.glob("*.zip"))
    print(f"Processing {len(zip_files)} zip files for contact analysis...")

    for zf_path in zip_files:
        try:
            with zipfile.ZipFile(zf_path, "r") as zf:
                # Find all model CIF files
                cif_files = [n for n in zf.namelist() if n.endswith(".cif")
                             and "model_" in n and "template" not in n]

                # Group by job directory
                jobs = {}
                for cf in cif_files:
                    parts = cf.split("/")
                    if len(parts) >= 2:
                        job_dir = parts[0]
                        if job_dir not in jobs:
                            jobs[job_dir] = []
                        jobs[job_dir].append(cf)

                for job_dir, cifs in jobs.items():
                    peptide_id = normalize_peptide_id(job_dir)
                    if peptide_id in seen:
                        continue
                    seen.add(peptide_id)

                    # Find the best model CIF
                    best_idx = best_models.get(peptide_id, 0)
                    best_cif = None
                    for cf in cifs:
                        if f"model_{best_idx}.cif" in cf:
                            best_cif = cf
                            break
                    if not best_cif and cifs:
                        best_cif = cifs[0]

                    if not best_cif:
                        continue

                    try:
                        cif_content = zf.read(best_cif)
                        predicted = get_predicted_contacts_from_cif(cif_content)
                        scores = contact_recapitulation_score(predicted, gold_standard)
                        scores["peptide_id"] = peptide_id
                        # Convert list to string for CSV
                        scores["recapitulated_residues"] = str(scores["recapitulated_residues"])
                        contact_results.append(scores)
                    except Exception as e:
                        print(f"  Error processing {peptide_id}: {e}")

        except zipfile.BadZipFile:
            print(f"  ERROR: Bad zip file {zf_path.name}, skipping")

    return pd.DataFrame(contact_results)


def classify_peptide(peptide_id: str) -> str:
    """Classify peptide by source tool."""
    pid = peptide_id.lower()
    if "scrambled" in pid:
        return "scrambled_control"
    elif "random" in pid:
        return "random_control"
    elif "pepmlm" in pid:
        return "PepMLM"
    elif "rfdiff" in pid:
        return "RFdiffusion"
    elif "rfpep" in pid:
        return "RFpeptides"
    elif "bindcraft" in pid:
        return "BindCraft"
    else:
        return "unknown"


def main():
    print("=" * 60)
    print("Contact Recapitulation Analysis — BMPR1A vs 1REW Gold Standard")
    print("=" * 60)

    gold_standard = load_gold_standard()
    print(f"Gold standard: {len(gold_standard)} BMPR1A interface residues from 1REW")
    print(f"Residues: {sorted(gold_standard)}")

    # Load AF3 scores for best model selection
    scores_df = pd.read_csv(DATA_DIR / "af3_results" / "round1_scores.csv")

    # Process all zips
    df = process_all_zips(RESULTS_DIR, gold_standard, scores_df)

    if df.empty:
        print("\nNo contact results generated!")
        return

    # Add source classification
    df["source_tool"] = df["peptide_id"].apply(classify_peptide)

    # Save
    out_path = DATA_DIR / "results" / "contact_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} contact scores to {out_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("CONTACT RECAPITULATION BY SOURCE TOOL")
    print("=" * 60)
    summary = df.groupby("source_tool").agg(
        count=("peptide_id", "count"),
        recap_mean=("recapitulation_fraction", "mean"),
        recap_max=("recapitulation_fraction", "max"),
        contacts_mean=("total_predicted_contacts", "mean"),
    ).round(3)
    print(summary.to_string())

    # Top 10
    print(f"\n{'=' * 60}")
    print("TOP 10 BY CONTACT RECAPITULATION")
    print("=" * 60)
    top10 = df.sort_values("recapitulation_fraction", ascending=False).head(10)[
        ["peptide_id", "source_tool", "recapitulation_fraction",
         "contacts_recapitulated", "total_predicted_contacts", "novel_contacts"]
    ]
    print(top10.to_string(index=False))

    # Candidates vs controls
    print(f"\n{'=' * 60}")
    print("CANDIDATES vs CONTROLS")
    print("=" * 60)
    candidates = df[~df["source_tool"].str.contains("control")]
    controls = df[df["source_tool"].str.contains("control")]
    print(f"Candidates (n={len(candidates)}): "
          f"recap={candidates['recapitulation_fraction'].mean():.3f} +/- "
          f"{candidates['recapitulation_fraction'].std():.3f}")
    print(f"Controls   (n={len(controls)}): "
          f"recap={controls['recapitulation_fraction'].mean():.3f} +/- "
          f"{controls['recapitulation_fraction'].std():.3f}")


if __name__ == "__main__":
    main()
