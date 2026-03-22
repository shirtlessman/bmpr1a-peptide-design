"""Parse AlphaFold 3 Server results from batch download zips.

Extracts all scoring metrics (ipTM, PAE, pLDDT, contact_probs, ranking_score)
from AF3 output across multiple zip files. Each zip may contain multiple jobs.
AF3 produces 5 models per job — we extract the best-ranked model's metrics
plus averages across all 5.

Output: data/af3_results/round1_scores.csv
"""
import json
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Project paths
PROJECT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_DIR.parent / "alphafold results"
OUTPUT_DIR = PROJECT_DIR / "data" / "af3_results"

BMPR1A_CHAIN_LEN = 86  # residues in BMPR1A ECD (chain A)


def parse_summary_confidences(data: dict) -> dict:
    """Parse summary_confidences JSON into flat metrics."""
    metrics = {}
    metrics["iptm"] = data.get("iptm")
    metrics["ptm"] = data.get("ptm")
    metrics["ranking_score"] = data.get("ranking_score")
    metrics["fraction_disordered"] = data.get("fraction_disordered")
    metrics["has_clash"] = data.get("has_clash")
    metrics["num_recycles"] = data.get("num_recycles")

    # Chain-pair ipTM (A-B interaction)
    cp_iptm = data.get("chain_pair_iptm", [])
    if len(cp_iptm) >= 2 and len(cp_iptm[0]) >= 2:
        metrics["chain_pair_iptm_AB"] = cp_iptm[0][1]  # A→B
        metrics["chain_pair_iptm_BA"] = cp_iptm[1][0]  # B→A

    # Chain-pair PAE min
    cp_pae = data.get("chain_pair_pae_min", [])
    if len(cp_pae) >= 2 and len(cp_pae[0]) >= 2:
        metrics["chain_pair_pae_min_AB"] = cp_pae[0][1]
        metrics["chain_pair_pae_min_BA"] = cp_pae[1][0]

    # Chain-level metrics
    chain_iptm = data.get("chain_iptm", [])
    if len(chain_iptm) >= 2:
        metrics["chain_iptm_A"] = chain_iptm[0]
        metrics["chain_iptm_B"] = chain_iptm[1]

    chain_ptm = data.get("chain_ptm", [])
    if len(chain_ptm) >= 2:
        metrics["chain_ptm_A"] = chain_ptm[0]
        metrics["chain_ptm_B"] = chain_ptm[1]

    return metrics


def parse_full_data(data: dict, bmpr1a_len: int = BMPR1A_CHAIN_LEN) -> dict:
    """Parse full_data JSON for pLDDT, PAE matrix, and contact probs."""
    metrics = {}

    # Token-level chain info
    chain_ids = data.get("token_chain_ids", [])
    n_tokens = len(chain_ids)
    n_chain_a = sum(1 for c in chain_ids if c == "A")
    n_chain_b = sum(1 for c in chain_ids if c == "B")
    metrics["n_residues_A"] = n_chain_a
    metrics["n_residues_B"] = n_chain_b

    # Atom-level pLDDT
    atom_chain_ids = data.get("atom_chain_ids", [])
    atom_plddts = data.get("atom_plddts", [])
    if atom_plddts and atom_chain_ids:
        a_plddts = [p for c, p in zip(atom_chain_ids, atom_plddts) if c == "A"]
        b_plddts = [p for c, p in zip(atom_chain_ids, atom_plddts) if c == "B"]
        metrics["plddt_mean_all"] = float(np.mean(atom_plddts))
        metrics["plddt_mean_A"] = float(np.mean(a_plddts)) if a_plddts else None
        metrics["plddt_mean_B"] = float(np.mean(b_plddts)) if b_plddts else None

    # PAE matrix — cross-chain
    pae = data.get("pae", [])
    if pae and n_chain_a > 0 and n_chain_b > 0:
        pae_matrix = np.array(pae)
        # Cross-chain PAE: A→B and B→A
        cross_AB = pae_matrix[:n_chain_a, n_chain_a:]
        cross_BA = pae_matrix[n_chain_a:, :n_chain_a]
        metrics["pae_cross_AB_min"] = float(np.min(cross_AB))
        metrics["pae_cross_AB_mean"] = float(np.mean(cross_AB))
        metrics["pae_cross_BA_min"] = float(np.min(cross_BA))
        metrics["pae_cross_BA_mean"] = float(np.mean(cross_BA))
        # Interface PAE: mean of both cross-chain blocks
        cross_all = np.concatenate([cross_AB.flatten(), cross_BA.flatten()])
        metrics["pae_interface_mean"] = float(np.mean(cross_all))

    # Contact probabilities
    contact_probs = data.get("contact_probs", [])
    if contact_probs and n_chain_a > 0:
        cp_matrix = np.array(contact_probs)
        if cp_matrix.ndim == 2 and cp_matrix.shape[0] >= n_chain_a:
            cross_contacts = cp_matrix[:n_chain_a, n_chain_a:]
            metrics["contact_prob_max"] = float(np.max(cross_contacts))
            metrics["contact_prob_mean"] = float(np.mean(cross_contacts))
            # Number of residue pairs with contact prob > 0.5
            metrics["n_confident_contacts"] = int(np.sum(cross_contacts > 0.5))
            # Number with > 0.8
            metrics["n_high_contacts"] = int(np.sum(cross_contacts > 0.8))

    return metrics


def normalize_peptide_id(raw_id: str) -> str:
    """Normalize AF3 subdirectory name back to our canonical peptide ID.

    AF3 Server lowercases job names, so we restore original casing.
    Also strips '_2' suffix from duplicate submissions.
    """
    import re
    # Strip trailing _2, _3 etc. (duplicate AF3 submissions)
    pid = re.sub(r'_(\d)$', '', raw_id) if re.search(r'(?<=\d)_\d$', raw_id) else raw_id

    # Restore casing: pepmlm_l15 -> pepmlm_L15, pepmlm_l20 -> pepmlm_L20, etc.
    pid = re.sub(r'pepmlm_l(\d+)', lambda m: f'pepmlm_L{m.group(1)}', pid)
    # scrambled_pepmlm -> scrambled_PepMLM
    pid = pid.replace('scrambled_pepmlm', 'scrambled_PepMLM')
    # scrambled_rfdiffusion -> scrambled_RFdiffusion
    pid = pid.replace('scrambled_rfdiffusion', 'scrambled_RFdiffusion')
    # scrambled_rfpeptides -> scrambled_RFpeptides
    pid = pid.replace('scrambled_rfpeptides', 'scrambled_RFpeptides')

    return pid


def parse_all_zips(results_dir: Path) -> pd.DataFrame:
    """Parse all AF3 result zips and return combined DataFrame."""
    all_results = {}  # peptide_id -> best model metrics

    zip_files = sorted(results_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files in {results_dir}")

    for zf_path in zip_files:
        print(f"\nProcessing: {zf_path.name}")
        try:
            with zipfile.ZipFile(zf_path, "r") as zf:
                # Find all summary_confidences files
                summary_files = [
                    n for n in zf.namelist()
                    if "summary_confidences" in n and n.endswith(".json")
                ]

                # Group by job (subdirectory)
                jobs = defaultdict(list)
                for sf in summary_files:
                    parts = sf.split("/")
                    if len(parts) >= 2:
                        job_dir = parts[0]
                        jobs[job_dir].append(sf)

                print(f"  Found {len(jobs)} jobs")

                for job_dir, summaries in jobs.items():
                    peptide_id = normalize_peptide_id(job_dir)

                    # Skip if already parsed from another zip
                    if peptide_id in all_results:
                        continue

                    # Parse all 5 models, pick best by ranking_score
                    best_model = None
                    best_ranking = -1
                    all_model_metrics = []

                    for sf in sorted(summaries):
                        model_idx = sf.split("_")[-1].replace(".json", "")
                        try:
                            summary_data = json.loads(zf.read(sf))
                            summary_metrics = parse_summary_confidences(summary_data)

                            # Find matching full_data file
                            full_data_name = sf.replace(
                                "summary_confidences", "full_data"
                            )
                            if full_data_name in zf.namelist():
                                full_data = json.loads(zf.read(full_data_name))
                                full_metrics = parse_full_data(full_data)
                                summary_metrics.update(full_metrics)

                            all_model_metrics.append(summary_metrics)

                            rs = summary_metrics.get("ranking_score", -1) or -1
                            if rs > best_ranking:
                                best_ranking = rs
                                best_model = summary_metrics.copy()
                                best_model["best_model_idx"] = int(model_idx)

                        except Exception as e:
                            print(f"    Error parsing {sf}: {e}")

                    if best_model and all_model_metrics:
                        # Add averages across all models
                        for key in [
                            "iptm", "ptm", "ranking_score", "plddt_mean_all",
                            "plddt_mean_B", "pae_interface_mean"
                        ]:
                            vals = [
                                m[key] for m in all_model_metrics
                                if key in m and m[key] is not None
                            ]
                            if vals:
                                best_model[f"avg_{key}"] = float(np.mean(vals))

                        best_model["peptide_id"] = peptide_id
                        best_model["n_models"] = len(all_model_metrics)
                        best_model["source_zip"] = zf_path.name
                        all_results[peptide_id] = best_model

        except zipfile.BadZipFile:
            print(f"  ERROR: Bad zip file, skipping")

    df = pd.DataFrame(all_results.values())
    if not df.empty:
        # Drop duplicate peptide IDs (keep best ranking_score)
        df = df.sort_values("ranking_score", ascending=False)
        df = df.drop_duplicates(subset="peptide_id", keep="first")
        df = df.reset_index(drop=True)

    return df


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AF3 Result Parser — BMPR1A Peptide Design Study")
    print("=" * 60)

    df = parse_all_zips(RESULTS_DIR)

    if df.empty:
        print("\nNo results parsed!")
        return

    # Add source classification
    df["source_tool"] = df["peptide_id"].apply(classify_peptide)

    # Save full results
    out_path = OUTPUT_DIR / "round1_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved {len(df)} results to {out_path}")

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("SUMMARY BY SOURCE TOOL")
    print("=" * 60)
    summary = df.groupby("source_tool").agg(
        count=("peptide_id", "count"),
        iptm_mean=("iptm", "mean"),
        iptm_max=("iptm", "max"),
        ranking_score_mean=("ranking_score", "mean"),
        ranking_score_max=("ranking_score", "max"),
        pae_interface_mean=("pae_interface_mean", lambda x: x.mean()),
        plddt_peptide_mean=("plddt_mean_B", lambda x: x.mean()),
    ).round(3)
    print(summary.to_string())

    # Top 10 candidates
    print(f"\n{'=' * 60}")
    print("TOP 10 CANDIDATES BY RANKING SCORE")
    print("=" * 60)
    top10 = df.head(10)[
        ["peptide_id", "source_tool", "ranking_score", "iptm", "ptm",
         "pae_interface_mean", "plddt_mean_B", "n_confident_contacts"]
    ]
    print(top10.to_string(index=False))

    # Controls comparison
    print(f"\n{'=' * 60}")
    print("CANDIDATES vs CONTROLS")
    print("=" * 60)
    candidates = df[~df["source_tool"].str.contains("control")]
    controls = df[df["source_tool"].str.contains("control")]
    print(f"Candidates (n={len(candidates)}): "
          f"ipTM={candidates['iptm'].mean():.3f} +/- {candidates['iptm'].std():.3f}")
    print(f"Controls   (n={len(controls)}): "
          f"ipTM={controls['iptm'].mean():.3f} +/- {controls['iptm'].std():.3f}")


if __name__ == "__main__":
    main()
