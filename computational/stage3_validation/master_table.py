"""Task 11: Build master scoring table with physicochemical filtering.

Merges all validation scores (AF3, contact recapitulation, PyRosetta, FoldX)
into a single ranked table. Adds physicochemical properties and applies
filters to produce a final shortlist of top candidates.

Output:
  - data/results/master_scores.csv (all 290 peptides, full feature set)
  - data/results/top_candidates.csv (filtered + ranked shortlist)
"""
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"


# ── Physicochemical property calculations ────────────────────────────

# Kyte-Doolittle hydrophobicity scale
KD_HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}

# Amino acid molecular weights (Da)
AA_MW = {
    'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
    'E': 147.13, 'Q': 146.15, 'G': 75.03, 'H': 155.16, 'I': 131.17,
    'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
    'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
}

# pKa values for isoelectric point calculation
PK_NTERM = 9.69
PK_CTERM = 2.34
PK_SIDE = {'D': 3.65, 'E': 4.25, 'C': 8.18, 'Y': 10.07,
            'H': 6.00, 'K': 10.53, 'R': 12.48}


def compute_gravy(seq: str) -> float:
    """Grand average of hydropathicity (GRAVY) score."""
    vals = [KD_HYDROPHOBICITY.get(aa, 0) for aa in seq]
    return np.mean(vals) if vals else 0.0


def compute_net_charge(seq: str, ph: float = 7.4) -> float:
    """Net charge at given pH using Henderson-Hasselbalch."""
    charge = 0.0
    # N-terminus (positive)
    charge += 1.0 / (1.0 + 10 ** (ph - PK_NTERM))
    # C-terminus (negative)
    charge -= 1.0 / (1.0 + 10 ** (PK_CTERM - ph))
    for aa in seq:
        if aa in ('D', 'E', 'C', 'Y'):
            charge -= 1.0 / (1.0 + 10 ** (PK_SIDE[aa] - ph))
        elif aa in ('H', 'K', 'R'):
            charge += 1.0 / (1.0 + 10 ** (ph - PK_SIDE[aa]))
    return charge


def compute_molecular_weight(seq: str) -> float:
    """Molecular weight in Daltons."""
    if not seq:
        return 0.0
    mw = sum(AA_MW.get(aa, 0) for aa in seq)
    mw -= 18.015 * (len(seq) - 1)  # subtract water for peptide bonds
    return mw


def compute_instability_index(seq: str) -> float:
    """Guruprasad instability index. Values > 40 = unstable."""
    # DIWV weight matrix (Guruprasad et al., 1990)
    # Simplified: use BioPython-equivalent lookup
    DIWV = {
        'A': {'A': 1.0, 'C': 44.94, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': 1.0, 'H': -7.49, 'I': 1.0, 'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'P': 20.26, 'Q': 1.0, 'R': 1.0, 'S': 1.0, 'T': 1.0, 'V': 1.0, 'W': 1.0, 'Y': 1.0},
        'C': {'A': 1.0, 'C': 1.0, 'D': 20.26, 'E': 1.0, 'F': 1.0, 'G': 1.0, 'H': 33.60, 'I': 1.0, 'K': 1.0, 'L': 20.26, 'M': 33.60, 'N': 1.0, 'P': 20.26, 'Q': -6.54, 'R': 1.0, 'S': 1.0, 'T': 33.60, 'V': -6.54, 'W': 24.68, 'Y': 1.0},
        'D': {'A': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': -6.54, 'G': 1.0, 'H': 1.0, 'I': 1.0, 'K': -7.49, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'P': 1.0, 'Q': 1.0, 'R': -6.54, 'S': 20.26, 'T': -14.03, 'V': 1.0, 'W': 1.0, 'Y': 1.0},
        'E': {'A': 1.0, 'C': 44.94, 'D': 20.26, 'E': 33.60, 'F': 1.0, 'G': 1.0, 'H': -6.54, 'I': 20.26, 'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'P': 20.26, 'Q': 20.26, 'R': 1.0, 'S': 20.26, 'T': 1.0, 'V': 1.0, 'W': -14.03, 'Y': 1.0},
        'F': {'A': 1.0, 'C': 1.0, 'D': 13.34, 'E': 1.0, 'F': 1.0, 'G': 1.0, 'H': 1.0, 'I': 1.0, 'K': -14.03, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'P': 20.26, 'Q': 1.0, 'R': 1.0, 'S': 1.0, 'T': 1.0, 'V': 1.0, 'W': 1.0, 'Y': 33.60},
        'G': {'A': -7.49, 'C': 1.0, 'D': 1.0, 'E': -6.54, 'F': 1.0, 'G': 13.34, 'H': 1.0, 'I': -7.49, 'K': -7.49, 'L': 1.0, 'M': 1.0, 'N': -7.49, 'P': 1.0, 'Q': 1.0, 'R': 1.0, 'S': 1.0, 'T': -7.49, 'V': 1.0, 'W': 13.34, 'Y': -7.49},
        'H': {'A': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': -9.37, 'G': -9.37, 'H': 1.0, 'I': 44.94, 'K': 24.68, 'L': 1.0, 'M': 1.0, 'N': 24.68, 'P': -1.88, 'Q': 1.0, 'R': 1.0, 'S': 1.0, 'T': -6.54, 'V': 1.0, 'W': -1.88, 'Y': 44.94},
        'I': {'A': 1.0, 'C': 1.0, 'D': 1.0, 'E': 44.94, 'F': 1.0, 'G': 1.0, 'H': 13.34, 'I': 1.0, 'K': -7.49, 'L': 20.26, 'M': 1.0, 'N': 1.0, 'P': -1.88, 'Q': 1.0, 'R': 1.0, 'S': 1.0, 'T': 1.0, 'V': -7.49, 'W': 1.0, 'Y': 1.0},
        'K': {'A': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': -7.49, 'H': 1.0, 'I': -7.49, 'K': 1.0, 'L': -7.49, 'M': 33.60, 'N': 1.0, 'P': -6.54, 'Q': 24.64, 'R': 33.60, 'S': 1.0, 'T': 1.0, 'V': -7.49, 'W': 1.0, 'Y': 1.0},
        'L': {'A': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': 20.26, 'H': 1.0, 'I': 1.0, 'K': -7.49, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'P': 20.26, 'Q': 33.60, 'R': 20.26, 'S': 1.0, 'T': 1.0, 'V': 1.0, 'W': 24.68, 'Y': 1.0},
        'M': {'A': 13.34, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': 1.0, 'H': 58.28, 'I': 1.0, 'K': 1.0, 'L': 1.0, 'M': -1.88, 'N': 1.0, 'P': 44.94, 'Q': -6.54, 'R': -6.54, 'S': 44.94, 'T': -1.88, 'V': 1.0, 'W': 1.0, 'Y': 24.68},
        'N': {'A': 1.0, 'C': -1.88, 'D': 1.0, 'E': 1.0, 'F': -14.03, 'G': -7.49, 'H': 1.0, 'I': 44.94, 'K': 24.68, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'P': -1.88, 'Q': -6.54, 'R': 1.0, 'S': 1.0, 'T': -7.49, 'V': 1.0, 'W': -9.37, 'Y': 1.0},
        'P': {'A': 20.26, 'C': -6.54, 'D': -6.54, 'E': 18.38, 'F': 20.26, 'G': 1.0, 'H': 1.0, 'I': 1.0, 'K': 1.0, 'L': 1.0, 'M': -6.54, 'N': 1.0, 'P': 20.26, 'Q': 20.26, 'R': -6.54, 'S': 20.26, 'T': 1.0, 'V': 20.26, 'W': -1.88, 'Y': 1.0},
        'Q': {'A': 1.0, 'C': -6.54, 'D': 20.26, 'E': 20.26, 'F': -6.54, 'G': 1.0, 'H': 1.0, 'I': 1.0, 'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'P': 20.26, 'Q': 20.26, 'R': 1.0, 'S': 44.94, 'T': 1.0, 'V': -6.54, 'W': 1.0, 'Y': -6.54},
        'R': {'A': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': -7.49, 'H': 1.0, 'I': 1.0, 'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': 13.34, 'P': 20.26, 'Q': 20.26, 'R': 58.28, 'S': 44.94, 'T': 1.0, 'V': 1.0, 'W': 58.28, 'Y': -6.54},
        'S': {'A': 1.0, 'C': 33.60, 'D': 1.0, 'E': 20.26, 'F': 1.0, 'G': 1.0, 'H': 1.0, 'I': 1.0, 'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'P': 44.94, 'Q': 20.26, 'R': 20.26, 'S': 20.26, 'T': 1.0, 'V': 1.0, 'W': 1.0, 'Y': 1.0},
        'T': {'A': 1.0, 'C': 1.0, 'D': 1.0, 'E': 20.26, 'F': 13.34, 'G': -7.49, 'H': 1.0, 'I': 1.0, 'K': 1.0, 'L': 1.0, 'M': 1.0, 'N': -14.03, 'P': 1.0, 'Q': -6.54, 'R': 1.0, 'S': 1.0, 'T': 1.0, 'V': 1.0, 'W': -14.03, 'Y': 1.0},
        'V': {'A': 1.0, 'C': 1.0, 'D': -14.03, 'E': 1.0, 'F': 1.0, 'G': -7.49, 'H': 1.0, 'I': 1.0, 'K': -1.88, 'L': 1.0, 'M': 1.0, 'N': 1.0, 'P': 20.26, 'Q': 1.0, 'R': 1.0, 'S': 1.0, 'T': -7.49, 'V': 1.0, 'W': 1.0, 'Y': -6.54},
        'W': {'A': -14.03, 'C': 1.0, 'D': 1.0, 'E': 1.0, 'F': 1.0, 'G': -9.37, 'H': 24.68, 'I': 1.0, 'K': 1.0, 'L': 13.34, 'M': 24.68, 'N': 13.34, 'P': 1.0, 'Q': 1.0, 'R': 1.0, 'S': 1.0, 'T': -14.03, 'V': -7.49, 'W': 1.0, 'Y': 1.0},
        'Y': {'A': 24.68, 'C': 1.0, 'D': 24.68, 'E': -6.54, 'F': 1.0, 'G': -7.49, 'H': 13.34, 'I': 1.0, 'K': 1.0, 'L': 1.0, 'M': 44.94, 'N': 1.0, 'P': 13.34, 'Q': 1.0, 'R': -15.91, 'S': 1.0, 'T': -7.49, 'V': 1.0, 'W': -9.37, 'Y': 13.34},
    }
    if len(seq) < 2:
        return 0.0
    total = 0.0
    for i in range(len(seq) - 1):
        aa1, aa2 = seq[i], seq[i + 1]
        if aa1 in DIWV and aa2 in DIWV[aa1]:
            total += DIWV[aa1][aa2]
    return (10.0 / len(seq)) * total


def compute_isoelectric_point(seq: str) -> float:
    """Estimate isoelectric point by bisection."""
    def charge_at_ph(ph):
        return compute_net_charge(seq, ph)

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if charge_at_ph(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def count_cysteines(seq: str) -> int:
    return seq.count('C')


def fraction_charged(seq: str) -> float:
    """Fraction of charged residues (D, E, K, R, H)."""
    charged = sum(1 for aa in seq if aa in 'DEKRH')
    return charged / len(seq) if seq else 0.0


def compute_physicochemical(seq: str) -> dict:
    """Compute all physicochemical properties for a peptide sequence."""
    return {
        'length': len(seq),
        'molecular_weight': round(compute_molecular_weight(seq), 1),
        'net_charge_7_4': round(compute_net_charge(seq, 7.4), 2),
        'gravy': round(compute_gravy(seq), 3),
        'isoelectric_point': round(compute_isoelectric_point(seq), 2),
        'instability_index': round(compute_instability_index(seq), 2),
        'n_cysteines': count_cysteines(seq),
        'fraction_charged': round(fraction_charged(seq), 3),
    }


# ── Sequence extraction ─────────────────────────────────────────────

def load_all_sequences() -> dict:
    """Load peptide sequences from metadata and AF3 input JSONs."""
    seqs = {}

    # 1. Designed candidates from metadata
    meta_path = DATA_DIR / "candidates" / "all_candidates_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        for _, row in meta.iterrows():
            if pd.notna(row.get("sequence")):
                seqs[row["peptide_id"]] = row["sequence"]

    # 2. Controls + BindCraft from AF3 input JSONs
    for jf in glob.glob(str(DATA_DIR / "af3_inputs" / "**" / "*.json"), recursive=True):
        with open(jf) as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    continue
                for job in data:
                    name = job.get("name", "")
                    chains = job.get("sequences", [])
                    if len(chains) >= 2 and name not in seqs:
                        pep_seq = chains[1].get("proteinChain", {}).get("sequence", "")
                        if pep_seq:
                            seqs[name] = pep_seq
            except (json.JSONDecodeError, KeyError):
                pass

    return seqs


# ── Main pipeline ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Task 11: Master Scoring Table + Physicochemical Filtering")
    print("=" * 60)

    # Load all score sources
    af3 = pd.read_csv(DATA_DIR / "af3_results" / "round1_scores.csv")
    rosetta = pd.read_csv(DATA_DIR / "energy_scores" / "rosetta_results.csv")
    foldx = pd.read_csv(DATA_DIR / "energy_scores" / "foldx_results.csv")

    contact_path = DATA_DIR / "results" / "contact_scores.csv"
    contact = pd.read_csv(contact_path) if contact_path.exists() else pd.DataFrame()

    print(f"AF3: {len(af3)}, Rosetta: {len(rosetta)}, FoldX: {len(foldx)}, Contact: {len(contact)}")

    # Select key columns from each source
    af3_cols = af3[["peptide_id", "source_tool", "iptm", "ptm", "ranking_score",
                     "pae_interface_mean", "plddt_mean_B", "n_confident_contacts",
                     "n_high_contacts", "has_clash", "fraction_disordered"]].copy()

    rosetta_cols = rosetta[["peptide_id", "dG_separated", "dSASA", "packstat",
                             "nres_interface", "n_unsatisfied_hbonds",
                             "dG_per_dSASA"]].copy()
    rosetta_cols.columns = ["peptide_id", "rosetta_dG", "rosetta_dSASA",
                            "rosetta_packstat", "rosetta_nres_interface",
                            "rosetta_unsat_hbonds", "rosetta_dG_per_dSASA"]

    foldx_cols = foldx[["peptide_id", "interaction_energy", "van_der_waals",
                         "sidechain_hbond", "electrostatics",
                         "interface_residues"]].copy()
    foldx_cols.columns = ["peptide_id", "foldx_dG", "foldx_vdw",
                          "foldx_sc_hbond", "foldx_elec",
                          "foldx_interface_res"]

    # Merge all scores
    master = af3_cols.copy()
    master = master.merge(rosetta_cols, on="peptide_id", how="left")
    master = master.merge(foldx_cols, on="peptide_id", how="left")

    if not contact.empty:
        contact_cols = contact[["peptide_id", "recapitulation_fraction",
                                 "total_predicted_contacts"]].copy()
        contact_cols.columns = ["peptide_id", "contact_recap", "contact_total"]
        master = master.merge(contact_cols, on="peptide_id", how="left")

    print(f"Merged table: {len(master)} rows, {len(master.columns)} columns")

    # Add sequences and physicochemical properties
    seqs = load_all_sequences()
    print(f"Loaded {len(seqs)} peptide sequences")

    # Normalize peptide IDs for matching (AF3 lowercases)
    seq_lower = {k.lower(): v for k, v in seqs.items()}

    physchem_rows = []
    for _, row in master.iterrows():
        pid = row["peptide_id"]
        seq = seqs.get(pid) or seq_lower.get(pid.lower(), "")
        if seq:
            props = compute_physicochemical(seq)
            props["peptide_id"] = pid
            props["sequence"] = seq
            physchem_rows.append(props)
        else:
            physchem_rows.append({"peptide_id": pid, "sequence": ""})

    physchem_df = pd.DataFrame(physchem_rows)
    master = master.merge(physchem_df, on="peptide_id", how="left")

    matched = master["sequence"].notna() & (master["sequence"] != "")
    print(f"Sequences matched: {matched.sum()}/{len(master)}")

    # ── Composite ranking ────────────────────────────────────────────
    # Rank by 3 key metrics (lower rank = better)
    # ipTM: higher is better → rank ascending=False
    # rosetta_dG: more negative is better → rank ascending=True
    # foldx_dG: more negative is better → rank ascending=True
    master["rank_iptm"] = master["iptm"].rank(ascending=False, method="min")
    master["rank_rosetta"] = master["rosetta_dG"].rank(ascending=True, method="min")
    master["rank_foldx"] = master["foldx_dG"].rank(ascending=True, method="min")
    master["composite_rank"] = (
        master[["rank_iptm", "rank_rosetta", "rank_foldx"]].mean(axis=1)
    )

    # Sort by composite rank
    master = master.sort_values("composite_rank").reset_index(drop=True)

    # Save full master table
    out_dir = DATA_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_dir / "master_scores.csv", index=False)
    print(f"\nSaved master_scores.csv ({len(master)} rows)")

    # ── Physicochemical filtering ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PHYSICOCHEMICAL FILTERING")
    print("=" * 60)

    filtered = master.copy()
    n_start = len(filtered)

    # Filter 1: Remove extreme net charge (|charge| > 8)
    mask = filtered["net_charge_7_4"].abs() <= 8
    n_charge = (~mask).sum()
    filtered = filtered[mask]

    # Filter 2: Remove high hydrophobicity (GRAVY > 0.5 = aggregation-prone)
    mask = filtered["gravy"] <= 0.5
    n_gravy = (~mask).sum()
    filtered = filtered[mask]

    # Filter 3: Remove unstable peptides (instability index > 40)
    mask = filtered["instability_index"] <= 40
    n_unstable = (~mask).sum()
    filtered = filtered[mask]

    # Filter 4: Keep only candidates (remove controls for shortlist)
    candidates = filtered[~filtered["source_tool"].str.contains("control")]

    print(f"Starting: {n_start}")
    print(f"  Removed {n_charge} for extreme charge (|charge| > 8)")
    print(f"  Removed {n_gravy} for high hydrophobicity (GRAVY > 0.5)")
    print(f"  Removed {n_unstable} for instability (index > 40)")
    print(f"After filters: {len(filtered)} ({len(candidates)} candidates, "
          f"{len(filtered) - len(candidates)} controls)")

    # Save filtered shortlist (candidates only)
    candidates = candidates.reset_index(drop=True)
    candidates.to_csv(out_dir / "top_candidates.csv", index=False)
    print(f"\nSaved top_candidates.csv ({len(candidates)} candidates)")

    # ── Summary statistics ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("COMPOSITE RANKING — TOP 20 CANDIDATES")
    print("=" * 60)
    top20 = candidates.head(20)[
        ["peptide_id", "source_tool", "composite_rank", "iptm",
         "rosetta_dG", "foldx_dG", "gravy", "net_charge_7_4",
         "instability_index", "length"]
    ]
    print(top20.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("FILTERED CANDIDATES BY SOURCE TOOL")
    print("=" * 60)
    tool_summary = candidates.groupby("source_tool").agg(
        count=("peptide_id", "count"),
        composite_rank_mean=("composite_rank", "mean"),
        iptm_mean=("iptm", "mean"),
        rosetta_dG_mean=("rosetta_dG", "mean"),
        foldx_dG_mean=("foldx_dG", "mean"),
    ).round(3)
    print(tool_summary.to_string())

    print(f"\n{'=' * 60}")
    print("FILTER SURVIVAL RATE BY TOOL")
    print("=" * 60)
    for tool in master["source_tool"].unique():
        total = len(master[master["source_tool"] == tool])
        survived = len(candidates[candidates["source_tool"] == tool])
        rate = survived / total * 100 if total > 0 else 0
        if "control" not in tool:
            print(f"  {tool}: {survived}/{total} passed ({rate:.0f}%)")


if __name__ == "__main__":
    main()
