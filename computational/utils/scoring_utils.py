"""Utility functions for score parsing and filtering."""
import json
from pathlib import Path
import pandas as pd
import numpy as np


def parse_af3_json(json_path: Path, bmpr1a_chain_len: int = 0) -> dict:
    """Parse AF3 server output JSON for key metrics.

    Extracts all four spec-required metrics:
    1. chain_pair_iptm
    2. chain_pair_pae_min
    3. local interface pLDDT
    4. contact_probs
    """
    data = json.loads(json_path.read_text())

    metrics = {}
    metrics["chain_pair_iptm"] = data.get("iptm", None)
    metrics["ptm"] = data.get("ptm", None)

    plddt = data.get("atom_plddts", data.get("plddt", []))
    metrics["plddt_mean"] = float(np.mean(plddt)) if plddt else None

    if plddt and bmpr1a_chain_len > 0:
        peptide_plddt = plddt[bmpr1a_chain_len:]
        metrics["peptide_plddt_mean"] = float(np.mean(peptide_plddt)) if peptide_plddt else None
    else:
        metrics["peptide_plddt_mean"] = None

    pae = data.get("pae", data.get("predicted_aligned_error", []))
    if pae and bmpr1a_chain_len > 0:
        pae_matrix = np.array(pae)
        cross_pae = pae_matrix[:bmpr1a_chain_len, bmpr1a_chain_len:]
        metrics["chain_pair_pae_min"] = float(np.min(cross_pae)) if cross_pae.size > 0 else None
        metrics["chain_pair_pae_mean"] = float(np.mean(cross_pae)) if cross_pae.size > 0 else None
    else:
        metrics["chain_pair_pae_min"] = None
        metrics["chain_pair_pae_mean"] = None

    return metrics


def combine_scores(af3_df: pd.DataFrame, rosetta_df: pd.DataFrame,
                    foldx_df: pd.DataFrame) -> pd.DataFrame:
    """Merge scoring dataframes on peptide_id."""
    merged = af3_df.merge(rosetta_df, on="peptide_id", how="left")
    merged = merged.merge(foldx_df, on="peptide_id", how="left")
    return merged


def filter_by_consensus(df: pd.DataFrame,
                        iptm_min: float = 0.4,
                        pae_max: float = 15.0,
                        energy_col: str = "dG_separated",
                        energy_max: float = 0.0) -> pd.DataFrame:
    """Filter candidates requiring AF3 + at least one energy function pass."""
    af3_pass = (df["chain_pair_iptm"] >= iptm_min) & (df["chain_pair_pae_min"] <= pae_max)
    rosetta_pass = df.get("dG_separated", pd.Series(dtype=float)).lt(energy_max)
    foldx_pass = df.get("foldx_energy", pd.Series(dtype=float)).lt(energy_max)
    energy_pass = rosetta_pass | foldx_pass
    return df[af3_pass & energy_pass].copy()
