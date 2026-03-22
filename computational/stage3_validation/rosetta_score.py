"""Run PyRosetta InterfaceAnalyzer on AF3-predicted peptide-BMPR1A complexes.

Pipeline: load CIF → FastRelax (constrained) → InterfaceAnalyzer → score.
Relaxation resolves steric clashes and bond geometry issues in raw AF3
predictions, producing physically meaningful binding energies (negative dG
= favorable binding).

Output: data/energy_scores/rosetta_results.csv
"""
import json
import zipfile
import tempfile
import re
import pandas as pd
import numpy as np
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.constraint_generator import (
    CoordinateConstraintGenerator,
    AddConstraints,
)

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR.parent / "alphafold results"


def normalize_peptide_id(raw_id: str) -> str:
    """Normalize AF3 subdirectory name back to canonical peptide ID."""
    pid = re.sub(r'_(\d)$', '', raw_id) if re.search(r'(?<=\d)_\d$', raw_id) else raw_id
    pid = re.sub(r'pepmlm_l(\d+)', lambda m: f'pepmlm_L{m.group(1)}', pid)
    pid = pid.replace('scrambled_pepmlm', 'scrambled_PepMLM')
    pid = pid.replace('scrambled_rfdiffusion', 'scrambled_RFdiffusion')
    pid = pid.replace('scrambled_rfpeptides', 'scrambled_RFpeptides')
    return pid


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
    return "unknown"


def relax_pose(pose, sfxn):
    """Constrained FastRelax to fix clashes while preserving AF3 prediction.

    Uses coordinate constraints (0.5 Å std dev) to keep atoms near their
    predicted positions. This resolves steric clashes and bad bond geometry
    without distorting the overall fold.
    """
    # Add coordinate constraints to anchor atoms near starting positions
    cg = CoordinateConstraintGenerator()
    cg.set_sd(0.5)  # 0.5 Å standard deviation — tight constraint
    add_csts = AddConstraints()
    add_csts.add_generator(cg)
    add_csts.apply(pose)

    # Score function with constraint weight
    sfxn_cst = sfxn.clone()
    sfxn_cst.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0
    )

    # FastRelax with constrained score function
    relax = FastRelax()
    relax.set_scorefxn(sfxn_cst)
    relax.constrain_relax_to_start_coords(True)
    relax.max_iter(200)

    # Allow all backbone and sidechain movement
    mm = MoveMap()
    mm.set_bb(True)
    mm.set_chi(True)
    relax.set_movemap(mm)

    relax.apply(pose)
    return pose


def score_interface(cif_content: bytes, sfxn) -> dict:
    """Relax and score a peptide-BMPR1A complex using PyRosetta."""
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(cif_content)
        tmp_path = tmp.name

    metrics = {}
    try:
        pose = pyrosetta.pose_from_file(tmp_path)

        # Pre-relaxation score for reference
        sfxn(pose)
        metrics["pre_relax_total"] = pose.energies().total_energy()

        # Constrained relaxation
        pose = relax_pose(pose, sfxn)

        # Post-relaxation score
        sfxn(pose)
        metrics["post_relax_total"] = pose.energies().total_energy()

        # Interface analysis on relaxed structure
        iam = InterfaceAnalyzerMover()
        iam.set_interface("A_B")
        iam.set_scorefunction(sfxn)
        iam.set_compute_packstat(True)
        iam.set_compute_interface_energy(True)
        iam.set_compute_interface_sc(True)
        iam.set_compute_interface_delta_hbond_unsat(True)
        iam.set_pack_separated(True)
        iam.apply(pose)

        metrics["dG_separated"] = iam.get_interface_dG()
        metrics["dSASA"] = iam.get_interface_delta_sasa()
        metrics["packstat"] = iam.get_interface_packstat()
        metrics["nres_interface"] = iam.get_num_interface_residues()
        metrics["n_unsatisfied_hbonds"] = iam.get_interface_delta_hbond_unsat()
        metrics["complex_energy"] = iam.get_complex_energy()
        metrics["separated_energy"] = iam.get_separated_interface_energy()
        metrics["crossterm_energy"] = iam.get_crossterm_interface_energy()

        # dG/dSASA ratio
        if metrics["dSASA"] and metrics["dSASA"] > 0:
            metrics["dG_per_dSASA"] = metrics["dG_separated"] / metrics["dSASA"]
        else:
            metrics["dG_per_dSASA"] = None

    except Exception as e:
        metrics["error"] = str(e)[:200]
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return metrics


def main():
    print("=" * 60)
    print("PyRosetta Interface Energy Scoring")
    print("=" * 60)

    # Initialize PyRosetta
    pyrosetta.init(extra_options="-mute all -ignore_unrecognized_res")
    sfxn = ScoreFunctionFactory.create_score_function("ref2015")

    # Load AF3 scores for best model selection
    scores_df = pd.read_csv(DATA_DIR / "af3_results" / "round1_scores.csv")
    best_models = {}
    for _, row in scores_df.iterrows():
        best_models[row["peptide_id"]] = int(row.get("best_model_idx", 0))

    # Load existing results to skip already-scored peptides
    existing_csv = DATA_DIR / "energy_scores" / "rosetta_results.csv"
    existing_results = []
    already_scored = set()
    if existing_csv.exists():
        existing_df = pd.read_csv(existing_csv)
        existing_results = existing_df.to_dict("records")
        already_scored = set(existing_df["peptide_id"].dropna())
        print(f"Found {len(already_scored)} already-scored peptides, skipping them.")

    # Process all zips
    all_results = list(existing_results)
    seen = set(already_scored)
    zip_files = sorted(RESULTS_DIR.glob("*.zip"))
    total_jobs = len(scores_df)
    new_jobs = total_jobs - len(already_scored)
    processed = 0

    print(f"Processing {len(zip_files)} zip files ({new_jobs} new jobs to score)...\n")

    for zf_path in zip_files:
        try:
            with zipfile.ZipFile(zf_path, "r") as zf:
                cif_files = [n for n in zf.namelist()
                             if n.endswith(".cif") and "model_" in n
                             and "template" not in n]

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
                        metrics = score_interface(cif_content, sfxn)
                        metrics["peptide_id"] = peptide_id
                        metrics["source_tool"] = classify_peptide(peptide_id)
                        all_results.append(metrics)
                        processed += 1
                        if processed % 25 == 0:
                            print(f"  Scored {processed}/{total_jobs}...")
                    except Exception as e:
                        print(f"  Error on {peptide_id}: {e}")

        except zipfile.BadZipFile:
            print(f"  Bad zip: {zf_path.name}")

    df = pd.DataFrame(all_results)

    # Save results
    out_dir = DATA_DIR / "energy_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rosetta_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} results to {out_path}")

    # Summary
    scored = df[df["error"].isna()] if "error" in df.columns else df
    print(f"Successfully scored: {len(scored)}/{len(df)}")

    if not scored.empty:
        print(f"\n{'=' * 60}")
        print("ENERGY SCORES BY SOURCE TOOL")
        print("=" * 60)
        summary = scored.groupby("source_tool").agg(
            count=("peptide_id", "count"),
            dG_mean=("dG_separated", "mean"),
            dG_min=("dG_separated", "min"),
            packstat_mean=("packstat", "mean"),
            nres_mean=("nres_interface", "mean"),
        ).round(3)
        print(summary.to_string())

        print(f"\n{'=' * 60}")
        print("TOP 10 BY BINDING ENERGY (most negative = strongest)")
        print("=" * 60)
        top10 = scored.nsmallest(10, "dG_separated")[
            ["peptide_id", "source_tool", "dG_separated",
             "dSASA", "packstat", "nres_interface"]
        ]
        print(top10.to_string(index=False))

        # Candidates vs controls
        print(f"\n{'=' * 60}")
        print("CANDIDATES vs CONTROLS")
        print("=" * 60)
        candidates = scored[~scored["source_tool"].str.contains("control")]
        controls = scored[scored["source_tool"].str.contains("control")]
        print(f"Candidates (n={len(candidates)}): "
              f"dG={candidates['dG_separated'].mean():.2f} +/- "
              f"{candidates['dG_separated'].std():.2f} REU")
        print(f"Controls   (n={len(controls)}): "
              f"dG={controls['dG_separated'].mean():.2f} +/- "
              f"{controls['dG_separated'].std():.2f} REU")


if __name__ == "__main__":
    main()
