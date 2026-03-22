"""Run FoldX AnalyseComplex on AF3-predicted peptide-BMPR1A complexes.

Pipeline: extract CIF from zip → convert CIF→PDB (BioPython) → FoldX
RepairPDB (relaxation) → FoldX AnalyseComplex → parse interaction energy.

RepairPDB resolves steric clashes and optimizes rotamers using FoldX's
own energy function, producing physically meaningful binding energies.
Without this step, raw AF3 structures yield inflated positive energies
due to unresolved clashes — same issue as with PyRosetta.

FoldX provides physics-based binding energy in kcal/mol, independent
from Rosetta's REU scoring. Agreement between the two strengthens
the validation.

Output: data/energy_scores/foldx_results.csv
"""
import zipfile
import tempfile
import subprocess
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR.parent / "alphafold results"
FOLDX_BIN = PROJECT_DIR.parent / "foldx" / "foldx_20270131"
FOLDX_MOLECULES = PROJECT_DIR.parent / "foldx" / "molecules"


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


def cif_to_pdb(cif_content: bytes, pdb_path: Path):
    """Convert CIF content to PDB file using BioPython."""
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(cif_content)
        tmp_cif = tmp.name

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("complex", tmp_cif)
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_path))
    finally:
        Path(tmp_cif).unlink(missing_ok=True)


def repair_pdb(pdb_path: Path, work_dir: Path) -> Path:
    """Run FoldX RepairPDB to fix clashes and optimize rotamers.

    Returns path to the repaired PDB file.
    FoldX names repaired files as '{stem}_Repair.pdb'.
    """
    cmd = [
        str(FOLDX_BIN),
        "--command=RepairPDB",
        f"--pdb={pdb_path.name}",
        f"--pdb-dir={pdb_path.parent}",
        f"--output-dir={work_dir}",
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                   cwd=str(work_dir))

    repaired = work_dir / f"{pdb_path.stem}_Repair.pdb"
    if repaired.exists():
        return repaired
    # Fallback to original if repair failed
    return pdb_path


def run_foldx_analyse(pdb_path: Path, work_dir: Path) -> dict:
    """Run FoldX RepairPDB + AnalyseComplex and parse results."""
    metrics = {}

    # Step 1: Repair (relax) the structure
    repaired_path = repair_pdb(pdb_path, work_dir)
    metrics["was_repaired"] = repaired_path != pdb_path

    # Step 2: AnalyseComplex on repaired structure
    pdb_name = repaired_path.name

    cmd = [
        str(FOLDX_BIN),
        "--command=AnalyseComplex",
        f"--pdb={pdb_name}",
        "--analyseComplexChains=A,B",
        f"--pdb-dir={repaired_path.parent}",
        f"--output-dir={work_dir}",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(work_dir)
        )

        # Parse Interaction_ output file
        # FoldX names files like Interaction_{stem}_AC.fxout
        # After RepairPDB, stem is {original}_Repair
        interaction_files = list(work_dir.glob(f"Interaction_{repaired_path.stem}*"))
        if interaction_files:
            lines = interaction_files[0].read_text().strip().split("\n")
            # Find the header line (contains "Pdb") and the data line after it
            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith("Pdb\t"):
                    header_idx = i
                    break
            if header_idx is not None and header_idx + 1 < len(lines):
                header = lines[header_idx].split("\t")
                values = lines[header_idx + 1].split("\t")
                foldx_data = dict(zip(header, values))

                # Key metrics
                for key in ["Interaction Energy", "Backbone Hbond",
                             "Sidechain Hbond", "Van der Waals",
                             "Electrostatics", "Solvation Polar",
                             "Solvation Hydrophobic", "entropy sidechain",
                             "entropy mainchain", "Number of Residues",
                             "Interface Residues"]:
                    if key in foldx_data:
                        try:
                            metrics[key.replace(" ", "_").lower()] = float(foldx_data[key])
                        except ValueError:
                            pass

        # Parse Summary_ output file for stability metrics
        summary_files = list(work_dir.glob(f"Summary_{repaired_path.stem}*"))
        if summary_files:
            lines = summary_files[0].read_text().strip().split("\n")
            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith("Pdb\t"):
                    header_idx = i
                    break
            if header_idx is not None and header_idx + 1 < len(lines):
                header = lines[header_idx].split("\t")
                values = lines[header_idx + 1].split("\t")
                summary_data = dict(zip(header, values))
                for key in ["StabilityGroup1", "StabilityGroup2",
                             "IntraclashesGroup1", "IntraclashesGroup2"]:
                    if key in summary_data:
                        try:
                            metrics[key.lower()] = float(summary_data[key])
                        except ValueError:
                            pass

        if not metrics:
            # Check stderr for errors
            if result.returncode != 0:
                metrics["error"] = result.stderr[:200] if result.stderr else "FoldX returned non-zero"

    except subprocess.TimeoutExpired:
        metrics["error"] = "timeout"
    except Exception as e:
        metrics["error"] = str(e)[:200]

    return metrics


def main():
    print("=" * 60)
    print("FoldX 5.1 AnalyseComplex Scoring")
    print("=" * 60)

    if not FOLDX_BIN.exists():
        print(f"ERROR: FoldX not found at {FOLDX_BIN}")
        return

    # Load AF3 scores for best model selection
    scores_df = pd.read_csv(DATA_DIR / "af3_results" / "round1_scores.csv")
    best_models = {}
    for _, row in scores_df.iterrows():
        best_models[row["peptide_id"]] = int(row.get("best_model_idx", 0))

    # Load existing results to skip already-scored peptides
    existing_csv = DATA_DIR / "energy_scores" / "foldx_results.csv"
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

    with tempfile.TemporaryDirectory() as work_dir:
        work_dir = Path(work_dir)

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
                            # Extract CIF and convert to PDB
                            cif_content = zf.read(best_cif)
                            pdb_path = work_dir / f"{peptide_id}.pdb"
                            cif_to_pdb(cif_content, pdb_path)

                            # Run FoldX
                            metrics = run_foldx_analyse(pdb_path, work_dir)
                            metrics["peptide_id"] = peptide_id
                            metrics["source_tool"] = classify_peptide(peptide_id)
                            all_results.append(metrics)

                            # Clean up PDB and FoldX output files
                            pdb_path.unlink(missing_ok=True)
                            for f in work_dir.glob(f"*{peptide_id}*"):
                                f.unlink(missing_ok=True)
                            # Clean any remaining FoldX temp files
                            for f in work_dir.glob("*.fxout"):
                                f.unlink(missing_ok=True)
                            for f in work_dir.glob("*_Repair*"):
                                f.unlink(missing_ok=True)

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
    out_path = out_dir / "foldx_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} results to {out_path}")

    # Summary
    has_energy = "interaction_energy" in df.columns
    if has_energy:
        scored = df[df["interaction_energy"].notna()]
    elif "error" in df.columns:
        scored = df[df["error"].isna()]
    else:
        scored = df

    print(f"Successfully scored: {len(scored)}/{len(df)}")

    if has_energy and not scored.empty:
        print(f"\n{'=' * 60}")
        print("FOLDX INTERACTION ENERGY BY SOURCE TOOL (kcal/mol)")
        print("=" * 60)
        summary = scored.groupby("source_tool").agg(
            count=("peptide_id", "count"),
            energy_mean=("interaction_energy", "mean"),
            energy_min=("interaction_energy", "min"),
        ).round(3)
        print(summary.to_string())

        print(f"\n{'=' * 60}")
        print("TOP 10 BY INTERACTION ENERGY (most negative = strongest)")
        print("=" * 60)
        cols = ["peptide_id", "source_tool", "interaction_energy"]
        if "van_der_waals" in scored.columns:
            cols.append("van_der_waals")
        if "sidechain_hbond" in scored.columns:
            cols.append("sidechain_hbond")
        if "electrostatics" in scored.columns:
            cols.append("electrostatics")
        top10 = scored.nsmallest(10, "interaction_energy")[cols]
        print(top10.to_string(index=False))

        # Candidates vs controls
        print(f"\n{'=' * 60}")
        print("CANDIDATES vs CONTROLS")
        print("=" * 60)
        candidates = scored[~scored["source_tool"].str.contains("control")]
        controls = scored[scored["source_tool"].str.contains("control")]
        print(f"Candidates (n={len(candidates)}): "
              f"ΔG={candidates['interaction_energy'].mean():.2f} +/- "
              f"{candidates['interaction_energy'].std():.2f} kcal/mol")
        print(f"Controls   (n={len(controls)}): "
              f"ΔG={controls['interaction_energy'].mean():.2f} +/- "
              f"{controls['interaction_energy'].std():.2f} kcal/mol")


if __name__ == "__main__":
    main()
