"""Prepare AlphaFold 3 Server submission inputs.

AF3 Server accepts JSON with sequences. Each job = one peptide + BMPR1A ECD.
Round 1: 1 submission per candidate (coarse filter).
Round 2: 5 submissions per top 20% (final scoring).
"""
import json
import pandas as pd
from pathlib import Path
from Bio import SeqIO

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def bmpr1a_sequence() -> str:
    """Read BMPR1A ECD sequence from Stage 1 FASTA."""
    fasta_path = DATA_DIR / "structures" / "bmpr1a_ecd_sequence.fasta"
    record = next(SeqIO.parse(str(fasta_path), "fasta"))
    return str(record.seq)


def make_af3_job(peptide_id: str, peptide_seq: str,
                 target_seq: str, job_name: str = None) -> dict:
    """Create an AF3 Server-compatible job JSON.

    AF3 Server format: https://alphafoldserver.com
    Each job specifies protein chains as sequences.
    """
    if job_name is None:
        job_name = peptide_id

    # AF3 Server expects this format
    return {
        "name": job_name,
        "modelSeeds": [],  # empty = server picks random seed
        "sequences": [
            {
                "proteinChain": {
                    "sequence": target_seq,
                    "count": 1,
                }
            },
            {
                "proteinChain": {
                    "sequence": peptide_seq,
                    "count": 1,
                }
            },
        ],
    }


def main():
    target_seq = bmpr1a_sequence()
    print(f"BMPR1A ECD: {len(target_seq)} residues")

    # Load all candidates + controls
    candidates_meta = pd.read_csv(
        DATA_DIR / "candidates" / "all_candidates_metadata.csv"
    )
    scrambled_meta = pd.read_csv(DATA_DIR / "controls" / "scrambled_metadata.csv")
    random_meta = pd.read_csv(DATA_DIR / "controls" / "random_metadata.csv")

    all_peptides = pd.concat([candidates_meta, scrambled_meta, random_meta],
                             ignore_index=True)

    # Round 1: 1 job per peptide
    out_dir = DATA_DIR / "af3_inputs" / "round1"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for _, row in all_peptides.iterrows():
        job = make_af3_job(row["peptide_id"], row["sequence"], target_seq)
        job_path = out_dir / f"{row['peptide_id']}.json"
        job_path.write_text(json.dumps(job, indent=2))
        jobs.append(row["peptide_id"])

    print(f"\nRound 1: {len(jobs)} AF3 jobs prepared in {out_dir}")
    print(f"  Candidates: {len(candidates_meta)}")
    print(f"  Scrambled controls: {len(scrambled_meta)}")
    print(f"  Random controls: {len(random_meta)}")

    # Also create a batch summary for manual submission tracking
    summary = all_peptides[["peptide_id", "sequence", "length", "tool"]].copy()
    summary["af3_submitted"] = False
    summary["af3_result_downloaded"] = False
    summary.to_csv(out_dir / "submission_tracker.csv", index=False)
    print(f"\nSubmission tracker saved to {out_dir / 'submission_tracker.csv'}")

    # Print submission instructions
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  AF3 SERVER SUBMISSION INSTRUCTIONS                      ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. Go to https://alphafoldserver.com                    ║
║  2. Sign in with Google account                          ║
║  3. For each job:                                        ║
║     - Click "New job"                                    ║
║     - Add protein chain 1: BMPR1A ECD sequence           ║
║     - Add protein chain 2: peptide sequence              ║
║     - Name the job with the peptide_id                   ║
║     - Submit                                             ║
║  4. Download results when ready                          ║
║  5. Save to data/af3_results/round1/<peptide_id>/        ║
║                                                          ║
║  Rate limit: ~20 jobs per day (may vary)                 ║
║  At 266 jobs, expect ~14 days for Round 1                ║
║                                                          ║
║  TIP: Start with a few candidates + controls to          ║
║  verify the pipeline works before bulk submission.        ║
╚══════════════════════════════════════════════════════════╝
""")

    # Create a quick-start batch: top 5 candidates + 5 controls
    quickstart_dir = out_dir / "quickstart_batch"
    quickstart_dir.mkdir(exist_ok=True)

    # Top candidates by tool diversity
    top_pepmlm = candidates_meta[candidates_meta["tool"] == "PepMLM"].head(2)
    top_rfdiff = candidates_meta[candidates_meta["tool"] == "RFdiffusion"].head(2)
    top_rfpep = candidates_meta[candidates_meta["tool"] == "RFpeptides"].head(1)
    top_scrambled = scrambled_meta.head(3)
    top_random = random_meta.head(2)

    quickstart = pd.concat([top_pepmlm, top_rfdiff, top_rfpep,
                            top_scrambled, top_random], ignore_index=True)

    for _, row in quickstart.iterrows():
        job = make_af3_job(row["peptide_id"], row["sequence"], target_seq)
        job_path = quickstart_dir / f"{row['peptide_id']}.json"
        job_path.write_text(json.dumps(job, indent=2))

    print(f"Quick-start batch: {len(quickstart)} jobs in {quickstart_dir}")
    print("Submit these first to verify the pipeline works.\n")

    # Print the quick-start sequences for easy copy-paste into AF3 server
    print("=" * 60)
    print("QUICK-START: Copy-paste these into AF3 Server")
    print("=" * 60)
    print(f"\nBMPR1A ECD (Chain 1 for ALL jobs):")
    print(f"  {target_seq}\n")
    for _, row in quickstart.iterrows():
        print(f"Job: {row['peptide_id']} ({row.get('tool', 'control')})")
        print(f"  Chain 2: {row['sequence']}")
        print()


if __name__ == "__main__":
    main()
