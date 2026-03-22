"""Merge candidates from all tools into a unified dataset."""
import pandas as pd
from pathlib import Path
from Bio import SeqIO

CANDIDATES_DIR = Path(__file__).parent.parent.parent / "data" / "candidates"

TOOL_FILES = {
    "PepMLM": ("pepmlm_candidates.fasta", "pepmlm_metadata.csv"),
    "RFdiffusion": ("rfdiffusion_candidates.fasta", "rfdiffusion_metadata.csv"),
    "RFpeptides": ("rfpeptides_candidates.fasta", "rfpeptides_metadata.csv"),
    "BindCraft": ("bindcraft_candidates.fasta", "bindcraft_metadata.csv"),
}


def main():
    all_records = []
    all_metadata = []

    for tool, (fasta_file, meta_file) in TOOL_FILES.items():
        fasta_path = CANDIDATES_DIR / fasta_file
        if not fasta_path.exists():
            print(f"  {tool}: not found — skipping")
            continue

        meta_path = CANDIDATES_DIR / meta_file
        tool_meta = pd.read_csv(meta_path) if meta_path.exists() else None

        records = list(SeqIO.parse(str(fasta_path), "fasta"))
        print(f"  {tool}: {len(records)} candidates")

        for rec in records:
            all_records.append(rec)
            row = {
                "peptide_id": rec.id,
                "sequence": str(rec.seq),
                "length": len(rec.seq),
                "tool": tool,
            }
            if tool_meta is not None and rec.id in tool_meta["peptide_id"].values:
                extra = tool_meta[tool_meta["peptide_id"] == rec.id].iloc[0].to_dict()
                for k, v in extra.items():
                    if k not in row:
                        row[k] = v
            all_metadata.append(row)

    # Write merged FASTA
    merged_fasta = CANDIDATES_DIR / "all_candidates.fasta"
    SeqIO.write(all_records, str(merged_fasta), "fasta")

    # Write merged metadata
    df = pd.DataFrame(all_metadata)
    df.to_csv(CANDIDATES_DIR / "all_candidates_metadata.csv", index=False)

    print(f"\nTotal merged: {len(df)} candidates")
    print(f"Per tool:\n{df['tool'].value_counts().to_string()}")
    print(f"Length distribution:\n{df['length'].describe().to_string()}")


if __name__ == "__main__":
    main()
