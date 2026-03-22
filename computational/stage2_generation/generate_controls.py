"""Generate scrambled decoys and random peptide baselines."""
import random
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

CANDIDATES_DIR = Path(__file__).parent.parent.parent / "data" / "candidates"
CONTROLS_DIR = Path(__file__).parent.parent.parent / "data" / "controls"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def scramble_sequence(seq: str) -> str:
    """Shuffle amino acid order, preserving composition."""
    aa_list = list(seq)
    random.shuffle(aa_list)
    return "".join(aa_list)


def random_peptide(length: int) -> str:
    """Generate a random peptide of given length."""
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))


def main():
    random.seed(42)  # Reproducibility
    CONTROLS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all candidates
    meta = pd.read_csv(CANDIDATES_DIR / "all_candidates_metadata.csv")

    # --- Scrambled decoys: 20 per tool ---
    scrambled_records = []
    scrambled_meta = []
    for tool in meta["tool"].unique():
        tool_seqs = meta[meta["tool"] == tool]["sequence"].tolist()
        sample = random.sample(tool_seqs, min(20, len(tool_seqs)))
        for i, seq in enumerate(sample):
            scrambled = scramble_sequence(seq)
            pid = f"scrambled_{tool}_{i:04d}"
            scrambled_records.append(
                SeqRecord(Seq(scrambled), id=pid, description="")
            )
            scrambled_meta.append({
                "peptide_id": pid,
                "sequence": scrambled,
                "length": len(scrambled),
                "tool": f"scrambled_{tool}",
                "control_type": "scrambled",
                "source_tool": tool,
            })

    SeqIO.write(scrambled_records,
                str(CONTROLS_DIR / "scrambled_decoys.fasta"), "fasta")
    pd.DataFrame(scrambled_meta).to_csv(
        CONTROLS_DIR / "scrambled_metadata.csv", index=False
    )
    print(f"Scrambled decoys: {len(scrambled_records)}")

    # --- Random baseline: 50 peptides matching length distribution ---
    lengths = meta["length"].tolist()
    random_records = []
    random_meta = []
    for i in range(50):
        length = random.choice(lengths)
        seq = random_peptide(length)
        pid = f"random_{i:04d}"
        random_records.append(SeqRecord(Seq(seq), id=pid, description=""))
        random_meta.append({
            "peptide_id": pid,
            "sequence": seq,
            "length": length,
            "tool": "random",
            "control_type": "random",
        })

    SeqIO.write(random_records,
                str(CONTROLS_DIR / "random_baseline.fasta"), "fasta")
    pd.DataFrame(random_meta).to_csv(
        CONTROLS_DIR / "random_metadata.csv", index=False
    )
    print(f"Random baselines: {len(random_records)}")

    # --- Summary ---
    total_controls = len(scrambled_records) + len(random_records)
    print(f"\nTotal controls: {total_controls}")
    print(f"Total candidates + controls: {len(meta) + total_controls}")


if __name__ == "__main__":
    main()
