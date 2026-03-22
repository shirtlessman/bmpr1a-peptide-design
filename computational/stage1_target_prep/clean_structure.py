"""Extract and clean relevant chains from PDB structures.

Chain assignments verified by inspection:
  1REW: A,B = BMP-2 dimer; C,D = BMPR1A ECD (two copies). Use C.
  2H62: A,B = BMP-2 dimer; C = BMPR1A ECD; D = ActRII ECD.
  3EVS: B = BMP-2; C = BMPR1B ECD.
  3MTF: Kinase domain only (res 202+), NOT the ECD. ACVR1 ECD will be
        obtained from UniProt sequence for AF3 modeling instead.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pdb_utils import extract_chain, STRUCTURES_DIR

EXTRACTIONS = [
    # 1REW — primary target
    ("1REW.pdb", "A", "1REW_BMP2_chainA.pdb"),
    ("1REW.pdb", "C", "1REW_BMPR1A.pdb"),
    # 2H62 — ternary complex
    ("2H62.pdb", "C", "2H62_BMPR1A.pdb"),
    ("2H62.pdb", "D", "2H62_ActRII.pdb"),
    ("2H62.pdb", "A", "2H62_BMP2_chainA.pdb"),
    # 3EVS — BMPR1B for specificity counter-screen
    ("3EVS.pdb", "C", "3EVS_BMPR1B.pdb"),
    # 3MTF — ACVR1 ECD not available in this structure (kinase domain only)
    # ACVR1 ECD sequence from UniProt Q04771 (residues 21-123) used instead
]


def main():
    cleaned_dir = STRUCTURES_DIR / "cleaned"
    cleaned_dir.mkdir(exist_ok=True)

    for pdb_file, chain_id, output_name in EXTRACTIONS:
        pdb_path = STRUCTURES_DIR / pdb_file
        out_path = cleaned_dir / output_name
        print(f"Extracting chain {chain_id} from {pdb_file} -> {output_name}")
        extract_chain(pdb_path, chain_id, out_path)
        print(f"  -> {out_path}")

    print("\nAll chains extracted.")
    print("\nNOTE: ACVR1 (ALK2) ECD is not in 3MTF (kinase domain only).")
    print("      Use UniProt Q04771 residues 21-123 for ACVR1 ECD sequence.")


if __name__ == "__main__":
    main()
