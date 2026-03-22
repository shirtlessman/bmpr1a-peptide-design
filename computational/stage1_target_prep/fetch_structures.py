"""Fetch all required PDB structures for the study."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pdb_utils import fetch_pdb

REQUIRED_STRUCTURES = {
    "1REW": "BMP-2:BMPR1A ECD binary complex",
    "2H62": "BMP-2:BMPR1A:ActRII ternary complex",
}

# BMPR1B and ACVR1 for specificity counter-screen
SPECIFICITY_STRUCTURES = {
    "3EVS": "BMPR1B (ALK6) ECD with BMP-2",
    "3MTF": "ACVR1 (ALK2) ECD with BMP-6",
}


def main():
    all_structures = {**REQUIRED_STRUCTURES, **SPECIFICITY_STRUCTURES}
    for pdb_id, description in all_structures.items():
        print(f"Fetching {pdb_id}: {description}...")
        path = fetch_pdb(pdb_id)
        print(f"  -> {path}")
    print("\nAll structures fetched.")


if __name__ == "__main__":
    main()
