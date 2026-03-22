"""Map BMP-2:BMPR1A interface residues from 1REW — the gold-standard contact set."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pdb_utils import get_interface_residues, STRUCTURES_DIR

# Chain IDs verified from PDB inspection
BMP2_CHAIN = "A"
BMPR1A_CHAIN = "C"
DISTANCE_CUTOFF = 5.0  # Angstroms, standard for interface definition


def main():
    pdb_path = STRUCTURES_DIR / "1REW.pdb"
    print(f"Mapping BMP-2 (chain {BMP2_CHAIN}) : BMPR1A (chain {BMPR1A_CHAIN}) "
          f"interface at {DISTANCE_CUTOFF} A cutoff...")

    interface = get_interface_residues(
        pdb_path,
        chain_a=BMP2_CHAIN,
        chain_b=BMPR1A_CHAIN,
        distance_cutoff=DISTANCE_CUTOFF,
    )

    result = {
        "pdb_id": "1REW",
        "bmp2_chain": BMP2_CHAIN,
        "bmpr1a_chain": BMPR1A_CHAIN,
        "distance_cutoff_angstroms": DISTANCE_CUTOFF,
        "bmp2_interface_residues": interface["chain_a_residues"],
        "bmpr1a_interface_residues": interface["chain_b_residues"],
        "num_bmp2_contacts": len(interface["chain_a_residues"]),
        "num_bmpr1a_contacts": len(interface["chain_b_residues"]),
    }

    out_path = STRUCTURES_DIR / "bmpr1a_interface_residues.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nInterface mapped:")
    print(f"  BMP-2 interface residues: {result['num_bmp2_contacts']}")
    print(f"  BMPR1A interface residues: {result['num_bmpr1a_contacts']}")
    print(f"  Saved to {out_path}")

    # Print residues for visual verification in PyMOL
    bmpr1a_resnums = [r[1] for r in interface["chain_b_residues"]]
    bmp2_resnums = [r[1] for r in interface["chain_a_residues"]]
    print(f"\nPyMOL commands for visual verification:")
    print(f"  select bmpr1a_hotspot, chain {BMPR1A_CHAIN} and resi "
          f"{'+'.join(map(str, bmpr1a_resnums))}")
    print(f"  select bmp2_interface, chain {BMP2_CHAIN} and resi "
          f"{'+'.join(map(str, bmp2_resnums))}")
    print(f"  color red, bmpr1a_hotspot")
    print(f"  color blue, bmp2_interface")

    # Print the actual residue identities
    print(f"\nBMPR1A hotspot residues:")
    for resname, resid in interface["chain_b_residues"]:
        print(f"  {resname} {resid}")


if __name__ == "__main__":
    main()
