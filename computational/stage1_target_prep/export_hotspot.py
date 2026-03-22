"""Export hotspot definitions in formats required by each AI design tool,
and extract BMPR1A ECD FASTA sequence for AF3/PepMLM inputs."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pdb_utils import STRUCTURES_DIR, extract_sequence_from_pdb
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def main():
    interface_path = STRUCTURES_DIR / "bmpr1a_interface_residues.json"
    interface = json.loads(interface_path.read_text())

    bmpr1a_chain = interface["bmpr1a_chain"]
    hotspot_resnums = [r[1] for r in interface["bmpr1a_interface_residues"]]

    # --- Extract and save BMPR1A ECD FASTA ---
    bmpr1a_pdb = STRUCTURES_DIR / "cleaned" / "1REW_BMPR1A.pdb"
    bmpr1a_seq = extract_sequence_from_pdb(bmpr1a_pdb)
    fasta_path = STRUCTURES_DIR / "bmpr1a_ecd_sequence.fasta"
    record = SeqRecord(Seq(bmpr1a_seq), id="BMPR1A_ECD_1REW",
                       description=f"BMPR1A ECD from 1REW chain {bmpr1a_chain}, {len(bmpr1a_seq)} residues")
    SeqIO.write([record], str(fasta_path), "fasta")
    print(f"BMPR1A ECD sequence: {len(bmpr1a_seq)} residues")
    print(f"Saved to {fasta_path}")
    print(f"Sequence: {bmpr1a_seq}\n")

    # --- Extract sequences for specificity counter-screen ---
    # BMPR1B from 3EVS
    bmpr1b_pdb = STRUCTURES_DIR / "cleaned" / "3EVS_BMPR1B.pdb"
    bmpr1b_seq = extract_sequence_from_pdb(bmpr1b_pdb)
    bmpr1b_fasta = STRUCTURES_DIR / "bmpr1b_ecd_sequence.fasta"
    SeqIO.write([SeqRecord(Seq(bmpr1b_seq), id="BMPR1B_ECD_3EVS",
                           description=f"BMPR1B ECD from 3EVS, {len(bmpr1b_seq)} residues")],
                str(bmpr1b_fasta), "fasta")
    print(f"BMPR1B ECD sequence: {len(bmpr1b_seq)} residues -> {bmpr1b_fasta}")

    # ACVR1 ECD from UniProt Q04771 (residues 21-123)
    # Since 3MTF only has the kinase domain, use the canonical UniProt sequence
    acvr1_ecd_seq = (
        "ETRECIYYNANWELERTNQSGLERCEGEQDKRLHCYASWRNSSGTIELVKK"
        "GCWLDDFNCYDRQECVATEENPQVYFCCCEGNFCNERFTHLPEAGGPE"
    )
    acvr1_fasta = STRUCTURES_DIR / "acvr1_ecd_sequence.fasta"
    SeqIO.write([SeqRecord(Seq(acvr1_ecd_seq), id="ACVR1_ECD_UniProt_Q04771",
                           description=f"ACVR1 (ALK2) ECD from UniProt Q04771 res 21-123, {len(acvr1_ecd_seq)} residues")],
                str(acvr1_fasta), "fasta")
    print(f"ACVR1 ECD sequence: {len(acvr1_ecd_seq)} residues -> {acvr1_fasta}")

    # --- Extract 2H62 sequences for ternary assembly ---
    actrii_pdb = STRUCTURES_DIR / "cleaned" / "2H62_ActRII.pdb"
    actrii_seq = extract_sequence_from_pdb(actrii_pdb)
    actrii_fasta = STRUCTURES_DIR / "actrii_ecd_sequence.fasta"
    SeqIO.write([SeqRecord(Seq(actrii_seq), id="ActRII_ECD_2H62",
                           description=f"ActRII ECD from 2H62, {len(actrii_seq)} residues")],
                str(actrii_fasta), "fasta")
    print(f"ActRII ECD sequence: {len(actrii_seq)} residues -> {actrii_fasta}")

    # --- Tool-specific hotspot configs ---
    rf_hotspot = [f"{bmpr1a_chain}{r}" for r in hotspot_resnums]
    configs = {
        "rfdiffusion": {
            "target_pdb": "data/structures/cleaned/1REW_BMPR1A.pdb",
            "hotspot_residues": rf_hotspot,
            "contig_note": "Use these as --hotspot_res in RFdiffusion contigmap",
        },
        "pepmlm": {
            "target_structure": "data/structures/cleaned/1REW_BMPR1A.pdb",
            "target_sequence": bmpr1a_seq,
            "binding_site_residues": hotspot_resnums,
        },
        "bindcraft": {
            "target_pdb": "data/structures/cleaned/1REW_BMPR1A.pdb",
            "target_chain": bmpr1a_chain,
            "interface_residues": hotspot_resnums,
        },
        "rfpeptides": {
            "target_pdb": "data/structures/cleaned/1REW_BMPR1A.pdb",
            "hotspot_residues": rf_hotspot,
        },
    }

    out_dir = STRUCTURES_DIR / "hotspot_configs"
    out_dir.mkdir(exist_ok=True)
    for name, config in configs.items():
        path = out_dir / f"{name}_config.json"
        path.write_text(json.dumps(config, indent=2))
        print(f"Exported {name} config -> {path}")


if __name__ == "__main__":
    main()
