"""Utility functions for PDB structure handling."""
from pathlib import Path
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
import numpy as np

STRUCTURES_DIR = Path(__file__).parent.parent.parent / "data" / "structures"


def fetch_pdb(pdb_id: str, output_dir: Path = STRUCTURES_DIR) -> Path:
    """Download a PDB file from RCSB."""
    import requests
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    out_path = output_dir / f"{pdb_id.upper()}.pdb"
    if out_path.exists():
        return out_path
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out_path.write_text(resp.text)
    return out_path


def extract_chain(pdb_path: Path, chain_id: str, output_path: Path) -> Path:
    """Extract a single chain from a PDB file, removing water and heteroatoms."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))

    class ChainSelect(Select):
        def accept_chain(self, chain):
            return chain.id == chain_id

        def accept_residue(self, residue):
            return is_aa(residue, standard=True)

    io = PDBIO()
    io.set_structure(structure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(output_path), ChainSelect())
    return output_path


def get_interface_residues(pdb_path: Path, chain_a: str, chain_b: str,
                           distance_cutoff: float = 5.0) -> dict:
    """Find interface residues between two chains within distance cutoff.

    Returns dict with keys 'chain_a_residues' and 'chain_b_residues',
    each a list of (resname, resid) tuples.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = structure[0]

    atoms_a = [a for a in model[chain_a].get_atoms()]
    atoms_b = [a for a in model[chain_b].get_atoms()]

    ns = NeighborSearch(atoms_b)

    interface_a = set()
    interface_b_atoms = set()
    for atom in atoms_a:
        nearby = ns.search(atom.coord, distance_cutoff)
        if nearby:
            res = atom.get_parent()
            if is_aa(res, standard=True):
                interface_a.add((res.get_resname(), res.id[1]))
            for nb in nearby:
                res_b = nb.get_parent()
                if is_aa(res_b, standard=True):
                    interface_b_atoms.add((res_b.get_resname(), res_b.id[1]))

    return {
        "chain_a_residues": sorted(interface_a, key=lambda x: x[1]),
        "chain_b_residues": sorted(interface_b_atoms, key=lambda x: x[1]),
    }


def cif_to_pdb(cif_path: Path, pdb_path: Path) -> Path:
    """Convert mmCIF to PDB format using BioPython.

    Required because AF3 outputs CIF but Rosetta/FoldX need PDB.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(pdb_path))
    return pdb_path


def extract_sequence_from_pdb(pdb_path: Path, chain_id: str = None) -> str:
    """Extract amino acid sequence from a PDB file."""
    from Bio.PDB.Polypeptide import PPBuilder
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    ppb = PPBuilder()

    if chain_id:
        chain = structure[0][chain_id]
        peptides = ppb.build_peptides(chain)
    else:
        peptides = ppb.build_peptides(structure[0])

    seq = ""
    for pp in peptides:
        seq += str(pp.get_sequence())
    return seq
