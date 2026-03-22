# AI-Guided Design of Candidate BMPR1A-Binding Peptides for Cartilage Regeneration

A multi-tool computational benchmarking study comparing four generative AI protein design tools for designing peptide binders targeting the BMPR1A extracellular domain.

**Preprint:** [bioRxiv link coming soon]

## Overview

This repository contains the complete computational pipeline, data, and figures for a study benchmarking **RFdiffusion**, **BindCraft**, **PepMLM**, and **RFpeptides** for designing BMPR1A-binding peptides aimed at cartilage regeneration. 192 candidate peptides and 98 controls (290 total) were evaluated using:

- **AlphaFold 3** structure prediction (ipTM, ranking score, PAE)
- **PyRosetta** binding energy (dG_separated)
- **FoldX** binding energy (ΔG)
- **Contact recapitulation** against the crystallographic BMP-2:BMPR1A interface (PDB: 1REW)

A four-metric composite ranking identified **pepmlm_L15_0026** (15 residues) as the top candidate, and physicochemical filtering yielded a shortlist of 54 candidates.

## Repository Structure

```
├── computational/             # Complete pipeline scripts
│   ├── stage1_target_prep/    # PDB fetching, chain extraction, interface mapping
│   ├── stage2_generation/     # Control generation, candidate merging
│   ├── stage3_validation/     # AF3 parsing, energy scoring, statistical analysis
│   └── utils/                 # Shared utilities
├── data/
│   ├── candidates/            # Designed peptide sequences (FASTA + metadata)
│   ├── controls/              # Scrambled and random control sequences
│   ├── structures/            # Target PDB files, interface residues, hotspot configs
│   ├── af3_inputs/            # AlphaFold 3 Server input JSONs (all 290 jobs)
│   └── results/               # Master scores, contact scores, top candidates
├── figures/                   # All manuscript figures (PDF + PNG)
│   └── structures/            # Interactive 3D viewers (HTML + py3Dmol)
```

## Key Results

| Tool | n | Mean ipTM | Mean FoldX ΔG | Mean Contact Recap | Top 10 |
|------|---|-----------|---------------|-------------------|--------|
| PepMLM | 96 | 0.547 | -8.4 | 0.249 | 6 |
| RFdiffusion | 64 | 0.546 | -10.0 | 0.194 | 3 |
| BindCraft | 24 | 0.595 | -12.8 | 0.224 | 1 |
| RFpeptides | 8 | 0.450 | -3.1 | 0.233 | 0 |

Designed candidates significantly outperformed controls on ipTM (p = 0.002) and FoldX ΔG (p < 0.001).

## Interactive 3D Structures

Open `figures/structures/index.html` in a browser to explore the top peptide-BMPR1A complexes in 3D (requires internet for py3Dmol CDN).

## Requirements

```
python >= 3.11
biopython >= 1.86
scipy >= 1.13
numpy >= 1.26
pandas >= 2.1
matplotlib >= 3.8
seaborn >= 0.13
```

Energy scoring additionally requires:
- **PyRosetta** (academic license from RosettaCommons)
- **FoldX** v5.1 (academic license from FoldX website)

## Citation

If you use this data or code, please cite:

> Ahmadov A, Ahmadov O. AI-guided design of candidate BMPR1A-binding peptides for cartilage regeneration: a multi-tool computational benchmarking study. *bioRxiv*. 2026. doi: [coming soon]

## License

This project is released under the MIT License. PDB structure files are from the RCSB Protein Data Bank (public domain).
