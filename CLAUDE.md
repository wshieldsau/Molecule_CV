# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Molecule_CV is a Python cheminformatics project for molecular visualization and analysis using the BACE (Beta-Secretase) inhibitor dataset. It uses RDKit to convert SMILES chemical notation strings into 2D molecular structure images.

## Development Environment

- **Python**: 3.12 via virtual environment managed with `uv`
- **Key dependencies**: rdkit, pandas, numpy, pillow

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the main script
python image_creation.py
```

## Core Workflow

SMILES string → `Chem.MolFromSmiles()` → `Draw.MolToImage()` → PNG output

## Data

`bace.csv` contains 1,514 molecular compound records with 595 columns including SMILES structures (`mol` column), potency measures (`pIC50`), physicochemical properties (`MW`, `AlogP`, `HBA`, `HBD`), and molecular descriptors.
