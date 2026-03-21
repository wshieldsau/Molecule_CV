# generate_images.py
#
# Generate 2D molecular structure PNGs from SMILES strings in bace.csv.
#
# Reads the BACE dataset, converts each SMILES string to a 2D molecular
# image using RDKit, and saves it as a PNG. Uses the CID column (e.g.
# "BACE_1") as the filename for easy identification.
#
# Already-generated images are skipped, so you can resume after interruption.
#
# Usage:
#   source .venv/bin/activate
#   python generate_images.py

import csv
import os

from rdkit import Chem
from rdkit.Chem import Draw

# --- Configuration ---
CSV_PATH = "bace.csv"
OUTPUT_DIR = "molecule_images"
IMAGE_SIZE = (400, 400)  # width x height in pixels
SMILES_COL = "mol"       # column containing SMILES strings
ID_COL = "CID"           # column used for filenames (e.g. BACE_1)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all molecules from the CSV
with open(CSV_PATH, newline="") as f:
    molecules = [
        {"id": row[ID_COL], "smiles": row[SMILES_COL]}
        for row in csv.DictReader(f)
    ]

total = len(molecules)
print(f"Found {total} molecules in {CSV_PATH}")

# Track progress
generated = 0
skipped = 0
failed = 0

for i, mol_info in enumerate(molecules, start=1):
    filename = f"{mol_info['id']}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Skip if image already exists (enables resume after interruption)
    if os.path.exists(output_path):
        skipped += 1
        continue

    # Convert SMILES to molecule and render as image
    mol = Chem.MolFromSmiles(mol_info["smiles"])
    if mol is None:
        failed += 1
        print(f"  WARNING: Could not parse SMILES for {mol_info['id']}")
        continue

    img = Draw.MolToImage(mol, size=IMAGE_SIZE)
    img.save(output_path)
    generated += 1

    # Print progress every 100 molecules
    if generated % 100 == 0:
        print(f"  Progress: {i}/{total} processed, {generated} generated")

# Final summary
print(f"\nDone! Generated: {generated}, Skipped (existing): {skipped}, Failed: {failed}")
print(f"Images saved to: {OUTPUT_DIR}/")
