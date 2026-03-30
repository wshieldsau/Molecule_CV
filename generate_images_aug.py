# generate_images_aug.py
#
# Generate augmented 2D molecular structure PNGs from SMILES strings in bace.csv.
#
# Loops over combinations of RDKit drawing options to produce multiple
# visual variants of each molecule.
#
# Augmentation parameters:
#   a (annotations): all on or all off — kekulize, stereo, atom indices,
#                    explicit methyl, bond indices
#   b (rotation):    0, 90, 180, 270 degrees
#   c (bond width):  1.0 or 3.0
#   d (style):       comic, bw palette, black background, or none
#
# Usage:
#   source .venv/bin/activate
#   python generate_images_aug.py

import csv
import itertools
import os

from rdkit import Chem
from rdkit.Chem import Draw

# --- Configuration ---
CSV_PATH = "bace.csv"
OUTPUT_DIR = "molecule_images_aug"
IMAGE_SIZE = (400, 400)
SMILES_COL = "mol"
ID_COL = "CID"

# --- Augmentation grid ---
A_VALUES = [False, True]          # annotations on/off
B_VALUES = [0, 90, 180, 270]     # rotation degrees
C_VALUES = [1.0, 3.0]            # bond line width
D_VALUES = ["none", "comic", "bw", "blackbg"]  # style

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(CSV_PATH, newline="") as f:
    molecules = [
        {"id": row[ID_COL], "smiles": row[SMILES_COL]}
        for row in csv.DictReader(f)
    ]

total = len(molecules)
combos = list(itertools.product(A_VALUES, B_VALUES, C_VALUES, D_VALUES))
print(f"Found {total} molecules, {len(combos)} variants each = {total * len(combos)} images")

generated = 0
skipped = 0
failed = 0

for i, mol_info in enumerate(molecules, start=1):
    mol = Chem.MolFromSmiles(mol_info["smiles"])
    if mol is None:
        failed += len(combos)
        print(f"  WARNING: Could not parse SMILES for {mol_info['id']}")
        continue

    for annot, rot, bw, style in combos:
        tag = f"a{int(annot)}_r{rot}_w{bw:.0f}_{style}"
        filename = f"{mol_info['id']}_{tag}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(output_path):
            skipped += 1
            continue

        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(*IMAGE_SIZE)
        opts = drawer.drawOptions()

        # a: annotations
        opts.addStereoAnnotation = annot
        opts.addAtomIndices = annot
        opts.explicitMethyl = annot
        opts.addBondIndices = annot

        # b: rotation
        opts.rotate = rot

        # c: bond line width
        opts.bondLineWidth = bw

        # d: style
        if style == "comic":
            opts.comicMode = True
        elif style == "bw":
            opts.useBWAtomPalette()
        elif style == "blackbg":
            opts.setBackgroundColour((0, 0, 0, 1))

        # kekulize variant (part of a)
        if annot:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=False)
            except Exception:
                pass
            drawer.drawOptions().prepareMolsBeforeDrawing = False

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        with open(output_path, "wb") as f:
            f.write(drawer.GetDrawingText())

        generated += 1

    if i % 100 == 0:
        print(f"  Progress: {i}/{total} molecules processed, {generated} generated")

print(f"\nDone! Generated: {generated}, Skipped: {skipped}, Failed: {failed}")
print(f"Images saved to: {OUTPUT_DIR}/")
