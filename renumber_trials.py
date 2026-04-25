"""Renumber the `trial` column in tuning_history_<model>.csv.

PerTrialLogger in tune.py has a latent bug: keras-tuner deepcopies the
callback per trial, which resets self.trial to -1 each time, so every trial
ends up logged as trial=0. Trial boundaries are still recoverable because
`epoch` resets to 0 at the start of each trial.

This script reads a tuning_history file, derives the correct trial numbers
from epoch-reset boundaries, and writes a sibling `<name>_renumbered.csv`.
It does not modify the original (so it is safe to run while tune.py is
still appending to a file).

Usage:
    python renumber_trials.py path/to/tuning_history_<model>.csv [...]
"""

import os
import sys
import pandas as pd


def renumber(df: pd.DataFrame) -> pd.DataFrame:
    new_trial = (df['epoch'] == 0).cumsum() - 1
    out = df.copy()
    out['trial'] = new_trial.clip(lower=0)
    return out


def main(paths):
    for path in paths:
        df = pd.read_csv(path)
        before = df['trial'].nunique()
        fixed = renumber(df)
        after = fixed['trial'].nunique()
        base, ext = os.path.splitext(path)
        out = base + '_renumbered' + ext
        fixed.to_csv(out, index=False)
        print(f"[ok] {path}: trials {before} -> {after}; rows={len(df)} -> {out}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit("usage: python renumber_trials.py <csv> [<csv> ...]")
    main(sys.argv[1:])
