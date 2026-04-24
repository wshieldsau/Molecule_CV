# Model Registry Proposal

A lightweight harness for running multiple CNN architecture variants on
BACE reproducibly. Each variant is a self-contained `.py` file; one
runner iterates over them and writes per-model results.

## Directory Layout

```
Molecule_CV/
├── models/
│   ├── __init__.py
│   ├── _data.py                  # shared: load images, split, z-score target
│   ├── admet_fixed.py            # one architecture + metadata per file
│   ├── admet_tuned.py
│   ├── toxic_colors_fixed.py
│   ├── toxic_colors_no_bn.py     # ablation: drop BN
│   ├── custom_cnn_fixed.py
│   └── custom_cnn_300px.py       # ablation: larger IMG_SIZE
├── run_all.py                    # iterate registry, skip completed
├── run_one.py                    # run a single model by name
└── molecule_cv_results/
    ├── admet_fixed/
    │   ├── metrics.json
    │   ├── predictions.csv
    │   ├── history.csv
    │   └── config.json
    ├── admet_tuned/
    ├── toxic_colors_fixed/
    └── ...
```

## Contract

Every model file exposes three things: `META`, `TRAIN`, `build(input_shape)`.

Example `models/admet_fixed.py`:

```python
"""ADMET Fixed — Shi 2019 with he_normal init, LeakyReLU, BatchNorm."""
from tensorflow import keras

META = {
    "name": "admet_fixed",
    "family": "admet",
    "description": "Shi 2019 with modern init/activation fixes",
    "paper": "https://doi.org/...",
    "changes_from_baseline": ["he_normal", "LeakyReLU(0.1)", "BatchNorm after conv1"],
}

TRAIN = {
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "img_size": (180, 180),
    "standardize_target": True,
}

def build(input_shape):
    return keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(16, (21, 21), kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.1),
        keras.layers.MaxPooling2D((14, 14), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(0.1),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1),
    ])
```

## Runner

`run_all.py` (~80 lines) iterates the registry:

```python
for module in discover_models("models/"):
    out = results_dir / module.META["name"]
    if (out / "metrics.json").exists():
        continue                      # idempotent; skip completed runs
    X_train, y_train, ... = load_data(module.TRAIN["img_size"])
    model = module.build(input_shape)
    model.compile(Adam(module.TRAIN["learning_rate"]), 'mse', ['mae'])
    history = model.fit(X_train, y_train_z, ...)
    preds = invert_zscore(model.predict(X_test))
    save_metrics(out, preds, y_test, c_test, module.META, module.TRAIN)
```

## Why This Is Worth It

- **Idempotent.** Re-running `run_all.py` only trains what is missing.
  Adding a new ablation means dropping a new file in `models/`; previous
  runs are not touched.
- **Each `.py` is self-documenting.** `META["changes_from_baseline"]` is
  the ablation log. Anyone reading `admet_no_bn.py` can see what it tests.
- **Stats layer is downstream.** `stats_comparison.ipynb` reads
  `results/*/predictions.csv` and does not care how they were produced.
  Wilcoxon, DeLong, residual plots, bootstrapped CIs can be added without
  touching training code.
- **Plays well with AI assist.** "Copy `admet_fixed.py` to `admet_300px.py`
  and change `img_size` to (300, 300)" is a one-shot ask. The author
  still owns the scientific decision; the file is small and reviewable.
- **Fits the "don't go crazy" constraint.** The whole harness is
  ~100 lines of glue, smaller than the current three training notebooks
  combined. It replaces copy-paste rather than adding complexity.

## What It Does Not Do

- No parallel runs, no GPU scheduling, no experiment UI. One Python
  process, sequential, stdout. If that becomes insufficient, MLflow or
  Weights & Biases slot in at the `save_metrics` boundary.
- No agentic architecture search. The author writes the ablations
  (optionally with AI help on individual files). The registry records
  what was tried.

## Open Questions for Teammate

- Is this in scope given "don't go crazy"?
- Should tuned models (from `tune.py`) be committed as separate registry
  entries, or is the best-hparams JSON enough?
- Do we want a CLI flag for IMG_SIZE=300 overrides, or is one file per
  variant the right granularity?
