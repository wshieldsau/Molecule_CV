# Session Notes

Untracked working notes. Keep out of git unless explicitly added.

## fANOVA — Hutter, Hoos, Leyton-Brown 2014

Citation: Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2014). "An Efficient
Approach for Assessing Hyperparameter Importance." ICML 2014.

**What it does:** Given a set of observed (hparams, objective) pairs from
any tuner, fits a surrogate model (random forest) over the hparam space,
then decomposes the surrogate's predicted variance into contributions from
individual hparams and their pairwise interactions via functional ANOVA.

**Output:** importance score per hparam (and per interaction term) that
sums to 1. Interpretable as "fraction of performance variance attributable
to this knob." Unlike simple sensitivity analysis, it accounts for the
interactions between hparams and works in unevenly-sampled spaces (which
Bayesian optimization always produces — it oversamples good regions).

**Why it's more principled than parallel-coords eyeballing:** a parallel
coordinates plot can look like "lr matters a lot" when actually the tuner
happened to try bad values of lr only when dropout was also high. fANOVA
conditions on the full surrogate, so it doesn't get fooled by correlated
samples.

**Implementations:**
- Original: `fanova` Python package — github.com/automl/fanova
- Optuna bundles it: `optuna.importance.FanovaImportanceEvaluator`
- The core algorithm is ~50 lines of sklearn if avoiding the dep:
  1. Fit `RandomForestRegressor` on (hparams, val_mae)
  2. For each hparam, integrate the forest's prediction over all other
     hparams, then take variance of the resulting 1D function
  3. Normalize against total variance of the full surrogate prediction

**Input from current run:** keras-tuner writes per-trial JSONs to
`keras_tuner_runs/<model>/trial_<n>/trial.json`. Parse those into a long
dataframe (one row per trial with all hparams + best val_mae), feed to a
fANOVA implementation.

**Notebook angle:** per-model fANOVA tables would be a strong figure for
the writeup — shows which knobs actually drive performance for each
architecture, which is the kind of diagnostic a stats grader would
appreciate beyond "here's the best config."
