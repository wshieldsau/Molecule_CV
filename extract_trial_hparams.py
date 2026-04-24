"""Extract per-trial hparams from keras-tuner dirs and join with tuning history.

Reads each keras_tuner_runs_v2/<model>/trial_*/trial.json, writes:
  - trial_hparams_<model>.csv        one row per trial, hparams + score + status
  - tuning_history_with_hparams_<model>.csv   tuning_history joined on trial

Run after tuning finishes; independent of tune.py.
"""

import os
import json
import glob
import argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def extract_model(model_name, tuner_dir, history_path, out_dir):
    model_tuner_dir = os.path.join(tuner_dir, model_name)
    if not os.path.isdir(model_tuner_dir):
        print(f"[skip] {model_name}: no tuner dir at {model_tuner_dir}")
        return

    rows = []
    for trial_json in sorted(glob.glob(os.path.join(model_tuner_dir, 'trial_*', 'trial.json'))):
        with open(trial_json) as f:
            tj = json.load(f)
        row = {'trial': int(tj['trial_id']), 'status': tj.get('status')}
        row.update(tj.get('hyperparameters', {}).get('values', {}))
        row['score'] = tj.get('score')
        row['best_step'] = tj.get('best_step')
        rows.append(row)

    if not rows:
        print(f"[skip] {model_name}: no trials found")
        return

    hp_df = pd.DataFrame(rows).sort_values('trial').reset_index(drop=True)
    hp_path = os.path.join(out_dir, f'trial_hparams_{model_name}.csv')
    hp_df.to_csv(hp_path, index=False)
    print(f"[ok] {model_name}: {len(hp_df)} trials -> {hp_path}")

    if not os.path.exists(history_path):
        print(f"[warn] {model_name}: history not found at {history_path}, skipping join")
        return

    hist_df = pd.read_csv(history_path)
    joined = hist_df.merge(hp_df, on='trial', how='left')
    joined_path = os.path.join(out_dir, f'tuning_history_with_hparams_{model_name}.csv')
    joined.to_csv(joined_path, index=False)
    print(f"[ok] {model_name}: joined history -> {joined_path} ({len(joined)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tuner-dir', default=os.path.join(HERE, 'keras_tuner_runs_v2'))
    p.add_argument('--out-dir', default=os.path.join(HERE, 'molecule_cv_results_tuned_v2'))
    p.add_argument('--models', nargs='+', default=None,
                   help='Defaults to all subdirs of --tuner-dir')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models = args.models or sorted(
        d for d in os.listdir(args.tuner_dir)
        if os.path.isdir(os.path.join(args.tuner_dir, d))
    )

    for m in models:
        history_path = os.path.join(args.out_dir, f'tuning_history_{m}.csv')
        extract_model(m, args.tuner_dir, history_path, args.out_dir)


if __name__ == '__main__':
    main()
