"""Compute per-model summary statistics from a tune.py results directory.

Reads:
  history_<model>_tuned.csv   per-epoch train/val/test loss from final refit
  predictions_tuned.csv       per-row test predictions (one column per model)
  model_metrics_tuned.csv     final test MSE/MAE/AUC per model

Writes:
  analysis_summary.csv        one row per model with refit and prediction stats

Usage:
  python analyze_results.py [--results-dir DIR]

DIR defaults to molecule_cv_results_tuned_v3.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


def refit_summary(history_df):
    n = len(history_df)
    half = n // 2
    bv = history_df['val_loss'].idxmin()
    bt = history_df['test_loss'].idxmin()
    return {
        'refit_n_epochs': n,
        'best_val_epoch': int(history_df['epoch'].iloc[bv]),
        'val_loss_at_best_val': float(history_df['val_loss'].iloc[bv]),
        'test_loss_at_best_val': float(history_df['test_loss'].iloc[bv]),
        'best_test_epoch': int(history_df['epoch'].iloc[bt]),
        'val_loss_at_best_test': float(history_df['val_loss'].iloc[bt]),
        'test_loss_at_best_test': float(history_df['test_loss'].iloc[bt]),
        'epoch_gap_val_vs_test': int(abs(history_df['epoch'].iloc[bv] -
                                         history_df['epoch'].iloc[bt])),
        'mean_val_minus_test_2ndhalf': float(
            (history_df['val_loss'].iloc[half:] -
             history_df['test_loss'].iloc[half:]).mean()),
        'final_train_loss': float(history_df['loss'].iloc[-1]),
        'final_val_loss': float(history_df['val_loss'].iloc[-1]),
        'final_test_loss': float(history_df['test_loss'].iloc[-1]),
    }


def prediction_summary(preds, y_true, y_class):
    s = pd.Series(preds)
    pearson_r, pearson_p = pearsonr(preds, y_true)
    spearman_r, spearman_p = spearmanr(preds, y_true)
    return {
        'pred_min': float(s.min()),
        'pred_max': float(s.max()),
        'pred_std': float(s.std()),
        'pred_n_unique': int(s.nunique()),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'auc_vs_class': float(roc_auc_score(y_class, preds)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir',
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'molecule_cv_results_tuned_v3'))
    args = p.parse_args()

    rd = args.results_dir
    metrics = pd.read_csv(os.path.join(rd, 'model_metrics_tuned.csv'))
    preds = pd.read_csv(os.path.join(rd, 'predictions_tuned.csv'))
    with open(os.path.join(rd, 'best_hparams.json')) as f:
        best_hp = json.load(f)

    rows = []
    for model in best_hp.keys():
        row = {'model': model}

        m = metrics[metrics['model'] == model]
        if len(m) == 1:
            for k in ['test_mse', 'test_mae', 'auc_roc']:
                row[k] = float(m.iloc[0][k])

        hist_path = os.path.join(rd, f'history_{model}_tuned.csv')
        if os.path.exists(hist_path):
            row.update(refit_summary(pd.read_csv(hist_path)))

        pred_col = f'pred_{model}'
        if pred_col in preds.columns:
            row.update(prediction_summary(
                preds[pred_col].values,
                preds['pIC50_actual'].values,
                preds['Class'].values,
            ))
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = os.path.join(rd, 'analysis_summary.csv')
    out.to_csv(out_path, index=False)

    print(f"wrote {out_path}\n")
    with pd.option_context('display.max_columns', None,
                           'display.width', 200,
                           'display.float_format', lambda x: f'{x:.4f}'):
        print(out.to_string(index=False))


if __name__ == '__main__':
    main()
