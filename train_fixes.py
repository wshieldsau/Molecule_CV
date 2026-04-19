"""Train Fixes — Revive ADMET and Toxic Colors (local run).

Python port of train_fixes.ipynb with Colab/Drive plumbing stripped out.
Reads data from the project root and writes to ./molecule_cv_results_fixed/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = HERE
IMG_DIR = os.path.join(DATA_DIR, 'molecule_images')
OUTPUT_DIR = os.path.join(DATA_DIR, 'molecule_cv_results_fixed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (180, 180)
EPOCHS = 100
BATCH_SIZE = 128

print(f"TF: {tf.__version__}")
print(f"Devices: {tf.config.list_physical_devices()}")
print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")


def load_images_and_targets(df, img_dir, img_size=IMG_SIZE):
    images, pic50s, classes, cids = [], [], [], []
    missing = 0
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, f"{row['CID']}.png")
        if not os.path.exists(img_path):
            missing += 1
            continue
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        images.append(img.numpy())
        pic50s.append(row['pIC50'])
        classes.append(row['Class'])
        cids.append(row['CID'])
    if missing:
        print(f"Warning: {missing} images not found")
    return np.array(images), np.array(pic50s), np.array(classes), cids


df = pd.read_csv(os.path.join(DATA_DIR, 'bace.csv'))
images, pic50, labels, cids = load_images_and_targets(df, IMG_DIR)
print(f"Loaded {len(images)} images at {IMG_SIZE}")

X_temp, X_test, y_temp, y_test, c_temp, c_test, cid_temp, cid_test = train_test_split(
    images, pic50, labels, cids, test_size=0.2, random_state=SEED)
X_train, X_val, y_train, y_val, c_train, c_val, cid_train, cid_val = train_test_split(
    X_temp, y_temp, c_temp, cid_temp, test_size=0.5, random_state=SEED)

Y_MEAN = float(y_train.mean())
Y_STD = float(y_train.std())
y_train_z = (y_train - Y_MEAN) / Y_STD
y_val_z = (y_val - Y_MEAN) / Y_STD
y_test_z = (y_test - Y_MEAN) / Y_STD

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"pIC50 train stats — mean: {Y_MEAN:.3f}, std: {Y_STD:.3f}")


def build_admet_fixed(input_shape=IMG_SIZE + (3,)):
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
        keras.layers.Dense(1)
    ])


def build_toxic_colors_fixed(input_shape=IMG_SIZE + (3,)):
    return keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(12, (16, 16), kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(0.1),
        keras.layers.Dropout(0.4),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(40, kernel_initializer='he_normal'),
        keras.layers.LeakyReLU(0.1),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1)
    ])


MODELS = {
    'ADMET Fixed': build_admet_fixed,
    'Toxic Colors Fixed': build_toxic_colors_fixed,
}

results = {}

for name, builder in MODELS.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    model = builder()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae'],
    )

    history = model.fit(
        X_train, y_train_z,
        validation_data=(X_val, y_val_z),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    preds_z = model.predict(X_test, verbose=0).flatten()
    preds = preds_z * Y_STD + Y_MEAN

    test_mae = float(np.mean(np.abs(preds - y_test)))
    test_mse = float(np.mean((preds - y_test) ** 2))

    results[name] = {
        'history': history.history,
        'predictions': preds,
        'test_loss': test_mse,
        'test_mae': test_mae,
    }
    print(f"\n{name} — Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
    print(f"  Prediction range: {preds.min():.4f} to {preds.max():.4f}")
    print(f"  Unique predictions: {len(np.unique(preds.round(4)))}")

    col = name.lower().replace(' ', '_')
    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = 'epoch'
    hist_df.to_csv(os.path.join(OUTPUT_DIR, f'history_{col}.csv'))

pred_df = pd.DataFrame({'CID': cid_test, 'pIC50_actual': y_test, 'Class': c_test})
for name, res in results.items():
    col = name.lower().replace(' ', '_')
    pred_df[f'pred_{col}'] = res['predictions']
pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions_fixed.csv'), index=False)
print(f"\nSaved predictions_fixed.csv ({len(pred_df)} rows)")

metrics_rows = []
for name, res in results.items():
    auc = roc_auc_score(c_test, res['predictions'])
    metrics_rows.append({
        'model': name,
        'test_mse': res['test_loss'],
        'test_mae': res['test_mae'],
        'auc_roc': round(auc, 4),
    })
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_metrics_fixed.csv'), index=False)
print("Saved model_metrics_fixed.csv")
print(metrics_df.to_string(index=False))

with open(os.path.join(OUTPUT_DIR, 'run_config.json'), 'w') as f:
    json.dump({
        'img_size': list(IMG_SIZE),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'seed': SEED,
        'y_mean': Y_MEAN,
        'y_std': Y_STD,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'tf_version': tf.__version__,
    }, f, indent=2)

print("\nDone.")
