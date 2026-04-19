"""Hyperparameter tuning for all three BACE CNN architectures.

Uses keras-tuner Bayesian optimization to search over learning rate,
dropout, and a few architecture knobs per model. For each model:
  1. Search `MAX_TRIALS` configurations at `TUNE_EPOCHS` epochs each
  2. Refit the best config at `FINAL_EPOCHS` on z-scored pIC50
  3. Save test metrics, predictions, best hparams, and training history

Writes to molecule_cv_results_tuned/. Architectures mirror the
"fixed" variants (he_normal + LeakyReLU + BatchNorm) so tuning
builds on the revived baselines, not the collapsed originals.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = HERE
IMG_DIR = os.path.join(DATA_DIR, 'molecule_images')
OUTPUT_DIR = os.path.join(DATA_DIR, 'molecule_cv_results_tuned')
TUNER_DIR = os.path.join(DATA_DIR, 'keras_tuner_runs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (180, 180)
MAX_TRIALS = 10       # Bayesian trials per model
TUNE_EPOCHS = 15      # epochs per trial (short; just to rank configs)
FINAL_EPOCHS = 100    # epochs for final retrain of best config
BATCH_SIZE = 64

print(f"TF: {tf.__version__} | keras-tuner: {kt.__version__}")
print(f"Devices: {tf.config.list_physical_devices()}")


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

INPUT_SHAPE = IMG_SIZE + (3,)


def _head(hp, flat):
    """Shared dense head with tunable dropout."""
    drop = hp.Float('dropout', 0.2, 0.6, step=0.1)
    dense_units = hp.Choice('dense_units', [64, 128, 256, 512])
    x = keras.layers.Dense(dense_units, kernel_initializer='he_normal')(flat)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Dropout(drop)(x)
    return keras.layers.Dense(1)(x)


def _compile(model, hp):
    lr = hp.Float('lr', 1e-5, 1e-3, sampling='log')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae'],
    )
    return model


def build_admet_hp(hp):
    filters = hp.Choice('filters', [8, 16, 32])
    kernel = hp.Choice('kernel_size', [15, 21, 27])
    inp = keras.layers.Input(shape=INPUT_SHAPE)
    x = keras.layers.Conv2D(filters, (kernel, kernel),
                            kernel_initializer='he_normal')(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.MaxPooling2D((14, 14), padding='same')(x)
    x = keras.layers.Flatten()(x)
    out = _head(hp, x)
    return _compile(keras.Model(inp, out), hp)


def build_toxic_colors_hp(hp):
    filters = hp.Choice('filters', [8, 12, 24])
    kernel = hp.Choice('kernel_size', [11, 16, 21])
    use_bn = hp.Boolean('use_bn')
    inp = keras.layers.Input(shape=INPUT_SHAPE)
    x = keras.layers.Conv2D(filters, (kernel, kernel),
                            kernel_initializer='he_normal')(inp)
    if use_bn:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Dropout(hp.Float('conv_dropout', 0.2, 0.5, step=0.1))(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    out = _head(hp, x)
    return _compile(keras.Model(inp, out), hp)


def build_custom_cnn_hp(hp):
    base = hp.Choice('base_filters', [16, 32, 64])
    inp = keras.layers.Input(shape=INPUT_SHAPE)
    x = keras.layers.Conv2D(base, (3, 3), kernel_initializer='he_normal')(inp)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(base * 2, (3, 3), kernel_initializer='he_normal')(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(base * 4, (3, 3), kernel_initializer='he_normal')(x)
    x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    if hp.Boolean('extra_block'):
        x = keras.layers.Conv2D(base * 8, (3, 3),
                                kernel_initializer='he_normal')(x)
        x = keras.layers.LeakyReLU(0.1)(x)
    x = keras.layers.Flatten()(x)
    out = _head(hp, x)
    return _compile(keras.Model(inp, out), hp)


MODELS = {
    'admet': build_admet_hp,
    'toxic_colors': build_toxic_colors_hp,
    'custom_cnn': build_custom_cnn_hp,
}


def summarize(name, model):
    preds_z = model.predict(X_test, verbose=0).flatten()
    preds = preds_z * Y_STD + Y_MEAN
    mse = float(np.mean((preds - y_test) ** 2))
    mae = float(np.mean(np.abs(preds - y_test)))
    auc = float(roc_auc_score(c_test, preds))
    return preds, {'model': name, 'test_mse': mse, 'test_mae': mae,
                   'auc_roc': round(auc, 4),
                   'pred_min': float(preds.min()),
                   'pred_max': float(preds.max()),
                   'unique_preds': int(len(np.unique(preds.round(4))))}


all_metrics = []
pred_df = pd.DataFrame({'CID': cid_test, 'pIC50_actual': y_test, 'Class': c_test})
best_hps = {}

for name, builder in MODELS.items():
    print(f"\n{'='*60}\nTuning: {name}\n{'='*60}")
    tuner = kt.BayesianOptimization(
        builder,
        objective='val_mae',
        max_trials=MAX_TRIALS,
        seed=SEED,
        directory=TUNER_DIR,
        project_name=name,
        overwrite=True,
    )
    tuner.search(
        X_train, y_train_z,
        validation_data=(X_val, y_val_z),
        epochs=TUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_hps[name] = best_hp.values
    print(f"Best hparams for {name}: {best_hp.values}")

    print(f"Refitting best {name} for {FINAL_EPOCHS} epochs...")
    model = builder(best_hp)
    history = model.fit(
        X_train, y_train_z,
        validation_data=(X_val, y_val_z),
        epochs=FINAL_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    preds, metrics = summarize(name, model)
    metrics['best_hparams'] = json.dumps(best_hp.values)
    all_metrics.append(metrics)
    pred_df[f'pred_{name}'] = preds

    hist_df = pd.DataFrame(history.history)
    hist_df.index.name = 'epoch'
    hist_df.to_csv(os.path.join(OUTPUT_DIR, f'history_{name}_tuned.csv'))
    print(f"{name}: MSE={metrics['test_mse']:.3f} MAE={metrics['test_mae']:.3f} AUC={metrics['auc_roc']}")

pred_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions_tuned.csv'), index=False)
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_metrics_tuned.csv'), index=False)
print("\nFinal metrics:")
print(metrics_df.to_string(index=False))

with open(os.path.join(OUTPUT_DIR, 'best_hparams.json'), 'w') as f:
    json.dump(best_hps, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'run_config.json'), 'w') as f:
    json.dump({
        'img_size': list(IMG_SIZE),
        'max_trials': MAX_TRIALS,
        'tune_epochs': TUNE_EPOCHS,
        'final_epochs': FINAL_EPOCHS,
        'batch_size': BATCH_SIZE,
        'seed': SEED,
        'y_mean': Y_MEAN,
        'y_std': Y_STD,
        'tf_version': tf.__version__,
        'keras_tuner_version': kt.__version__,
    }, f, indent=2)

print("\nDone.")
