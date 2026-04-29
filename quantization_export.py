from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ─────────────────────────────────────────────────────────────
# OPTIONAL
# ─────────────────────────────────────────────────────────────

try:
    import m2cgen as m2c
    HAS_M2C = True
except:
    HAS_M2C = False

try:
    import tensorflow as tf
    HAS_TF = True
except:
    HAS_TF = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except:
    HAS_LGBM = False


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

RANDOM_STATE = 42
ARDUINO_SRAM = 8 * 1024
ESP32_SRAM = 400 * 1024
OUT = Path("quant_outputs")
OUT.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADER (REAL ONLY)
# ─────────────────────────────────────────────────────────────

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if "label" not in df:
        raise ValueError("Dataset must contain 'label'")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def eval_model(model, X, y):
    pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        prob = pred

    return {
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, prob)
    }


# ─────────────────────────────────────────────────────────────
# ARDUINO EXPORT (C / m2cgen)
# ─────────────────────────────────────────────────────────────

def export_c(model, name, X_test, y_test):
    if not HAS_M2C:
        return {"error": "m2cgen missing"}

    metrics = eval_model(model, X_test, y_test)

    c_code = m2c.export_to_c(model)

    path = OUT / f"{name}.c"
    path.write_text(c_code)

    return {
        "model": name,
        "target": "Arduino Mega",
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "size_kb": path.stat().st_size / 1024
    }


# ─────────────────────────────────────────────────────────────
# MLP → TFLITE INT8 (REAL CALIBRATION)
# ─────────────────────────────────────────────────────────────

def mlp_to_tflite(model, X_calib):
    if not HAS_TF:
        raise RuntimeError("TensorFlow required")

    calib = X_calib[:200]

    def rep():
        for x in calib:
            yield [x.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def export_mlp(model, name, X_train, X_test, y_test):
    metrics = eval_model(model, X_test, y_test)

    tflite = mlp_to_tflite(model, X_train)

    path = OUT / f"{name}.tflite"
    path.write_bytes(tflite)

    return {
        "model": name,
        "target": "ESP32-C3",
        "size_kb": len(tflite)/1024,
        "f1": metrics["f1"]
    }


# ─────────────────────────────────────────────────────────────
# XGBOOST / LIGHTGBM (REAL SIZE + SRAM CHECK)
# ─────────────────────────────────────────────────────────────

def model_size_kb(model):
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    joblib.dump(model, tmp.name)
    return Path(tmp.name).stat().st_size / 1024


def export_tree(model, name, X_test, y_test):
    metrics = eval_model(model, X_test, y_test)

    size = model_size_kb(model)
    fits = size * 1024 < ESP32_SRAM

    return {
        "model": name,
        "target": "ESP32-C3",
        "size_kb": size,
        "fits_sram": fits,
        "f1": metrics["f1"]
    }


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────

def run(X_train, X_test, y_train, y_test):

    results = []

    # ── Arduino: Logistic Regression
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    results.append(export_c(lr, "logreg", X_test, y_test))

    # ── Arduino: Decision Tree shallow (4-5)
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    results.append(export_c(dt, "tree", X_test, y_test))

    # ── ESP32: MLP
    if HAS_TF:
        mlp = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        mlp.compile(optimizer="adam", loss="binary_crossentropy")
        mlp.fit(X_train, y_train, epochs=10, verbose=0)

        results.append(export_mlp(mlp, "mlp", X_train, X_test, y_test))

    # ── ESP32: XGBoost
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            eval_metric="logloss"
        )
        xgb.fit(X_train, y_train)

        results.append(export_tree(xgb, "xgb", X_test, y_test))

    # ── ESP32: LightGBM
    if HAS_LGBM:
        lgb = LGBMClassifier(n_estimators=100, num_leaves=15)
        lgb.fit(X_train, y_train)

        results.append(export_tree(lgb, "lgbm", X_test, y_test))

    df = pd.DataFrame(results)
    df.to_csv(OUT / "summary.csv", index=False)

    return df


# ─────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    DATA = "data/dataset.csv"

    X_train, X_test, y_train, y_test = load_data(DATA)

    df = run(X_train, X_test, y_train, y_test)

    print(df)