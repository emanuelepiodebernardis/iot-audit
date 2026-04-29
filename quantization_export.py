from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ─────────────────────────────────────────────
# OPTIONAL DEPENDENCIES
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ARDUINO_SRAM = 8 * 1024
ESP32_SRAM = 400 * 1024

OUT = Path("quant_outputs")
OUT.mkdir(exist_ok=True)

OUTPUT_DIR = OUT


# ─────────────────────────────────────────────
# METRICS (FIXED)
# ─────────────────────────────────────────────
def eval_model(model, X, y):
    pred = model.predict(X)

    pred = np.asarray(pred).ravel()

    # binarizzazione robusta
    if pred.min() < 0 or pred.max() > 1:
        pred_class = pred.astype(int)
    else:
        pred_class = (pred >= 0.5).astype(int)

    # probabilità per ROC-AUC
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        prob = prob[:, 1] if prob.ndim > 1 else prob.ravel()
    else:
        prob = pred.astype(float)

    return {
        "f1": f1_score(y, pred_class),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.0
    }


# ─────────────────────────────────────────────
# ARDUINO EXPORT (C)
# ─────────────────────────────────────────────
def export_arduino(model, name, X_test, y_test):
    if not HAS_M2C:
        return {"error": "m2cgen missing"}

    metrics = eval_model(model, X_test, y_test)

    c_code = m2c.export_to_c(model)

    path = OUT / f"{name}.c"
    path.write_text(c_code)

    # APPROX SRAM (non file size)
    approx_sram = len(c_code.encode("utf-8"))

    return {
        "model": name,
        "target": "Arduino Mega 2560",
        "format": "C (m2cgen)",
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "sram_bytes": approx_sram,
        "fits_sram": approx_sram < ARDUINO_SRAM
    }


# ─────────────────────────────────────────────
# TFLITE → C ARRAY (ESP32-C3)
# ─────────────────────────────────────────────
def tflite_to_c_array(tflite_model):
    hex_array = ', '.join(f'0x{b:02x}' for b in tflite_model)
    return f"const unsigned char model[] = {{{hex_array}}};"


def mlp_to_tflite(model, X_calib):
    if not HAS_TF:
        raise RuntimeError("TensorFlow required")

    calib = X_calib[:200].astype(np.float32)

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

    c_array = tflite_to_c_array(tflite)

    path = OUT / f"{name}.h"
    path.write_text(c_array)

    return {
        "model": name,
        "target": "ESP32-C3",
        "format": "TFLite Micro (C array)",
        "size_kb": len(tflite) / 1024,
        "f1": metrics["f1"]
    }


# ─────────────────────────────────────────────
# TREE EXPORT (ESP32)
# ─────────────────────────────────────────────
def model_size_kb(model):
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    joblib.dump(model, tmp.name)
    return Path(tmp.name).stat().st_size / 1024


def export_tree(model, name, X_test, y_test):
    metrics = eval_model(model, X_test, y_test)

    size_kb = model_size_kb(model)

    return {
        "model": name,
        "target": "ESP32-C3",
        "format": "tree model",
        "size_kb": size_kb,
        "fits_sram": size_kb * 1024 < ESP32_SRAM,
        "f1": metrics["f1"]
    }


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────
def run_quantization_pipeline(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names: List[str] = None
):

    results = []

    # ── Arduino
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    results.append(export_arduino(lr, "logreg", X_test, y_test))

    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    results.append(export_arduino(dt, "tree_depth5", X_test, y_test))

    # ── ESP32 MLP
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

    # ── XGBoost (pseudo INT8)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            tree_method="hist"
        )
        xgb.fit(X_train, y_train)

        results.append(export_tree(xgb, "xgboost", X_test, y_test))

    # ── LightGBM
    if HAS_LGBM:
        lgb = LGBMClassifier(
            n_estimators=100,
            num_leaves=15,
            max_bin=255
        )
        lgb.fit(X_train, y_train)

        results.append(export_tree(lgb, "lightgbm", X_test, y_test))

    df = pd.DataFrame(results)
    df.to_csv(OUT / "summary.csv", index=False)

    return df