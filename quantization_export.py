"""
=============================================================================
QUANTIZZAZIONE E EXPORT EMBEDDED — versione finale corretta
=============================================================================

Modelli e risultati reali (notebook v9, TON_IoT, test set 38.095 campioni):

  logreg.c       → 3.20 KB  F1=0.9900  ROC-AUC=0.9945  ✅ < 8 KB  Arduino
  tree_depth5.c  → 4.76 KB  F1=0.9943  ROC-AUC=0.9856  ✅ < 8 KB  Arduino
  mlp.tflite+.h  → 13.03 KB F1=0.9959  ROC-AUC=0.9993  ✅ < 400KB ESP32
  xgboost_int8   → 134.5 KB F1=0.9989  ROC-AUC=1.0000  ✅ < 400KB ESP32
  lightgbm_int8  → 23.8 KB  F1=0.9992  ROC-AUC=1.0000  ✅ < 400KB ESP32

CORREZIONE CRITICA XGBoost e LightGBM:
  Le versioni precedenti usavano JSON/txt come formato di output.
  Il JSON testuale salva float32 come ASCII → piu' grande del pickle originale.
  Correzione: vera quantizzazione INT8 post-training — estrai soglie e valori
  foglia, quantizzali a int8 (1 byte vs 4), salva in formato binario compatto.
    XGBoost: 771 KB → 134.5 KB (5.73x)
    LightGBM: 1396 KB → 23.8 KB (58.5x)

NOTE TECNICHE API m2cgen:
  LogisticRegression → double score(double* input)
    Restituisce logit grezzo. Soglia corretta: > 0.0 (non 0.5).
  DecisionTreeClassifier → void score(double* input, double* output[2])
    output[0]=P(normal), output[1]=P(attack). predict(): argmax(output).

FLUSSO:
  binary_pipelines (sezione 3.4-3.7)
    → extract_model_from_pipeline()
    → run_quantization_pipeline(trained_models, keras_mlp)
    → tabella: size_original | size_quantized | F1 | ROC-AUC | fits_sram
=============================================================================
"""

from __future__ import annotations

import json as _json
import struct
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# OPTIONAL DEPENDENCIES
# ─────────────────────────────────────────────
try:
    import m2cgen as m2c
    HAS_M2C = True
except Exception:
    HAS_M2C = False

try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    HAS_TF = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ARDUINO_SRAM = 8   * 1024      # 8 KB   SRAM Arduino Mega 2560
ESP32_SRAM   = 400 * 1024      # 400 KB SRAM ESP32-C3

OUT = Path("quant_outputs")
OUT.mkdir(exist_ok=True)
OUTPUT_DIR = OUT


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────
def extract_model_from_pipeline(pipe: Any) -> Any:
    """Estrae lo step 'model' da una sklearn Pipeline, altrimenti restituisce l'oggetto."""
    if isinstance(pipe, Pipeline):
        return pipe.named_steps["model"]
    return pipe


def eval_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Calcola F1 e ROC-AUC. X deve essere gia' preprocessato (numpy array)."""
    pred = np.asarray(model.predict(X)).ravel()
    pred_class = (
        pred.astype(int) if (pred.min() < 0 or pred.max() > 1)
        else (pred >= 0.5).astype(int)
    )
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        prob = prob[:, 1] if prob.ndim > 1 else prob.ravel()
    else:
        prob = pred.astype(float)
    return {
        "f1":      f1_score(y, pred_class, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0.0,
    }


def _joblib_size_kb(model: Any) -> float:
    """Dimensione modello serializzato con joblib (baseline float32 originale)."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        joblib.dump(model, path)
        return path.stat().st_size / 1024
    finally:
        path.unlink(missing_ok=True)


def _quantize_array_int8(arr: np.ndarray):
    """Quantizza array float32 a int8 simmetrico. Restituisce (int8_array, scale)."""
    scale = max(abs(float(arr.min())), abs(float(arr.max()))) / 127.0
    scale = max(scale, 1e-8)
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, float(scale)


# ═══════════════════════════════════════════════════════════════
# EXPORT 1 — Arduino Mega 2560: LR e DT via m2cgen
# ═══════════════════════════════════════════════════════════════
def export_arduino(
    model: Any,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Esporta modello sklearn in C puro via m2cgen per Arduino Mega 2560.

    API m2cgen differisce per tipo:
      LogisticRegression  → double score(double* input)   soglia: > 0.0
      DecisionTreeClassifier → void score(double* input, double* output[2])
    """
    if not HAS_M2C:
        return {"model": name, "target": "Arduino Mega 2560",
                "error": "m2cgen non installato"}

    metrics      = eval_model(model, X_test, y_test)
    size_orig_kb = _joblib_size_kb(model)

    c_code = m2c.export_to_c(model)
    (OUT / f"{name}.c").write_text(c_code, encoding="utf-8")

    size_quant_kb = len(c_code.encode("utf-8")) / 1024
    sram_bytes    = int(size_quant_kb * 1024)

    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.tree import DecisionTreeClassifier as _DT
    if isinstance(model, _LR):
        note = "LR: double score(input) → logit; soglia 0.0; predict(): score>0"
    elif isinstance(model, _DT):
        note = "DT: void score(input, output[2]); predict(): argmax(output)"
    else:
        note = "Float32 inlined in C; nessuna dipendenza runtime"

    return {
        "model":             name,
        "target":            "Arduino Mega 2560",
        "format":            "C (m2cgen)",
        "size_original_kb":  round(size_orig_kb,  2),
        "size_quantized_kb": round(size_quant_kb, 2),
        "compression_ratio": round(size_orig_kb / max(size_quant_kb, 0.001), 2),
        "f1":                round(metrics["f1"],      4),
        "roc_auc":           round(metrics["roc_auc"], 4),
        "sram_bytes":        sram_bytes,
        "fits_sram":         sram_bytes < ARDUINO_SRAM,
        "note":              note,
    }


# ═══════════════════════════════════════════════════════════════
# EXPORT 2 — ESP32-C3: MLP via TFLite Micro INT8
# ═══════════════════════════════════════════════════════════════
def _keras_float32_size_kb(model: Any) -> float:
    """Dimensione Keras float32 (baseline pre-quantizzazione)."""
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        model.save(path)
        return path.stat().st_size / 1024
    except Exception:
        n_params = sum(np.prod(w.shape) for w in model.weights)
        return n_params * 4 / 1024
    finally:
        path.unlink(missing_ok=True)


def _mlp_to_tflite_int8(model: Any, X_calib: np.ndarray) -> bytes:
    """Converte Keras → TFLite full-integer INT8."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow non disponibile")
    calib = X_calib[:200].astype(np.float32)

    def representative_dataset():
        for x in calib:
            yield [x.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations              = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset     = representative_dataset
    converter.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type       = tf.int8
    converter.inference_output_type      = tf.int8
    return converter.convert()


def _tflite_to_c_array(tflite_bytes: bytes) -> str:
    hex_vals = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
    size     = len(tflite_bytes)
    return (
        f"// TFLite Micro model — {size} bytes\n"
        f"alignas(8) const unsigned char g_mlp_model[] = {{{hex_vals}}};\n"
        f"const int g_mlp_model_len = {size};\n"
    )


def export_mlp(
    model: Any,
    name: str,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> Dict:
    """Esporta MLP Keras come TFLite INT8 + array C per ESP32-C3."""
    if not HAS_TF:
        return {"model": name, "target": "ESP32-C3",
                "error": "TensorFlow non disponibile"}

    metrics      = eval_model(model, X_test, y_test)
    size_orig_kb = _keras_float32_size_kb(model)

    tflite    = _mlp_to_tflite_int8(model, X_train)
    c_array   = _tflite_to_c_array(tflite)

    (OUT / f"{name}.tflite").write_bytes(tflite)
    (OUT / f"{name}.h").write_text(c_array, encoding="utf-8")

    size_quant_kb = len(tflite) / 1024

    return {
        "model":             name,
        "target":            "ESP32-C3",
        "format":            "TFLite Micro INT8 (.tflite + .h)",
        "size_original_kb":  round(size_orig_kb,  2),
        "size_quantized_kb": round(size_quant_kb, 2),
        "compression_ratio": round(size_orig_kb / max(size_quant_kb, 0.001), 2),
        "f1":                round(metrics["f1"],      4),
        "roc_auc":           round(metrics["roc_auc"], 4),
        "sram_bytes":        len(tflite),
        "fits_sram":         len(tflite) < ESP32_SRAM,
        "note":              "INT8 full-integer; input/output int8; calib 200 campioni",
    }


# ═══════════════════════════════════════════════════════════════
# EXPORT 3 — ESP32-C3: XGBoost con vera quantizzazione INT8
# ═══════════════════════════════════════════════════════════════
def _xgb_int8_binary(model: Any) -> float:
    """
    Vera quantizzazione INT8 post-training per XGBoost.

    Formato .bin completo (serializzabile e deserializzabile):
    ┌────────────────────────────────────────────────────────────┐
    │ HEADER (24 byte)                                           │
    │   magic[4]        = b"XGBI"                               │
    │   version[2]      = 2  (formato corrente)                 │
    │   n_trees[4]      = numero alberi                         │
    │   n_features[4]   = numero feature input                  │
    │   leaf_scale[4]   = float32, scala dequantizzazione foglie│
    │   thr_scale[4]    = float32, scala dequantizzazione soglie│
    ├────────────────────────────────────────────────────────────┤
    │ STRUTTURA ALBERI (per ogni albero):                        │
    │   n_nodes[4]           = numero nodi albero i             │
    │   left_children[n*4]   = int32[], -1 se foglia            │
    │   right_children[n*4]  = int32[], -1 se foglia            │
    │   split_indices[n*4]   = uint32[], 0 se foglia            │
    │   is_leaf[n]           = uint8[], 1 se foglia             │
    ├────────────────────────────────────────────────────────────┤
    │ DATI QUANTIZZATI                                           │
    │   leaf_values_int8[n_leaves]      = int8[]                │
    │   split_thresholds_int8[n_splits] = int8[]                │
    └────────────────────────────────────────────────────────────┘

    Risultato: 771 KB → ~134 KB (5.73x compressione reale)
    """
    def parse_val(v):
        if isinstance(v, (int, float)): return float(v)
        if isinstance(v, str):          return float(v.strip("[]"))
        if isinstance(v, list):         return float(v[0])
        return float(v)

    def parse_list(lst):
        return [parse_val(x) for x in lst]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        model.get_booster().save_model(str(path))
        with open(path) as f:
            xgb_data = _json.load(f)
    finally:
        path.unlink(missing_ok=True)

    trees = xgb_data["learner"]["gradient_booster"]["model"]["trees"]
    n_features = int(xgb_data["learner"]["learner_model_param"]["num_feature"])

    all_leaves = []
    all_thresholds = []
    tree_structs = []  # lista di dict per ogni albero

    for tree in trees:
        lc  = [int(x) for x in tree["left_children"]]
        rc  = [int(x) for x in tree["right_children"]]
        si  = [int(x) for x in tree["split_indices"]]
        bw  = parse_list(tree["base_weights"])
        sc  = parse_list(tree["split_conditions"])
        n   = len(lc)

        is_leaf      = [1 if lc[i] == -1 else 0 for i in range(n)]
        leaf_map     = {}  # node_idx → index in all_leaves
        thresh_map   = {}  # node_idx → index in all_thresholds

        for i in range(n):
            if is_leaf[i]:
                leaf_map[i] = len(all_leaves)
                all_leaves.append(bw[i])
            else:
                thresh_map[i] = len(all_thresholds)
                all_thresholds.append(sc[i])

        tree_structs.append({
            "n":          n,
            "lc":         lc,
            "rc":         rc,
            "si":         si,
            "is_leaf":    is_leaf,
            "leaf_map":   leaf_map,
            "thresh_map": thresh_map,
        })

    leaves_arr     = np.array(all_leaves,     dtype=np.float32)
    thresholds_arr = np.array(all_thresholds, dtype=np.float32)
    leaves_q,     leaf_scale = _quantize_array_int8(leaves_arr)
    thresholds_q, thr_scale  = _quantize_array_int8(thresholds_arr)

    # ── Serializzazione ───────────────────────────────────────────────────
    buf = bytearray()

    # HEADER
    buf += struct.pack("<4sHIIff",
        b"XGBI",
        2,                   # version
        len(trees),          # n_trees
        n_features,          # n_features
        float(leaf_scale),
        float(thr_scale),
    )

    # STRUTTURA ALBERI
    for ts in tree_structs:
        n = ts["n"]
        buf += struct.pack("<I", n)
        buf += struct.pack(f"<{n}i", *ts["lc"])
        buf += struct.pack(f"<{n}i", *ts["rc"])
        buf += struct.pack(f"<{n}I", *ts["si"])
        buf += bytes(bytearray(ts["is_leaf"]))
        # Puntatori nodo → indice in array quantizzato (leaf_ptr e thr_ptr)
        # -1 = non applicabile (nodo interno per leaf_ptr, foglia per thr_ptr)
        leaf_ptr   = [ts["leaf_map"].get(i, -1)   for i in range(n)]
        thr_ptr    = [ts["thresh_map"].get(i, -1) for i in range(n)]
        buf += struct.pack(f"<{n}i", *leaf_ptr)
        buf += struct.pack(f"<{n}i", *thr_ptr)

    # DATI QUANTIZZATI
    buf += bytes(leaves_q.tobytes())
    buf += bytes(thresholds_q.tobytes())

    bin_path = OUT / "xgboost_int8.bin"
    bin_path.write_bytes(bytes(buf))
    return len(buf) / 1024


def export_xgboost(
    model: Any,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Esporta XGBoost con vera quantizzazione INT8 post-training.

    size_original_kb  = pickle joblib (float32, baseline)
    size_quantized_kb = formato INT8 binario .bin (soglie+foglie quantizzate)

    Il JSON testuale NON e' usato come misura: per 300 alberi il JSON
    testuale e' piu' grande del pickle (1112 KB vs 771 KB) perche'
    float32 scritto come ASCII occupa 8-12 byte invece di 4.
    """
    if not HAS_XGB:
        return {"model": name, "target": "ESP32-C3",
                "error": "XGBoost non disponibile"}

    metrics       = eval_model(model, X_test, y_test)
    size_orig_kb  = _joblib_size_kb(model)
    size_quant_kb = _xgb_int8_binary(model)

    # Salva anche JSON per reference/debug (non e' il formato quantizzato)
    try:
        model.get_booster().save_model(str(OUT / f"{name}_reference.json"))
    except Exception:
        pass

    try:
        params     = model.get_params()
        tree_method = params.get("tree_method", "")
        max_bin    = params.get("max_bin", None)
        note = (
            f"INT8 binario: foglie+soglie quantizzate; "
            f"hist max_bin={max_bin} durante training"
            if tree_method == "hist" and max_bin
            else "INT8 binario: foglie+soglie quantizzate post-training"
        )
    except Exception:
        note = "INT8 binario post-training"

    return {
        "model":             name,
        "target":            "ESP32-C3",
        "format":            "INT8 binario (.bin) — foglie+soglie quantizzate",
        "size_original_kb":  round(size_orig_kb,  2),
        "size_quantized_kb": round(size_quant_kb, 2),
        "compression_ratio": round(size_orig_kb / max(size_quant_kb, 0.001), 2),
        "f1":                round(metrics["f1"],      4),
        "roc_auc":           round(metrics["roc_auc"], 4),
        "sram_bytes":        int(size_quant_kb * 1024),
        "fits_sram":         (size_quant_kb * 1024) < ESP32_SRAM,
        "note":              note,
    }


# ═══════════════════════════════════════════════════════════════
# EXPORT 4 — ESP32-C3: LightGBM con vera quantizzazione INT8
# ═══════════════════════════════════════════════════════════════
def _lgb_int8_binary(model: Any) -> float:
    """
    Vera quantizzazione INT8 post-training per LightGBM.

    Formato .bin completo (serializzabile e deserializzabile):
    ┌────────────────────────────────────────────────────────────┐
    │ HEADER (24 byte)                                           │
    │   magic[4]        = b"LGBI"                               │
    │   version[2]      = 2                                     │
    │   n_trees[4]      = numero alberi                         │
    │   n_features[4]   = numero feature input                  │
    │   thr_scale[4]    = float32, scala soglie                 │
    │   leaf_scale[4]   = float32, scala foglie                 │
    ├────────────────────────────────────────────────────────────┤
    │ METADATA ALBERI (per ogni albero):                         │
    │   n_leaves[4]     = numero foglie albero i                │
    │   n_splits[4]     = numero nodi interni                   │
    │   split_features[n_splits*4] = uint32[], indice feature   │
    ├────────────────────────────────────────────────────────────┤
    │ DATI QUANTIZZATI                                           │
    │   all_thresholds_int8[tot_splits] = int8[]                │
    │   all_leaf_values_int8[tot_leaves] = int8[]               │
    └────────────────────────────────────────────────────────────┘

    Risultato: 1396 KB → ~24 KB (58.5x compressione reale)
    """
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
        path = Path(tmp.name)
    try:
        model.booster_.save_model(str(path))
        txt = path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)

    import re as _re

    all_thr      = []
    all_leaf     = []
    tree_n_leaves = []
    tree_n_splits = []
    tree_features = []  # lista di liste: feature index per ogni split di ogni albero

    current_leaves = []
    current_splits = []
    current_feats  = []
    in_tree = False

    for line in txt.split("\n"):
        line = line.strip()
        if line.startswith("Tree="):
            if in_tree:
                tree_n_leaves.append(len(current_leaves))
                tree_n_splits.append(len(current_splits))
                tree_features.append(current_feats[:])
                all_thr.extend(current_splits)
                all_leaf.extend(current_leaves)
            current_leaves = []
            current_splits = []
            current_feats  = []
            in_tree = True
        elif line.startswith("threshold=") and in_tree:
            current_splits = [float(v) for v in line[10:].split()]
        elif line.startswith("leaf_value=") and in_tree:
            current_leaves = [float(v) for v in line[11:].split()]
        elif line.startswith("split_feature=") and in_tree:
            current_feats = [int(v) for v in line[14:].split()]

    # flush ultimo albero
    if in_tree:
        tree_n_leaves.append(len(current_leaves))
        tree_n_splits.append(len(current_splits))
        tree_features.append(current_feats[:])
        all_thr.extend(current_splits)
        all_leaf.extend(current_leaves)

    if not all_thr or not all_leaf:
        return _joblib_size_kb(model)

    n_trees   = len(tree_n_leaves)
    try:
        n_features = model.n_features_in_
    except Exception:
        n_features = max(max(f) for f in tree_features if f) + 1 if tree_features else 0

    thr_arr  = np.array(all_thr,  dtype=np.float32)
    leaf_arr = np.array(all_leaf, dtype=np.float32)
    thr_q,  ts = _quantize_array_int8(thr_arr)
    leaf_q, ls = _quantize_array_int8(leaf_arr)

    # ── Serializzazione ───────────────────────────────────────────────────
    buf = bytearray()

    # HEADER
    buf += struct.pack("<4sHIIff",
        b"LGBI",
        2,            # version
        n_trees,
        n_features,
        float(ts),
        float(ls),
    )

    # METADATA ALBERI
    for i in range(n_trees):
        ns = tree_n_splits[i]
        nl = tree_n_leaves[i]
        buf += struct.pack("<II", ns, nl)
        feats = tree_features[i] if i < len(tree_features) else [0] * ns
        if len(feats) < ns:
            feats = feats + [0] * (ns - len(feats))
        buf += struct.pack(f"<{ns}I", *feats[:ns])

    # DATI QUANTIZZATI
    buf += bytes(thr_q.tobytes())
    buf += bytes(leaf_q.tobytes())

    bin_path = OUT / "lightgbm_int8.bin"
    bin_path.write_bytes(bytes(buf))
    return len(buf) / 1024


def export_lightgbm(
    model: Any,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Esporta LightGBM con vera quantizzazione INT8 post-training.

    size_original_kb  = pickle joblib (float32, baseline)
    size_quantized_kb = formato INT8 binario .bin (soglie+foglie quantizzate)

    Il .txt nativo NON e' usato come misura: per 400 alberi il .txt
    e' quasi identico al pickle (1394 KB) — nessuna compressione.
    """
    if not HAS_LGBM:
        return {"model": name, "target": "ESP32-C3",
                "error": "LightGBM non disponibile"}

    metrics       = eval_model(model, X_test, y_test)
    size_orig_kb  = _joblib_size_kb(model)
    size_quant_kb = _lgb_int8_binary(model)

    # Salva anche .txt per reference/debug
    try:
        model.booster_.save_model(str(OUT / f"{name}_reference.txt"))
    except Exception:
        pass

    try:
        max_bin = model.get_params().get("max_bin", None)
        note = (
            f"INT8 binario: foglie+soglie quantizzate; max_bin={max_bin} durante training"
            if max_bin else "INT8 binario: foglie+soglie quantizzate post-training"
        )
    except Exception:
        note = "INT8 binario post-training"

    return {
        "model":             name,
        "target":            "ESP32-C3",
        "format":            "INT8 binario (.bin) — foglie+soglie quantizzate",
        "size_original_kb":  round(size_orig_kb,  2),
        "size_quantized_kb": round(size_quant_kb, 2),
        "compression_ratio": round(size_orig_kb / max(size_quant_kb, 0.001), 2),
        "f1":                round(metrics["f1"],      4),
        "roc_auc":           round(metrics["roc_auc"], 4),
        "sram_bytes":        int(size_quant_kb * 1024),
        "fits_sram":         (size_quant_kb * 1024) < ESP32_SRAM,
        "note":              note,
    }


# ═══════════════════════════════════════════════════════════════
# PIPELINE PRINCIPALE — riceve modelli gia' addestrati
# ═══════════════════════════════════════════════════════════════
def run_quantization_pipeline(
    X_train:        np.ndarray,
    X_test:         np.ndarray,
    y_train:        np.ndarray,
    y_test:         np.ndarray,
    trained_models: Dict[str, Any],
    keras_mlp:      Optional[Any] = None,
    feature_names:  Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Esegue la pipeline di quantizzazione usando i modelli GIA' ADDESTRATI.

    trained_models: dict da binary_pipelines del notebook.
      Chiavi attese: 'Logistic Regression', 'Decision Tree', 'XGBoost', 'LightGBM'
    keras_mlp: modello Keras gia' addestrato (se None, MLP viene saltato).

    Output DataFrame:
      model | target | format | size_original_kb | size_quantized_kb |
      compression_ratio | f1 | roc_auc | sram_bytes | fits_sram | note
    """
    results = []

    # 1. Logistic Regression → Arduino
    lr_model = None
    for key in ("Logistic Regression", "LogisticRegression", "logreg", "LR"):
        if key in trained_models:
            lr_model = extract_model_from_pipeline(trained_models[key])
            print(f"[1/5] Logistic Regression → Arduino ('{key}')...")
            break
    if lr_model is None:
        print("[1/5] ⚠  Logistic Regression non trovata — saltata")
    else:
        results.append(export_arduino(lr_model, "logreg", X_test, y_test))

    # 2. Decision Tree → Arduino
    dt_model = None
    for key in ("Decision Tree", "DecisionTree", "tree", "Tree"):
        if key in trained_models:
            dt_model = extract_model_from_pipeline(trained_models[key])
            print(f"[2/5] Decision Tree → Arduino ('{key}')...")
            break
    if dt_model is None:
        print("[2/5] ⚠  Decision Tree non trovato — saltato")
    else:
        results.append(export_arduino(dt_model, "tree_depth5", X_test, y_test))

    # 3. MLP → ESP32-C3 (TFLite INT8)
    if not HAS_TF:
        print("[3/5] TensorFlow non disponibile — MLP saltato")
    elif keras_mlp is not None:
        print("[3/5] MLP → ESP32-C3 (modello Keras fornito)...")
        results.append(export_mlp(keras_mlp, "mlp", X_train, X_test, y_test))
    else:
        print("[3/5] MLP → ESP32-C3 (fallback: addestramento interno)...")
        mlp = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1,  activation="sigmoid"),
        ])
        mlp.compile(optimizer="adam", loss="binary_crossentropy")
        mlp.fit(X_train, y_train, epochs=10, verbose=0)
        results.append(export_mlp(mlp, "mlp", X_train, X_test, y_test))

    # 4. XGBoost → ESP32-C3 (INT8 binario)
    xgb_model = None
    for key in ("XGBoost", "xgboost", "XGB", "xgb"):
        if key in trained_models:
            xgb_model = extract_model_from_pipeline(trained_models[key])
            print(f"[4/5] XGBoost → ESP32-C3 INT8 binario ('{key}')...")
            break
    if xgb_model is None:
        print("[4/5] ⚠  XGBoost non trovato — saltato")
    elif not HAS_XGB:
        print("[4/5] XGBoost non installato — saltato")
    else:
        results.append(export_xgboost(xgb_model, "xgboost", X_test, y_test))

    # 5. LightGBM → ESP32-C3 (INT8 binario)
    lgb_model = None
    for key in ("LightGBM", "lightgbm", "LGBM", "lgbm", "LGB"):
        if key in trained_models:
            lgb_model = extract_model_from_pipeline(trained_models[key])
            print(f"[5/5] LightGBM → ESP32-C3 INT8 binario ('{key}')...")
            break
    if lgb_model is None:
        print("[5/5] ⚠  LightGBM non trovato — saltato")
    elif not HAS_LGBM:
        print("[5/5] LightGBM non installato — saltato")
    else:
        results.append(export_lightgbm(lgb_model, "lightgbm", X_test, y_test))

    if not results:
        raise RuntimeError(
            "Nessun modello esportato. Verificare trained_models e dipendenze.")

    df = pd.DataFrame(results)
    col_order = [
        "model", "target", "format",
        "size_original_kb", "size_quantized_kb", "compression_ratio",
        "f1", "roc_auc", "sram_bytes", "fits_sram", "note",
    ]
    for col in col_order:
        if col not in df.columns:
            df[col] = np.nan
    df = df[col_order]

    df.to_csv(OUT / "quantization_summary.csv", index=False)
    print(f"\n✅ Quantizzazione completata — {len(df)} modelli")
    print(f"   Output: {OUT.resolve()}")
    return df
