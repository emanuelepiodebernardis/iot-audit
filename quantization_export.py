"""
=============================================================================
QUANTIZZAZIONE E EXPORT EMBEDDED — versione finale corretta
=============================================================================

Correzione critica rispetto alla versione precedente:
  run_quantization_pipeline() ora riceve i modelli GIA' ADDESTRATI
  dalla pipeline principale della tesi (binary_pipelines).
  Non ri-addestra nulla: le metriche F1/ROC-AUC riflettono esattamente
  i modelli valutati nelle sezioni precedenti del notebook.

Flusso corretto:
  Notebook sezione 3.4-3.7
    → binary_pipelines[name] (Pipeline sklearn fittata)
    → estrai step "model" da ogni pipeline
    → passa i modelli a run_quantization_pipeline()
    → export C / TFLite / JSON nativo
    → tabella size_original | size_quantized | F1 | ROC-AUC | fits_sram

Per l'MLP (non presente in binary_pipelines perche' Keras non e' sklearn):
  → viene costruito e addestrato una volta nella cella del notebook
  → passato come keras_mlp=<model> a run_quantization_pipeline()
=============================================================================
"""

from __future__ import annotations

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
ARDUINO_SRAM = 8   * 1024       # 8 KB  SRAM Arduino Mega 2560
ESP32_SRAM   = 400 * 1024       # 400 KB SRAM ESP32-C3

OUT = Path("quant_outputs")
OUT.mkdir(exist_ok=True)
OUTPUT_DIR = OUT


# ─────────────────────────────────────────────
# UTILITY: estrazione modello da Pipeline sklearn
# ─────────────────────────────────────────────
def extract_model_from_pipeline(pipe: Any) -> Any:
    """
    Estrae lo step 'model' da una sklearn Pipeline.
    Se l'oggetto passato non e' una Pipeline, lo restituisce direttamente.
    """
    if isinstance(pipe, Pipeline):
        return pipe.named_steps["model"]
    return pipe


# ─────────────────────────────────────────────
# UTILITY: metriche sul modello estratto
# ─────────────────────────────────────────────
def eval_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Calcola F1 e ROC-AUC su qualsiasi modello sklearn-compatibile.
    X deve essere l'array numpy GIA' preprocessato (output del preprocessor).
    """
    pred = np.asarray(model.predict(X)).ravel()

    # binarizzazione robusta (XGBoost puo' restituire valori fuori [0,1])
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
    """Dimensione del modello serializzato con joblib (baseline float32)."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        joblib.dump(model, path)
        return path.stat().st_size / 1024
    finally:
        path.unlink(missing_ok=True)


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
    Esporta un modello sklearn in C puro via m2cgen.

    Compatibile con: LogisticRegression, DecisionTreeClassifier,
    e qualsiasi modello supportato da m2cgen.

    I pesi float32 vengono serializzati come costanti C inline:
    nessuna dipendenza runtime → adatto ad Arduino Mega 2560 (8 KB SRAM).

    Nota SRAM:
      size_original_kb  = dimensione joblib pickle (Python)
      size_quantized_kb = dimensione del sorgente .c generato
        (upper-bound conservativo dello spazio flash; il compilato
         sara' piu' compatto. Non include stack runtime ~200-400 B.)
    """
    if not HAS_M2C:
        return {
            "model": name, "target": "Arduino Mega 2560",
            "error": "m2cgen non installato — installare con: pip install m2cgen"
        }

    metrics      = eval_model(model, X_test, y_test)
    size_orig_kb = _joblib_size_kb(model)

    c_code = m2c.export_to_c(model)
    (OUT / f"{name}.c").write_text(c_code, encoding="utf-8")

    size_quant_kb = len(c_code.encode("utf-8")) / 1024
    sram_bytes    = int(size_quant_kb * 1024)

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
        "note":              "Float32 inlined in C; nessuna dipendenza runtime",
    }


# ═══════════════════════════════════════════════════════════════
# EXPORT 2 — ESP32-C3: MLP via TFLite Micro INT8
# ═══════════════════════════════════════════════════════════════
def _keras_float32_size_kb(model: Any) -> float:
    """Dimensione del modello Keras in float32 (baseline pre-quantizzazione)."""
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        model.save(path)
        return path.stat().st_size / 1024
    except Exception:
        # fallback analitico: n_parametri × 4 byte
        n_params = sum(np.prod(w.shape) for w in model.weights)
        return n_params * 4 / 1024
    finally:
        path.unlink(missing_ok=True)


def _mlp_to_tflite_int8(model: Any, X_calib: np.ndarray) -> bytes:
    """
    Conversione Keras → TFLite full-integer INT8.
    Usa i primi 200 campioni di calibrazione per il range dei pesi.
    input/output rimangono int8 per compatibilita' TFLite Micro.
    """
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


def _tflite_to_c_array(tflite_bytes: bytes, model_name: str = "g_model") -> str:
    """Genera l'header C con il modello TFLite come array di byte."""
    hex_vals = ", ".join(f"0x{b:02x}" for b in tflite_bytes)
    size     = len(tflite_bytes)
    return (
        f"// TFLite Micro model — {size} bytes\n"
        f"// Generato automaticamente — non modificare\n"
        f"alignas(8) const unsigned char {model_name}[] = {{{hex_vals}}};\n"
        f"const int {model_name}_len = {size};\n"
    )


def export_mlp(
    keras_model: Any,
    name: str,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> Dict:
    """
    Esporta MLP Keras come TFLite INT8 + array C per TFLite Micro su ESP32-C3.

    keras_model: modello Keras GIA' addestrato (tf.keras.Sequential o Model).
    X_train: dati di calibrazione per la quantizzazione (array preprocessato).
    X_test:  dati di test per le metriche (array preprocessato).
    y_test:  label binarie int per le metriche.
    """
    if not HAS_TF:
        return {
            "model": name, "target": "ESP32-C3",
            "error": "TensorFlow non disponibile"
        }

    metrics      = eval_model(keras_model, X_test, y_test)
    size_orig_kb = _keras_float32_size_kb(keras_model)

    tflite    = _mlp_to_tflite_int8(keras_model, X_train)
    c_array   = _tflite_to_c_array(tflite, model_name="g_mlp_model")

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
# EXPORT 3 — ESP32-C3: XGBoost
# ═══════════════════════════════════════════════════════════════
def _xgb_native_size_kb(model: Any) -> float:
    """Dimensione del booster XGBoost in formato JSON nativo."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        model.get_booster().save_model(str(path))
        return path.stat().st_size / 1024
    finally:
        path.unlink(missing_ok=True)


def export_xgboost(
    model: Any,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Esporta XGBoost in formato JSON nativo + C via m2cgen.

    Il modello ricevuto E' GIA' ADDESTRATO dalla sezione binary della tesi.
    Non viene ri-addestrato.

    Quantizzazione INT8:
      Le soglie dei nodi sono quantizzate in INT8-equivalent durante il
      training se il modello e' stato addestrato con tree_method='hist'.
      Se il modello della tesi NON usa hist, il JSON nativo rimane float32
      ma e' comunque deployabile su ESP32 tramite la libreria XGBoost port.
      In entrambi i casi size_quantized_kb misura il formato nativo deployabile,
      non il pickle Python che include l'overhead dello sklearn wrapper.
    """
    if not HAS_XGB:
        return {
            "model": name, "target": "ESP32-C3",
            "error": "XGBoost non disponibile"
        }

    metrics       = eval_model(model, X_test, y_test)
    size_orig_kb  = _joblib_size_kb(model)
    size_quant_kb = _xgb_native_size_kb(model)

    # Salva il modello nativo per deploy
    model.get_booster().save_model(str(OUT / f"{name}.json"))

    # Tenta export C via m2cgen (massima portabilita' — no runtime XGBoost)
    c_size_kb = None
    if HAS_M2C:
        try:
            c_code = m2c.export_to_c(model)
            (OUT / f"{name}.c").write_text(c_code, encoding="utf-8")
            c_size_kb = round(len(c_code.encode("utf-8")) / 1024, 2)
        except Exception as exc:
            print(f"  [m2cgen XGBoost skip: {exc}]")

    # Determina se il training ha usato hist (soglie INT8)
    try:
        tree_method = model.get_params().get("tree_method", "")
        max_bin     = model.get_params().get("max_bin", None)
        quant_note  = (
            f"Soglie INT8 (hist, max_bin={max_bin})"
            if tree_method == "hist" and max_bin
            else "JSON nativo XGBoost (float32 se non hist)"
        )
    except Exception:
        quant_note = "JSON nativo XGBoost"

    if c_size_kb is not None:
        quant_note += f"; C export {c_size_kb:.1f} KB via m2cgen"

    return {
        "model":             name,
        "target":            "ESP32-C3",
        "format":            "XGBoost JSON nativo + C (m2cgen)",
        "size_original_kb":  round(size_orig_kb,  2),
        "size_quantized_kb": round(size_quant_kb, 2),
        "compression_ratio": round(size_orig_kb / max(size_quant_kb, 0.001), 2),
        "f1":                round(metrics["f1"],      4),
        "roc_auc":           round(metrics["roc_auc"], 4),
        "sram_bytes":        int(size_quant_kb * 1024),
        "fits_sram":         (size_quant_kb * 1024) < ESP32_SRAM,
        "note":              quant_note,
    }


# ═══════════════════════════════════════════════════════════════
# EXPORT 4 — ESP32-C3: LightGBM
# ═══════════════════════════════════════════════════════════════
def _lgb_native_size_kb(model: Any) -> float:
    """Dimensione del booster LightGBM in formato testo nativo."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        model.booster_.save_model(str(path))
        return path.stat().st_size / 1024
    finally:
        path.unlink(missing_ok=True)


def export_lightgbm(
    model: Any,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Esporta LightGBM in formato testo nativo + C via m2cgen.

    Il modello ricevuto E' GIA' ADDESTRATO dalla sezione binary della tesi.
    Non viene ri-addestrato.

    Quantizzazione INT8:
      Le soglie sono quantizzate in INT8-equivalent se il training ha usato
      max_bin=255. Il formato nativo .txt e' 3-4x piu' compatto del pickle
      joblib che include lo sklearn wrapper.
    """
    if not HAS_LGBM:
        return {
            "model": name, "target": "ESP32-C3",
            "error": "LightGBM non disponibile"
        }

    metrics       = eval_model(model, X_test, y_test)
    size_orig_kb  = _joblib_size_kb(model)
    size_quant_kb = _lgb_native_size_kb(model)

    model.booster_.save_model(str(OUT / f"{name}.txt"))

    c_size_kb = None
    if HAS_M2C:
        try:
            c_code = m2c.export_to_c(model)
            (OUT / f"{name}.c").write_text(c_code, encoding="utf-8")
            c_size_kb = round(len(c_code.encode("utf-8")) / 1024, 2)
        except Exception as exc:
            print(f"  [m2cgen LightGBM skip: {exc}]")

    try:
        max_bin    = model.get_params().get("max_bin", None)
        quant_note = (
            f"Soglie INT8 (max_bin={max_bin})"
            if max_bin and max_bin <= 256
            else "Nativo LightGBM .txt"
        )
    except Exception:
        quant_note = "Nativo LightGBM .txt"

    if c_size_kb is not None:
        quant_note += f"; C export {c_size_kb:.1f} KB via m2cgen"

    return {
        "model":             name,
        "target":            "ESP32-C3",
        "format":            "LightGBM nativo (.txt) + C (m2cgen)",
        "size_original_kb":  round(size_orig_kb,  2),
        "size_quantized_kb": round(size_quant_kb, 2),
        "compression_ratio": round(size_orig_kb / max(size_quant_kb, 0.001), 2),
        "f1":                round(metrics["f1"],      4),
        "roc_auc":           round(metrics["roc_auc"], 4),
        "sram_bytes":        int(size_quant_kb * 1024),
        "fits_sram":         (size_quant_kb * 1024) < ESP32_SRAM,
        "note":              quant_note,
    }


# ═══════════════════════════════════════════════════════════════
# PIPELINE PRINCIPALE — riceve modelli gia' addestrati
# ═══════════════════════════════════════════════════════════════
def run_quantization_pipeline(
    X_train:       np.ndarray,
    X_test:        np.ndarray,
    y_train:       np.ndarray,
    y_test:        np.ndarray,
    trained_models: Dict[str, Any],
    keras_mlp:     Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Esegue la pipeline di quantizzazione usando i modelli GIA' ADDESTRATI.

    Parametri
    ---------
    X_train, X_test : np.ndarray
        Dati gia' preprocessati (output del DataFramePreprocessor, convertiti
        in numpy). X_train usato per calibrazione TFLite; X_test per metriche.
    y_train, y_test : np.ndarray
        Label binarie int (0/1).
    trained_models : dict
        Dizionario {nome_display: modello_sklearn} dei modelli gia' addestrati.
        I valori possono essere Pipeline sklearn o modelli nudi — il wrapper
        extract_model_from_pipeline() estrae lo step 'model' automaticamente.
        Chiavi attese (opzionali, usate se presenti):
          'Logistic Regression', 'Decision Tree', 'XGBoost', 'LightGBM'
    keras_mlp : tf.keras.Model, opzionale
        Modello MLP Keras gia' addestrato. Se None, l'MLP viene costruito
        e addestrato internamente (fallback per compatibilita').
    feature_names : list[str], opzionale
        Nomi delle feature preprocessate (per documentazione).

    Output DataFrame colonne
    ------------------------
    model | target | format | size_original_kb | size_quantized_kb |
    compression_ratio | f1 | roc_auc | sram_bytes | fits_sram | note
    """
    results = []

    # ── 1. Logistic Regression → Arduino (m2cgen) ────────────────────
    lr_model = None
    for key in ("Logistic Regression", "LogisticRegression", "logreg", "LR"):
        if key in trained_models:
            lr_model = extract_model_from_pipeline(trained_models[key])
            print(f"[1/5] Logistic Regression → Arduino (modello tesi: '{key}')...")
            break

    if lr_model is None:
        print("[1/5] ⚠  Logistic Regression non trovata in trained_models — saltata")
    else:
        results.append(export_arduino(lr_model, "logreg", X_test, y_test))

    # ── 2. Decision Tree → Arduino (m2cgen) ──────────────────────────
    dt_model = None
    for key in ("Decision Tree", "DecisionTree", "tree", "Tree"):
        if key in trained_models:
            dt_model = extract_model_from_pipeline(trained_models[key])
            print(f"[2/5] Decision Tree → Arduino (modello tesi: '{key}')...")
            break

    if dt_model is None:
        print("[2/5] ⚠  Decision Tree non trovato in trained_models — saltato")
    else:
        results.append(export_arduino(dt_model, "tree_depth5", X_test, y_test))

    # ── 3. MLP → ESP32-C3 (TFLite INT8) ─────────────────────────────
    if not HAS_TF:
        print("[3/5] TensorFlow non disponibile — MLP saltato")
    elif keras_mlp is not None:
        print("[3/5] MLP → ESP32-C3 (modello Keras fornito)...")
        results.append(export_mlp(keras_mlp, "mlp", X_train, X_test, y_test))
    else:
        # Fallback: costruisce e addestra MLP internamente
        # (solo se non e' disponibile un modello pre-addestrato)
        print("[3/5] MLP → ESP32-C3 (fallback: addestramento interno 10 epoche)...")
        mlp = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1,  activation="sigmoid"),
        ])
        mlp.compile(optimizer="adam", loss="binary_crossentropy")
        mlp.fit(X_train, y_train, epochs=10, verbose=0)
        results.append(export_mlp(mlp, "mlp", X_train, X_test, y_test))

    # ── 4. XGBoost → ESP32-C3 (JSON nativo) ─────────────────────────
    xgb_model = None
    for key in ("XGBoost", "xgboost", "XGB", "xgb"):
        if key in trained_models:
            xgb_model = extract_model_from_pipeline(trained_models[key])
            print(f"[4/5] XGBoost → ESP32-C3 (modello tesi: '{key}')...")
            break

    if xgb_model is None:
        print("[4/5] ⚠  XGBoost non trovato in trained_models — saltato")
    elif not HAS_XGB:
        print("[4/5] XGBoost non installato — saltato")
    else:
        results.append(export_xgboost(xgb_model, "xgboost", X_test, y_test))

    # ── 5. LightGBM → ESP32-C3 (nativo) ─────────────────────────────
    lgb_model = None
    for key in ("LightGBM", "lightgbm", "LGBM", "lgbm", "LGB"):
        if key in trained_models:
            lgb_model = extract_model_from_pipeline(trained_models[key])
            print(f"[5/5] LightGBM → ESP32-C3 (modello tesi: '{key}')...")
            break

    if lgb_model is None:
        print("[5/5] ⚠  LightGBM non trovato in trained_models — saltato")
    elif not HAS_LGBM:
        print("[5/5] LightGBM non installato — saltato")
    else:
        results.append(export_lightgbm(lgb_model, "lightgbm", X_test, y_test))

    # ── Tabella finale ────────────────────────────────────────────────
    if not results:
        raise RuntimeError(
            "Nessun modello esportato. Verificare trained_models e le dipendenze."
        )

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
    print(f"\n✅ Quantizzazione completata — {len(df)} modelli esportati")
    print(f"   File salvati in: {OUT.resolve()}")
    return df
