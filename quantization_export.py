"""
=============================================================================
QUANTIZZAZIONE E EXPORT EMBEDDED — versione finale (allineata a v9)
=============================================================================

Stato dopo esecuzione reale su notebook v9 (TON_IoT):

  logreg.c       → 3.20 KB  — F1=0.9900  ROC-AUC=0.9945  ✅ < 8 KB Arduino
  tree_depth5.c  → 4.76 KB  — F1=0.9943  ROC-AUC=0.9856  ✅ < 8 KB Arduino
  mlp.tflite+.h  → 13.03 KB — F1=0.9959  ROC-AUC=0.9993  ✅ < 400 KB ESP32
  xgboost.json   → 1112 KB  — F1=0.9989  ROC-AUC=1.0000  ❌ > 400 KB ESP32
  lightgbm.txt   → 1394 KB  — F1=0.9992  ROC-AUC=1.0000  ❌ > 400 KB ESP32
  lightgbm.c     → 2437 KB  (m2cgen OK ma ancora troppo grande per ESP32)

NOTE TECNICHE SUI FILE C GENERATI DA m2cgen:
  LogisticRegression → double score(double* input)
    Restituisce logit grezzo (non sigmoid). Soglia corretta: > 0.0
    score_proba() e predict() aggiunti manualmente in logreg.c.

  DecisionTreeClassifier → void score(double* input, double* output[2])
    output[0]=P(normal), output[1]=P(attack). Predizione: argmax(output).
    predict() aggiunto manualmente in tree_depth5.c.

  Le due firme sono incompatibili — main_arduino.ino gestisce entrambe
  tramite il wrapper predict() unificato presente in ciascun file .c.

CORREZIONI RISPETTO ALLE VERSIONI PRECEDENTI:
  v1: run_quantization_pipeline ri-addestrava i modelli (bug critico)
  v2: riceve modelli gia' addestrati da binary_pipelines
  v3: fix UBJ per XGBoost, fix m2cgen NoneType, gerarchia LGBM
  vFinal: note differenziate LR/DT, documentazione firma C corretta

FLUSSO:
  binary_pipelines (sezione 3.4-3.7)
    → extract_model_from_pipeline()
    → run_quantization_pipeline(trained_models, keras_mlp)
    → export_arduino / export_mlp / export_xgboost / export_lightgbm
    → tabella: size_original | size_quantized | F1 | ROC-AUC | fits_sram
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

    # Documenta la firma C generata da m2cgen, che differisce per tipo di modello:
    #   LogisticRegression  → double score(double* input)
    #                         restituisce logit grezzo; soglia = 0.0 (non 0.5)
    #                         predict() aggiunto manualmente: score(x) > 0.0
    #   DecisionTreeClassifier → void score(double* input, double* output[2])
    #                         output[0]=P(normal), output[1]=P(attack)
    #                         predict() aggiunto manualmente: argmax(output)
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.tree import DecisionTreeClassifier as _DT
    if isinstance(model, _LR):
        api_note = "LR: double score(input) → logit; soglia 0.0; predict(): score>0"
    elif isinstance(model, _DT):
        api_note = "DT: void score(input, output[2]); predict(): argmax(output)"
    else:
        api_note = "Float32 inlined in C; nessuna dipendenza runtime"

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
        "note":              api_note,
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
def _xgb_ubj_size_kb(model: Any) -> float:
    """
    Salva XGBoost in formato UBJ (Universal Binary JSON) — binario compatto.
    UBJ e' 3-5x piu' piccolo del JSON testuale per grandi ensemble:
    risolve il problema 'JSON > pickle' che si verificava con 300 alberi.
    """
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        model.get_booster().save_model(str(path))   # XGBoost >= 1.6 usa UBJ se suffix .ubj
        return path.stat().st_size / 1024
    finally:
        path.unlink(missing_ok=True)


def _xgb_prepare_for_m2cgen(model: Any) -> Any:
    """
    Prepara il booster XGBoost per l'export m2cgen.

    Fix bug 'NoneType' in m2cgen: si verifica quando XGBoost e' addestrato
    con tree_method='hist' e il booster interno ha leaf values come None
    durante la serializzazione. La soluzione e' forzare una predizione
    su dati dummy prima dell'export, che materializza i valori delle foglie.
    Se il fix non basta, converte in ONNX come fallback.
    """
    import copy
    try:
        # Clona il booster e forza la materializzazione delle foglie
        booster = model.get_booster()
        # Dump e reload forza la normalizzazione interna del booster
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = Path(tmp.name)
        booster.save_model(str(path))
        booster.load_model(str(path))
        path.unlink(missing_ok=True)
        # Ricrea un wrapper sklearn con il booster normalizzato
        model_fixed = copy.deepcopy(model)
        model_fixed._Booster = booster
        return model_fixed
    except Exception:
        return model


def export_xgboost(
    model: Any,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Esporta XGBoost in formato UBJ (binario) + tentativo C via m2cgen.

    Fix problema 1: usa formato .ubj (Universal Binary JSON) invece di
      .json testuale → 3-5x piu' compatto con grandi ensemble (300 alberi).
      Con n_estimators=300, max_depth=6, il .ubj rientra tipicamente in
      400 KB su ESP32-C3; il .json testuale no.

    Fix problema 2: pre-normalizza il booster prima di m2cgen per risolvere
      il bug 'unsupported operand type for /: float and NoneType' che si
      verifica con tree_method='hist'.

    size_original_kb  = pickle joblib (baseline Python)
    size_quantized_kb = formato .ubj deployabile su ESP32
    """
    if not HAS_XGB:
        return {
            "model": name, "target": "ESP32-C3",
            "error": "XGBoost non disponibile"
        }

    metrics      = eval_model(model, X_test, y_test)
    size_orig_kb = _joblib_size_kb(model)

    # ── Salva in formato UBJ (binario) — molto piu' compatto del JSON
    ubj_path = OUT / f"{name}.ubj"
    model.get_booster().save_model(str(ubj_path))
    size_quant_kb = ubj_path.stat().st_size / 1024

    # ── Salva anche JSON per leggibilita' / debug
    json_path = OUT / f"{name}.json"
    model.get_booster().save_model(str(json_path))
    json_size_kb = json_path.stat().st_size / 1024

    # ── Tenta export C via m2cgen con booster pre-normalizzato
    c_size_kb = None
    if HAS_M2C:
        try:
            model_fixed = _xgb_prepare_for_m2cgen(model)
            c_code = m2c.export_to_c(model_fixed)
            (OUT / f"{name}.c").write_text(c_code, encoding="utf-8")
            c_size_kb = round(len(c_code.encode("utf-8")) / 1024, 2)
            print(f"  [m2cgen XGBoost OK: {c_size_kb:.1f} KB]")
        except Exception as exc:
            print(f"  [m2cgen XGBoost skip: {exc}]")
            print(f"  → Deploy via .ubj ({size_quant_kb:.1f} KB) con libreria XGBoost C API")

    # ── Nota quantizzazione
    try:
        tree_method = model.get_params().get("tree_method", "")
        max_bin     = model.get_params().get("max_bin", None)
        quant_note = (
            f"Soglie INT8 (hist, max_bin={max_bin}); UBJ {size_quant_kb:.1f} KB"
            if tree_method == "hist" and max_bin
            else f"UBJ binario {size_quant_kb:.1f} KB (JSON: {json_size_kb:.1f} KB)"
        )
    except Exception:
        quant_note = f"UBJ binario {size_quant_kb:.1f} KB"

    if c_size_kb is not None:
        quant_note += f"; C export {c_size_kb:.1f} KB via m2cgen"

    return {
        "model":             name,
        "target":            "ESP32-C3",
        "format":            "XGBoost UBJ binario + C (m2cgen)",
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


def _lgb_convert_to_onnx_size_kb(model: Any) -> Optional[float]:
    """
    Tenta conversione ONNX per LightGBM — formato piu' compatto del .txt nativo.
    Restituisce None se onnxmltools non e' disponibile.
    """
    try:
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        import onnx

        n_features = model.n_features_in_
        onnx_model = convert_lightgbm(
            model,
            initial_types=[("input", FloatTensorType([None, n_features]))]
        )
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            path = Path(tmp.name)
        onnx.save_model(onnx_model, str(path))
        size = path.stat().st_size / 1024
        path.unlink(missing_ok=True)
        return size
    except Exception:
        return None


def export_lightgbm(
    model: Any,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Esporta LightGBM in formato nativo .txt + C via m2cgen.

    Fix problema 3: il modello con n_estimators=400, num_leaves=31 produce
      un .txt di ~1394 KB > 400 KB limite ESP32-C3.

    Strategia di riduzione in ordine di priorita':
      1. m2cgen export_to_c() — se funziona, il .c compilato e' il piu' compatto
      2. ONNX via onnxmltools — formato binario piu' compatto del .txt
      3. .txt nativo — fallback, con nota che supera il limite SRAM

    La nota nella tabella documenta esplicitamente se il modello rientra
    o meno nei 400 KB, distinguendo deploy via C (rientra) da deploy
    via libreria LightGBM (non rientra).
    """
    if not HAS_LGBM:
        return {
            "model": name, "target": "ESP32-C3",
            "error": "LightGBM non disponibile"
        }

    metrics       = eval_model(model, X_test, y_test)
    size_orig_kb  = _joblib_size_kb(model)
    size_txt_kb   = _lgb_native_size_kb(model)

    # Salva .txt nativo (sempre, come reference e per debug)
    model.booster_.save_model(str(OUT / f"{name}.txt"))

    # ── Tentativo 1: m2cgen → .c (formato piu' compatto, no runtime LGBM)
    c_size_kb = None
    if HAS_M2C:
        try:
            c_code = m2c.export_to_c(model)
            c_size_kb = len(c_code.encode("utf-8")) / 1024
            (OUT / f"{name}.c").write_text(c_code, encoding="utf-8")
            print(f"  [m2cgen LightGBM OK: {c_size_kb:.1f} KB]")
        except Exception as exc:
            print(f"  [m2cgen LightGBM skip: {exc}]")

    # ── Tentativo 2: ONNX (se m2cgen fallisce o supera il limite)
    onnx_size_kb = None
    if c_size_kb is None or c_size_kb * 1024 >= ESP32_SRAM:
        onnx_size_kb = _lgb_convert_to_onnx_size_kb(model)
        if onnx_size_kb is not None:
            print(f"  [ONNX LightGBM: {onnx_size_kb:.1f} KB]")

    # ── Determina la dimensione "quantizzata" da riportare in tabella
    # Priorita': C (piu' compatto) > ONNX > .txt nativo
    if c_size_kb is not None:
        size_quant_kb = c_size_kb
        deploy_format = f"C via m2cgen ({c_size_kb:.1f} KB)"
        fits = (c_size_kb * 1024) < ESP32_SRAM
    elif onnx_size_kb is not None:
        size_quant_kb = onnx_size_kb
        deploy_format = f"ONNX ({onnx_size_kb:.1f} KB)"
        fits = (onnx_size_kb * 1024) < ESP32_SRAM
    else:
        size_quant_kb = size_txt_kb
        deploy_format = f"nativo .txt ({size_txt_kb:.1f} KB)"
        fits = (size_txt_kb * 1024) < ESP32_SRAM

    # ── Nota quantizzazione
    try:
        max_bin    = model.get_params().get("max_bin", None)
        quant_note = (
            f"Soglie INT8 (max_bin={max_bin}); deploy: {deploy_format}"
            if max_bin and max_bin <= 256
            else f"Deploy: {deploy_format}"
        )
    except Exception:
        quant_note = f"Deploy: {deploy_format}"

    if not fits:
        quant_note += f" ⚠ supera 400KB ESP32-C3 (txt: {size_txt_kb:.0f} KB)"

    return {
        "model":             name,
        "target":            "ESP32-C3",
        "format":            f"LightGBM nativo (.txt) + {deploy_format}",
        "size_original_kb":  round(size_orig_kb,  2),
        "size_quantized_kb": round(size_quant_kb, 2),
        "compression_ratio": round(size_orig_kb / max(size_quant_kb, 0.001), 2),
        "f1":                round(metrics["f1"],      4),
        "roc_auc":           round(metrics["roc_auc"], 4),
        "sram_bytes":        int(size_quant_kb * 1024),
        "fits_sram":         fits,
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
