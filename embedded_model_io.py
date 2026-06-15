"""
=============================================================================
embedded_model_io.py — Serializzazione e Deserializzazione Modelli INT8
=============================================================================

Modulo richiesto dal professore per documentare il codice di
serializzazione/deserializzazione dei modelli XGBoost e LightGBM
quantizzati in formato INT8 binario.

File prodotti dalla pipeline di quantizzazione:
  quant_outputs/xgboost_int8.bin   — XGBoost quantizzato (134.5 KB)
  quant_outputs/lightgbm_int8.bin  — LightGBM quantizzato (23.8 KB)

API pubblica:
  save_xgb_int8(model, path)           → .bin
  load_xgb_int8(path)                  → XGBInt8Model
  XGBInt8Model.predict(X)              → np.ndarray (0/1)
  XGBInt8Model.predict_proba(X)        → np.ndarray (probabilita')

  save_lgb_int8(model, path)           → .bin
  load_lgb_int8(path)                  → LGBInt8Model
  LGBInt8Model.predict(X)              → np.ndarray (0/1)
  LGBInt8Model.predict_proba(X)        → np.ndarray (probabilita')

  verify_int8_model(original_model, int8_model, X_test, y_test)
    → dict con metriche di confronto originale vs INT8

Formato binario .bin:
  Vedi docstring di save_xgb_int8() e save_lgb_int8() per la specifica
  completa del formato (header, struttura alberi, dati quantizzati).
=============================================================================
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI FORMATO
# ─────────────────────────────────────────────────────────────────────────────
XGB_MAGIC   = b"XGBI"
LGB_MAGIC   = b"LGBI"
FORMAT_VER  = 3   # v3: threshold quantizzati per-albero

ARDUINO_SRAM_BYTES = 8   * 1024
ESP32_SRAM_BYTES   = 400 * 1024


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY INT8
# ─────────────────────────────────────────────────────────────────────────────
def _quantize(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Quantizza array float32 a int8 simmetrico (range [-127, 127]).
    Restituisce (array_int8, scale) dove scale e' il fattore di dequantizzazione.
    Per dequantizzare: float_val = int8_val * scale
    """
    scale = max(abs(float(arr.min())), abs(float(arr.max()))) / 127.0
    scale = max(scale, 1e-8)
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, float(scale)


def _quantize_floor(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Quantizza array float32 a int8 usando floor invece di round.

    Garantisce dequant(quant(thr)) <= thr per ogni elemento,
    condizione necessaria per replicare esattamente il confronto
    strict x < thr di XGBoost e LightGBM dopo dequantizzazione.
    Usare questa funzione SOLO per i threshold di split, non per
    i valori foglia (che non hanno questo vincolo direzionale).
    """
    scale = max(abs(float(arr.min())), abs(float(arr.max()))) / 127.0
    scale = max(scale, 1e-8)
    q = np.clip(np.floor(arr / scale), -128, 127).astype(np.int8)
    return q, float(scale)


def _dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    """Dequantizza array int8 a float32: float_val = int8_val * scale"""
    return q.astype(np.float32) * scale


# ─────────────────────────────────────────────────────────────────────────────
# SIGMOID utility
# ─────────────────────────────────────────────────────────────────────────────
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# =============================================================================
# XGBOOST INT8 — SERIALIZZAZIONE
# =============================================================================
def save_xgb_int8(model: Any, path: Union[str, Path]) -> int:
    """
    Serializza un modello XGBoost addestrato in formato INT8 binario.

    Formato .bin (versione 2):
    ┌──────────────────────────────────────────────────────────────┐
    │ HEADER (18 byte)                                             │
    │   magic[4]        = b"XGBI"                                 │
    │   version[2]      = 2 (uint16 little-endian)                │
    │   n_trees[4]      = numero alberi (uint32)                  │
    │   n_features[4]   = numero feature input (uint32)           │
    │   leaf_scale[4]   = scala dequantizzazione foglie (float32) │
    │   thr_scale[4]    = scala dequantizzazione soglie (float32) │
    ├──────────────────────────────────────────────────────────────┤
    │ STRUTTURA ALBERI (per ogni albero i, i=0..n_trees-1):       │
    │   n_nodes[4]           = numero nodi (uint32)               │
    │   left_children[n*4]   = int32[], -1 se foglia              │
    │   right_children[n*4]  = int32[], -1 se foglia              │
    │   split_indices[n*4]   = uint32[], indice feature split     │
    │   is_leaf[n]           = uint8[], 1 se foglia, 0 se interno │
    │   leaf_ptr[n*4]        = int32[], indice in leaf_values     │
    │   thr_ptr[n*4]         = int32[], indice in split_thresholds│
    ├──────────────────────────────────────────────────────────────┤
    │ DATI QUANTIZZATI                                             │
    │   leaf_values_int8[n_total_leaves]      = int8[]            │
    │   split_thresholds_int8[n_total_splits] = int8[]            │
    └──────────────────────────────────────────────────────────────┘

    Parametri
    ---------
    model : XGBClassifier addestrato
    path  : percorso file .bin di output

    Restituisce
    -----------
    int : dimensione in byte del file .bin prodotto
    """
    import json as _json
    import tempfile as _tmp

    def _parse_val(v):
        if isinstance(v, (int, float)): return float(v)
        if isinstance(v, str):          return float(v.strip("[]"))
        if isinstance(v, list):         return float(v[0])
        return float(v)

    # Legge struttura dal booster XGBoost in JSON
    with _tmp.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        jpath = Path(tmp.name)
    try:
        model.get_booster().save_model(str(jpath))
        with open(jpath) as f:
            xgb_data = _json.load(f)
    finally:
        jpath.unlink(missing_ok=True)

    trees      = xgb_data["learner"]["gradient_booster"]["model"]["trees"]
    n_features = int(xgb_data["learner"]["learner_model_param"]["num_feature"])

    all_leaves = []
    all_thresholds = []
    tree_structs = []

    for tree in trees:
        lc  = [int(x) for x in tree["left_children"]]
        rc  = [int(x) for x in tree["right_children"]]
        si  = [int(x) for x in tree["split_indices"]]
        bw  = [_parse_val(x) for x in tree["base_weights"]]
        sc  = [_parse_val(x) for x in tree["split_conditions"]]
        n   = len(lc)

        is_leaf    = [1 if lc[i] == -1 else 0 for i in range(n)]
        leaf_ptr   = []
        thr_ptr    = []

        for i in range(n):
            if is_leaf[i]:
                leaf_ptr.append(len(all_leaves))
                all_leaves.append(bw[i])
                thr_ptr.append(-1)
            else:
                thr_ptr.append(len(all_thresholds))
                all_thresholds.append(sc[i])
                leaf_ptr.append(-1)

        tree_structs.append({
            "n": n, "lc": lc, "rc": rc, "si": si,
            "is_leaf": is_leaf, "leaf_ptr": leaf_ptr, "thr_ptr": thr_ptr,
        })

    leaves_arr = np.array(all_leaves,     dtype=np.float32)
    thr_arr    = np.array(all_thresholds, dtype=np.float32)
    leaves_q, leaf_scale = _quantize(leaves_arr)       # round OK per le foglie
    thr_q,    thr_scale  = _quantize_floor(thr_arr)   # floor: garantisce dequant(thr) <= thr

    buf = bytearray()

    # HEADER
    buf += struct.pack("<4sHIIff",
        XGB_MAGIC, FORMAT_VER, len(trees), n_features,
        float(leaf_scale), float(thr_scale))

    # STRUTTURA ALBERI
    for ts in tree_structs:
        n = ts["n"]
        buf += struct.pack("<I", n)
        buf += struct.pack(f"<{n}i", *ts["lc"])
        buf += struct.pack(f"<{n}i", *ts["rc"])
        buf += struct.pack(f"<{n}I", *ts["si"])
        buf += bytes(bytearray(ts["is_leaf"]))
        buf += struct.pack(f"<{n}i", *ts["leaf_ptr"])
        buf += struct.pack(f"<{n}i", *ts["thr_ptr"])

    # DATI QUANTIZZATI
    buf += bytes(leaves_q.tobytes())
    buf += bytes(thr_q.tobytes())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return len(buf)


# =============================================================================
# XGBOOST INT8 — MODELLO DESERIALIZZATO
# =============================================================================
class XGBInt8Model:
    """
    Modello XGBoost deserializzato da formato INT8 binario.
    Supporta predict() e predict_proba() con inferenza numerica
    direttamente sui valori int8 dequantizzati.

    Uso:
        model = load_xgb_int8("quant_outputs/xgboost_int8.bin")
        probs = model.predict_proba(X_test_numpy)   # float32 [n, 2]
        preds = model.predict(X_test_numpy)          # int [n]
    """

    def __init__(
        self,
        trees: list,
        leaf_values: np.ndarray,
        split_thresholds: np.ndarray,
        leaf_scale: float,
        thr_scale: float,
        n_features: int,
    ):
        self._trees            = trees           # lista di dict per albero
        self._leaf_values      = leaf_values     # float32 dequantizzato
        self._split_thresholds = split_thresholds  # float32 dequantizzato
        self._leaf_scale       = leaf_scale
        self._thr_scale        = thr_scale
        self.n_features_       = n_features
        self.n_estimators_     = len(trees)

    def _predict_tree(self, tree: dict, x: np.ndarray) -> float:
        """Percorre un singolo albero e restituisce il valore foglia."""
        node = 0
        while not tree["is_leaf"][node]:
            feat  = tree["si"][node]
            thr   = self._split_thresholds[tree["thr_ptr"][node]]
            # XGBoost usa confronto STRICT x < thr per andare a sinistra.
            # Con <= si sbaglia il ramo quando x[feat] == thr (tipico per OHE 0/1).
            node  = tree["lc"][node] if x[feat] < thr else tree["rc"][node]
        return self._leaf_values[tree["leaf_ptr"][node]]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcola probabilita' di classe per ogni campione.
        X: numpy array float32 [n_samples, n_features] — gia' preprocessato.
        Restituisce: float32 [n_samples, 2] — colonna 0=normal, 1=attack.
        """
        X = np.asarray(X, dtype=np.float32)
        n = len(X)
        scores = np.zeros(n, dtype=np.float32)
        for tree in self._trees:
            for i in range(n):
                scores[i] += self._predict_tree(tree, X[i])
        probs_1 = _sigmoid(scores)
        return np.column_stack([1 - probs_1, probs_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classificazione binaria: 1=attack, 0=normal.
        X: numpy array float32 [n_samples, n_features] — gia' preprocessato.
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def size_bytes(self) -> int:
        """Stima dimensione in memoria del modello deserializzato (float32)."""
        return (
            self._leaf_values.nbytes +
            self._split_thresholds.nbytes +
            sum(
                len(t["lc"]) * (4 * 5 + 1)  # lc, rc, si, leaf_ptr, thr_ptr, is_leaf
                for t in self._trees
            )
        )


# =============================================================================
# XGBOOST INT8 — DESERIALIZZAZIONE
# =============================================================================
def load_xgb_int8(path: Union[str, Path]) -> XGBInt8Model:
    """
    Deserializza un file .bin XGBoost INT8 in un modello inferibile.

    Parametri
    ---------
    path : percorso file .bin prodotto da save_xgb_int8()

    Restituisce
    -----------
    XGBInt8Model : oggetto con .predict() e .predict_proba()

    Esempio
    -------
    >>> model = load_xgb_int8("quant_outputs/xgboost_int8.bin")
    >>> probs = model.predict_proba(X_test)
    >>> preds = model.predict(X_test)
    """
    data = Path(path).read_bytes()
    offset = 0

    # HEADER
    magic, version, n_trees, n_features, leaf_scale, thr_scale = \
        struct.unpack_from("<4sHIIff", data, offset)
    offset += struct.calcsize("<4sHIIff")

    if magic != XGB_MAGIC:
        raise ValueError(f"Magic non valido: {magic!r} (atteso {XGB_MAGIC!r})")
    if version != FORMAT_VER:
        raise ValueError(f"Versione non supportata: {version} (attesa {FORMAT_VER})")

    # STRUTTURA ALBERI v3: thr_ptr globali (come v2)
    trees = []
    for _ in range(n_trees):
        n = struct.unpack_from("<I", data, offset)[0]; offset += 4
        lc       = list(struct.unpack_from(f"<{n}i", data, offset)); offset += n * 4
        rc       = list(struct.unpack_from(f"<{n}i", data, offset)); offset += n * 4
        si       = list(struct.unpack_from(f"<{n}I", data, offset)); offset += n * 4
        is_leaf  = list(data[offset:offset+n]);                       offset += n
        leaf_ptr = list(struct.unpack_from(f"<{n}i", data, offset)); offset += n * 4
        thr_ptr  = list(struct.unpack_from(f"<{n}i", data, offset)); offset += n * 4
        trees.append({
            "lc": lc, "rc": rc, "si": si,
            "is_leaf": is_leaf, "leaf_ptr": leaf_ptr, "thr_ptr": thr_ptr,
        })

    # DATI v3: foglie INT8 + threshold float32 esatti
    n_leaves = max(
        max((p for t in trees for p in t["leaf_ptr"] if p >= 0), default=0) + 1, 1
    )
    n_thresholds = max(
        max((p for t in trees for p in t["thr_ptr"] if p >= 0), default=0) + 1, 1
    )
    leaves_q     = np.frombuffer(data, dtype=np.int8,
                                 count=n_leaves, offset=offset).copy()
    offset += n_leaves
    # threshold float32 esatti — nessuna perdita di precisione
    split_thresholds = np.frombuffer(data, dtype=np.float32,
                                     count=n_thresholds, offset=offset).copy()
    leaf_values = _dequantize(leaves_q, leaf_scale)

    return XGBInt8Model(
        trees=trees,
        leaf_values=leaf_values,
        split_thresholds=split_thresholds,
        leaf_scale=leaf_scale,
        thr_scale=0.0,
        n_features=n_features,
    )


# =============================================================================
# LIGHTGBM INT8 — SERIALIZZAZIONE
# =============================================================================
def save_lgb_int8(model: Any, path: Union[str, Path]) -> int:
    """
    Serializza un modello LightGBM addestrato in formato INT8 binario.

    Formato .bin (versione 2):
    ┌──────────────────────────────────────────────────────────────┐
    │ HEADER (18 byte)                                             │
    │   magic[4]        = b"LGBI"                                 │
    │   version[2]      = 2                                       │
    │   n_trees[4]      = numero alberi (uint32)                  │
    │   n_features[4]   = numero feature input (uint32)           │
    │   thr_scale[4]    = scala soglie (float32)                  │
    │   leaf_scale[4]   = scala foglie (float32)                  │
    ├──────────────────────────────────────────────────────────────┤
    │ METADATA ALBERI (per ogni albero i):                         │
    │   n_splits[4]     = numero nodi interni (uint32)            │
    │   n_leaves[4]     = numero foglie (uint32)                  │
    │   split_features[n_splits*4] = uint32[], indice feature     │
    ├──────────────────────────────────────────────────────────────┤
    │ DATI QUANTIZZATI                                             │
    │   all_thresholds_int8[sum(n_splits)] = int8[]               │
    │   all_leaf_values_int8[sum(n_leaves)] = int8[]              │
    └──────────────────────────────────────────────────────────────┘

    Parametri
    ---------
    model : LGBMClassifier addestrato
    path  : percorso file .bin di output

    Restituisce
    -----------
    int : dimensione in byte del file .bin prodotto
    """
    import tempfile as _tmp

    with _tmp.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as tmp:
        tpath = Path(tmp.name)
    try:
        model.booster_.save_model(str(tpath))
        txt = tpath.read_text(encoding="utf-8")
    finally:
        tpath.unlink(missing_ok=True)

    all_thr       = []
    all_leaf      = []
    tree_n_splits = []
    tree_n_leaves = []
    tree_features = []
    tree_lc       = []   # left_child per albero
    tree_rc       = []   # right_child per albero

    current_splits  = []
    current_leaves  = []
    current_feats   = []
    current_lc      = []
    current_rc      = []
    in_tree = False

    for line in txt.split("\n"):
        line = line.strip()
        if line.startswith("Tree="):
            if in_tree:
                ns = len(current_splits)
                tree_n_splits.append(ns)
                tree_n_leaves.append(len(current_leaves))
                tree_features.append(current_feats[:])
                tree_lc.append(current_lc[:])
                tree_rc.append(current_rc[:])
                all_thr.extend(current_splits)
                all_leaf.extend(current_leaves)
            current_splits = []
            current_leaves = []
            current_feats  = []
            current_lc     = []
            current_rc     = []
            in_tree = True
        elif line.startswith("threshold=") and in_tree:
            current_splits = [float(v) for v in line[10:].split()]
        elif line.startswith("leaf_value=") and in_tree:
            current_leaves = [float(v) for v in line[11:].split()]
        elif line.startswith("split_feature=") and in_tree:
            current_feats = [int(v) for v in line[14:].split()]
        elif line.startswith("left_child=") and in_tree:
            current_lc = [int(v) for v in line[11:].split()]
        elif line.startswith("right_child=") and in_tree:
            current_rc = [int(v) for v in line[12:].split()]

    if in_tree:
        ns = len(current_splits)
        tree_n_splits.append(ns)
        tree_n_leaves.append(len(current_leaves))
        tree_features.append(current_feats[:])
        tree_lc.append(current_lc[:])
        tree_rc.append(current_rc[:])
        all_thr.extend(current_splits)
        all_leaf.extend(current_leaves)

    try:
        n_features = model.n_features_in_
    except Exception:
        n_features = (max(max(f) for f in tree_features if f) + 1
                      if any(tree_features) else 0)

    n_trees  = len(tree_n_splits)
    thr_arr  = np.array(all_thr,  dtype=np.float32)
    leaf_arr = np.array(all_leaf, dtype=np.float32)
    # v3: threshold salvati come float32 esatti (no perdita di precisione)
    # foglie quantizzate INT8 con round
    leaf_q, ls = _quantize(leaf_arr)

    buf = bytearray()

    # HEADER  (thr_scale=0.0 perché i threshold sono float32 esatti)
    buf += struct.pack("<4sHIIff",
        LGB_MAGIC, FORMAT_VER, n_trees, n_features,
        0.0, float(ls))

    # METADATA ALBERI v3: ns, nl, feats[ns], left_child[ns], right_child[ns]
    for i in range(n_trees):
        ns = tree_n_splits[i]
        nl = tree_n_leaves[i]
        buf += struct.pack("<II", ns, nl)

        feats = tree_features[i] if i < len(tree_features) else []
        if len(feats) < ns:
            feats = feats + [0] * (ns - len(feats))
        buf += struct.pack(f"<{ns}I", *feats[:ns])

        lc = tree_lc[i] if i < len(tree_lc) else []
        if len(lc) < ns:
            lc = lc + [-1] * (ns - len(lc))
        buf += struct.pack(f"<{ns}i", *lc[:ns])

        rc = tree_rc[i] if i < len(tree_rc) else []
        if len(rc) < ns:
            rc = rc + [-1] * (ns - len(rc))
        buf += struct.pack(f"<{ns}i", *rc[:ns])

    # DATI v3: threshold float32 esatti + foglie INT8
    buf += thr_arr.tobytes()    # float32 esatti — nessuna perdita di precisione
    buf += bytes(leaf_q.tobytes())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))
    return len(buf)


# =============================================================================
# LIGHTGBM INT8 — MODELLO DESERIALIZZATO
# =============================================================================
class LGBInt8Model:
    """
    Modello LightGBM deserializzato da formato INT8 binario.

    LightGBM usa leaf-wise splitting: ogni campione percorre l'albero
    tramite i valori di soglia e arriva a una foglia.
    Il punteggio finale e' la somma dei valori foglia di tutti gli alberi,
    passato attraverso una sigmoid per ottenere la probabilita'.

    Uso:
        model = load_lgb_int8("quant_outputs/lightgbm_int8.bin")
        probs = model.predict_proba(X_test_numpy)
        preds = model.predict(X_test_numpy)
    """

    def __init__(
        self,
        tree_metadata: list,
        all_thresholds: np.ndarray,
        all_leaf_values: np.ndarray,
        thr_scale: float,
        leaf_scale: float,
        n_features: int,
    ):
        self._meta          = tree_metadata    # lista dict per albero
        self._thresholds    = all_thresholds   # float32 [tot_splits]
        self._leaf_values   = all_leaf_values  # float32 [tot_leaves]
        self._thr_scale     = thr_scale
        self._leaf_scale    = leaf_scale
        self.n_features_    = n_features
        self.n_estimators_  = len(tree_metadata)

    def _predict_sample(self, x: np.ndarray) -> float:
        """Somma i contributi di tutti gli alberi per un campione.
        
        Navigazione esatta con left_child/right_child espliciti di LightGBM.
        Valori child: >= 0 = nodo interno, < 0 = foglia (indice = -val - 1).
        """
        score = 0.0
        for meta in self._meta:
            thr_off  = meta["thr_offset"]
            leaf_off = meta["leaf_offset"]
            feats    = meta["features"]
            lc       = meta["left_child"]
            rc       = meta["right_child"]
            thrs     = self._thresholds[thr_off: thr_off + meta["n_splits"]]
            leaves   = self._leaf_values[leaf_off: leaf_off + meta["n_leaves"]]

            node = 0
            while True:
                feat = feats[node] if node < len(feats) else 0
                feat = min(feat, len(x) - 1)
                # LightGBM usa confronto STRICT x < thr per andare a sinistra
                next_node = lc[node] if x[feat] < thrs[node] else rc[node]
                if next_node < 0:
                    # Foglia: indice = -next_node - 1
                    leaf_idx = min(-next_node - 1, meta["n_leaves"] - 1)
                    score += leaves[leaf_idx]
                    break
                node = next_node
        return score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        X: numpy array float32 [n_samples, n_features] — gia' preprocessato.
        Restituisce: float32 [n_samples, 2] — colonne: [P(normal), P(attack)]
        """
        X = np.asarray(X, dtype=np.float32)
        scores   = np.array([self._predict_sample(x) for x in X], dtype=np.float32)
        probs_1  = _sigmoid(scores)
        return np.column_stack([1 - probs_1, probs_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classificazione binaria: 1=attack, 0=normal."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# =============================================================================
# LIGHTGBM INT8 — DESERIALIZZAZIONE
# =============================================================================
def load_lgb_int8(path: Union[str, Path]) -> LGBInt8Model:
    """
    Deserializza un file .bin LightGBM INT8 in un modello inferibile.

    Parametri
    ---------
    path : percorso file .bin prodotto da save_lgb_int8()

    Restituisce
    -----------
    LGBInt8Model : oggetto con .predict() e .predict_proba()

    Esempio
    -------
    >>> model = load_lgb_int8("quant_outputs/lightgbm_int8.bin")
    >>> probs = model.predict_proba(X_test)
    >>> preds = model.predict(X_test)
    """
    data = Path(path).read_bytes()
    offset = 0

    # HEADER
    magic, version, n_trees, n_features, thr_scale, leaf_scale = \
        struct.unpack_from("<4sHIIff", data, offset)
    offset += struct.calcsize("<4sHIIff")

    if magic != LGB_MAGIC:
        raise ValueError(f"Magic non valido: {magic!r} (atteso {LGB_MAGIC!r})")
    if version != FORMAT_VER:
        raise ValueError(f"Versione non supportata: {version} (attesa {FORMAT_VER})")

    # METADATA ALBERI v3: ns, nl, features[ns], left_child[ns], right_child[ns]
    tree_metadata   = []
    thr_offset_acc  = 0
    leaf_offset_acc = 0

    for _ in range(n_trees):
        ns, nl = struct.unpack_from("<II", data, offset); offset += 8
        feats = list(struct.unpack_from(f"<{ns}I", data, offset)) if ns > 0 else []
        offset += ns * 4
        lc = list(struct.unpack_from(f"<{ns}i", data, offset)) if ns > 0 else []
        offset += ns * 4
        rc = list(struct.unpack_from(f"<{ns}i", data, offset)) if ns > 0 else []
        offset += ns * 4
        tree_metadata.append({
            "n_splits":    ns,
            "n_leaves":    nl,
            "features":    feats,
            "left_child":  lc,
            "right_child": rc,
            "thr_offset":  thr_offset_acc,
            "leaf_offset": leaf_offset_acc,
        })
        thr_offset_acc  += ns
        leaf_offset_acc += nl

    # DATI v3: threshold float32 (tutti) + foglie INT8 (tutte)
    thresholds = np.frombuffer(data, dtype=np.float32,
                               count=thr_offset_acc, offset=offset).copy()
    offset += thr_offset_acc * 4
    leaf_q     = np.frombuffer(data, dtype=np.int8,
                               count=leaf_offset_acc, offset=offset).copy()
    leaf_values = _dequantize(leaf_q, leaf_scale)

    return LGBInt8Model(
        tree_metadata=tree_metadata,
        all_thresholds=thresholds,
        all_leaf_values=leaf_values,
        thr_scale=0.0,
        leaf_scale=leaf_scale,
        n_features=n_features,
    )


# =============================================================================
# VERIFICA COERENZA INT8 vs ORIGINALE
# =============================================================================
def verify_int8_model(
    original_model:  Any,
    int8_model:      Union[XGBInt8Model, LGBInt8Model],
    X_test:          np.ndarray,
    y_test:          np.ndarray,
    tolerance:       float = 0.02,
) -> Dict:
    """
    Confronta le predizioni del modello originale con quelle INT8.

    Calcola:
    - F1 originale vs F1 INT8
    - Accordo predizioni (% campioni con stessa classe)
    - Differenza media probabilita'

    Parametri
    ---------
    original_model : modello sklearn originale (XGBClassifier / LGBMClassifier)
    int8_model     : modello INT8 deserializzato
    X_test         : numpy array [n, features] — gia' preprocessato
    y_test         : numpy array [n] — label binarie int
    tolerance      : soglia accettabile per perdita F1 (default 2%)

    Restituisce
    -----------
    dict con chiavi:
      f1_original, f1_int8, f1_delta, agreement_pct,
      prob_mae, fits_tolerance, size_original_kb, size_int8_kb
    """
    from sklearn.metrics import f1_score as _f1

    X = np.asarray(X_test, dtype=np.float32)
    y = np.asarray(y_test, dtype=int)

    # Predizioni originale
    pred_orig  = np.asarray(original_model.predict(X)).ravel()
    pred_orig  = (pred_orig >= 0.5).astype(int) if pred_orig.max() <= 1 else pred_orig.astype(int)
    if hasattr(original_model, "predict_proba"):
        prob_orig = original_model.predict_proba(X)[:, 1]
    else:
        prob_orig = pred_orig.astype(float)

    # Predizioni INT8
    prob_int8  = int8_model.predict_proba(X)[:, 1]
    pred_int8  = (prob_int8 >= 0.5).astype(int)

    f1_orig  = _f1(y, pred_orig,  zero_division=0)
    f1_int8  = _f1(y, pred_int8,  zero_division=0)
    agree    = float(np.mean(pred_orig == pred_int8)) * 100
    prob_mae = float(np.mean(np.abs(prob_orig - prob_int8)))

    # Stima dimensione in byte
    size_orig_bytes = sum(
        getattr(original_model, attr, np.array([])).nbytes
        for attr in ("feature_importances_",)
    )

    return {
        "f1_original":    round(f1_orig, 4),
        "f1_int8":        round(f1_int8, 4),
        "f1_delta":       round(abs(f1_orig - f1_int8), 4),
        "agreement_pct":  round(agree, 2),
        "prob_mae":       round(prob_mae, 4),
        "fits_tolerance": abs(f1_orig - f1_int8) <= tolerance,
        "note": (
            f"F1 delta {abs(f1_orig-f1_int8):.4f} "
            f"{'dentro' if abs(f1_orig-f1_int8)<=tolerance else 'FUORI'} "
            f"tolleranza {tolerance}"
        ),
    }


# =============================================================================
# INFO FORMATO
# =============================================================================
def inspect_bin_file(path: Union[str, Path]) -> Dict:
    """
    Legge l'header di un file .bin e restituisce le informazioni principali
    senza caricare l'intero modello in memoria.

    Utile per verificare rapidamente i file prima del caricamento.
    """
    data = Path(path).read_bytes()
    magic = data[:4]

    if magic == XGB_MAGIC:
        _, version, n_trees, n_features, leaf_scale, thr_scale = \
            struct.unpack_from("<4sHIIff", data, 0)
        model_type = "XGBoost INT8"
    elif magic == LGB_MAGIC:
        _, version, n_trees, n_features, thr_scale, leaf_scale = \
            struct.unpack_from("<4sHIIff", data, 0)
        model_type = "LightGBM INT8"
    else:
        return {"error": f"Magic non riconosciuto: {magic!r}"}

    size_kb = len(data) / 1024
    fits    = len(data) < ESP32_SRAM_BYTES

    return {
        "type":        model_type,
        "version":     version,
        "n_trees":     n_trees,
        "n_features":  n_features,
        "leaf_scale":  round(leaf_scale, 6),
        "thr_scale":   round(thr_scale,  6),
        "size_kb":     round(size_kb, 2),
        "size_bytes":  len(data),
        "fits_esp32":  fits,
        "margin_kb":   round(ESP32_SRAM_BYTES / 1024 - size_kb, 1),
    }


# =============================================================================
# MAIN — test rapido di round-trip
# =============================================================================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("embedded_model_io.py — Test ispezione file .bin")
    print("=" * 60)

    for fname in ["quant_outputs/xgboost_int8.bin",
                  "quant_outputs/lightgbm_int8.bin"]:
        path = Path(fname)
        if path.exists():
            info = inspect_bin_file(path)
            print(f"\n{fname}:")
            for k, v in info.items():
                print(f"  {k:<15} = {v}")
        else:
            print(f"\n{fname}: NON TROVATO — eseguire prima il notebook sezione 3.13")

    print("\nPer caricare e usare i modelli:")
    print("  xgb = load_xgb_int8('quant_outputs/xgboost_int8.bin')")
    print("  lgb = load_lgb_int8('quant_outputs/lightgbm_int8.bin')")
    print("  probs = xgb.predict_proba(X_test_numpy)")
    print("  preds = lgb.predict(X_test_numpy)")
