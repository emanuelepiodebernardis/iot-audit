# =========================================================================
# UTILS — MODELING, PREPROCESSING, EVALUATION
# =========================================================================

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# STANDARD LIB
# ─────────────────────────────────────────────────────────────
import time
import tempfile
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, List

# ─────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ─────────────────────────────────────────────────────────────
# DEFAULT CONFIG
# ─────────────────────────────────────────────────────────────
DEFAULT_RANDOM_STATE = 42
DEFAULT_CARDINALITY_THRESHOLD = 20


# =========================================================================
# FIX CORE: DataFramePreprocessor
# =========================================================================

class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    """
    Wrapper attorno a ColumnTransformer che preserva i nomi delle feature.

    ColumnTransformer.transform() restituisce un ndarray. Quando questo
    viene passato a LightGBM (e in alcuni casi a XGBoost) dopo un fit()
    in cui il modello ha memorizzato i feature names, scikit-learn lancia:
        UserWarning: X does not have valid feature names, but
        LGBMClassifier was fitted with feature names.

    Questo wrapper intercetta l'output di transform() e lo converte in un
    pd.DataFrame con i nomi corretti da get_feature_names_out(), eliminando
    il warning alla radice senza alterare la logica del preprocessing.

    Parametri
    ----------
    column_transformer : ColumnTransformer
        L'istanza (già configurata, non ancora fittata) da avvolgere.

    Note
    ----
    - sklearn >= 1.0 richiesta per get_feature_names_out().
    - I nomi prodotti da ColumnTransformer seguono il pattern
      "pipeline_name__feature_name", es. "num__bytes_total",
      "cat__proto_unified_tcp". Questo è il comportamento standard e
      non viene alterato.
    - sparse_threshold=0 è impostato internamente per garantire output
      denso (necessario per la conversione in DataFrame).
    """

    def __init__(self, column_transformer: ColumnTransformer):
        self.column_transformer = column_transformer

    def fit(self, X, y=None):
        # Forza output denso: senza questo, con OneHotEncoder sparse,
        # il toarray() implicito in DataFrame() fallirebbe.
        self.column_transformer.set_params(sparse_threshold=0)
        self.column_transformer.fit(X, y)
        self._feature_names = self.column_transformer.get_feature_names_out()
        return self

    def transform(self, X):
        arr = self.column_transformer.transform(X)
        # Converti in array denso se sparse (fallback di sicurezza)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        return pd.DataFrame(arr, columns=self._feature_names)

    def get_feature_names_out(self, input_features=None):
        return self._feature_names


# =========================================================================
# FEATURE HANDLING
# =========================================================================

def infer_feature_types(
    X: pd.DataFrame,
    threshold: int = DEFAULT_CARDINALITY_THRESHOLD
) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and low-cardinality categorical features.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = []

    for col in X.columns:
        if col in numeric_features:
            continue
        if X[col].nunique(dropna=True) <= threshold:
            categorical_features.append(col)

    return numeric_features, categorical_features


def build_preprocessor(
    X: pd.DataFrame,
    threshold: int = DEFAULT_CARDINALITY_THRESHOLD
) -> Tuple[DataFramePreprocessor, List[str], List[str]]:
    """
    Build preprocessing pipeline (numeric + categorical).

    Restituisce un DataFramePreprocessor (non un ColumnTransformer raw)
    in modo che l'output di transform() sia sempre un DataFrame con nomi
    di colonna validi. Questo elimina i UserWarning di LightGBM/XGBoost
    sul mismatch tra feature names al fit e alla predizione.
    """
    numeric_features, categorical_features = infer_feature_types(X, threshold)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,          # sempre denso
    )

    preprocessor = DataFramePreprocessor(ct)
    return preprocessor, numeric_features, categorical_features


# =========================================================================
# MODEL FACTORY
# =========================================================================

def get_models(
    task: str = "binary",
    n_classes: int = 2,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Dict[str, Any]:
    """
    Return a dictionary of baseline models.
    """
    models = {}

    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,
        random_state=random_state,
    )

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss" if task == "binary" else "mlogloss",
        )

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,               # silenzia log LightGBM interni
        )

    return models


# =========================================================================
# PERFORMANCE UTILITIES
# =========================================================================

def measure_inference_time_ms(
    pipeline: Pipeline,
    X_eval: pd.DataFrame,
    repeats: int = 10,
    sample_size: int = 1000,
    random_state: int = DEFAULT_RANDOM_STATE
) -> float:
    """
    Measure inference latency (ms per 1000 samples).
    """
    sample_size = min(len(X_eval), sample_size)

    if len(X_eval) > sample_size:
        X_sample = X_eval.sample(n=sample_size, random_state=random_state)
    else:
        X_sample = X_eval.copy()

    # warm-up
    _ = pipeline.predict(X_sample.iloc[:min(10, len(X_sample))])

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = pipeline.predict(X_sample)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    mean_ms = float(np.mean(timings))
    return mean_ms * (1000.0 / len(X_sample))


def save_model_and_get_size_mb(
    pipeline: Pipeline,
    path: Path
) -> float:
    """
    Save model and return size in MB.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path.stat().st_size / (1024 ** 2)


def estimate_pipeline_size_mb(pipeline: Pipeline) -> float:
    """
    Estimate pipeline size without saving permanently.
    """
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        joblib.dump(pipeline, tmp_path)
        return tmp_path.stat().st_size / (1024 ** 2)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# =========================================================================
# EVALUATION
# =========================================================================

def evaluate_binary_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    save_path: Path | None = None
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluate binary classification pipeline.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "inference_ms_per_1k": measure_inference_time_ms(pipeline, X_test),
    }

    if save_path is not None:
        result["model_size_mb"] = save_model_and_get_size_mb(pipeline, save_path)
    else:
        result["model_size_mb"] = estimate_pipeline_size_mb(pipeline)

    return result, y_pred, y_prob


def evaluate_multiclass_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
):
    """
    Evaluate multiclass classification pipeline.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    labels = list(pipeline.named_steps["model"].classes_)
    y_test_bin = label_binarize(y_test, classes=labels)

    if y_test_bin.shape[1] == 1:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc_macro": roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr"),
        "pr_auc_macro": average_precision_score(y_test_bin, y_prob, average="macro"),
        "inference_ms_per_1k": measure_inference_time_ms(pipeline, X_test),
    }

    result["model_size_mb"] = estimate_pipeline_size_mb(pipeline)

    return result, y_pred, y_prob, labels


# =========================================================================
# CROSS-VALIDATION
# =========================================================================

def cross_validate_binary_models(
    X: pd.DataFrame,
    y: pd.Series,
    task_name: str,
    n_splits: int = 5,
    models: Dict[str, Any] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE
):
    """
    Perform stratified cross-validation.

    NOTA SUL FIX WARNING:
    In precedenza build_preprocessor restituiva un ColumnTransformer raw.
    Il suo output .transform() era un ndarray senza nomi. LightGBM
    memorizza i feature names al fit() e li confronta al predict():
    se non coincidono emette UserWarning.

    Con DataFramePreprocessor il transform() restituisce un DataFrame
    con i nomi corretti, e il warning non viene mai generato.
    """
    if models is None:
        models = get_models()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_fold_rows = []
    summary_rows = []

    for model_name, estimator in models.items():
        fold_rows = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # build_preprocessor ora restituisce DataFramePreprocessor
            # → output di transform() è sempre pd.DataFrame con colnames
            preprocessor, _, _ = build_preprocessor(X_train)

            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", estimator),
            ])

            pipe.fit(X_train, y_train)

            res, _, _ = evaluate_binary_pipeline(
                pipe, X_val, y_val, model_name
            )

            res["fold"] = fold_idx
            res["task"] = task_name

            fold_rows.append(res)

        fold_df = pd.DataFrame(fold_rows)
        all_fold_rows.append(fold_df)

        summary = {"model": model_name, "task": task_name}
        for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            summary[f"{col}_mean"] = fold_df[col].mean()
            summary[f"{col}_std"] = fold_df[col].std()

        summary_rows.append(summary)

    return (
        pd.concat(all_fold_rows, ignore_index=True),
        pd.DataFrame(summary_rows).sort_values("f1_mean", ascending=False)
    )


# =========================================================================
# VISUALIZATION
# =========================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, title, filename, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_metric_comparison(df_results, metric_col, title, filename, sort_desc=True):
    plot_df = df_results[["model", metric_col]].copy()
    plot_df = plot_df.sort_values(metric_col, ascending=not sort_desc)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_df["model"], plot_df[metric_col])

    ax.set_title(title)
    ax.set_xlabel("Modello")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=20)

    for i, v in enumerate(plot_df[metric_col]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_cv_metric_comparison(df_results, metric_col, title, filename, sort_desc=True):
    mean_col = f"{metric_col}_mean"
    std_col = f"{metric_col}_std"

    plot_df = df_results[["model", mean_col, std_col]].copy()
    plot_df = plot_df.sort_values(mean_col, ascending=not sort_desc)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        plot_df["model"],
        plot_df[mean_col],
        yerr=plot_df[std_col],
        capsize=4,
    )

    ax.set_title(title)
    ax.set_xlabel("Modello")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=20)

    for i, v in enumerate(plot_df[mean_col]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def get_shap_class_index(class_name: str, label_encoder):
    """
    Get index of a class from LabelEncoder for SHAP multiclass.
    """
    if label_encoder is None:
        raise ValueError("label_encoder richiesto per il task multiclass.")

    classes = list(label_encoder.classes_)

    if class_name not in classes:
        raise ValueError(f"Classe {class_name} non trovata in {classes}")

    return int(np.where(label_encoder.classes_ == class_name)[0][0])
