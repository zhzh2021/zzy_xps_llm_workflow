"""
Training and evaluation helpers for XAS deep learning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .dl_utils import apply_plot_style


def get_focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    import tensorflow as tf

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_onehot * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = tf.reduce_sum(weight * ce, axis=-1)
        return tf.reduce_mean(loss)

    return focal_loss

def compile_model(model, task_type: str, learning_rate: float, loss_override: Optional[str] = None):
    from tensorflow.keras.optimizers import Adam

    if task_type == "regression":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_override or "mse", metrics=["mse", "mae"])
    elif task_type == "classification":
        loss = loss_override or "sparse_categorical_crossentropy"
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=["accuracy"])
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    return model


def train_keras_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    class_weight: Optional[Dict[int, float]] = None,
):
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.h5"
    callbacks = [
        ModelCheckpoint(str(best_model_path), monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history, best_model_path


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def plot_training_history(history, output_path: Path, title: str) -> None:
    apply_plot_style()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Val Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    apply_plot_style()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_v = float(min(y_true.min(), y_pred.min()))
    max_v = float(max(y_true.max(), y_pred.max()))
    plt.plot([min_v, max_v], [min_v, max_v], "--", color="red", label="Ideal")
    plt.title(title)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    apply_plot_style()
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_error_hist(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    apply_plot_style()
    import matplotlib.pyplot as plt

    residuals = y_true - y_pred
    plt.figure(figsize=(7, 6))
    plt.hist(residuals, bins=30, color="#1f77b4", alpha=0.8)
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    apply_plot_style()
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
