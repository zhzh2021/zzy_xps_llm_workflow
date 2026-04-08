"""
Model builders for XAS deep learning.
"""

from __future__ import annotations

from typing import List, Optional


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "TensorFlow is required for deep learning models. "
            "Install with: pip install tensorflow"
        ) from exc


def build_mlp(
    input_dim: int,
    hidden_layers: List[int],
    task_type: str,
    n_classes: Optional[int] = None,
    l2: float = 0.0,
    dropout: float = 0.0,
):
    _require_tensorflow()
    from tensorflow.keras import Sequential, regularizers
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential()
    for i, units in enumerate(hidden_layers):
        if i == 0:
            model.add(
                Dense(
                    units,
                    input_dim=input_dim,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2) if l2 else None,
                )
            )
        else:
            model.add(
                Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2) if l2 else None,
                )
            )
        if dropout and dropout > 0:
            model.add(Dropout(dropout))

    if task_type == "regression":
        model.add(Dense(1))
    elif task_type == "classification":
        if n_classes is None:
            raise ValueError("n_classes is required for classification")
        model.add(Dense(n_classes, activation="softmax"))
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return model


def build_cnn(
    input_length: int,
    task_type: str,
    n_classes: Optional[int] = None,
    filters: Optional[List[int]] = None,
    kernel_size: int = 5,
    pool_size: int = 2,
    dense_units: Optional[List[int]] = None,
    dropout: float = 0.0,
    use_global_avg_pool: bool = False,
    dilations: Optional[List[int]] = None,
):
    _require_tensorflow()
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import (
        Conv1D,
        MaxPooling1D,
        Flatten,
        GlobalAveragePooling1D,
        Dense,
        Dropout,
    )

    if filters is None:
        filters = [32, 64]
    if dense_units is None:
        dense_units = [128, 64]

    model = Sequential()
    if dilations is None:
        dilations = [1] * len(filters)
    model.add(
        Conv1D(
            filters[0],
            kernel_size=kernel_size,
            activation="relu",
            dilation_rate=dilations[0],
            input_shape=(input_length, 1),
        )
    )
    model.add(MaxPooling1D(pool_size=pool_size))
    for idx, f in enumerate(filters[1:], start=1):
        model.add(Conv1D(f, kernel_size=kernel_size, activation="relu", dilation_rate=dilations[idx] if idx < len(dilations) else 1))
        model.add(MaxPooling1D(pool_size=pool_size))
    if use_global_avg_pool:
        model.add(GlobalAveragePooling1D())
    else:
        model.add(Flatten())
    for units in dense_units:
        model.add(Dense(units, activation="relu"))
        if dropout and dropout > 0:
            model.add(Dropout(dropout))

    if task_type == "regression":
        model.add(Dense(1))
    elif task_type == "classification":
        if n_classes is None:
            raise ValueError("n_classes is required for classification")
        model.add(Dense(n_classes, activation="softmax"))
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return model
