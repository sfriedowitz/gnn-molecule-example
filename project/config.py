from dataclasses import dataclass
import numpy as np

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

MODEL_PARAMETER_GRID = {
    "batch_size": [32, 128, 64],
    "learning_rate": [0.1, 0.05, 0.01, 0.001],
    "weight_decay": [0.0001, 0.00001, 0.001],
    "sgd_momentum": [0.9, 0.8, 0.5],
    "scheduler_gamma": [0.995, 0.9, 0.8, 0.5, 1],
    "pos_weight": [1.0],
    "embedding_size": [8, 16, 32, 64, 128],
    "n_heads": [1, 2, 3, 4],
    "n_layers": [3],
    "dropout_rate": [0.2, 0.5, 0.9],
    "top_k_ratio": [0.2, 0.5, 0.8, 0.9],
    "top_k_every_n": [0],
    "dense_neurons": [16, 128, 64, 256, 32],
}

INPUT_SCHEMA = Schema(
    [
        TensorSpec(np.dtype(np.float32), (-1, 30), name="x"),
        TensorSpec(np.dtype(np.float32), (-1, 11), name="edge_attr"),
        TensorSpec(np.dtype(np.int32), (2, 30), name="edge_index"),
        TensorSpec(np.dtype(np.int32), (-1, 1), name="batch_index"),
    ]
)

OUTPUT_SCHEMA = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])

MODEL_SIGNATURE = ModelSignature(inputs=INPUT_SCHEMA, outputs=OUTPUT_SCHEMA)


@dataclass
class ModelParameters:
    """Best model parameter defaults obtained from hyperparameter optimization."""

    batch_size: int = 128
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    sgd_momentum: float = 0.8
    scheduler_gamma: float = 0.8
    pos_weight: float = 1.3
    embedding_size: int = 64
    n_heads: int = 3
    n_layers: int = 4
    dropout_rate: float = 0.2
    top_k_ratio: float = 0.5
    top_k_every_n: int = 1
    dense_neurons: int = 256
