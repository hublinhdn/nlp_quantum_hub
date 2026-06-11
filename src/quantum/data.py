"""Load DisCoCat diagrams từ pickle + mini-batching cho quantum training."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np


DEFAULT_PROCESSED = Path("data/processed")


def load_split(
    reader: str, split: str, processed_dir: Path = DEFAULT_PROCESSED
) -> list[dict]:
    """Load `diagrams_{reader}_{split}.pkl`.

    Returns
    -------
    list of dict: {source_id, text, label, diagram}
    """
    path = processed_dir / f"diagrams_{reader}_{split}.pkl"
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. "
            f"Chạy: python scripts/05_parse_diagrams.py --reader {reader}"
        )
    with path.open("rb") as f:
        return pickle.load(f)


def to_onehot(labels: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Convert int labels → one-hot float matrix (N, n_classes)."""
    arr = np.zeros((len(labels), n_classes), dtype=np.float32)
    arr[np.arange(len(labels)), labels] = 1.0
    return arr


def iter_batches(
    circuits: list[Any],
    labels_onehot: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[tuple[list[Any], np.ndarray]]:
    """Yield mini-batches (batch_circuits, batch_labels_onehot).

    `circuits` là list các diagram đã chuyển bằng ansatz. `labels_onehot` shape (N, 2).
    """
    n = len(circuits)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        batch_circuits = [circuits[i] for i in idx]
        batch_y = labels_onehot[idx]
        yield batch_circuits, batch_y
