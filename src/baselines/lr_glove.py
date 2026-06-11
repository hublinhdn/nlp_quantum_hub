"""Logistic Regression + averaged GloVe (dense baseline).

So sánh với LR+TF-IDF (sparse) — cho thấy việc dùng dense embedding có giúp
hơn không trên phrase ngắn của SST.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.baselines.common import Metrics, SplitData, compute_metrics
from src.baselines.glove import DEFAULT_DIM, DEFAULT_GLOVE_PATH, encode_texts, load_glove


DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass
class LRGloveResult:
    seed: int
    best_C: float
    embedding_dim: int
    dev_metrics: Metrics
    grid_dev_accs: dict[float, float]


def train_lr_glove(
    train: SplitData,
    dev: SplitData,
    glove_path: Path = DEFAULT_GLOVE_PATH,
    dim: int = DEFAULT_DIM,
    seed: int = 0,
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
    max_iter: int = 1000,
    embeddings: dict[str, np.ndarray] | None = None,
) -> LRGloveResult:
    """Tham số ``embeddings`` cho phép cache giữa các seed (load 1 lần)."""
    if embeddings is None:
        embeddings = load_glove(glove_path, dim=dim)

    X_train = encode_texts(train.texts, embeddings, dim=dim)
    X_dev = encode_texts(dev.texts, embeddings, dim=dim)

    grid_accs: dict[float, float] = {}
    best_C = C_grid[0]
    best_acc = -1.0
    for C in C_grid:
        clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, solver="liblinear")
        clf.fit(X_train, train.labels)
        acc = float(clf.score(X_dev, dev.labels))
        grid_accs[C] = acc
        if acc > best_acc:
            best_acc = acc
            best_C = C

    final = LogisticRegression(C=best_C, max_iter=max_iter, random_state=seed, solver="liblinear")
    final.fit(X_train, train.labels)
    dev_pred = final.predict(X_dev)
    dev_score = final.predict_proba(X_dev)[:, 1]
    dev_metrics = compute_metrics(dev.labels, dev_pred, dev_score)

    return LRGloveResult(
        seed=seed,
        best_C=best_C,
        embedding_dim=dim,
        dev_metrics=dev_metrics,
        grid_dev_accs=grid_accs,
    )


def result_to_dict(r: LRGloveResult) -> dict[str, Any]:
    return {
        "seed": r.seed,
        "best_C": r.best_C,
        "embedding_dim": r.embedding_dim,
        "grid_dev_accs": {str(k): v for k, v in r.grid_dev_accs.items()},
        "dev_metrics": r.dev_metrics.to_dict(),
    }
