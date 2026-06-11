"""Logistic Regression baseline với TF-IDF features.

Pipeline:
    1. TfidfVectorizer(ngram=(1,2), min_df=2)  — fit trên train
    2. Grid search C ∈ {0.01, 0.1, 1, 10, 100} → chọn best theo dev accuracy
    3. Refit với best C trên train
    4. Eval trên dev (KHÔNG eval test — để dành Phase 6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.baselines.common import Metrics, SplitData, compute_metrics


DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass
class LRTfidfResult:
    """Kết quả 1 run LR+TF-IDF."""

    seed: int
    best_C: float
    n_features: int
    dev_metrics: Metrics
    grid_dev_accs: dict[float, float]  # C → dev accuracy


def train_lr_tfidf(
    train: SplitData,
    dev: SplitData,
    seed: int = 0,
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_iter: int = 1000,
) -> LRTfidfResult:
    """Train một LR+TF-IDF với grid search trên dev.

    Parameters
    ----------
    train, dev : SplitData
    seed : int
        Random seed cho LogisticRegression (LR là deterministic nhưng vẫn truyền seed cho clarity).
    C_grid : tuple
        Các giá trị C để thử.
    ngram_range : tuple
        N-gram range cho TfidfVectorizer. (1,1) = unigram only; (1,2) = uni + bi.
    min_df : int
        Drop token xuất hiện < min_df trên train.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range, lowercase=False, min_df=min_df
    )  # text đã được lowercase từ Phase 2.5 — lowercase=False để tránh double work
    X_train = vectorizer.fit_transform(train.texts)
    X_dev = vectorizer.transform(dev.texts)

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

    # Refit với best C
    final = LogisticRegression(C=best_C, max_iter=max_iter, random_state=seed, solver="liblinear")
    final.fit(X_train, train.labels)
    dev_pred = final.predict(X_dev)
    dev_score = final.predict_proba(X_dev)[:, 1]
    dev_metrics = compute_metrics(dev.labels, dev_pred, dev_score)

    return LRTfidfResult(
        seed=seed,
        best_C=best_C,
        n_features=X_train.shape[1],
        dev_metrics=dev_metrics,
        grid_dev_accs=grid_accs,
    )


def result_to_dict(r: LRTfidfResult) -> dict[str, Any]:
    return {
        "seed": r.seed,
        "best_C": r.best_C,
        "n_features": r.n_features,
        "grid_dev_accs": {str(k): v for k, v in r.grid_dev_accs.items()},
        "dev_metrics": r.dev_metrics.to_dict(),
    }
