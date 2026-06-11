"""Shared utilities for Phase 3 baselines.

Phase 3 quy ước:
    - Train trên `qnlp_train.csv`
    - Tune hyperparam trên `qnlp_dev.csv`
    - KHÔNG đánh giá `qnlp_test.csv` (để dành Phase 6)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_PROCESSED = Path("data/processed")


@dataclass
class SplitData:
    """Một split (train/dev/test) đã load."""

    texts: list[str]
    labels: np.ndarray
    df: pd.DataFrame

    def __len__(self) -> int:
        return len(self.texts)


def load_qnlp(
    processed_dir: Path = DEFAULT_PROCESSED, prefix: str = "qnlp"
) -> dict[str, SplitData]:
    """Load qnlp_{train,dev,test}.csv thành SplitData.

    Returns
    -------
    dict {"train": SplitData, "dev": SplitData, "test": SplitData}
    """
    out: dict[str, SplitData] = {}
    for name in ("train", "dev", "test"):
        path = processed_dir / f"{prefix}_{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy {path}. "
                f"Chạy: python scripts/02b_subsample_qnlp.py"
            )
        df = pd.read_csv(path)
        out[name] = SplitData(
            texts=df["text"].astype(str).tolist(),
            labels=df["label"].astype(int).to_numpy(),
            df=df,
        )
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    """Container metrics chuẩn cho binary classification."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    auc_roc: float | None
    confusion_matrix: list[list[int]]
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None
) -> Metrics:
    """Tính metrics chuẩn. y_score (probability of class 1) optional cho AUC."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    auc: float | None = None
    if y_score is not None:
        try:
            auc = float(roc_auc_score(y_true, np.asarray(y_score).ravel()))
        except ValueError:
            auc = None  # ví dụ chỉ có 1 class trong y_true

    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        auc_roc=auc,
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        n_samples=int(len(y_true)),
    )


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


@dataclass
class AggregatedMetrics:
    """Mean / std qua nhiều seed."""

    n_seeds: int
    accuracy_mean: float
    accuracy_std: float
    f1_macro_mean: float
    f1_macro_std: float
    auc_roc_mean: float | None
    auc_roc_std: float | None
    per_seed: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate(metrics_list: list[Metrics]) -> AggregatedMetrics:
    accs = np.array([m.accuracy for m in metrics_list])
    f1s = np.array([m.f1_macro for m in metrics_list])
    aucs = [m.auc_roc for m in metrics_list]
    aucs_valid = [a for a in aucs if a is not None]

    return AggregatedMetrics(
        n_seeds=len(metrics_list),
        accuracy_mean=float(accs.mean()),
        accuracy_std=float(accs.std(ddof=0)),
        f1_macro_mean=float(f1s.mean()),
        f1_macro_std=float(f1s.std(ddof=0)),
        auc_roc_mean=(float(np.mean(aucs_valid)) if aucs_valid else None),
        auc_roc_std=(float(np.std(aucs_valid, ddof=0)) if aucs_valid else None),
        per_seed=[m.to_dict() for m in metrics_list],
    )


def format_metrics_row(name: str, agg: AggregatedMetrics) -> str:
    """Một dòng cho bảng tổng kết."""
    auc_str = f"{agg.auc_roc_mean:.4f}±{agg.auc_roc_std:.4f}" if agg.auc_roc_mean is not None else "  N/A"
    return (
        f"{name:<20} "
        f"acc={agg.accuracy_mean:.4f}±{agg.accuracy_std:.4f}  "
        f"f1m={agg.f1_macro_mean:.4f}±{agg.f1_macro_std:.4f}  "
        f"auc={auc_str}  "
        f"(n_seeds={agg.n_seeds})"
    )
