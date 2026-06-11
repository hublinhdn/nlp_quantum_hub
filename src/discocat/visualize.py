"""Visualize DisCoCat string diagrams.

Lưu thành PNG dùng matplotlib Agg backend (chạy được trên SSH headless).

Diagram của lambeq có method ``.draw(path=..., figsize=...)`` — gọi trực tiếp
là cách an toàn nhất, không cần quản lý figure thủ công.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_diagram(diagram: Any, path: Path, figsize: tuple[float, float] = (6.0, 4.0)) -> None:
    """Lưu 1 diagram thành PNG.

    Ưu tiên dùng API ``diagram.draw(path=...)`` của lambeq để né bug figure leak.
    Sau khi save, đóng MỌI figure để tránh `More than 20 figures` warning.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        diagram.draw(path=str(path), figsize=figsize)
    except Exception:
        # Fallback: tự tạo figure rồi save
        fig, ax = plt.subplots(figsize=figsize)
        try:
            diagram.draw(ax=ax)
        except TypeError:
            diagram.draw()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    plt.close("all")  # quan trọng: đóng mọi figure lambeq tạo ngầm


def save_sample_grid(
    diagrams: list[Any],
    texts: list[str],
    labels: list[int],
    out_dir: Path,
    n_samples: int = 20,
    prefix: str = "sample",
) -> list[Path]:
    """Lưu n_samples diagrams (mỗi cái 1 PNG) vào out_dir.

    Tên file: {prefix}_{idx:03d}_{label}_{slug}.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, (diag, text, label) in enumerate(zip(diagrams[:n_samples], texts, labels)):
        if diag is None:
            continue
        label_str = "POS" if label == 1 else "NEG"
        slug = _slugify(text)[:30]
        filename = f"{prefix}_{i:03d}_{label_str}_{slug}.png"
        path = out_dir / filename
        try:
            save_diagram(diag, path)
            saved.append(path)
        except Exception as e:
            print(f"  [warn] Không vẽ được {text!r}: {type(e).__name__}: {e}")
        # Cứ mỗi 10 file đóng hết figure để chắc chắn
        if (i + 1) % 10 == 0:
            plt.close("all")
    plt.close("all")
    return saved


def _slugify(text: str) -> str:
    """Chuyển text thành tên file an toàn."""
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in " -_":
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "phrase"
