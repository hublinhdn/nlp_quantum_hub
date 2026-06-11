"""GloVe pretrained vector loader.

Tải qua scripts/00_download_glove.py — file kỳ vọng tại:
    data/raw/glove/glove.6B.50d.txt

Format: mỗi dòng = `token v_1 v_2 ... v_d` (space-separated, float).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_GLOVE_PATH = Path("data/raw/glove/glove.6B.50d.txt")
DEFAULT_DIM = 50


def load_glove(
    path: Path = DEFAULT_GLOVE_PATH, dim: int = DEFAULT_DIM
) -> dict[str, np.ndarray]:
    """Load GloVe text file thành dict {token: np.ndarray(dim,)}."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Chạy: python scripts/00_download_glove.py"
        )
    embeddings: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip().split(" ")
            tok = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.shape[0] != dim:
                raise ValueError(
                    f"Dòng {line_no}: token {tok!r} có {vec.shape[0]} chiều, "
                    f"không khớp dim={dim}"
                )
            embeddings[tok] = vec
    return embeddings


def average_embedding(
    text: str,
    embeddings: dict[str, np.ndarray],
    dim: int = DEFAULT_DIM,
) -> np.ndarray:
    """Trung bình các token vector. Token OOV bị bỏ qua. Nếu tất cả OOV → zero vector."""
    vecs = [embeddings[tok] for tok in text.lower().split() if tok in embeddings]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def encode_texts(
    texts: list[str],
    embeddings: dict[str, np.ndarray],
    dim: int = DEFAULT_DIM,
) -> np.ndarray:
    """Encode toàn bộ texts thành ma trận (N, dim)."""
    return np.stack([average_embedding(t, embeddings, dim) for t in texts], axis=0)
