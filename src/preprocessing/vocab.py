"""Vocab pruning + OOV filtering cho QNLP subset.

Mục đích: giảm vocab từ ~14k xuống ~500 token để mạch lượng tử khả thi.

Quy ước:
    - Token đếm sau khi `.lower().split()` (không lemmatize, để khớp lambeq).
    - Vocab build CHỈ trên train. Dev/test áp cùng vocab → bị drop phrase nào có
      token OOV.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản: lowercase + whitespace split.

    Để khớp với cách lambeq parse (Bobcat parser cũng dùng whitespace).
    """
    return str(text).lower().split()


def build_vocab(
    texts: list[str] | pd.Series,
    top_k: int | None = 500,
    min_freq: int = 1,
) -> dict[str, int]:
    """Build vocab từ tập text.

    Parameters
    ----------
    texts
        Iterable các câu/phrase.
    top_k
        Giữ tối đa top-K token theo tần suất. None = không giới hạn.
    min_freq
        Bỏ token có tần suất < min_freq.

    Returns
    -------
    dict {token: frequency} — theo thứ tự giảm dần (Python 3.7+ giữ order).
    """
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    most_common = counter.most_common()
    most_common = [(tok, c) for tok, c in most_common if c >= min_freq]
    if top_k is not None:
        most_common = most_common[:top_k]
    return dict(most_common)


def phrase_in_vocab(text: str, vocab: dict[str, int] | set[str]) -> bool:
    """True nếu MỌI token của phrase đều có trong vocab."""
    if isinstance(vocab, dict):
        keys = vocab.keys()
    else:
        keys = vocab
    return all(tok in keys for tok in tokenize(text))


def filter_phrases_in_vocab(
    df: pd.DataFrame,
    vocab: dict[str, int] | set[str],
    text_col: str = "text",
) -> pd.DataFrame:
    """Giữ lại các row mà text chứa toàn token trong vocab."""
    keep = df[text_col].apply(lambda t: phrase_in_vocab(t, vocab))
    return df.loc[keep].reset_index(drop=True)


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    """Ghi vocab ra file `token <TAB> count` mỗi dòng, sort theo tần suất giảm."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# token\tcount\n")
        for tok, count in vocab.items():
            f.write(f"{tok}\t{count}\n")


def load_vocab(path: Path) -> dict[str, int]:
    """Ngược của save_vocab."""
    vocab: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            tok, count = line.split("\t")
            vocab[tok] = int(count)
    return vocab
