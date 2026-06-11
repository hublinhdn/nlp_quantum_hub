"""Loading + filtering Stanford Sentiment Treebank (SST).

Stanford SST gốc gồm các file (sau khi giải nén `stanfordSentimentTreebank.zip`):

    stanfordSentimentTreebank/
        datasetSentences.txt   # id <TAB> sentence  — tất cả câu gốc
        datasetSplit.txt       # sentence_id,split  — 1=train, 2=test, 3=dev
        dictionary.txt         # phrase|phrase_id   — TẤT CẢ phrase (sub-string)
        sentiment_labels.txt   # phrase_id|score    — sentiment ∈ [0,1]
        SOStr.txt, STree.txt   # parse structure (không dùng ở Phase 2)
        original_rt_snippets.txt  # raw RT review (không dùng)

Module này cung cấp các hàm thuần (pure functions) để:
    1. Load 4 file chính thành DataFrame
    2. Join phrase + score
    3. Filter theo độ dài
    4. Binarize sentiment (drop neutral)
    5. Split train/dev/test
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


SPLIT_MAP = {1: "train", 2: "test", 3: "dev"}


def load_sentences(sst_dir: Path) -> pd.DataFrame:
    """Đọc datasetSentences.txt.

    Returns
    -------
    DataFrame[sentence_id: int, sentence: str]
    """
    path = sst_dir / "datasetSentences.txt"
    df = pd.read_csv(path, sep="\t", header=0)
    df.columns = ["sentence_id", "sentence"]
    df["sentence_id"] = df["sentence_id"].astype(int)
    df["sentence"] = df["sentence"].astype(str).str.strip()
    return df


def load_split(sst_dir: Path) -> pd.DataFrame:
    """Đọc datasetSplit.txt — map sentence_id → split (train/dev/test).

    Returns
    -------
    DataFrame[sentence_id: int, split: str]
    """
    path = sst_dir / "datasetSplit.txt"
    df = pd.read_csv(path, header=0)
    df.columns = ["sentence_id", "splitset_label"]
    df["sentence_id"] = df["sentence_id"].astype(int)
    df["split"] = df["splitset_label"].map(SPLIT_MAP)
    return df[["sentence_id", "split"]]


def load_dictionary(sst_dir: Path) -> pd.DataFrame:
    """Đọc dictionary.txt — tất cả phrase + phrase_id.

    File có dạng ``phrase|phrase_id``. Phrase có thể chứa ký tự ``|`` không?
    Theo định dạng gốc của Stanford thì KHÔNG — ``|`` chỉ là delimiter.
    Tuy vậy ta split từ phải để an toàn.

    Returns
    -------
    DataFrame[phrase: str, phrase_id: int]
    """
    path = sst_dir / "dictionary.txt"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            phrase, _, pid = line.rpartition("|")
            rows.append((phrase, int(pid)))
    return pd.DataFrame(rows, columns=["phrase", "phrase_id"])


def load_sentiment_labels(sst_dir: Path) -> pd.DataFrame:
    """Đọc sentiment_labels.txt — phrase_id + score ∈ [0, 1].

    Returns
    -------
    DataFrame[phrase_id: int, score: float]
    """
    path = sst_dir / "sentiment_labels.txt"
    df = pd.read_csv(path, sep="|", header=0)
    df.columns = ["phrase_id", "score"]
    df["phrase_id"] = df["phrase_id"].astype(int)
    df["score"] = df["score"].astype(float)
    return df


def binarize(
    df: pd.DataFrame,
    score_col: str = "score",
    neutral_band: tuple[float, float] = (0.4, 0.6),
) -> pd.DataFrame:
    """Drop phrase trung tính + gán label nhị phân.

    Quy ước theo paper SST gốc (Socher et al. 2013):
        score < 0.4         → label 0 (negative)
        0.4 ≤ score ≤ 0.6   → drop (neutral)
        score > 0.6         → label 1 (positive)
    """
    lo, hi = neutral_band
    mask = (df[score_col] < lo) | (df[score_col] > hi)
    out = df.loc[mask].copy()
    out["label"] = (out[score_col] > hi).astype(int)
    return out


def count_words(text: str) -> int:
    """Đếm token theo whitespace."""
    return len(text.split())


def filter_by_length(
    df: pd.DataFrame,
    max_words: int = 5,
    min_words: int = 1,
    text_col: str = "phrase",
) -> pd.DataFrame:
    """Giữ lại row có ``min_words ≤ len(text.split()) ≤ max_words``."""
    n_words = df[text_col].map(count_words)
    return df.loc[(n_words >= min_words) & (n_words <= max_words)].copy()


# ---------------------------------------------------------------------------
# High-level builders
# ---------------------------------------------------------------------------


def build_sentence_level(sst_dir: Path) -> pd.DataFrame:
    """Sentence-level: chỉ giữ các CÂU gốc (~11.8k) + score + split.

    Logic: tìm phrase_id của câu trong dictionary.txt (full sentence text =
    một phrase trong dictionary), lấy score, ghép split.

    Returns
    -------
    DataFrame[sentence_id, text, score, split]
    """
    sentences = load_sentences(sst_dir)
    split = load_split(sst_dir)
    dictionary = load_dictionary(sst_dir)
    sent_labels = load_sentiment_labels(sst_dir)

    # join: sentence → phrase_id qua text match
    merged = sentences.merge(
        dictionary, left_on="sentence", right_on="phrase", how="left"
    )
    merged = merged.merge(sent_labels, on="phrase_id", how="left")
    merged = merged.merge(split, on="sentence_id", how="left")

    out = merged[["sentence_id", "sentence", "score", "split"]].rename(
        columns={"sentence": "text"}
    )
    # Câu nào không match được phrase_id (lỗi encoding hiếm) → drop
    out = out.dropna(subset=["score", "split"]).reset_index(drop=True)
    return out


def build_phrase_level(sst_dir: Path) -> pd.DataFrame:
    """Phrase-level: tất cả ~239k phrase + score.

    Phrase-level không có split sẵn (split chỉ áp cho sentence).
    Trả về DataFrame chưa có cột ``split`` — caller tự chia.

    Returns
    -------
    DataFrame[phrase_id, text, score]
    """
    dictionary = load_dictionary(sst_dir)
    sent_labels = load_sentiment_labels(sst_dir)
    merged = dictionary.merge(sent_labels, on="phrase_id", how="left")
    out = merged.rename(columns={"phrase": "text"})[["phrase_id", "text", "score"]]
    out = out.dropna(subset=["score"]).reset_index(drop=True)
    return out


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label",
    train: float = 0.8,
    dev: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified split DataFrame thành train/dev/test theo tỉ lệ.

    Thêm cột ``split`` ∈ {train, dev, test}. Bảo toàn tỉ lệ label.
    """
    assert abs(train + dev + (1 - train - dev) - 1.0) < 1e-9
    test = 1.0 - train - dev

    pieces = []
    for label_value, group in df.groupby(label_col, sort=False):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = int(n * train)
        n_dev = int(n * dev)
        shuffled["split"] = "test"
        shuffled.loc[: n_train - 1, "split"] = "train"
        shuffled.loc[n_train : n_train + n_dev - 1, "split"] = "dev"
        pieces.append(shuffled)

    out = pd.concat(pieces, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def prepare_short_sst2(
    sst_dir: Path,
    source: Literal["sentences", "phrases"] = "phrases",
    max_words: int = 5,
    min_words: int = 1,
    neutral_band: tuple[float, float] = (0.4, 0.6),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """End-to-end: load → filter → binarize → split.

    Returns
    -------
    dict {"train": df, "dev": df, "test": df} — mỗi df có columns
    [text, label, score, source_id].
    """
    if source == "sentences":
        df = build_sentence_level(sst_dir)
        df = filter_by_length(df, max_words=max_words, min_words=min_words, text_col="text")
        df = binarize(df, score_col="score", neutral_band=neutral_band)
        df = df.rename(columns={"sentence_id": "source_id"})
        df = df[["text", "label", "score", "split", "source_id"]]
    elif source == "phrases":
        df = build_phrase_level(sst_dir)
        df = filter_by_length(df, max_words=max_words, min_words=min_words, text_col="text")
        df = binarize(df, score_col="score", neutral_band=neutral_band)
        df = df.rename(columns={"phrase_id": "source_id"})
        df = stratified_split(df, label_col="label", train=0.8, dev=0.1, seed=seed)
        df = df[["text", "label", "score", "split", "source_id"]]
    else:
        raise ValueError(f"source phải là 'sentences' hoặc 'phrases', không phải {source!r}")

    return {name: df[df["split"] == name].reset_index(drop=True) for name in ("train", "dev", "test")}
