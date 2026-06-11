"""BiLSTM baseline cho Phase 3.

Kiến trúc:
    Token IDs → Embedding(emb_dim=32) → BiLSTM(hidden=64, 1 layer)
            → mean-pool over time → Dropout(0.3) → Linear(2*hidden → 1)
    Loss: BCEWithLogitsLoss
    Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
    Early stopping trên dev loss, patience=5.

Vocab dùng `data/processed/qnlp_vocab.tsv` (top-1000 token chuẩn của Phase 2.5)
+ thêm 2 special token: <pad>, <unk>.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.baselines.common import Metrics, SplitData, compute_metrics
from src.preprocessing.vocab import load_vocab


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


# ---------------------------------------------------------------------------
# Vocab → token IDs
# ---------------------------------------------------------------------------


def build_token_id_map(vocab_path: Path) -> dict[str, int]:
    """Map token → id (0 = pad, 1 = unk, rồi đến các token từ vocab)."""
    vocab = load_vocab(vocab_path)
    token_to_id: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok in vocab:
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)
    return token_to_id


def encode_text(text: str, token_to_id: dict[str, int]) -> list[int]:
    """Lowercase tokenize + lookup → list ID."""
    unk_id = token_to_id[UNK_TOKEN]
    return [token_to_id.get(tok, unk_id) for tok in text.lower().split()]


# ---------------------------------------------------------------------------
# Dataset + collate
# ---------------------------------------------------------------------------


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, token_to_id: dict[str, int]):
        self.input_ids = [encode_text(t, token_to_id) for t in texts]
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[list[int], float]:
        return self.input_ids[idx], float(self.labels[idx])


def collate(batch: list[tuple[list[int], float]]) -> dict[str, torch.Tensor]:
    ids_list, labels = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids_list], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(ids_list), max_len), dtype=torch.long)  # pad = 0
    for i, ids in enumerate(ids_list):
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return {
        "input_ids": padded,
        "lengths": lengths,
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)  # (B, T, E)
        # Pack để LSTM bỏ qua padding
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T, 2H)

        # Mean pool theo độ dài thực (không tính pad)
        mask = (input_ids != 0).float().unsqueeze(-1)  # (B, T, 1)
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts  # (B, 2H)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(-1)  # (B,)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class BiLSTMResult:
    seed: int
    best_epoch: int
    best_dev_loss: float
    dev_metrics: Metrics
    train_curve: list[dict[str, float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_bilstm(
    train: SplitData,
    dev: SplitData,
    vocab_path: Path,
    seed: int = 0,
    emb_dim: int = 32,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 5,
    device: str | None = None,
    verbose: bool = True,
) -> BiLSTMResult:
    """Train BiLSTM với early stopping."""
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    token_to_id = build_token_id_map(vocab_path)
    vocab_size = len(token_to_id)

    train_ds = TextDataset(train.texts, train.labels, token_to_id)
    dev_ds = TextDataset(dev.texts, dev.labels, token_to_id)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = BiLSTMClassifier(vocab_size, emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    curve: list[dict[str, float]] = []
    best_dev_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(batch["labels"])
            n_seen += len(batch["labels"])
        train_loss = total_loss / max(n_seen, 1)

        dev_loss, dev_metrics = _eval(model, dev_loader, criterion, device)
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "dev_acc": dev_metrics.accuracy,
            }
        )
        if verbose:
            print(
                f"    epoch {epoch:>3d}  "
                f"train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}  "
                f"dev_acc={dev_metrics.accuracy:.4f}"
            )

        if dev_loss < best_dev_loss - 1e-6:
            best_dev_loss = dev_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    early stop tại epoch {epoch} (best epoch={best_epoch})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    _, final_metrics = _eval(model, dev_loader, criterion, device)

    return BiLSTMResult(
        seed=seed,
        best_epoch=best_epoch,
        best_dev_loss=best_dev_loss,
        dev_metrics=final_metrics,
        train_curve=curve,
    )


@torch.no_grad()
def _eval(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, Metrics]:
    model.eval()
    total_loss = 0.0
    n_seen = 0
    all_scores: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
        labels_dev = batch["labels"].to(device)
        loss = criterion(logits, labels_dev)
        total_loss += loss.item() * len(batch["labels"])
        n_seen += len(batch["labels"])
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (scores >= 0.5).astype(int)
        all_scores.extend(scores.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["labels"].numpy().astype(int).tolist())
    avg_loss = total_loss / max(n_seen, 1)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores)
    )
    return avg_loss, metrics


def result_to_dict(r: BiLSTMResult) -> dict[str, Any]:
    return {
        "seed": r.seed,
        "best_epoch": r.best_epoch,
        "best_dev_loss": r.best_dev_loss,
        "dev_metrics": r.dev_metrics.to_dict(),
        "train_curve": r.train_curve,
    }
