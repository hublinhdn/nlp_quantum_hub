"""Training loop cho quantum model (PennyLane + PyTorch autograd)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.quantum.data import iter_batches, to_onehot


@dataclass
class TrainResult:
    """Kết quả 1 quantum training run."""

    config: dict[str, Any]
    best_epoch: int
    best_dev_loss: float
    best_dev_acc: float
    n_params: int
    train_time: float
    train_curve: list[dict[str, float]] = field(default_factory=list)
    best_state_dict: dict[str, torch.Tensor] | None = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, circuits, labels_np: np.ndarray) -> tuple[float, float]:
    """Trả về (loss, accuracy) trên 1 set."""
    if not circuits:
        return 0.0, 0.0
    y_onehot = torch.tensor(to_onehot(labels_np))
    outputs = model(circuits)
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(np.asarray(outputs))
    eps = 1e-9
    loss = -((y_onehot * torch.log(outputs + eps)).sum(dim=1)).mean().item()
    preds = outputs.argmax(dim=1).cpu().numpy()
    acc = float((preds == labels_np).mean())
    return loss, acc


def train_one_config(
    train_circuits: list[Any],
    train_labels: np.ndarray,
    dev_circuits: list[Any],
    dev_labels: np.ndarray,
    model,
    config: dict[str, Any],
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 20,
    seed: int = 0,
    grad_clip: float | None = 1.0,
    use_scheduler: bool = True,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 7,
    min_lr: float = 1e-5,
    verbose: bool = True,
) -> TrainResult:
    """Train 1 config với early stopping trên dev_loss.

    Stability features:
        - Gradient clipping (max_norm) — tránh explosion
        - ReduceLROnPlateau scheduler — giảm lr khi dev_loss stuck → break barren plateau
        - Longer patience — cho phép fluctuate trước khi early stop

    Trả về best checkpoint (state_dict tại epoch dev_loss thấp nhất).
    """
    set_seed(seed)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )

    train_y_onehot = to_onehot(train_labels)

    curve: list[dict[str, float]] = []
    best_dev_loss = float("inf")
    best_dev_acc = 0.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()

        # Mini-batch SGD
        epoch_loss = 0.0
        epoch_n = 0
        for batch_c, batch_y_oh in iter_batches(
            train_circuits, train_y_onehot,
            batch_size=batch_size, shuffle=True, seed=seed * 1000 + epoch,
        ):
            optimizer.zero_grad()
            outputs = model(batch_c)
            if not isinstance(outputs, torch.Tensor):
                outputs = torch.tensor(np.asarray(outputs))
            batch_y_t = torch.tensor(batch_y_oh)
            eps = 1e-9
            loss = -((batch_y_t * torch.log(outputs + eps)).sum(dim=1)).mean()
            loss.backward()
            # Gradient clipping for stability
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_c)
            epoch_n += len(batch_c)
        train_loss = epoch_loss / max(epoch_n, 1)

        # Eval
        model.eval()
        dev_loss, dev_acc = evaluate(model, dev_circuits, dev_labels)
        train_eval_loss, train_acc = evaluate(model, train_circuits, train_labels)

        # LR scheduler step (based on dev_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(dev_loss)

        elapsed = time.time() - t0
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
                "lr": current_lr,
                "time_s": elapsed,
            }
        )
        if verbose:
            print(
                f"    epoch {epoch:>3d}  loss={train_loss:.4f}  "
                f"train_acc={train_acc:.4f}  dev_loss={dev_loss:.4f}  "
                f"dev_acc={dev_acc:.4f}  lr={current_lr:.1e}  ({elapsed:.1f}s)"
            )

        # Early stopping
        if dev_loss < best_dev_loss - 1e-6:
            best_dev_loss = dev_loss
            best_dev_acc = dev_acc
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    early stop tại epoch {epoch} (best epoch={best_epoch})")
                break

    train_time = time.time() - t_start

    return TrainResult(
        config=config,
        best_epoch=best_epoch,
        best_dev_loss=best_dev_loss,
        best_dev_acc=best_dev_acc,
        n_params=n_params,
        train_time=train_time,
        train_curve=curve,
        best_state_dict=best_state,
    )
