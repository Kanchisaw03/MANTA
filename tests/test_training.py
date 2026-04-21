"""Tests for MANTA training utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from manta.training.loss import FocalBCELoss
from manta.training.trainer import MANTATrainer
from manta.utils.config import MANTAConfig


class _TinyDataset(Dataset):
    def __init__(self, n: int = 64) -> None:
        self.x = torch.randn(n, 1, 32)
        self.y = (self.x.mean(dim=(1, 2)) > 0).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]
        return {
            "global_view": x,
            "local_view": x[:, :16],
            "freq_bands": torch.cat([x, x, x], dim=0),
            "stellar_params": torch.tensor([5777.0, 4.4, 0.0, 20.0]),
            "label": y,
            "kepler_id": "0",
        }


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        return self.net(batch["global_view"])


def test_loss_decrease() -> None:
    dataset = _TinyDataset(n=64)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = _TinyModel()
    optimizer = AdamW(model.parameters(), lr=1.0e-2)
    loss_fn = FocalBCELoss(gamma=2.0, alpha=0.75)

    losses = []
    for _ in range(5):
        for batch in loader:
            pred = model(batch)
            target = batch["label"].view(-1, 1)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

    assert losses[-1] < losses[0]


def test_checkpoint_save_load(tmp_path: Path) -> None:
    config = MANTAConfig()
    model = _TinyModel()
    optimizer = AdamW(model.parameters(), lr=1.0e-3)
    loss_fn = FocalBCELoss()

    trainer = MANTATrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        device="cpu",
        config=config,
        checkpoint_dir=tmp_path,
    )

    sample = {
        "global_view": torch.randn(4, 1, 32),
        "local_view": torch.randn(4, 1, 16),
        "freq_bands": torch.randn(4, 3, 32),
        "stellar_params": torch.randn(4, 4),
        "label": torch.randint(0, 2, (4,), dtype=torch.float32),
    }

    with torch.no_grad():
        before = model(sample).clone()

    ckpt = trainer.save_checkpoint(epoch=1, metrics={"val_auc_roc": 0.5}, tag="unit")

    # Change weights to verify restoration.
    for p in model.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.1)

    trainer.load_checkpoint(ckpt)

    with torch.no_grad():
        after = model(sample)

    assert torch.allclose(before, after, atol=1.0e-7)


def test_focal_loss_hard_example_weighting() -> None:
    loss_fn = FocalBCELoss(gamma=2.0, alpha=0.75)

    easy_pred = torch.tensor([[0.95], [0.05]], dtype=torch.float32)
    easy_true = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    hard_pred = torch.tensor([[0.55], [0.45]], dtype=torch.float32)
    hard_true = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    easy_loss = float(loss_fn(easy_pred, easy_true).item())
    hard_loss = float(loss_fn(hard_pred, hard_true).item())

    assert hard_loss > easy_loss
