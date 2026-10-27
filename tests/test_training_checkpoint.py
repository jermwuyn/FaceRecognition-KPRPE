"""Checkpoint regression tests; synthetic CPU models, no LFW training."""
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts import train
from src.face_recognition.training.trainer import FaceRecognitionTrainer
from src.face_recognition.evaluation.verification import evaluate_verification


def tiny_trainer():
    trainer = FaceRecognitionTrainer.__new__(FaceRecognitionTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = nn.Linear(2, 2)
    trainer.arcface = nn.Linear(2, 2)
    trainer.optimizer = torch.optim.AdamW([
        {"params": trainer.model.parameters()},
        {"params": trainer.arcface.parameters()},
    ])
    return trainer


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_roundtrip(tmp_path, legacy):
    trainer = tiny_trainer()
    trainer.arcface(trainer.model(torch.ones(1, 2))).sum().backward()
    trainer.optimizer.step()
    path = tmp_path / "checkpoint.pth"
    trainer.save_checkpoint(path, epoch=7, loss=1.25, best_auc=0.85)
    saved = torch.load(path, weights_only=True)
    if legacy:
        saved.pop("best_auc")
        torch.save(saved, path)
    restored = tiny_trainer()
    info = restored.load_checkpoint(path)
    assert info == {"epoch": 7, "train_loss": 1.25,
                    "best_auc": None if legacy else 0.85}
    for name in ("model", "arcface"):
        for a, b in zip(getattr(trainer, name).parameters(),
                        getattr(restored, name).parameters()):
            torch.testing.assert_close(a, b)
    for key, state in trainer.optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, restored.optimizer.state_dict()["state"][key][field])
    assert restored.optimizer.param_groups[0]["lr"] == trainer.optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("score", [0.7, 0.85, 0.9])
@pytest.mark.parametrize("legacy", [False, True])
def test_resume_selection(tmp_path, monkeypatch, score, legacy):
    trainer = tiny_trainer()
    resume = tmp_path / "epoch_007.pth"
    best = tmp_path / "best_model.pth"
    trainer.save_checkpoint(resume, 7, 1.0, 0.8)
    trainer.save_checkpoint(best, 6, 1.0, 0.85)
    if legacy:
        for path in (resume, best):
            checkpoint = torch.load(path, weights_only=True)
            checkpoint.pop("best_auc")
            torch.save(checkpoint, path)
    original_best = best.read_bytes()
    steps = []
    trainer.train_one_epoch = lambda loader: steps.append(1) or 0.5
    # Legacy scores are recomputed, then the single simulated epoch is evaluated.
    scores = iter([0.8, 0.85, score] if legacy else [score])
    monkeypatch.setattr(train, "evaluate_verification", lambda *a, **kw: {
        "roc_auc": next(scores), "eer": 0.2, "best_accuracy": 0.5,
    })
    monkeypatch.setattr(train, "FaceRecognitionTrainer", lambda **kw: trainer)
    monkeypatch.setattr(train, "create_dataloader", lambda **kw: SimpleNamespace(dataset=[0]))
    monkeypatch.setattr(sys, "argv", ["train", "--resume", str(resume),
                                      "--epochs", "8", "--output-dir", str(tmp_path)])
    train.main()
    assert steps == [1]
    saved = torch.load(tmp_path / "epoch_008.pth", weights_only=True)
    assert saved["epoch"] == 8
    assert saved["best_auc"] == max(0.85, score)
    if score <= 0.85:
        assert best.read_bytes() == original_best
    else:
        saved_best = torch.load(best, weights_only=True)
        assert saved_best["epoch"] == 8
        assert saved_best["best_auc"] == score


def test_validation_uses_eval_without_gradients():
    class Embeddings(nn.Module):
        def forward(self, images, landmarks):
            assert not self.training
            assert not torch.is_grad_enabled()
            return images

    model = Embeddings().train()
    metrics = evaluate_verification(model, [{
        "image": torch.tensor([[1., 0.], [1., 0.], [0., 1.]]),
        "landmarks": torch.zeros(3, 5, 2),
        "identity": ["a", "a", "b"],
        "image_path": ["a1", "a2", "b1"],
    }], "cpu")
    assert metrics["roc_auc"] == 1.0
    assert metrics["num_positive_pairs"] == 1
    assert metrics["num_negative_pairs"] == 2
