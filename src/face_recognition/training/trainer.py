import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.face_recognition.models.vit import FaceRecognitionModel
from src.face_recognition.models.arcface import ArcFaceHead


class FaceRecognitionTrainer:
    def __init__(
        self,
        device=None,
        learning_rate=1e-4,
        weight_decay=0.05,
        num_classes=296,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # ViT + KP-RPE + Face Embedding Head
        self.model = FaceRecognitionModel().to(self.device)

        # ArcFace classification head
        self.arcface = ArcFaceHead(
            embedding_dim=256,
            num_classes=num_classes,
        ).to(self.device)

        # Classification loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimize both recognition model and ArcFace weights
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.model.parameters(),
                },
                {
                    "params": self.arcface.parameters(),
                },
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    def train_one_epoch(self, dataloader):
        self.model.train()
        self.arcface.train()

        total_loss = 0.0
        total_samples = 0

        progress_bar = tqdm(
            dataloader,
            desc="Training",
            unit="batch",
        )
        for batch in progress_bar:
            images = batch["image"].to(
                self.device, non_blocking=True
            )
            landmarks = batch["landmarks"].to(
                self.device, non_blocking=True
            )
            labels = batch["label"].to(
                self.device, non_blocking=True
            )

            self.optimizer.zero_grad(set_to_none=True)

            embeddings = self.model(
                images,
                landmarks=landmarks,
            )

            logits = self.arcface(embeddings, labels)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}"
            )

        return total_loss / total_samples

    def save_checkpoint(self, path, epoch, loss):
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "arcface_state_dict": self.arcface.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": loss,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        from pathlib import Path

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.arcface.load_state_dict(
            checkpoint["arcface_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        return {
            "epoch": checkpoint["epoch"],
            "train_loss": checkpoint["train_loss"],
        }