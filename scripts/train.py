import argparse
from pathlib import Path

import torch

from src.face_recognition.data.dataloader import create_dataloader
from src.face_recognition.training.trainer import FaceRecognitionTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train ViT + KP-RPE + ArcFace from scratch"
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    dataloader = create_dataloader(
        split="train",
        batch_size=args.batch_size,
    )

    trainer = FaceRecognitionTrainer(
        learning_rate=args.learning_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1

    if args.resume is not None:
        checkpoint_info = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint_info["epoch"] + 1

        print(
            f"Resuming from epoch {checkpoint_info['epoch']}"
        )

    print(f"Device: {trainer.device}")
    print(f"Training samples: {len(dataloader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total epochs: {args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = trainer.train_one_epoch(dataloader)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}",
            flush=True,
        )

        checkpoint_path = output_dir / f"epoch_{epoch:03d}.pth"

        trainer.save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            loss=train_loss,
        )

    print("Training completed.")


if __name__ == "__main__":
    main()