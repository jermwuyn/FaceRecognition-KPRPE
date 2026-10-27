import argparse
import copy
from pathlib import Path
from src.face_recognition.evaluation.verification import evaluate_verification
import torch

from src.face_recognition.data.dataloader import create_dataloader
from src.face_recognition.training.trainer import FaceRecognitionTrainer


def checkpoint_best_auc(path, trainer, validation_loader):
    """Recover old checkpoints' score using the current validation protocol."""
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=True)
    if checkpoint.get("best_auc") is not None:
        return float(checkpoint["best_auc"])

    print(f"Checkpoint {path} has no best_auc; evaluating its model on validation.")
    model = copy.deepcopy(trainer.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate_verification(model, validation_loader, trainer.device)["roc_auc"]


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

    validation_loader = create_dataloader(
        split = "validation",
        batch_size = args.batch_size,
    )

    trainer = FaceRecognitionTrainer(
        learning_rate=args.learning_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_auc = float("-inf")
    best_paths = {output_dir / "best_model.pth"}

    if args.resume is not None:
        checkpoint_info = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint_info["epoch"] + 1

        saved_best_auc = checkpoint_info.get("best_auc")

        if saved_best_auc is not None:
            best_auc = float(saved_best_auc)
        else:
            best_auc = checkpoint_best_auc(args.resume, trainer, validation_loader)

        # A newer best model may exist when resuming an earlier epoch.
        best_paths.add(Path(args.resume).parent / "best_model.pth")

    for best_path in sorted({path.resolve() for path in best_paths}):
        if best_path.is_file():
            best_auc = max(best_auc, checkpoint_best_auc(
                best_path, trainer, validation_loader,
            ))

    if args.resume is not None:
        print(
            f"Resuming from epoch {checkpoint_info['epoch']} "
            f"| Best ROC-AUC: {best_auc:.4f}"
        )

    print(f"Device: {trainer.device}")
    print(f"Training samples: {len(dataloader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total epochs: {args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = trainer.train_one_epoch(dataloader)

        validation_metrics = evaluate_verification(
            model=trainer.model,
            dataloader=validation_loader,
            device=trainer.device,
        )
        print(
            f"Validation ROC-AUC: {validation_metrics['roc_auc']:.4f} | "
            f"EER: {validation_metrics['eer']:.4f} | "
            f"Best Accuracy: {validation_metrics['best_accuracy']:.4f}",
            flush=True,
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}",
            flush=True,
        )

        checkpoint_path = output_dir / f"epoch_{epoch:03d}.pth"
        current_auc = validation_metrics["roc_auc"]
        is_best = current_auc > best_auc
        best_auc = max(best_auc, current_auc)

        trainer.save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            loss=train_loss,
            best_auc=best_auc,
        )

        if is_best:
            trainer.save_checkpoint(
                path=output_dir / "best_model.pth",
                epoch = epoch,
                loss = train_loss,
                best_auc=best_auc,
            )

            print(
                f"New best model saved with ROC-AUC: {best_auc:.4f}",
                flush = True, 
            )

    print("Training completed.")


if __name__ == "__main__":
    main()
