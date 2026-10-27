import json
from datetime import datetime
from pathlib import Path

import torch

from src.face_recognition.models.vit import (
    PatchEmbedding,
    MultiHeadSelfAttention,
    TransformerEncoderBlock,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "tests"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def tensor_info(tensor):
    detached = tensor.detach()

    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "mean": float(detached.mean().item()),
        "std": float(detached.std().item()),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "all_finite": bool(torch.isfinite(detached).all().item()),
    }


def test_model_pipeline():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    patch_embedding = PatchEmbedding().to(device)
    attention = MultiHeadSelfAttention().to(device)

    # 1. Simulated RGB images
    images = torch.randn(
        2, 3, 112, 112,
        device=device,
        requires_grad=True,
    )

    # 2. Patch Embedding
    patches = patch_embedding(images)

    # 3. Multi-Head Self-Attention
    features = attention(patches)

    # 4. Verify expected shapes
    assert patches.shape == (2, 196, 192)
    assert features.shape == (2, 196, 192)

    # 5. Verify numerical stability
    assert torch.isfinite(patches).all().item()
    assert torch.isfinite(features).all().item()

    # 6. Backward through the entire pipeline
    loss = features.square().mean()
    loss.backward()

    assert images.grad is not None
    assert torch.isfinite(images.grad).all().item()

    modules = {
        "patch_embedding": patch_embedding,
        "multi_head_self_attention": attention,
    }

    gradient_report = {}

    for name, module in modules.items():
        grads = [
            p.grad for p in module.parameters()
            if p.requires_grad
        ]

        assert grads
        assert all(g is not None for g in grads)
        assert all(torch.isfinite(g).all().item() for g in grads)

        gradient_report[name] = {
            "parameters_with_gradients": len(grads),
            "all_gradients_finite": True,
        }

    # 7. Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "pytorch_version": torch.__version__,
        "stages": {
            "input_images": tensor_info(images),
            "patch_embedding_output": tensor_info(patches),
            "attention_output": tensor_info(features),
        },
        "parameters": {
            "patch_embedding": count_parameters(patch_embedding),
            "multi_head_self_attention": count_parameters(attention),
            "total": (
                count_parameters(patch_embedding)
                + count_parameters(attention)
            ),
        },
        "backward": {
            "loss": float(loss.item()),
            "input_gradient_finite": True,
            "modules": gradient_report,
        },
        "status": "PASSED",
    }

    # 8. Save results after all assertions pass
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "model_pipeline.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 9. Print readable summary
    print("\n=== MODEL PIPELINE TEST ===")
    print(f"Device: {device}")

    for name, info in report["stages"].items():
        print(
            f"{name}: "
            f"shape={info['shape']}, "
            f"mean={info['mean']:.6f}, "
            f"std={info['std']:.6f}"
        )

    print("\nParameters:")
    for name, count in report["parameters"].items():
        print(f"  {name}: {count:,}")

    print(f"\nBackward loss: {loss.item():.6f}")
    print("Gradient check: PASSED")
    print(f"Saved report: {output_file}")
    print("Overall status: PASSED")


if __name__ == "__main__":
    test_model_pipeline()