import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc

@torch.inference_mode()
def extract_embeddings(model, dataloader, device):
    model.eval()
    device = torch.device(device)

    all_embeddings = []
    all_identities = []
    all_image_paths = []

    for batch in tqdm(
        dataloader,
        desc="Extracting embeddings",
        unit="batch",
    ):
        images = batch["image"].to(
            device, non_blocking=True
        )
        landmarks = batch["landmarks"].to(
            device, non_blocking=True
        )

        embeddings = model(
            images,
            landmarks=landmarks,
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())
        all_identities.extend(batch["identity"])
        all_image_paths.extend(batch["image_path"])

    if not all_embeddings:
        raise ValueError("Dataloader contains no samples")

    return {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "identities": all_identities,
        "image_paths": all_image_paths,
    }

def create_verification_pairs(identities):
    if len(identities) < 2:
        raise ValueError("At least two images are required")

    pair_indices = torch.triu_indices(
        len(identities),
        len(identities),
        offset=1,
    )

    identity_ids = {
        identity: index
        for index, identity in enumerate(dict.fromkeys(identities))
    }

    labels = torch.tensor(
        [identity_ids[identity] for identity in identities],
        dtype=torch.long,
    )

    same_identity = (
        labels[pair_indices[0]] == labels[pair_indices[1]]
    )

    return {
        "pair_indices": pair_indices,
        "same_identity": same_identity,
    }

def compute_pair_similarities(embeddings, pair_indices):
    if embeddings.ndim != 2:
        raise ValueError("Expected embeddings [N, D]")

    if pair_indices.ndim != 2 or pair_indices.shape[0] != 2:
        raise ValueError("Expected pair_indices [2, P]")

    embeddings = F.normalize(embeddings, p=2, dim=1)

    if pair_indices.device != embeddings.device:
        pair_indices = pair_indices.to(embeddings.device)

    if pair_indices.dtype != torch.long:
        raise TypeError("pair_indices must be torch.long")

    first = embeddings[pair_indices[0]]
    second = embeddings[pair_indices[1]]

    similarities = (first * second).sum(dim=1)

    return similarities


def compute_verification_metrics(similarities, same_identity):
    scores = similarities.detach().cpu().numpy()
    labels = same_identity.detach().cpu().numpy().astype(int)

    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("Expected 1D scores and labels")

    if len(scores) != len(labels):
        raise ValueError("Scores and labels must have the same length")

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain non-finite values")

    if np.unique(labels).size != 2:
        raise ValueError("Both positive and negative pairs are required")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fnr = 1.0 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2.0

    positives = labels.sum()
    negatives = len(labels) - positives

    correct = (
        tpr * positives
        + (1.0 - fpr) * negatives
    )

    accuracies = correct / len(labels)

    best_index = np.argmax(accuracies)

    return {
        "roc_auc": float(roc_auc),
        "eer": float(eer),
        "eer_threshold": float(thresholds[eer_index]),
        "best_accuracy": float(accuracies[best_index]),
        "best_threshold": float(thresholds[best_index]),
        "num_pairs": int(len(labels)),
        "num_positive_pairs": int(positives),
        "num_negative_pairs": int(negatives),
    }


def evaluate_verification(model, dataloader, device):
    # Step 1: Extract face embeddings
    extracted = extract_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    # Step 2: Create positive and negative pairs
    pairs = create_verification_pairs(
        extracted["identities"]
    )

    # Step 3: Compute cosine similarities
    similarities = compute_pair_similarities(
        extracted["embeddings"],
        pairs["pair_indices"],
    )

    # Step 4: Calculate verification metrics
    metrics = compute_verification_metrics(
        similarities,
        pairs["same_identity"],
    )

    return metrics