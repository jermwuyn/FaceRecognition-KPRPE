import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        num_classes=296,
        scale=30.0,
        margin=0.5,
    ):
        super().__init__()

        if embedding_dim <= 0 or num_classes <= 1:
            raise ValueError("Invalid embedding_dim or num_classes")
        if scale <= 0 or not (0 < margin < math.pi / 2):
            raise ValueError("Invalid scale or margin")

        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.margin_correction = math.sin(
            math.pi - margin
        ) * margin

    def forward(self, embeddings, labels):
        """
        embeddings: [B, 256]
        labels: [B], integer IDs in [0, 295]

        returns: ArcFace logits [B, 296]
        """
        if embeddings.ndim != 2:
            raise ValueError("Expected embeddings [B, embedding_dim]")

        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("Expected labels [B]")

        if labels.dtype != torch.long:
            raise TypeError("Labels must be torch.long")

        if labels.device != embeddings.device:
            raise ValueError("Labels and embeddings must share device")

        if torch.any(labels < 0) or torch.any(labels >= self.weight.shape[0]):
            raise ValueError("Label out of class range")

        normalized_embeddings = F.normalize(
            embeddings, p=2, dim=1
        )
        normalized_weights = F.normalize(
            self.weight, p=2, dim=1
        )

        cosine = F.linear(
            normalized_embeddings,
            normalized_weights,
        ).clamp(-1.0, 1.0)

        sine = torch.sqrt(
            (1.0 - cosine.square()).clamp(min=0.0)
        )

        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(
            cosine > self.threshold,
            phi,
            cosine - self.margin_correction,
        )

        one_hot = F.one_hot(
            labels,
            num_classes=self.weight.shape[0],
        ).to(dtype=cosine.dtype)

        logits = (
            one_hot * phi
            + (1.0 - one_hot) * cosine
        ) * self.scale

        return logits