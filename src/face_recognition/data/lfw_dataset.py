import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "yunet"
    / "annotations"
    / "yunet_2026may"
)

LANDMARK_NAMES = (
    "left_eye",
    "right_eye",
    "nose",
    "left_mouth",
    "right_mouth",
)


class LFWDataset(Dataset):
    def __init__(
        self,
        split="train",
        image_size=112,
    ):
        super().__init__()

        if split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {split}")

        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.split = split
        self.image_size = image_size
        self.samples = []

        annotation_path = ANNOTATION_DIR / f"{split}.csv"

        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)

        with annotation_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["status"] != "ok":
                    continue

                image_path = (
                    PROJECT_ROOT / row["image_path"]
                ).resolve()

                # Prevent references outside the project.
                if not image_path.is_relative_to(PROJECT_ROOT.resolve()):
                    raise ValueError("Image path is outside project")

                if not image_path.is_file():
                    raise FileNotFoundError(image_path)

                if int(row["width"]) != 250 or int(row["height"]) != 250:
                    raise ValueError(
                        f"Unexpected source image size: {image_path}"
                    )

                landmarks = []

                for name in LANDMARK_NAMES:
                    x = float(row[f"{name}_x"])
                    y = float(row[f"{name}_y"])
                    landmarks.append([x, y])

                landmarks = np.asarray(
                    landmarks,
                    dtype=np.float32,
                )

                if not np.isfinite(landmarks).all():
                    raise ValueError("Non-finite landmark coordinates")

                if np.any(landmarks < 0) or np.any(landmarks > 250):
                    raise ValueError("Landmarks outside image bounds")

                if split == "train":
                    label = int(row["train_label"])

                    if not 0 <= label < 296:
                        raise ValueError("Invalid train label")
                else:
                    label = -1

                self.samples.append(
                    {
                        "image_path": image_path,
                        "relative_path": row["image_path"],
                        "landmarks": landmarks,
                        "label": label,
                        "identity": row["identity_id"],
                    }
                )

        if not self.samples:
            raise ValueError(f"No valid samples in {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = cv2.imread(
            str(sample["image_path"]),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {sample['image_path']}"
            )

        if image.shape != (250, 250, 3):
            raise RuntimeError("Unexpected image dimensions")

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # Resize full image without cropping.
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        image = torch.from_numpy(
            np.ascontiguousarray(image)
        ).permute(2, 0, 1).float() / 255.0

        # Original coordinates are measured on 250x250 images.
        # Uniform resizing preserves normalized positions.
        landmarks = torch.from_numpy(
            sample["landmarks"].copy()
        ) / 250.0

        label = torch.tensor(
            sample["label"],
            dtype=torch.long,
        )

        return {
            "image": image,
            "landmarks": landmarks,
            "label": label,
            "identity": sample["identity"],
            "image_path": sample["relative_path"],
        }