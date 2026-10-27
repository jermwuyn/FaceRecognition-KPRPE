from torch.utils.data import DataLoader

from src.face_recognition.data.lfw_dataset import LFWDataset


def create_dataloader(
    split="train",
    batch_size=16,
    num_workers=0,
):
    dataset = LFWDataset(split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader