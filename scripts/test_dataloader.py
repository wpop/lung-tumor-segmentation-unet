import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import IMAGES_TR_DIR, LABELS_TR_DIR
from src.data.dataset import LungTumorSliceDataset2D


def main() -> None:
    """
    Load the 2D dataset with a DataLoader and print information about one batch.
    """
    dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Load the first batch from the DataLoader.
    batch = next(iter(dataloader))
    images = batch["image"]
    masks = batch["mask"]

    # Convert NumPy arrays to PyTorch tensors if needed.
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).to(torch.float32)
    else:
        images = images.to(torch.float32)

    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).to(torch.uint8)
    else:
        masks = masks.to(torch.uint8)

    # Add a channel dimension: (batch, height, width) -> (batch, 1, height, width).
    if images.ndim == 3:
        images = images.unsqueeze(1)

    if masks.ndim == 3:
        masks = masks.unsqueeze(1)

    print(f"Images shape: {tuple(images.shape)}")
    print(f"Masks shape: {tuple(masks.shape)}")
    print(f"Image dtype: {images.dtype}")
    print(f"Mask dtype: {masks.dtype}")


if __name__ == "__main__":
    main()
