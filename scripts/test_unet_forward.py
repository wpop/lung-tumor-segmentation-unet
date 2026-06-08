import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import IMAGES_TR_DIR, LABELS_TR_DIR
from src.data.dataset import LungTumorSliceDataset2D
from src.models.unet2d import UNet2D


def main() -> None:
    """
    Load one batch and run it through the 2D U-Net model.
    """
    dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)
    dataloader = DataLoader(dataset, batch_size=4)

    # Load one batch from the DataLoader.
    batch = next(iter(dataloader))
    images = batch["image"]
    masks = batch["mask"]

    # Convert NumPy arrays to PyTorch tensors if needed.
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    images = images.to(torch.float32)
    masks = masks.to(torch.uint8)

    # Add a channel dimension: (batch, height, width) -> (batch, 1, height, width).
    if images.ndim == 3:
        images = images.unsqueeze(1)

    if masks.ndim == 3:
        masks = masks.unsqueeze(1)

    model = UNet2D()
    output = model(images)

    print(f"Input shape: {tuple(images.shape)}")
    print(f"Mask shape: {tuple(masks.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output dtype: {output.dtype}")


if __name__ == "__main__":
    main()
