import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import IMAGES_TR_DIR, LABELS_TR_DIR
from src.data.dataset import LungTumorSliceDataset2D
from src.models.losses import BCEDiceLoss
from src.models.unet2d import UNet2D


def main() -> None:
    """
    Run one simple training step with the 2D U-Net model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = UNet2D().to(device)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
    masks = masks.to(torch.float32)

    # Add a channel dimension: (batch, height, width) -> (batch, 1, height, width).
    if images.ndim == 3:
        images = images.unsqueeze(1)

    if masks.ndim == 3:
        masks = masks.unsqueeze(1)

    images = images.to(device)
    masks = masks.to(device)

    # Run one forward and backward training step.
    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs, masks)
    loss.backward()
    optimizer.step()

    print(f"Device: {device}")
    print(f"Input shape: {tuple(images.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
