import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    IMAGES_TR_DIR,
    LABELS_TR_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from src.data.dataset import LungTumorSliceDataset2D
from src.models.losses import BCEDiceLoss
from src.models.unet2d import UNet2D


def main() -> None:
    """
    Train a simple 2D U-Net for lung tumor segmentation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    model = UNet2D(in_channels=1, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            images = batch["image"].to(device).float()
            masks = batch["mask"].to(device).float()

            # Add a channel dimension: (batch, height, width) -> (batch, 1, height, width).
            if images.ndim == 3:
                images = images.unsqueeze(1)

            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Average loss: {average_loss:.4f}")

        checkpoint_path = CHECKPOINTS_DIR / f"unet2d_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
