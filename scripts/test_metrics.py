import torch
from torch.utils.data import DataLoader

from src.config import CHECKPOINTS_DIR, IMAGES_TR_DIR, LABELS_TR_DIR
from src.data.dataset import LungTumorSliceDataset2D
from src.models.unet2d import UNet2D
from src.training.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)


def main() -> None:
    """
    Run inference on one batch and print binary segmentation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINTS_DIR / "unet2d_epoch_5.pt"
    threshold = 0.1

    dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = UNet2D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load one batch from the DataLoader.
    batch = next(iter(dataloader))
    images = batch["image"].to(torch.float32)
    masks = batch["mask"].to(torch.float32)

    # Add a channel dimension: (batch, height, width) -> (batch, 1, height, width).
    if images.ndim == 3:
        images = images.unsqueeze(1)

    if masks.ndim == 3:
        masks = masks.unsqueeze(1)

    images = images.to(device)
    masks = masks.to(device)

    # Run inference without tracking gradients.
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.sigmoid(logits)

    # Apply the temporary threshold before passing predictions to metric helpers.
    predictions = (probabilities > threshold).float()

    print(f"Probability min: {probabilities.min().item():.4f}")
    print(f"Probability max: {probabilities.max().item():.4f}")
    print(f"Probability mean: {probabilities.mean().item():.4f}")
    print(f"Prediction threshold: {threshold:.1f}")
    print(f"Dice score: {dice_score(predictions, masks):.4f}")
    print(f"IoU score: {iou_score(predictions, masks):.4f}")
    print(f"Precision score: {precision_score(predictions, masks):.4f}")
    print(f"Recall score: {recall_score(predictions, masks):.4f}")


if __name__ == "__main__":
    main()
