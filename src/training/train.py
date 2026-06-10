import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    IMAGES_TR_DIR,
    LABELS_TR_DIR,
    LEARNING_RATE,
    METRICS_DIR,
    NUM_EPOCHS,
)
from src.data.dataset import LungTumorSliceDataset2D
from src.models.losses import BCEDiceLoss
from src.models.unet2d import UNet2D
from src.training.metrics import (
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)


def main() -> None:
    """
    Train a simple 2D U-Net for lung tumor segmentation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split the available labeled slices into training and validation sets.
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Training slices: {len(train_dataset)}")
    print(f"Validation slices: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    model = UNet2D(in_channels=1, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    epoch_metrics = []
    metric_threshold = 0.1
    max_metric_batches = 20
    best_val_dice = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

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

        average_loss = total_loss / len(train_loader)

        # Compute validation metrics on a few batches for speed.
        model.eval()
        total_dice = 0.0
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        evaluated_batches = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                if batch_index >= max_metric_batches:
                    break

                images = batch["image"].to(device).float()
                masks = batch["mask"].to(device).float()

                # Add a channel dimension: (batch, height, width) -> (batch, 1, height, width).
                if images.ndim == 3:
                    images = images.unsqueeze(1)

                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                outputs = model(images)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > metric_threshold).float()

                total_dice += dice_score(predictions, masks)
                total_iou += iou_score(predictions, masks)
                total_precision += precision_score(predictions, masks)
                total_recall += recall_score(predictions, masks)
                evaluated_batches += 1

        average_dice = total_dice / evaluated_batches
        average_iou = total_iou / evaluated_batches
        average_precision = total_precision / evaluated_batches
        average_recall = total_recall / evaluated_batches

        if average_dice > best_val_dice:
            best_val_dice = average_dice
            best_checkpoint_path = CHECKPOINTS_DIR / "best_unet2d.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": average_dice,
                    "val_iou": average_iou,
                    "val_precision": average_precision,
                    "val_recall": average_recall,
                },
                best_checkpoint_path,
            )
            print(f"New best model saved with Dice: {best_val_dice:.4f}")

        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "loss": average_loss,
                "dice": average_dice,
                "iou": average_iou,
                "precision": average_precision,
                "recall": average_recall,
            }
        )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
            f"Average loss: {average_loss:.4f} - "
            f"Dice: {average_dice:.4f} - "
            f"IoU: {average_iou:.4f} - "
            f"Precision: {average_precision:.4f} - "
            f"Recall: {average_recall:.4f}"
        )

        # Save training metrics after each epoch.
        metrics_path = METRICS_DIR / "train_metrics.csv"
        pd.DataFrame(epoch_metrics).to_csv(metrics_path, index=False)

        checkpoint_path = CHECKPOINTS_DIR / f"unet2d_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
