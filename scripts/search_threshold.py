import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CHECKPOINTS_DIR, IMAGES_TR_DIR, LABELS_TR_DIR, METRICS_DIR
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
    Search prediction thresholds for the best 2D U-Net checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINTS_DIR / "best_unet2d.pt"
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_batches = 50

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = UNet2D(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    results = []

    for threshold in thresholds:
        total_dice = 0.0
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        evaluated_batches = 0

        # Evaluate this threshold on a limited number of batches for speed.
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if batch_index >= max_batches:
                    break

                images = batch["image"].to(device).float()
                masks = batch["mask"].to(device).float()

                # Add a channel dimension: (batch, height, width) -> (batch, 1, height, width).
                if images.ndim == 3:
                    images = images.unsqueeze(1)

                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                logits = model(images)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > threshold).float()

                total_dice += dice_score(predictions, masks)
                total_iou += iou_score(predictions, masks)
                total_precision += precision_score(predictions, masks)
                total_recall += recall_score(predictions, masks)
                evaluated_batches += 1

        average_dice = total_dice / evaluated_batches
        average_iou = total_iou / evaluated_batches
        average_precision = total_precision / evaluated_batches
        average_recall = total_recall / evaluated_batches

        result = {
            "threshold": threshold,
            "dice": average_dice,
            "iou": average_iou,
            "precision": average_precision,
            "recall": average_recall,
        }
        results.append(result)

        print(
            f"Threshold {threshold:.1f} - "
            f"Dice: {average_dice:.4f} - "
            f"IoU: {average_iou:.4f} - "
            f"Precision: {average_precision:.4f} - "
            f"Recall: {average_recall:.4f}"
        )

    best_result = max(results, key=lambda item: item["dice"])
    print(
        f"Best threshold by Dice: {best_result['threshold']:.1f} "
        f"(Dice: {best_result['dice']:.4f})"
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / "threshold_search.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved threshold search results to: {output_path}")


if __name__ == "__main__":
    main()
