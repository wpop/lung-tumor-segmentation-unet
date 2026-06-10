import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CHECKPOINTS_DIR, IMAGES_TR_DIR, LABELS_TR_DIR, PREDICTIONS_DIR
from src.data.dataset import LungTumorSliceDataset2D
from src.models.unet2d import UNet2D


def make_overlay(image: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Create a red prediction overlay on top of the grayscale CT slice.
    """
    image_rgb = np.stack([image, image, image], axis=-1)
    overlay = image_rgb.copy()
    overlay[prediction > 0] = [1.0, 0.0, 0.0]

    return (0.7 * image_rgb) + (0.3 * overlay)


def main() -> None:
    """
    Save visual examples from the best 2D U-Net checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINTS_DIR / "best_unet2d.pt"
    output_dir = PREDICTIONS_DIR / "visual_examples"
    threshold = 0.5
    max_samples = 5

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet2D(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save visualizations for the first few samples.
    with torch.no_grad():
        for sample_index, batch in enumerate(dataloader):
            if sample_index >= max_samples:
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

            image_np = images.squeeze().cpu().numpy()
            mask_np = masks.squeeze().cpu().numpy()
            probability_np = probabilities.squeeze().cpu().numpy()
            prediction_np = predictions.squeeze().cpu().numpy()
            overlay_np = make_overlay(image_np, prediction_np)

            fig, axes = plt.subplots(1, 5, figsize=(20, 4))

            axes[0].imshow(image_np, cmap="gray")
            axes[0].set_title("CT Slice")
            axes[0].axis("off")

            axes[1].imshow(mask_np, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(probability_np, cmap="gray")
            axes[2].set_title("Probability Map")
            axes[2].axis("off")

            axes[3].imshow(prediction_np, cmap="gray")
            axes[3].set_title("Binary Prediction")
            axes[3].axis("off")

            axes[4].imshow(overlay_np)
            axes[4].set_title("Overlay")
            axes[4].axis("off")

            plt.tight_layout()
            output_path = output_dir / f"sample_{sample_index + 1}.png"
            plt.savefig(output_path)
            plt.close(fig)

            print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
