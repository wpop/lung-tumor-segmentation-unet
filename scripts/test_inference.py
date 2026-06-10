import matplotlib.pyplot as plt
import torch

from src.config import CHECKPOINTS_DIR, IMAGES_TR_DIR, LABELS_TR_DIR, PREDICTIONS_DIR
from src.data.dataset import LungTumorSliceDataset2D
from src.models.unet2d import UNet2D


def main() -> None:
    """
    Run inference on one 2D slice and save a visual example.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINTS_DIR / "unet2d_epoch_5.pt"

    model = UNet2D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)
    sample = dataset[0]

    image = torch.from_numpy(sample["image"]).float()
    mask = sample["mask"]

    # Add batch and channel dimensions: (height, width) -> (1, 1, height, width).
    image_batch = image.unsqueeze(0).unsqueeze(0).to(device)
    threshold = 0.1

    # Run inference without tracking gradients.
    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.sigmoid(logits)
        prediction = probabilities > threshold

    image_np = image.cpu().numpy()
    probability_np = probabilities.squeeze().cpu().numpy()
    prediction_np = prediction.squeeze().cpu().numpy()

    print(f"Probability min: {probability_np.min():.4f}")
    print(f"Probability max: {probability_np.max():.4f}")
    print(f"Probability mean: {probability_np.mean():.4f}")
    print(f"Prediction threshold: {threshold:.1f}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / "inference_epoch_5_threshold_01.png"

    # Save CT slice, ground truth mask, probability map, and binary mask side by side.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("CT Slice")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(probability_np, cmap="gray")
    axes[2].set_title("Prediction Probability Map")
    axes[2].axis("off")

    axes[3].imshow(prediction_np, cmap="gray")
    axes[3].set_title("Binary Prediction Mask")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved inference figure to: {output_path}")


if __name__ == "__main__":
    main()
