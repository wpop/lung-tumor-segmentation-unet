import nibabel as nib
import numpy as np
import torch

from src.config import CHECKPOINTS_DIR, HU_MAX, HU_MIN, IMAGES_TR_DIR, MASKS_DIR
from src.models.unet2d import UNet2D


def normalize_slice(image_slice: np.ndarray) -> np.ndarray:
    """
    Clip and normalize one CT slice to the [0, 1] range.
    """
    image_slice = np.clip(image_slice, HU_MIN, HU_MAX)
    image_slice = (image_slice - HU_MIN) / (HU_MAX - HU_MIN)

    return image_slice.astype(np.float32)


def main() -> None:
    """
    Predict a full 3D mask by running the 2D model slice by slice.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINTS_DIR / "best_unet2d.pt"
    input_path = IMAGES_TR_DIR / "lung_001.nii.gz"
    output_path = MASKS_DIR / "lung_001_predicted_mask.nii.gz"
    threshold = 0.5

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model = UNet2D(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_nifti = nib.load(input_path)
    image_volume = input_nifti.get_fdata()
    predicted_mask = np.zeros(image_volume.shape, dtype=np.uint8)

    print(f"Input shape: {image_volume.shape}")

    # Run inference on each axial slice and rebuild the full volume.
    with torch.no_grad():
        for slice_index in range(image_volume.shape[2]):
            image_slice = normalize_slice(image_volume[:, :, slice_index])
            image_tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.to(device)

            logits = model(image_tensor)
            probabilities = torch.sigmoid(logits)
            prediction = (probabilities > threshold).squeeze().cpu().numpy()

            predicted_mask[:, :, slice_index] = prediction.astype(np.uint8)

    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the predicted mask with the same affine matrix as the input CT volume.
    output_nifti = nib.Nifti1Image(predicted_mask, input_nifti.affine)
    nib.save(output_nifti, output_path)

    print(f"Output shape: {predicted_mask.shape}")
    print(f"Saved predicted mask to: {output_path}")


if __name__ == "__main__":
    main()
