import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from src.config import HU_MAX, HU_MIN, IMAGES_TR_DIR, MASKS_DIR, OUTPUTS_DIR


def normalize_ct_to_uint8(ct_slice: np.ndarray) -> np.ndarray:
    """
    Clip and normalize one CT slice to 8-bit grayscale.
    """
    ct_slice = np.clip(ct_slice, HU_MIN, HU_MAX)
    ct_slice = (ct_slice - HU_MIN) / (HU_MAX - HU_MIN)

    return (ct_slice * 255).astype(np.uint8)


def normalize_mask_to_uint8(mask_slice: np.ndarray) -> np.ndarray:
    """
    Convert one mask slice to 8-bit grayscale.
    """
    return (mask_slice > 0).astype(np.uint8) * 255


def main() -> None:
    """
    Export CT and mask slices for the future C++ Qt/OpenGL viewer.
    """
    patient_id = "lung_001"
    threshold = 0.5

    ct_path = IMAGES_TR_DIR / f"{patient_id}.nii.gz"
    mask_path = MASKS_DIR / f"{patient_id}_predicted_mask.nii.gz"

    export_dir = OUTPUTS_DIR / "viewer_export" / patient_id
    ct_output_dir = export_dir / "ct"
    mask_output_dir = export_dir / "masks"
    metadata_path = export_dir / "metadata.json"

    ct_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    ct_volume = nib.load(ct_path).get_fdata()
    mask_volume = nib.load(mask_path).get_fdata()

    if ct_volume.shape != mask_volume.shape:
        raise ValueError(
            f"CT and mask shapes do not match: {ct_volume.shape} vs {mask_volume.shape}"
        )

    width, height, num_slices = ct_volume.shape
    print(f"CT shape: {ct_volume.shape}")
    print(f"Mask shape: {mask_volume.shape}")

    # Export each axial CT and mask slice as a PNG image.
    for slice_index in range(num_slices):
        ct_slice = normalize_ct_to_uint8(ct_volume[:, :, slice_index])
        mask_slice = normalize_mask_to_uint8(mask_volume[:, :, slice_index])

        ct_slice_path = ct_output_dir / f"slice_{slice_index:03d}.png"
        mask_slice_path = mask_output_dir / f"mask_{slice_index:03d}.png"

        plt.imsave(ct_slice_path, ct_slice, cmap="gray", vmin=0, vmax=255)
        plt.imsave(mask_slice_path, mask_slice, cmap="gray", vmin=0, vmax=255)

    metadata = {
        "patient_id": patient_id,
        "width": width,
        "height": height,
        "num_slices": num_slices,
        "ct_folder": "ct",
        "mask_folder": "masks",
        "threshold": threshold,
        "source_volume": str(ct_path),
        "source_mask": str(mask_path),
    }

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Exported CT slices to: {ct_output_dir}")
    print(f"Exported mask slices to: {mask_output_dir}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
