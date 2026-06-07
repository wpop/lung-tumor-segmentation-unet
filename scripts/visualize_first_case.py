from src.config import IMAGES_TR_DIR, LABELS_TR_DIR
from src.data.preprocessing import load_nifti, normalize_to_01, get_slice
from src.visualization.plot_slices import plot_image_and_mask, plot_overlay


def main():
    image_path = IMAGES_TR_DIR / "lung_001.nii.gz"
    mask_path = LABELS_TR_DIR / "lung_001.nii.gz"

    image_volume, _ = load_nifti(image_path)
    mask_volume, _ = load_nifti(mask_path)

    image_volume = normalize_to_01(image_volume)

    # Use the known tumor slice for this case.
    slice_index = 240

    image_slice = get_slice(image_volume, slice_index, axis=2)
    mask_slice = get_slice(mask_volume, slice_index, axis=2)

    print(f"Showing slice index: {slice_index}")

    plot_image_and_mask(
        image_slice,
        mask_slice,
        title=f"lung_001, axial slice {slice_index}",
    )

    plot_overlay(
        image_slice,
        mask_slice,
        title=f"lung_001, axial slice {slice_index} overlay",
        output_path="outputs/overlays/lung_001_slice_240_overlay.png",
    )


if __name__ == "__main__":
    main()
