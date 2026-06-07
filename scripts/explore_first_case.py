from src.config import IMAGES_TR_DIR, LABELS_TR_DIR
from src.data.preprocessing import load_nifti


def main():
    image_path = IMAGES_TR_DIR / "lung_001.nii.gz"
    mask_path = LABELS_TR_DIR / "lung_001.nii.gz"

    image_volume, _ = load_nifti(image_path)
    mask_volume, _ = load_nifti(mask_path)

    print("IMAGE")
    print("path:", image_path)
    print("shape:", image_volume.shape)
    print("dtype:", image_volume.dtype)
    print("min:", image_volume.min())
    print("max:", image_volume.max())

    print()

    print("MASK")
    print("path:", mask_path)
    print("shape:", mask_volume.shape)
    print("dtype:", mask_volume.dtype)
    print("unique values:", sorted(set(mask_volume.flatten())))


if __name__ == "__main__":
    main()