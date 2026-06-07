import numpy as np

from src.config import LABELS_TR_DIR


def main():
    mask_path = LABELS_TR_DIR / "lung_001.nii.gz"

    import nibabel as nib
    mask = nib.load(str(mask_path)).get_fdata()

    tumor_slices = []

    for z in range(mask.shape[2]):
        mask_slice = mask[:, :, z]

        if np.sum(mask_slice) > 0:
            tumor_slices.append(z)

    print(f"Total tumor slices: {len(tumor_slices)}")

    if tumor_slices:
        print(f"First tumor slice: {tumor_slices[0]}")
        print(f"Middle tumor slice: {tumor_slices[len(tumor_slices) // 2]}")
        print(f"Last tumor slice: {tumor_slices[-1]}")
        print(f"All tumor slices: {tumor_slices}")


if __name__ == "__main__":
    main()
