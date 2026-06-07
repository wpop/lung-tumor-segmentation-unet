import numpy as np

from src.config import IMAGES_TR_DIR, LABELS_TR_DIR
from src.data.dataset import LungTumorSliceDataset2D


def main() -> None:
    """
    Load the 2D training dataset and print basic information about the first sample.
    """
    dataset = LungTumorSliceDataset2D(IMAGES_TR_DIR, LABELS_TR_DIR)

    print(f"Dataset length: {len(dataset)}")

    # Load the first tumor-containing 2D slice.
    sample = dataset[0]
    image_slice = sample["image"]
    mask_slice = sample["mask"]
    slice_index = sample["slice_index"]

    print(f"Image shape: {image_slice.shape}")
    print(f"Mask shape: {mask_slice.shape}")
    print(f"Image dtype: {image_slice.dtype}")
    print(f"Mask dtype: {mask_slice.dtype}")
    print(f"Image min value: {image_slice.min()}")
    print(f"Image max value: {image_slice.max()}")
    print(f"Unique mask values: {np.unique(mask_slice)}")
    print(f"Slice index: {slice_index}")


if __name__ == "__main__":
    main()
